"""A bounded Isaac Lab ray-pattern surrogate for the Livox Mid-360.

Livox does not publish the Mid-360's non-repetitive temporal scan trajectory,
and Isaac Lab 3.0 has no native Mid-360/RTX profile.  This pattern preserves
the published field of view, 200 kpoints/s first-return rate, 10 Hz frame rate,
and 0.15-degree 1-sigma angular-precision envelope while using a deterministic
low-discrepancy spatial distribution.  The same 20,000 directions repeat each
simulated frame; it is not a claim to reproduce Livox timing, occlusion
accumulation, or intensity.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils.configclass import configclass


def mid360_surrogate_pattern(
    cfg: "Mid360SurrogatePatternCfg", device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one low-discrepancy hemispherical scan in the sensor frame."""
    points_per_scan = int(round(cfg.points_per_second / cfg.scan_rate_hz))
    if points_per_scan < 1:
        raise ValueError("Mid-360 surrogate must produce at least one ray per scan.")

    index = torch.arange(points_per_scan, dtype=torch.float32, device=device)
    unit_index = (index + 0.5) / points_per_scan
    min_elevation = math.radians(cfg.vertical_fov_range_deg[0])
    max_elevation = math.radians(cfg.vertical_fov_range_deg[1])

    # Uniform-in-solid-angle elevation and a golden-ratio azimuth sequence give
    # even hemispherical coverage without pretending the real Livox trajectory
    # is a stack of spinning rings.
    sin_elevation = math.sin(min_elevation) + unit_index * (
        math.sin(max_elevation) - math.sin(min_elevation)
    )
    elevation = torch.asin(sin_elevation)
    golden_ratio_conjugate = (math.sqrt(5.0) - 1.0) / 2.0
    azimuth = -math.pi + 2.0 * math.pi * torch.remainder(
        index * golden_ratio_conjugate, 1.0
    )

    cos_elevation = torch.cos(elevation)
    directions = torch.stack(
        (
            cos_elevation * torch.cos(azimuth),
            cos_elevation * torch.sin(azimuth),
            torch.sin(elevation),
        ),
        dim=-1,
    )

    # Apply a fixed zero-mean Gaussian calibration perturbation in each ray's
    # tangent plane. Livox publishes angular precision as a one-sigma quantity,
    # not a hard bound, so retain Gaussian tails but clip this static surrogate
    # at a configurable number of sigma. Per-return temporal angular randomness
    # cannot be represented by a static RayCaster pattern.
    hash_a = torch.clamp(
        torch.remainder(
            torch.sin((index + cfg.seed) * 12.9898) * 43758.5453, 1.0
        ),
        min=1.0e-7,
    )
    hash_b = torch.remainder(
        torch.sin((index + cfg.seed) * 78.233) * 12345.6789, 1.0
    )
    hash_c = torch.remainder(
        torch.sin((index + cfg.seed) * 37.719) * 24634.6345, 1.0
    )
    standard_normal = torch.sqrt(-2.0 * torch.log(hash_a)) * torch.cos(
        2.0 * math.pi * hash_b
    )
    standard_normal = torch.clamp(
        standard_normal,
        -cfg.angular_error_clip_sigma,
        cfg.angular_error_clip_sigma,
    )
    angular_offset = math.radians(cfg.angular_error_std_deg) * standard_normal
    tangent_phase = 2.0 * math.pi * hash_c
    elevation_tangent = torch.stack(
        (
            -torch.sin(elevation) * torch.cos(azimuth),
            -torch.sin(elevation) * torch.sin(azimuth),
            torch.cos(elevation),
        ),
        dim=-1,
    )
    azimuth_tangent = torch.stack(
        (-torch.sin(azimuth), torch.cos(azimuth), torch.zeros_like(azimuth)),
        dim=-1,
    )
    tangent = (
        torch.cos(tangent_phase).unsqueeze(-1) * elevation_tangent
        + torch.sin(tangent_phase).unsqueeze(-1) * azimuth_tangent
    )
    directions = (
        torch.cos(angular_offset).unsqueeze(-1) * directions
        + torch.sin(angular_offset).unsqueeze(-1) * tangent
    )
    directions = torch.nn.functional.normalize(directions, dim=-1)
    return torch.zeros_like(directions), directions


@configclass
class Mid360SurrogatePatternCfg(PatternBaseCfg):
    """Published Mid-360 envelope represented through Isaac Lab's pattern API."""

    func: Callable = mid360_surrogate_pattern
    points_per_second: int = 200_000
    scan_rate_hz: float = 10.0
    vertical_fov_range_deg: tuple[float, float] = (-7.0, 52.0)
    angular_error_std_deg: float = 0.15
    angular_error_clip_sigma: float = 3.0
    seed: int = 360

    def __post_init__(self) -> None:
        if self.points_per_second <= 0 or self.scan_rate_hz <= 0.0:
            raise ValueError("Point and scan rates must be positive.")
        if not -90.0 <= self.vertical_fov_range_deg[0] < self.vertical_fov_range_deg[1] <= 90.0:
            raise ValueError("Vertical field of view must lie within [-90, 90] degrees.")
        if not 0.0 <= self.angular_error_std_deg <= 1.0:
            raise ValueError("Angular error standard deviation is outside the supported range.")
        if not 1.0 <= self.angular_error_clip_sigma <= 6.0:
            raise ValueError("Angular error clipping must lie within [1, 6] sigma.")
