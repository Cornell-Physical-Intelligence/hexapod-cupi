"""Deck-stability scoring and command-conditioned stance targets."""

from __future__ import annotations

import math

import torch


def normalized_deck_stability_reward(
    root_vertical_velocity_w: torch.Tensor,
    roll_pitch_angular_velocity: torch.Tensor,
    projected_gravity_xy: torch.Tensor,
    base_height_error: torch.Tensor,
    *,
    vertical_velocity_scale_mps: float,
    roll_pitch_rate_scale_rad_s: float,
    projected_gravity_xy_scale: float,
    height_error_scale_m: float,
) -> torch.Tensor:
    """Return an equal-weight, bounded deck-stability score in ``[0, 1]``.

    ``root_vertical_velocity_w`` is the world-frame Z velocity. Using the
    body-frame Z component would mix fore/aft motion into this platform-bob
    measure whenever the deck is pitched or rolled.

    Each component is a Cauchy score, ``1 / (1 + normalized_error**2)``.
    Vector components use their mean squared normalized error so roll/pitch and
    gravity tilt have the same total weight as vertical speed and base height.
    A normalization scale is therefore the scalar/RMS error at which that
    component contributes 0.5.  Non-finite state is treated as maximally bad.
    """

    scales = {
        "vertical_velocity_scale_mps": vertical_velocity_scale_mps,
        "roll_pitch_rate_scale_rad_s": roll_pitch_rate_scale_rad_s,
        "projected_gravity_xy_scale": projected_gravity_xy_scale,
        "height_error_scale_m": height_error_scale_m,
    }
    for name, scale in scales.items():
        if not math.isfinite(float(scale)) or float(scale) <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {scale!r}")

    def component(error: torch.Tensor, scale: float) -> torch.Tensor:
        normalized_error = torch.clamp(
            torch.nan_to_num(
                error / scale,
                nan=1.0e6,
                posinf=1.0e6,
                neginf=-1.0e6,
            ),
            min=-1.0e6,
            max=1.0e6,
        )
        normalized_error_l2 = torch.square(normalized_error)
        if normalized_error_l2.ndim > 1:
            normalized_error_l2 = torch.mean(normalized_error_l2, dim=-1)
        return torch.reciprocal(1.0 + normalized_error_l2)

    component_scores = torch.stack(
        (
            component(root_vertical_velocity_w, vertical_velocity_scale_mps),
            component(roll_pitch_angular_velocity, roll_pitch_rate_scale_rad_s),
            component(projected_gravity_xy, projected_gravity_xy_scale),
            component(base_height_error, height_error_scale_m),
        ),
        dim=0,
    )
    return torch.clamp(torch.mean(component_scores, dim=0), min=0.0, max=1.0)


def select_command_conditioned_nominal_height(
    commanded_planar_speed: torch.Tensor,
    *,
    moving_nominal_height_m: float,
    stand_nominal_height_m: float | None,
    moving_command_threshold_mps: float,
) -> torch.Tensor:
    """Select the moving or stand base-height target for each environment.

    ``None`` returns the moving/legacy target without inspecting the opt-in
    threshold. When a stand target is configured, planar command magnitudes at
    or below ``moving_command_threshold_mps`` use it; larger commands keep
    ``moving_nominal_height_m``.
    """

    moving_height = torch.full_like(
        commanded_planar_speed, float(moving_nominal_height_m)
    )
    if stand_nominal_height_m is None:
        return moving_height
    values = {
        "moving_nominal_height_m": moving_nominal_height_m,
        "stand_nominal_height_m": stand_nominal_height_m,
    }
    for name, value in values.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")
    if (
        not math.isfinite(float(moving_command_threshold_mps))
        or float(moving_command_threshold_mps) < 0.0
    ):
        raise ValueError(
            "moving_command_threshold_mps must be finite and non-negative, "
            f"got {moving_command_threshold_mps!r}"
        )
    stand_height = torch.full_like(
        commanded_planar_speed, float(stand_nominal_height_m)
    )
    return torch.where(
        commanded_planar_speed <= float(moving_command_threshold_mps),
        stand_height,
        moving_height,
    )


def select_support_contact_target(
    commanded_planar_speed: torch.Tensor,
    *,
    scalar_target: float,
    speed_conditioning_enabled: bool,
    low_speed_threshold_mps: float,
    high_speed_threshold_mps: float,
    low_speed_target: float,
    medium_speed_target: float,
    high_speed_target: float,
) -> torch.Tensor:
    """Select scalar or low/medium/high support targets per environment.

    Low speed includes its threshold, high speed includes its threshold, and
    the open interval between them uses the medium target.  When conditioning
    is disabled only ``scalar_target`` is used, preserving the legacy path.
    """

    if not speed_conditioning_enabled:
        return torch.full_like(commanded_planar_speed, float(scalar_target))
    if (
        not math.isfinite(float(low_speed_threshold_mps))
        or not math.isfinite(float(high_speed_threshold_mps))
        or float(low_speed_threshold_mps) < 0.0
        or float(high_speed_threshold_mps) <= float(low_speed_threshold_mps)
    ):
        raise ValueError(
            "support-contact speed thresholds must be finite, non-negative, "
            "and strictly increasing"
        )
    targets = (low_speed_target, medium_speed_target, high_speed_target)
    if any(not math.isfinite(float(target)) or float(target) < 0.0 for target in targets):
        raise ValueError("support-contact targets must be finite and non-negative")
    low_target = torch.full_like(commanded_planar_speed, float(low_speed_target))
    medium_target = torch.full_like(commanded_planar_speed, float(medium_speed_target))
    high_target = torch.full_like(commanded_planar_speed, float(high_speed_target))
    return torch.where(
        commanded_planar_speed <= low_speed_threshold_mps,
        low_target,
        torch.where(
            commanded_planar_speed >= high_speed_threshold_mps,
            high_target,
            medium_target,
        ),
    )
