"""Tripod gait phase, stance expectation, and swing clearance shaping."""

from __future__ import annotations

import math

import torch


def assign_tripod_pairs_from_foot_offsets(
    foot_offsets_b: torch.Tensor,
    *,
    lateral_axis_index: int = 0,
    forward_axis_index: int = 1,
    forward_sign: float = -1.0,
) -> torch.Tensor:
    """Split six feet into alternating tripods from base-frame geometry.

    Leg names in this robot do not encode physical sides, so the grouping is
    resolved from the default stance: feet are split by lateral sign, ranked
    fore-to-aft per side, and paired as tripod A = {left front, left hind,
    right mid} with tripod B as the complement.  Returns a boolean (6,) mask
    that is ``True`` for tripod A members, in input foot order.
    """

    if foot_offsets_b.ndim != 2 or foot_offsets_b.shape[0] != 6:
        raise ValueError(
            f"foot_offsets_b must have shape (6, 3), got {tuple(foot_offsets_b.shape)}"
        )
    if not torch.isfinite(foot_offsets_b).all():
        raise ValueError("foot_offsets_b must be finite")
    if forward_sign not in (-1.0, 1.0):
        raise ValueError(f"forward_sign must be -1.0 or 1.0, got {forward_sign!r}")
    lateral = foot_offsets_b[:, lateral_axis_index]
    forward = foot_offsets_b[:, forward_axis_index] * forward_sign
    left_side = lateral > 0.0
    if int(left_side.sum().item()) != 3:
        raise ValueError(
            "expected exactly three feet on each lateral side, got "
            f"{int(left_side.sum().item())} with positive lateral offset"
        )
    tripod_a = torch.zeros(6, dtype=torch.bool, device=foot_offsets_b.device)
    for side_mask, ranks_in_a in ((left_side, (0, 2)), (~left_side, (1,))):
        side_indices = torch.nonzero(side_mask, as_tuple=False).squeeze(-1)
        order = torch.argsort(forward[side_indices], descending=True)
        for rank in ranks_in_a:
            tripod_a[side_indices[order[rank]]] = True
    if int(tripod_a.sum().item()) != 3:
        raise ValueError("tripod assignment must select exactly three feet")
    return tripod_a


def tripod_expected_stance(
    gait_phase: torch.Tensor,
    tripod_a_mask: torch.Tensor,
    *,
    duty_factor: float,
    sharpness: float,
) -> torch.Tensor:
    """Smooth per-foot expected-stance weights in [0, 1] for a tripod clock.

    Tripod A's stance window starts at phase 0 and tripod B's at phase 0.5,
    each spanning ``duty_factor`` of the cycle.  The indicator is a sigmoid of
    circular distance from the window center, so the phase reward stays
    differentiable at the touchdown/liftoff transitions.
    """

    if gait_phase.ndim != 1:
        raise ValueError(f"gait_phase must be 1-D, got shape {tuple(gait_phase.shape)}")
    if tripod_a_mask.shape != (6,) or tripod_a_mask.dtype != torch.bool:
        raise ValueError("tripod_a_mask must be a boolean tensor of shape (6,)")
    if not 0.0 < duty_factor < 1.0:
        raise ValueError(f"duty_factor must be in (0, 1), got {duty_factor!r}")
    if not math.isfinite(sharpness) or sharpness <= 0.0:
        raise ValueError(f"sharpness must be finite and positive, got {sharpness!r}")
    offsets = torch.where(
        tripod_a_mask,
        torch.zeros(6, dtype=gait_phase.dtype, device=gait_phase.device),
        torch.full((6,), 0.5, dtype=gait_phase.dtype, device=gait_phase.device),
    )
    half_window = 0.5 * float(duty_factor)
    local_phase = torch.remainder(gait_phase.unsqueeze(-1) - offsets, 1.0)
    center_distance = torch.abs(local_phase - half_window)
    circular_distance = torch.minimum(center_distance, 1.0 - center_distance)
    return torch.sigmoid(float(sharpness) * (half_window - circular_distance))


def gait_phase_contact_reward(
    foot_contact: torch.Tensor,
    expected_stance: torch.Tensor,
) -> torch.Tensor:
    """Mean per-foot agreement between actual and expected contact, in [0, 1]."""

    if foot_contact.dtype != torch.bool:
        raise ValueError(f"foot_contact must be boolean, got {foot_contact.dtype}")
    if foot_contact.shape != expected_stance.shape:
        raise ValueError(
            "foot_contact and expected_stance must have matching shapes, got "
            f"{tuple(foot_contact.shape)} and {tuple(expected_stance.shape)}"
        )
    contact = foot_contact.to(expected_stance.dtype)
    agreement = expected_stance * contact + (1.0 - expected_stance) * (1.0 - contact)
    return torch.mean(agreement, dim=-1)


def swing_clearance_reward(
    foot_pad_heights_m: torch.Tensor,
    expected_stance: torch.Tensor,
    *,
    target_m: float,
    tolerance_m: float,
) -> torch.Tensor:
    """Swing-weighted Gaussian reward for lifting pads toward the apex target.

    Feet are weighted by their expected-swing probability so stance feet earn
    nothing; a foot dragged along the ground through its swing window scores
    near zero while a foot lifted to the target apex scores near one.
    """

    if foot_pad_heights_m.shape != expected_stance.shape:
        raise ValueError(
            "foot_pad_heights_m and expected_stance must have matching shapes, "
            f"got {tuple(foot_pad_heights_m.shape)} and {tuple(expected_stance.shape)}"
        )
    if not math.isfinite(target_m) or target_m <= 0.0:
        raise ValueError(f"target_m must be finite and positive, got {target_m!r}")
    if not math.isfinite(tolerance_m) or tolerance_m <= 0.0:
        raise ValueError(f"tolerance_m must be finite and positive, got {tolerance_m!r}")
    swing_weight = 1.0 - expected_stance
    normalized_error = (foot_pad_heights_m - float(target_m)) / float(tolerance_m)
    clearance_score = torch.exp(-torch.square(normalized_error))
    weighted = torch.sum(swing_weight * clearance_score, dim=-1)
    return weighted / torch.clamp(torch.sum(swing_weight, dim=-1), min=1.0e-6)


def advance_gait_phase(
    gait_phase: torch.Tensor,
    commanded_planar_speed: torch.Tensor,
    moving: torch.Tensor,
    *,
    step_dt: float,
    cycles_per_meter: float,
    min_frequency_hz: float,
    max_frequency_hz: float,
) -> torch.Tensor:
    """Advance and wrap the per-environment gait clock for moving commands.

    Frequency scales with commanded speed the way insect stride frequency
    does, bounded to a plausible band; standing environments hold phase so a
    later command resumes from a valid clock state.
    """

    if gait_phase.shape != commanded_planar_speed.shape or gait_phase.shape != moving.shape:
        raise ValueError("gait_phase, commanded_planar_speed, and moving must match shapes")
    if moving.dtype != torch.bool:
        raise ValueError(f"moving must be boolean, got {moving.dtype}")
    if not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ValueError(f"step_dt must be finite and positive, got {step_dt!r}")
    if not math.isfinite(cycles_per_meter) or cycles_per_meter <= 0.0:
        raise ValueError(f"cycles_per_meter must be finite and positive, got {cycles_per_meter!r}")
    if not (
        math.isfinite(min_frequency_hz)
        and math.isfinite(max_frequency_hz)
        and 0.0 < min_frequency_hz <= max_frequency_hz
    ):
        raise ValueError(
            f"invalid frequency band [{min_frequency_hz!r}, {max_frequency_hz!r}]"
        )
    frequency = torch.clamp(
        float(cycles_per_meter) * commanded_planar_speed,
        min=float(min_frequency_hz),
        max=float(max_frequency_hz),
    )
    advanced = gait_phase + float(step_dt) * frequency * moving.to(gait_phase.dtype)
    return torch.remainder(advanced, 1.0)
