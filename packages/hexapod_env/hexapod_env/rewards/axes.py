"""Signed-axis error shaping and reset-safe running statistics."""

from __future__ import annotations

import math

import torch


def reset_safe_exponential_moving_average(
    current: torch.Tensor,
    previous: torch.Tensor,
    has_previous: torch.Tensor,
    *,
    step_dt: float,
    tau_s: float,
) -> torch.Tensor:
    """Update an EMA without carrying history across reset boundaries.

    Entries without valid history initialize exactly to the current sample.
    The exponential coefficient is derived from physical time, so changing the
    policy rate does not change the filter time constant.
    """

    if current.shape != previous.shape or has_previous.shape != current.shape:
        raise ValueError(
            "current, previous, and has_previous must have matching shapes"
        )
    if has_previous.dtype != torch.bool:
        raise ValueError(f"has_previous must be boolean, got {has_previous.dtype}")
    if not math.isfinite(float(step_dt)) or float(step_dt) <= 0.0:
        raise ValueError(f"step_dt must be finite and positive, got {step_dt!r}")
    if not math.isfinite(float(tau_s)) or float(tau_s) <= 0.0:
        raise ValueError(f"tau_s must be finite and positive, got {tau_s!r}")
    alpha = -math.expm1(-float(step_dt) / float(tau_s))
    filtered = previous + alpha * (current - previous)
    return torch.where(has_previous, filtered, current)


def capped_normalized_axis_error(
    command: torch.Tensor,
    achieved: torch.Tensor,
    *,
    active_threshold: float,
    error_cap: float,
) -> torch.Tensor:
    """Return command-normalized absolute error, masked and capped.

    Dividing by the active command magnitude gives zero at the target and one
    at zero response.  The finite cap keeps a transient or wrong-sign outlier
    from dominating the acquisition objective.
    """

    if command.shape != achieved.shape:
        raise ValueError(
            f"command and achieved must have matching shapes, got "
            f"{command.shape!r} and {achieved.shape!r}"
        )
    if not math.isfinite(float(active_threshold)) or float(active_threshold) <= 0.0:
        raise ValueError(
            "active_threshold must be finite and positive, "
            f"got {active_threshold!r}"
        )
    if not math.isfinite(float(error_cap)) or float(error_cap) <= 0.0:
        raise ValueError(f"error_cap must be finite and positive, got {error_cap!r}")
    active = torch.abs(command) > float(active_threshold)
    denominator = torch.clamp(torch.abs(command), min=float(active_threshold))
    normalized_error = torch.abs(command - achieved) / denominator
    normalized_error = torch.nan_to_num(
        normalized_error,
        nan=float(error_cap),
        posinf=float(error_cap),
        neginf=float(error_cap),
    )
    normalized_error = torch.clamp(normalized_error, min=0.0, max=float(error_cap))
    return torch.where(active, normalized_error, torch.zeros_like(normalized_error))


def bounded_inactive_yaw_rate_slew_cost(
    yaw_rate: torch.Tensor,
    previous_yaw_rate: torch.Tensor,
    has_previous_yaw_rate: torch.Tensor,
    yaw_command_active: torch.Tensor,
    *,
    reference_rad_s_per_step: float,
) -> torch.Tensor:
    """Return a bounded yaw-rate slew cost for uncommanded-yaw steps.

    The Cauchy-shaped cost is 0.5 when the yaw-rate change equals
    ``reference_rad_s_per_step`` and approaches one for larger changes.  A
    sample contributes only when both the current and previous steps have an
    inactive yaw command.  Invalid history therefore masks reset boundaries
    and the first inactive step following an active yaw command.
    """

    if yaw_rate.shape != previous_yaw_rate.shape:
        raise ValueError("yaw_rate and previous_yaw_rate must have matching shapes")
    if has_previous_yaw_rate.shape != yaw_rate.shape:
        raise ValueError("has_previous_yaw_rate must match yaw_rate shape")
    if yaw_command_active.shape != yaw_rate.shape:
        raise ValueError("yaw_command_active must match yaw_rate shape")
    if has_previous_yaw_rate.dtype != torch.bool:
        raise ValueError(
            "has_previous_yaw_rate must be boolean, "
            f"got {has_previous_yaw_rate.dtype}"
        )
    if yaw_command_active.dtype != torch.bool:
        raise ValueError(
            f"yaw_command_active must be boolean, got {yaw_command_active.dtype}"
        )
    if (
        not math.isfinite(float(reference_rad_s_per_step))
        or float(reference_rad_s_per_step) <= 0.0
    ):
        raise ValueError(
            "reference_rad_s_per_step must be finite and positive, "
            f"got {reference_rad_s_per_step!r}"
        )

    normalized_delta = torch.clamp(
        torch.nan_to_num(
            (yaw_rate - previous_yaw_rate) / float(reference_rad_s_per_step),
            nan=1.0e6,
            posinf=1.0e6,
            neginf=-1.0e6,
        ),
        min=-1.0e6,
        max=1.0e6,
    )
    normalized_delta_l2 = torch.square(normalized_delta)
    unmasked_cost = normalized_delta_l2 / (1.0 + normalized_delta_l2)
    valid_inactive_history = has_previous_yaw_rate & ~yaw_command_active
    return torch.where(
        valid_inactive_history, unmasked_cost, torch.zeros_like(unmasked_cost)
    )
