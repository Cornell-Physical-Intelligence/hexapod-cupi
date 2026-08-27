"""Action-space conditioning: stand blending and joint-target slew limits."""

from __future__ import annotations

import math

import torch


def apply_command_conditioned_stand_action_scale(
    actions: torch.Tensor,
    commands: torch.Tensor,
    *,
    stand_action_scale: float,
    command_active_threshold: float,
) -> torch.Tensor:
    """Blend inactive-command policy actions toward the nominal joint pose.

    The returned tensor is the effective normalized action: rows with any
    x/y/yaw command strictly above ``command_active_threshold`` pass through
    unchanged, while fully inactive rows are multiplied by
    ``stand_action_scale``.  The inclusive threshold matches the task's
    existing inactive-axis convention.

    A scale of one is an exact legacy pass-through and intentionally does not
    inspect the dormant command or threshold arguments.
    """

    if (
        not math.isfinite(float(stand_action_scale))
        or not 0.0 <= float(stand_action_scale) <= 1.0
    ):
        raise ValueError(
            "stand_action_scale must be finite and in [0, 1], "
            f"got {stand_action_scale!r}"
        )
    if float(stand_action_scale) == 1.0:
        return actions
    if actions.ndim < 1 or actions.shape[-1] == 0:
        raise ValueError("actions must have a non-empty action dimension")
    if commands.ndim < 1 or commands.shape[-1] != 3:
        raise ValueError("commands must have exactly three x/y/yaw components")
    if actions.shape[:-1] != commands.shape[:-1]:
        raise ValueError("actions and commands must have matching leading dimensions")
    if (
        not math.isfinite(float(command_active_threshold))
        or float(command_active_threshold) < 0.0
    ):
        raise ValueError(
            "command_active_threshold must be finite and non-negative, "
            f"got {command_active_threshold!r}"
        )

    command_active = torch.any(
        torch.abs(commands) > float(command_active_threshold), dim=-1
    )
    scaled_stand_actions = actions * float(stand_action_scale)
    return torch.where(command_active.unsqueeze(-1), actions, scaled_stand_actions)


def limit_processed_joint_target_slew(
    target: torch.Tensor,
    previous_target: torch.Tensor,
    has_previous_target: torch.Tensor,
    *,
    max_delta_rad_per_20ms: float | None,
    step_dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Limit processed target change and return per-environment intervention.

    ``None`` is an exact pass-through that deliberately does not inspect the
    history tensors.  When enabled, the configured radian delta is normalized
    to a 20 ms policy step and scaled by the actual policy timestep.
    """

    if max_delta_rad_per_20ms is None:
        return target, torch.zeros(target.shape[:-1], dtype=target.dtype, device=target.device)
    if target.shape != previous_target.shape:
        raise ValueError("target and previous_target must have matching shapes")
    if has_previous_target.shape != target.shape[:-1]:
        raise ValueError(
            "has_previous_target must match the target leading dimensions"
        )
    if has_previous_target.dtype != torch.bool:
        raise ValueError(
            f"has_previous_target must be boolean, got {has_previous_target.dtype}"
        )
    if (
        not math.isfinite(float(max_delta_rad_per_20ms))
        or float(max_delta_rad_per_20ms) <= 0.0
    ):
        raise ValueError(
            "max_delta_rad_per_20ms must be finite and positive when enabled, "
            f"got {max_delta_rad_per_20ms!r}"
        )
    if not math.isfinite(float(step_dt)) or float(step_dt) <= 0.0:
        raise ValueError(f"step_dt must be finite and positive, got {step_dt!r}")

    allowed_delta = float(max_delta_rad_per_20ms) * float(step_dt) / 0.020
    bounded_delta = torch.clamp(
        target - previous_target, min=-allowed_delta, max=allowed_delta
    )
    candidate = previous_target + bounded_delta
    history_mask = has_previous_target.unsqueeze(-1)
    limited_target = torch.where(history_mask, candidate, target)
    joint_was_limited = history_mask & (torch.abs(target - limited_target) > 1.0e-12)
    limited_fraction = torch.mean(joint_was_limited.to(target.dtype), dim=-1)
    return limited_target, limited_fraction
