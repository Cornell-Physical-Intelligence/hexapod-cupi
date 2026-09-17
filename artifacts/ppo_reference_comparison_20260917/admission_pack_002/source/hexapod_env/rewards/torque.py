"""Actuator torque slew and rated-torque excess penalties."""

from __future__ import annotations

import math

import torch


def applied_torque_slew_l2(
    applied_torque: torch.Tensor,
    previous_applied_torque: torch.Tensor,
    has_previous_applied_torque: torch.Tensor,
) -> torch.Tensor:
    """Return squared torque-step change, masked across reset boundaries."""

    return torch.sum(
        torch.square(applied_torque - previous_applied_torque), dim=-1
    ) * has_previous_applied_torque.float()


def max_joint_rated_torque_excess_l2(
    computed_torque: torch.Tensor,
    *,
    rated_torque_nm: float,
) -> torch.Tensor:
    """Return each environment's worst single-joint squared torque excess.

    The excess is measured from raw computed PD demand, before actuator
    clipping. Taking the maximum across joints prevents one overloaded motor
    from being diluted by the 17 joints that remain below the continuous
    rating.
    """

    if computed_torque.ndim < 1 or computed_torque.shape[-1] == 0:
        raise ValueError("computed_torque must have a non-empty joint dimension")
    if not math.isfinite(float(rated_torque_nm)) or float(rated_torque_nm) < 0.0:
        raise ValueError("rated_torque_nm must be finite and non-negative")
    excess = torch.relu(torch.abs(computed_torque) - float(rated_torque_nm))
    return torch.amax(torch.square(excess), dim=-1)


def max_joint_rated_torque_excess_l1(
    computed_torque: torch.Tensor,
    *,
    rated_torque_nm: float,
) -> torch.Tensor:
    """Return each environment's worst single-joint linear torque excess."""

    if computed_torque.ndim < 1 or computed_torque.shape[-1] == 0:
        raise ValueError("computed_torque must have a non-empty joint dimension")
    if not math.isfinite(float(rated_torque_nm)) or float(rated_torque_nm) < 0.0:
        raise ValueError("rated_torque_nm must be finite and non-negative")
    excess = torch.relu(torch.abs(computed_torque) - float(rated_torque_nm))
    return torch.amax(excess, dim=-1)
