"""Torch-only RS05 servo model accepted for the canonical paper-walk runtime.

The arithmetic reproduces ``experiments/paper_walk/env.py:motor_force`` value
for value: a float32 PD demand, a float64 piecewise ceiling over the published
48 V speed knots, zero authority at and above 480 rpm, a clamp into
``[0, 1.6]`` N*m and a float32 result. The frozen admission scorer recomputes
the same three channels, so any change here breaks that comparison.

The 18 damping values are bound by joint name. The articulation reports its own
joint order, so the runtime gathers the damping for that order instead of
trusting a position. This module imports torch alone and runs on CPU.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "CEILING_NM_KNOTS",
    "DAMPING_NM_S_PER_RAD_BY_NAME",
    "MAX_APPLIED_NM",
    "SPEED_RPM_KNOTS",
    "STIFFNESS_NM_PER_RAD",
    "ZERO_AUTHORITY_RPM",
    "damping_vector",
    "effort_ceiling",
    "motor_effort",
]

#: Position gain of the accepted servo, newton metre per radian.
STIFFNESS_NM_PER_RAD = 12.0
#: Output speed knots of the provisional 48 V curve, revolutions per minute.
SPEED_RPM_KNOTS = (0.0, 70.0, 275.0, 340.0, 450.0, 477.0, 480.0)
#: Torque at each knot, newton metre, before the 1.6 N*m software clamp.
CEILING_NM_KNOTS = (5.5, 5.5, 4.0, 3.0, 1.6, 0.5, 0.0)
#: Provisional software clamp. The RS05 review keeps it an experimental value.
MAX_APPLIED_NM = 1.6
#: At and above this speed the motor delivers no torque.
ZERO_AUTHORITY_RPM = 480.0

#: Damping per named joint, newton metre second per radian. The values come
#: from the accepted prototype configuration and stay bound to these names.
DAMPING_NM_S_PER_RAD_BY_NAME = (
    ("lf_coxa_yaw", 0.44197696391810243),
    ("lf_femur_pitch", 0.2456657243160222),
    ("lf_tibia_pitch", 0.10552064613953134),
    ("lm_coxa_yaw", 0.4419789704596685),
    ("lm_femur_pitch", 0.24566676283633715),
    ("lm_tibia_pitch", 0.10552131615792909),
    ("lr_coxa_yaw", 0.441977634689341),
    ("lr_femur_pitch", 0.24566456721764315),
    ("lr_tibia_pitch", 0.10552100524837532),
    ("rf_coxa_yaw", 0.4419765017453795),
    ("rf_femur_pitch", 0.245667092769639),
    ("rf_tibia_pitch", 0.10552130774083751),
    ("rm_coxa_yaw", 0.44197471269634825),
    ("rm_femur_pitch", 0.24566700905274597),
    ("rm_tibia_pitch", 0.10552141334046235),
    ("rr_coxa_yaw", 0.4419772543481437),
    ("rr_femur_pitch", 0.24566767616660323),
    ("rr_tibia_pitch", 0.10552122889741367),
)


def damping_vector(joint_names, *, device=None, dtype=torch.float32):
    """Damping for ``joint_names`` in the order the caller gives."""

    table = dict(DAMPING_NM_S_PER_RAD_BY_NAME)
    names = list(joint_names)
    if len(names) != 18 or set(names) != set(table):
        raise ValueError("The paper-walk damping binds exactly the 18 named RS05 joints")
    return torch.tensor([table[name] for name in names], device=device, dtype=dtype)


def effort_ceiling(joint_velocity_rad_s):
    """Speed-dependent torque ceiling, newton metre, clamped into [0, 1.6]."""

    rpm = joint_velocity_rad_s.to(torch.float64).abs() * 60.0 / (2.0 * math.pi)
    device = joint_velocity_rad_s.device
    xp = torch.tensor(SPEED_RPM_KNOTS, dtype=torch.float64, device=device)
    yp = torch.tensor(CEILING_NM_KNOTS, dtype=torch.float64, device=device)
    index = torch.searchsorted(xp, rpm.contiguous(), right=True).clamp(1, 6)
    ceiling = yp[index - 1] + (rpm - xp[index - 1]) * (yp[index] - yp[index - 1]) / (xp[index] - xp[index - 1])
    ceiling = torch.where(rpm >= ZERO_AUTHORITY_RPM, 0.0, ceiling)
    return ceiling.clamp(min=0.0, max=MAX_APPLIED_NM).to(torch.float32)


def motor_effort(joint_pos, joint_vel, target_pos, damping):
    """Return the requested effort, the applied effort and the ceiling."""

    requested = STIFFNESS_NM_PER_RAD * (target_pos - joint_pos) - damping * joint_vel
    ceiling = effort_ceiling(joint_vel)
    return requested, torch.maximum(torch.minimum(requested, ceiling), -ceiling), ceiling
