"""Torch-only action, observation and reward mathematics of the RS05 task.

Every function here reproduces the accepted prototype in
``experiments/paper_walk/env.py``: the held target with its double-precision
slew bound, the 42-value proprioceptive row, the 231-value policy observation
with five history frames, the 234-value critic observation and the six reward
terms of one control step. CPU tests compare these values with that frozen
source, so a change here changes a recorded comparison.

This module imports torch and the shared frame helper alone. It loads on a
machine without Isaac Lab.
"""

from __future__ import annotations

import math

import torch

from ...command_sampling import body_to_navigation_frame

__all__ = [
    "ACTION_SCALE_RAD",
    "ACTION_WIDTH",
    "COMMAND_WIDTH",
    "CRITIC_OBSERVATION_WIDTH",
    "DECIMATION",
    "EPISODE_SECONDS",
    "ENV_SPACING_M",
    "HISTORY_LENGTH",
    "PHYSICS_DT_S",
    "POLICY_OBSERVATION_WIDTH",
    "PROPRIO_WIDTH",
    "SOLVER_POSITION_ITERATIONS",
    "SOLVER_VELOCITY_ITERATIONS",
    "TARGET_SLEW_RAD",
    "advance_history",
    "critic_observation",
    "emitted_target",
    "executed_action_feature",
    "policy_observation",
    "proprio_row",
    "reward_terms",
    "termination_flags",
]

PHYSICS_DT_S = 0.0025
DECIMATION = 8
EPISODE_SECONDS = 20.0
ENV_SPACING_M = 2.0
ACTION_SCALE_RAD = 0.35
#: Target travel allowed per 20 ms control step. Every formal comparison keeps
#: this limiter, so it never becomes a tuning parameter.
TARGET_SLEW_RAD = 0.040
SOLVER_POSITION_ITERATIONS = 32
SOLVER_VELOCITY_ITERATIONS = 0

ACTION_WIDTH = 18
COMMAND_WIDTH = 3
PROPRIO_WIDTH = 42
HISTORY_LENGTH = 5
POLICY_OBSERVATION_WIDTH = HISTORY_LENGTH * PROPRIO_WIDTH + COMMAND_WIDTH + ACTION_WIDTH
CRITIC_OBSERVATION_WIDTH = POLICY_OBSERVATION_WIDTH + 3

#: Body tilt at which the episode ends, radians from upright.
MAX_TILT_RAD = 0.85
#: Plate height at which the episode ends, metres above the local ground.
MIN_PLATE_HEIGHT_M = 0.045
#: Tolerance on a hard joint limit before the episode ends, radians.
JOINT_LIMIT_TOLERANCE_RAD = 2e-6


def emitted_target(joint_action, held, neutral, lower, upper,
                   scale=ACTION_SCALE_RAD, slew=TARGET_SLEW_RAD):
    """Hold-limited joint target for one control step.

    The requested target is the stance plus the scaled action, clamped to the
    hard URDF limits. The result then stays inside the previous target plus or
    minus the slew bound. The bound is applied in float64 and the two nextafter
    corrections keep the float32 result inside it.
    """

    requested = (neutral + scale * joint_action.clamp(-1, 1)).clamp(lower, upper)
    low = torch.maximum(lower.to(torch.float64), held.to(torch.float64) - slew)
    high = torch.minimum(upper.to(torch.float64), held.to(torch.float64) + slew)
    target = torch.maximum(torch.minimum(requested.to(torch.float64), high), low).to(torch.float32)
    target = torch.where(target.to(torch.float64) > high,
                         torch.nextafter(target, torch.full_like(target, -torch.inf)), target)
    target = torch.where(target.to(torch.float64) < low,
                         torch.nextafter(target, torch.full_like(target, torch.inf)), target)
    return target


def executed_action_feature(target, neutral, scale=ACTION_SCALE_RAD):
    """The held target expressed in action units, after clamp and slew."""

    return (target - neutral) / scale


def proprio_row(angular_body, gravity_body, joint_pos, joint_vel, neutral):
    """One 42-value proprioceptive frame in canonical joint order."""

    blocks = (
        body_to_navigation_frame(angular_body),
        body_to_navigation_frame(gravity_body),
        joint_pos - neutral,
        joint_vel,
    )
    row = torch.cat(blocks, dim=-1)
    if row.shape[-1] != PROPRIO_WIDTH:
        raise ValueError(f"The proprioceptive row must hold {PROPRIO_WIDTH} values")
    return row


def advance_history(history, row):
    """Drop the oldest frame and append ``row`` as the newest one."""

    if history.shape[-2:] != (HISTORY_LENGTH, PROPRIO_WIDTH) or row.shape[-1] != PROPRIO_WIDTH:
        raise ValueError("The history holds five 42-value frames")
    return torch.cat((history[:, 1:], row[:, None]), dim=1)


def policy_observation(history, commands, previous_action):
    """History frames, then the command, then the previous executed action."""

    if history.shape[-2:] != (HISTORY_LENGTH, PROPRIO_WIDTH):
        raise ValueError("The history holds five 42-value frames")
    if commands.shape[-1] != COMMAND_WIDTH or previous_action.shape[-1] != ACTION_WIDTH:
        raise ValueError("The command holds three values and the action holds 18")
    return torch.cat((history.flatten(1), commands, previous_action), dim=-1)


def critic_observation(policy, linear_body):
    """The policy observation plus the privileged base velocity."""

    if policy.shape[-1] != POLICY_OBSERVATION_WIDTH or linear_body.shape[-1] != 3:
        raise ValueError("The critic observation extends the policy observation by three values")
    return torch.cat((policy, body_to_navigation_frame(linear_body)), dim=-1)


def reward_terms(*, commands, linear_body, angular_body, gravity_body, torque_square_sum,
                 target, previous_target, terminated, substeps=DECIMATION, slew=TARGET_SLEW_RAD):
    """Return the six reward components and the total of one control step."""

    linear_navigation = body_to_navigation_frame(linear_body)
    angular_navigation = body_to_navigation_frame(angular_body)
    tracking = torch.exp(-((linear_navigation[:, :2] - commands[:, :2]).square().sum(-1)) / .01)
    yaw_tracking = torch.exp(-(angular_navigation[:, 2] - commands[:, 2]).square() / .25)
    tilt_cost = gravity_body[:, :2].square().sum(-1)
    torque_cost = torque_square_sum.mean(-1) / substeps
    slew_cost = ((target - previous_target) / slew).square().mean(-1)
    vertical_cost = linear_navigation[:, 2].square()
    total = (tracking + .3 * yaw_tracking - .5 * tilt_cost - .005 * torque_cost
             - .01 * slew_cost - .05 * vertical_cost)
    total = total - 2.0 * terminated
    components = {"tracking": tracking, "yaw_tracking": yaw_tracking, "tilt_cost": tilt_cost,
                  "torque_cost": torque_cost, "slew_cost": slew_cost, "vertical_cost": vertical_cost}
    return components, total


def termination_flags(*, root_height, gravity_body, joint_pos, lower, upper):
    """Return the termination flag and the three reasons behind it."""

    joint_violation = ((joint_pos < lower - JOINT_LIMIT_TOLERANCE_RAD)
                       | (joint_pos > upper + JOINT_LIMIT_TOLERANCE_RAD)).any(-1)
    low_plate = root_height < MIN_PLATE_HEIGHT_M
    tilted = gravity_body[:, 2] > -math.cos(MAX_TILT_RAD)
    reasons = {"low_plate": low_plate, "tilted": tilted, "joint_violation": joint_violation}
    return low_plate | tilted | joint_violation, reasons
