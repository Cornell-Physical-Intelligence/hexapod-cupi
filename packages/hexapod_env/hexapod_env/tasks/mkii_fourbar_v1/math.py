"""Torch-only physical motor-coordinate math; no simulator or gait clock."""
from __future__ import annotations

import torch

from hexapod_core import fourbar_v1 as contract
from ...command_sampling import body_to_navigation_frame
from ...rewards.actions import limit_processed_joint_target_slew


class MotorCoordinates:
    def __init__(self, tree_names, kinematics, *, device="cpu", dtype=torch.float32):
        self.names = contract.validate_tree_names(tree_names)
        self.kinematics = contract.validate_kinematics(kinematics)
        self.active_indices = torch.tensor(contract.active_indices(tree_names), device=device, dtype=torch.long)
        active_lookup = {name: index for index, name in enumerate(contract.ACTIVE_JOINT_NAMES)}
        sources, multipliers, offsets = [], [], []
        for name in self.names:
            relation = kinematics["passive_relations"].get(name)
            sources.append(active_lookup[relation["source_joint"] if relation else name])
            multipliers.append(relation["multiplier"] if relation else 1.)
            offsets.append(relation["offset_rad"] if relation else 0.)
        self.sources = torch.tensor(sources, device=device, dtype=torch.long)
        self.multipliers = torch.tensor(multipliers, device=device, dtype=dtype)
        self.offsets = torch.tensor(offsets, device=device, dtype=dtype)
        self.default = torch.tensor([kinematics["default_joint_positions_rad"][name]
                                     for name in contract.ACTIVE_JOINT_NAMES], device=device, dtype=dtype)
        self.soft_limits = torch.tensor(contract.soft_limits(kinematics), device=device, dtype=dtype)
        self.hard_limits = torch.tensor([kinematics["joint_limits_rad"][name] for name in self.names],
                                       device=device, dtype=dtype)

    def gather(self, value):
        if value.shape[-1] != 30:
            raise ValueError("Motor gathering requires all 30 tree coordinates")
        return value.index_select(-1, self.active_indices)

    def closed_reset(self, active_positions, active_velocities):
        if active_positions.shape != active_velocities.shape or active_positions.shape[-1] != 18:
            raise ValueError("Closed reset requires matching 18-motor position/velocity arrays")
        q = active_positions.index_select(-1, self.sources) * self.multipliers + self.offsets
        qd = active_velocities.index_select(-1, self.sources) * self.multipliers
        if not bool(torch.isfinite(q).all() & torch.isfinite(qd).all()):
            raise ValueError("Nonfinite closed reset state")
        if bool(((q < self.hard_limits[:, 0]) | (q > self.hard_limits[:, 1])).any()):
            raise ValueError("Closed reset violates a physical tree joint limit")
        return q, qd

    def closure_coordinate_error(self, tree_positions):
        if tree_positions.shape[-1] != 30:
            raise ValueError("Closure coordinates require all 30 tree joints")
        expected = self.gather(tree_positions).index_select(-1, self.sources) * self.multipliers + self.offsets
        return tree_positions - expected

    def process_action(self, action, prior_target, *, step_dt):
        if (action.shape[-1] != 18 or prior_target.shape != action.shape
                or not bool(torch.isfinite(action).all() & torch.isfinite(prior_target).all())):
            raise ValueError("Actions/target history must be finite matching 18-motor arrays")
        effective = action.clamp(-1., 1.)
        targets = (self.default + contract.ACTION_SCALE_RAD * effective).clamp(
            min=self.soft_limits[:, 0], max=self.soft_limits[:, 1])
        target, fraction = limit_processed_joint_target_slew(targets, prior_target,
            torch.ones(action.shape[:-1], dtype=torch.bool, device=action.device),
            max_delta_rad_per_20ms=contract.SLEW_RAD_PER_20MS, step_dt=step_dt)
        return effective, target, fraction


def observations(root_linear_body, root_angular_body, gravity_body, commands,
                 active_position_error, active_velocity, actions, burst_headroom):
    leading = commands.shape[:-1]
    blocks = [body_to_navigation_frame(root_linear_body), body_to_navigation_frame(root_angular_body),
              body_to_navigation_frame(gravity_body), commands, active_position_error, active_velocity,
              actions, burst_headroom]
    for (name, width), value in zip(contract.OBSERVATION_FIELDS, blocks):
        if value.shape != (*leading, width):
            raise ValueError(f"Incorrect physical observation shape for {name}")
    return torch.cat(blocks, dim=-1)


def reward_terms(*, command, linear_navigation, angular_navigation, gravity_body,
                 active_torque, active_velocity, active_acceleration, active_position,
                 soft_limits, action, previous_action, height, nominal_height,
                 nonfoot_contacts, support_count, foot_slip, clipping_nm, overload_nm):
    """Score motor coordinates only; passive reaction torques are not motors."""
    motor_arrays = (active_torque, active_velocity, active_acceleration, active_position,
                    action, previous_action, clipping_nm, overload_nm)
    if any(value.shape != active_torque.shape or value.shape[-1] != 18 for value in motor_arrays):
        raise ValueError("Motor rewards require exactly 18 active coordinates")
    violations = torch.relu(soft_limits[:, 0]-active_position) + torch.relu(active_position-soft_limits[:, 1])
    return {
        "track_linear": 3.0 * torch.exp(-((linear_navigation[:, :2]-command[:, :2]).square().sum(-1))/.0225),
        "track_yaw": .6 * torch.exp(-(angular_navigation[:, 2]-command[:, 2]).square()/.09),
        "vertical_velocity": -1.5 * linear_navigation[:, 2].square(),
        "roll_pitch_rate": -.1 * angular_navigation[:, :2].square().sum(-1),
        "orientation": -2. * gravity_body[:, :2].square().sum(-1),
        "height": -10. * (height-nominal_height).square(),
        "motor_torque": -1e-4 * active_torque.square().sum(-1),
        "motor_work": -1e-3 * (active_torque*active_velocity).abs().sum(-1),
        "motor_acceleration": -2.5e-7 * active_acceleration.square().sum(-1),
        "motor_clipping": -.01 * clipping_nm.square().sum(-1),
        "continuous_overload": -.01 * overload_nm.square().sum(-1),
        "action_rate": -.02 * (action-previous_action).square().sum(-1),
        "motor_limits": -.25 * violations.sum(-1),
        "nonfoot_contacts": -1. * nonfoot_contacts.float(),
        "support_shortfall": -.15 * (3-support_count).clamp(min=0).float(),
        "foot_slip": -.1 * foot_slip,
    }
