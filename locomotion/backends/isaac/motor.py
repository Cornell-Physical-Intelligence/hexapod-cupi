"""Explicit Isaac Lab actuator with the approved speed-dependent torque curve."""
import torch
from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg
from isaaclab.utils import configclass
from .config import JOINT_NAMES, motor_damping
from ...core.motor import motor_force


class Motor(ActuatorBase):
    """Compute the canonical PD torque once per 400 Hz scene write."""

    def reset(self, env_ids):
        self.computed_effort[env_ids] = 0
        self.applied_effort[env_ids] = 0

    def compute(self, control_action, joint_pos, joint_vel):
        if not bool(torch.isfinite(joint_pos).all() and torch.isfinite(joint_vel).all()):
            raise FloatingPointError("Nonfinite pre-servo state")
        self.pre_position = joint_pos.clone()
        self.pre_velocity = joint_vel.clone()
        self.computed_effort, self.applied_effort, self.ceiling = motor_force(
            joint_pos, joint_vel, control_action.joint_positions, self.damping)
        control_action.joint_positions = None
        control_action.joint_velocities = None
        control_action.joint_efforts = self.applied_effort
        return control_action


@configclass
class MotorCfg(ActuatorBaseCfg):
    class_type = Motor
    joint_names_expr = [".*"]
    stiffness = 12.0
    damping = dict(zip(JOINT_NAMES, motor_damping(JOINT_NAMES)))
    armature = 0.0
    effort_limit = 1.6
    effort_limit_sim = 1.6
