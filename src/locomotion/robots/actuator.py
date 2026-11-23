"""Explicit Isaac Lab actuator with the approved speed-dependent torque curve."""

import torch
from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg
from isaaclab.utils import configclass

from locomotion.robots import randomization as rnd
from locomotion.robots.hexapod import JOINT_NAMES, motor_damping
from locomotion.robots.rs05 import RATED_TORQUE_NM, STIFFNESS_NM_PER_RAD, motor_force


class Motor(ActuatorBase):
    """Compute the canonical PD torque once per 400 Hz scene write.

    ``draw`` is None for the nominal law. Under a Step 7 draw it holds float32 ``kp``, ``kd`` and
    ``strength`` tensors of shape (replicas, joints) in this actuator's joint order, and
    ``randomization.motor_force`` applies them. The environment owns and updates the draw.
    """

    draw = None

    def reset(self, env_ids):
        self.computed_effort[env_ids] = 0
        self.applied_effort[env_ids] = 0

    def compute(self, control_action, joint_pos, joint_vel):
        if not bool(torch.isfinite(joint_pos).all() and torch.isfinite(joint_vel).all()):
            raise FloatingPointError("Nonfinite pre-servo state")
        self.pre_position = joint_pos.clone()
        self.pre_velocity = joint_vel.clone()
        target = control_action.joint_positions
        if self.draw is None:
            self.computed_effort, self.applied_effort, self.ceiling = motor_force(
                joint_pos, joint_vel, target, self.damping
            )
        else:
            self.computed_effort, self.applied_effort, self.ceiling = rnd.motor_force(
                joint_pos,
                joint_vel,
                target,
                self.draw["kd"],
                self.draw["kp"],
                self.draw["strength"],
            )
        control_action.joint_positions = None
        control_action.joint_velocities = None
        control_action.joint_efforts = self.applied_effort
        return control_action


@configclass
class MotorCfg(ActuatorBaseCfg):
    class_type = Motor
    joint_names_expr = [".*"]
    stiffness = STIFFNESS_NM_PER_RAD
    damping = dict(zip(JOINT_NAMES, motor_damping(JOINT_NAMES)))
    armature = 0.0
    effort_limit = RATED_TORQUE_NM
    effort_limit_sim = RATED_TORQUE_NM
