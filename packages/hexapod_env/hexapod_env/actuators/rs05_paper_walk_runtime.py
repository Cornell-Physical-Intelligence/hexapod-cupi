"""Isaac Lab runtime binding. Import only after the simulation backend starts.

The actuator computes the accepted servo itself, so PhysX receives the applied
effort alone: the returned control action carries no position or velocity
target. Isaac Lab keeps the implicit drive gains of an explicit actuator at
zero, and this class refuses a configuration whose armature, friction or
stiffness differs from the accepted values.

``computed_effort``, ``applied_effort`` and ``effort_ceiling`` hold the three
per-physics-step channels the diagnostic capture records.
"""

from __future__ import annotations

import torch
from isaaclab.actuators import IdealPDActuator

from .rs05_paper_walk_model import STIFFNESS_NM_PER_RAD, damping_vector, effort_ceiling, motor_effort


class RS05PaperWalkActuator(IdealPDActuator):
    """Named-damping PD demand with the provisional speed-dependent ceiling."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        declared = list(getattr(cfg, "active_joint_names", []))
        if len(self.joint_names) != 18 or set(self.joint_names) != set(declared):
            raise ValueError("The resolved motor group must contain exactly the 18 declared RS05 joints")
        if cfg.stiffness != STIFFNESS_NM_PER_RAD:
            raise ValueError("The paper-walk servo stiffness is fixed at 12 N*m per radian")
        if cfg.armature != 0.0 or cfg.friction != 0.0:
            raise ValueError("The paper-walk actuator requires zero armature and zero joint friction")
        self.damping_nm_s_per_rad = damping_vector(
            self.joint_names, device=self.computed_effort.device, dtype=self.computed_effort.dtype
        )
        self.effort_ceiling = torch.zeros_like(self.computed_effort)
        self._motor_velocity = torch.zeros_like(self.computed_effort)

    def compute(self, control_action, joint_pos, joint_vel):
        self._motor_velocity = joint_vel
        requested, applied, ceiling = motor_effort(
            joint_pos, joint_vel, control_action.joint_positions, self.damping_nm_s_per_rad
        )
        self.computed_effort = requested
        self.applied_effort = applied
        self.effort_ceiling = ceiling
        control_action.joint_efforts = applied
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    def _clip_effort(self, effort):
        ceiling = effort_ceiling(self._motor_velocity)
        return torch.maximum(torch.minimum(effort, ceiling), -ceiling)
