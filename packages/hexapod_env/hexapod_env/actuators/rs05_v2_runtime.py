"""Isaac Lab runtime binding. Import only after the simulation backend starts."""
from __future__ import annotations

from isaaclab.actuators import IdealPDActuator

from hexapod_core.rs05_v2 import verify_runtime_cfg
from .rs05_v2_model import RS05V2BudgetModel, TELEMETRY_FIELDS


class RS05V2Actuator(IdealPDActuator):
    """PD raw demand plus a per-physics-step output/burst limiter.

    SDK ``joint_names``/``joint_indices`` describe actual resolved motor order.
    Environment observations must gather headroom by those names, not config
    regex order. Terminate when any ``invalid_input`` motor flag is set.
    """

    def __init__(self, cfg, *args, **kwargs):
        verify_runtime_cfg(cfg, cfg.active_joint_names)
        super().__init__(cfg, *args, **kwargs)
        if set(self.joint_names) != set(cfg.active_joint_names) or len(self.joint_names) != 18:
            raise ValueError("RS05 resolved motor group must contain exactly the18 declared active joints")
        self._budget = RS05V2BudgetModel(
            self.computed_effort.shape, device=self.computed_effort.device, dtype=self.computed_effort.dtype,
            physics_dt_s=cfg.physics_dt_s, assumed_bus_voltage_v=cfg.assumed_bus_voltage_v,
            reset_burst_headroom=cfg.reset_burst_headroom,
        )

    def reset(self, env_ids=None):
        self._budget.reset(env_ids)

    def compute(self, control_action, joint_pos, joint_vel):
        self._motor_velocity = joint_vel
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort):
        return self._budget.step(effort, self._motor_velocity)


def _telemetry_property(name):
    return property(lambda self: getattr(self._budget, name))


for _name in TELEMETRY_FIELDS:
    setattr(RS05V2Actuator, _name, _telemetry_property(_name))
