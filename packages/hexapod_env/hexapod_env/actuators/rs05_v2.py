"""Pre-Kit-safe configuration for the provisional RS05 v2 actuator.

Only the SDK config class is imported here. The runtime actuator is resolved
from ``class_type`` by Isaac Lab after the simulation backend starts.
"""
from __future__ import annotations

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.utils.configclass import configclass

from hexapod_core.rs05_v2 import ACTUATOR_CLASS, MODEL_ID, configuration_values, verify_runtime_cfg


@configclass
class RS05V2ActuatorCfg(IdealPDActuatorCfg):
    class_type: str = ACTUATOR_CLASS.rsplit(".", 1)[0] + ":RS05V2Actuator"
    active_joint_names: list[str] = []
    physics_dt_s: float = 0.005
    assumed_bus_voltage_v: float = 48.0
    reset_burst_headroom: float = 0.5
    model_id: str = MODEL_ID


def make_rs05_v2_cfg(active_joint_names, *, physics_dt_s=0.005, assumed_bus_voltage_v=48.0):
    """Bind only18 explicitly named motors; never absorb passive joints."""
    cfg = RS05V2ActuatorCfg(**configuration_values(active_joint_names, physics_dt_s=physics_dt_s,
                                                assumed_bus_voltage_v=assumed_bus_voltage_v))
    verify_runtime_cfg(cfg, active_joint_names)
    return cfg
