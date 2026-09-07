"""Versioned, stdlib-only identity and validation for the provisional RS05 v2.

This is a simulation contract, not a hardware safety or thermal calibration.
The untouched ``actuator.py`` remains the historical v1 contract.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType

CONFIG_PATH = Path(__file__).with_suffix(".json")
CONFIG = json.loads(CONFIG_PATH.read_text())
MODEL_ID = CONFIG["model_id"]
PEAK_TORQUE_NM = CONFIG["vendor"]["peak_output_torque_nm"]
STALL_CONTINUOUS_NM = CONFIG["vendor"]["stall_continuous_nm"]
ROTATING_CONTINUOUS_NM = CONFIG["vendor"]["rotating_continuous_nm"]
PHYSICS_DT_S = CONFIG["provisional"]["physics_dt_s"]
SUPPORTED_PHYSICS_DT_S = tuple(CONFIG["provisional"]["supported_physics_dt_s"])
ACTUATOR_CLASS = "hexapod_env.actuators.rs05_v2_runtime.RS05V2Actuator"
DEFAULT_CONTROLLER_PROFILE = "rs05_pd_default_v2"
LOWER_DAMPING_CONTROLLER_PROFILE = "mkii_pd_damping_030_v1"
# These profiles select explicit simulation PD gains. They do not alter the
# vendor envelope, thermal budget, friction, armature or hardware calibration.
CONTROLLER_PROFILES = MappingProxyType({
    DEFAULT_CONTROLLER_PROFILE: MappingProxyType({
        "stiffness_nm_per_rad": CONFIG["provisional"]["stiffness_nm_per_rad"],
        "damping_nm_s_per_rad": CONFIG["provisional"]["damping_nm_s_per_rad"],
    }),
    LOWER_DAMPING_CONTROLLER_PROFILE: MappingProxyType({
        "stiffness_nm_per_rad": 30.0,
        "damping_nm_s_per_rad": 0.30,
    }),
})
IMPLEMENTATION_PATHS = (
    "packages/hexapod_core/hexapod_core/rs05_v2.py",
    "packages/hexapod_core/hexapod_core/rs05_v2.json",
    "packages/hexapod_env/hexapod_env/actuators/rs05_v2.py",
    "packages/hexapod_env/hexapod_env/actuators/rs05_v2_model.py",
    "packages/hexapod_env/hexapod_env/actuators/rs05_v2_runtime.py",
)


def validate_operating_assumptions(physics_dt_s=PHYSICS_DT_S, assumed_bus_voltage_v=48.0):
    """Reject malformed settings; voltage-range validity is not calibration."""
    if not math.isfinite(physics_dt_s) or physics_dt_s not in SUPPORTED_PHYSICS_DT_S:
        raise ValueError(f"RS05 v2 contract requires an explicitly supported physics_dt_s: {SUPPORTED_PHYSICS_DT_S}")
    lo, hi = CONFIG["vendor"]["operating_voltage_range_v"]
    if not math.isfinite(assumed_bus_voltage_v) or not lo <= assumed_bus_voltage_v <= hi:
        raise ValueError("Assumed bus voltage must be finite and within15..60 V")


def resolve_controller_profile(controller_profile=DEFAULT_CONTROLLER_PROFILE):
    """Return a copy of an explicitly named PD profile; arbitrary gains fail."""
    if not isinstance(controller_profile, str) or controller_profile not in CONTROLLER_PROFILES:
        raise ValueError(f"Unknown RS05 controller profile: {controller_profile!r}")
    return dict(CONTROLLER_PROFILES[controller_profile])


def configuration_values(active_joint_names, *, physics_dt_s=PHYSICS_DT_S, assumed_bus_voltage_v=48.0,
                         controller_profile=DEFAULT_CONTROLLER_PROFILE):
    validate_operating_assumptions(physics_dt_s, assumed_bus_voltage_v)
    controller = resolve_controller_profile(controller_profile)
    names = list(active_joint_names)
    if len(names) != 18 or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != 18:
        raise ValueError("RS05 factory requires18 unique explicit actuator joint names")
    p = CONFIG["provisional"]
    return {
        "active_joint_names": names,
        "joint_names_expr": ["^" + re.escape(name) + "$" for name in names],
        "physics_dt_s": physics_dt_s,
        "assumed_bus_voltage_v": assumed_bus_voltage_v,
        "model_id": MODEL_ID,
        "controller_profile": controller_profile,
        "reset_burst_headroom": p["reset_burst_headroom"],
        "effort_limit": PEAK_TORQUE_NM,
        "effort_limit_sim": PEAK_TORQUE_NM,
        "velocity_limit": 480.0 * 2 * math.pi / 60,
        "velocity_limit_sim": 480.0 * 2 * math.pi / 60,
        "stiffness": controller["stiffness_nm_per_rad"],
        "damping": controller["damping_nm_s_per_rad"],
        "armature": p["armature_kg_m2"],
        "friction": p["friction"],
        "dynamic_friction": p["dynamic_friction"],
        "viscous_friction": p["viscous_friction"],
    }


def contract_manifest(*, physics_dt_s=PHYSICS_DT_S, assumed_bus_voltage_v=48.0, repo_root=None,
                      controller_profile=DEFAULT_CONTROLLER_PROFILE):
    """Hash actual source bytes, contract data, and operational assumptions.

    ``repo_root`` is explicit for deployed source bundles. Missing files fail
    closed; no installed-code fallback or silently omitted implementation hash.
    """
    validate_operating_assumptions(physics_dt_s, assumed_bus_voltage_v)
    controller = resolve_controller_profile(controller_profile)
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    result = {
        "model_id": MODEL_ID,
        "status": CONFIG["status"],
        "physics_dt_s": physics_dt_s,
        "assumed_bus_voltage_v": assumed_bus_voltage_v,
        "controller_profile": controller_profile,
        "controller_parameters": controller,
        "actual_battery_voltage_known": False,
        "implementation_sha256": {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                                   for path in IMPLEMENTATION_PATHS},
        "configuration": json.loads(json.dumps(CONFIG)),
    }
    result["contract_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return result


def verify_runtime_cfg(cfg, active_joint_names):
    """Bind the instantiated SDK config, rather than only the manifest's prose."""
    expected = configuration_values(active_joint_names, physics_dt_s=cfg.physics_dt_s,
                                    assumed_bus_voltage_v=cfg.assumed_bus_voltage_v,
                                    controller_profile=getattr(cfg, "controller_profile", None))
    for key, value in expected.items():
        if getattr(cfg, key) != value:
            raise ValueError(f"RS05 runtime configuration mismatch: {key}")
    cls = cfg.class_type
    actual = cls.replace(":", ".") if isinstance(cls, str) else f"{cls.__module__}.{cls.__name__}"
    if actual != ACTUATOR_CLASS:
        raise ValueError("RS05 runtime actuator class mismatch")
    return dict(expected, class_type=ACTUATOR_CLASS)
