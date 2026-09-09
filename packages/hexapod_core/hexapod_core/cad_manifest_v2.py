"""Explicit CAD-v2 simulation action contract; no checkpoint is admitted by it.

The old ``joints``/``action`` modules describe the archived mock. This module
stands beside those contracts. Names, not group indices, determine defaults
and limits. The policy order below was observed on the original CAD import;
every new articulation must report its names and pass the binding checks.

These are simulation limits, not measured motor stops or hardware approval.
The URDF hash pins the source model, not the generated USD: the separate
URDF/USD integrity gate must also pass before simulation is admitted.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence


SCHEMA_VERSION = 2
ASSET_NAME = "mkii_v2"
URDF_PATH = "robot/hexapod_mkii_assy/urdf/hexapod_mkii_serial.urdf"
URDF_SHA256 = "6109956e9e3a7a648bc4afa7310904c2fd3f7cb790e157d83212339a73b7fbc0"
LEG_NAMES = ("lf", "lm", "lr", "rf", "rm", "rr")
JOINT_GROUPS = ("coxa_yaw", "femur_pitch", "tibia_pitch")
RUNTIME_JOINT_NAMES = tuple(f"{leg}_{group}" for group in JOINT_GROUPS for leg in LEG_NAMES)
COMMAND_FRAME = "navigation"
BODY_TO_NAVIGATION_MATRIX = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
ACTION_SCALE_RAD = 0.30
STAND_ACTION_SCALE = 1.0
COMMAND_ACTIVE_THRESHOLD = 0.01
PHYSICS_DT_S = 0.005
DECIMATION = 4
POLICY_STEP_DT_S = PHYSICS_DT_S * DECIMATION
SLEW_LIMIT_RAD_PER_20MS = 0.040
SOFT_JOINT_POS_LIMIT_FACTOR = 0.95

_DEFAULTS = {"coxa_yaw": 0.0, "femur_pitch": -0.25, "tibia_pitch": -0.55}
_HARD_LIMITS = {
    "coxa_yaw": (-0.872665, 0.872665),
    "femur_pitch": (-1.745329, 0.55),
    "tibia_pitch": (-0.95, 1.75),
}
DEFAULT_JOINT_POSITIONS_BY_NAME = MappingProxyType({
    f"{leg}_{group}": _DEFAULTS[group] for group in JOINT_GROUPS for leg in LEG_NAMES
})
HARD_LIMITS_BY_NAME = MappingProxyType({
    f"{leg}_{group}": _HARD_LIMITS[group] for group in JOINT_GROUPS for leg in LEG_NAMES
})


def _soft_limits(bounds: tuple[float, float]) -> tuple[float, float]:
    """Isaac Lab shrinks the range about its midpoint, not about zero."""
    lower, upper = bounds
    center = (lower + upper) / 2.0
    half_range = (upper - lower) * SOFT_JOINT_POS_LIMIT_FACTOR / 2.0
    return (center - half_range, center + half_range)


SOFT_LIMITS_BY_NAME = MappingProxyType({
    name: _soft_limits(bounds) for name, bounds in HARD_LIMITS_BY_NAME.items()
})


def simulation_manifest() -> dict:
    """Return a fresh, JSON-serializable manifest with every action parameter.

    Future policies must carry this exact contract plus their own checkpoint,
    USD and observation provenance. This manifest alone is not a policy file.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "asset_name": ASSET_NAME,
        "urdf_path": URDF_PATH,
        "urdf_sha256": URDF_SHA256,
        "command_frame": COMMAND_FRAME,
        "body_to_navigation_matrix": [list(row) for row in BODY_TO_NAVIGATION_MATRIX],
        "policy_joint_names": list(RUNTIME_JOINT_NAMES),
        "default_joint_positions_rad": dict(DEFAULT_JOINT_POSITIONS_BY_NAME),
        "hard_joint_limits_rad": {name: list(bounds) for name, bounds in HARD_LIMITS_BY_NAME.items()},
        "soft_joint_limits_rad": {name: list(bounds) for name, bounds in SOFT_LIMITS_BY_NAME.items()},
        "soft_joint_pos_limit_factor": SOFT_JOINT_POS_LIMIT_FACTOR,
        "action_clip": [-1.0, 1.0],
        "action_scale_rad": ACTION_SCALE_RAD,
        "stand_action_scale": STAND_ACTION_SCALE,
        "command_active_threshold": COMMAND_ACTIVE_THRESHOLD,
        "slew_limit_rad_per_20ms": SLEW_LIMIT_RAD_PER_20MS,
        "physics_dt_s": PHYSICS_DT_S,
        "decimation": DECIMATION,
        "policy_step_dt_s": POLICY_STEP_DT_S,
        "scope": "simulation_action_contract_only",
    }


def _require_equal(actual: object, expected: object, field: str) -> None:
    """Reject missing/extra fields, nonfinite values and stale contract values."""
    if isinstance(expected, dict):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"{field}: manifest fields do not match CAD v2")
        for key, value in expected.items():
            _require_equal(actual[key], value, f"{field}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, (tuple, list)) or len(actual) != len(expected):
            raise ValueError(f"{field}: manifest sequence does not match CAD v2")
        for index, (value, reference) in enumerate(zip(actual, expected)):
            _require_equal(value, reference, f"{field}[{index}]")
    elif isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise ValueError(f"{field}: expected a finite CAD v2 number")
        if not math.isfinite(actual) or actual != expected:
            raise ValueError(f"{field}: value does not match CAD v2")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{field}: value does not match CAD v2")


def validate_simulation_manifest(manifest: Mapping) -> None:
    """Fail closed instead of silently applying a mock or stale CAD contract."""
    _require_equal(manifest, simulation_manifest(), "manifest")


def load_simulation_manifest(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    validate_simulation_manifest(manifest)
    return manifest


def verify_urdf_identity(path: str | Path) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != URDF_SHA256:
        raise ValueError(f"URDF SHA-256 does not match {ASSET_NAME}: {digest}")
    return digest


def validate_joint_names(names: Sequence[str], *, field: str) -> tuple[str, ...]:
    result = tuple(names)
    if (len(result) != 18 or any(not isinstance(name, str) for name in result)
            or len(set(result)) != 18 or set(result) != set(RUNTIME_JOINT_NAMES)):
        raise ValueError(f"{field} must identify the 18 distinct CAD joint names")
    return result
