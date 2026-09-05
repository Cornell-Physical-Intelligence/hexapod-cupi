"""Physical four-bar motor-coordinate contract, independent of Isaac and NumPy.

The knee is passive. The third commanded coordinate is the motor's pushlever
angle in the new CAD hinge frame, not the archived serial knee coordinate.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

MODEL_ID = "mkii_fourbar_v3"
TASK_ID = "Isaac-Velocity-Flat-Hexapod-MKII-Fourbar-V1-Direct-v0"
KINEMATICS_SCHEMA = "hexapod.mkii_fourbar_v3.kinematics.v1"
KINEMATICS_PATH = "configs/mkii_fourbar_v3_kinematics.json"
LEGS = ("lf", "lm", "lr", "rf", "rm", "rr")
ACTIVE_GROUPS = ("coxa_yaw", "femur_pitch", "tibia_lever_pivot")
ACTIVE_JOINT_NAMES = tuple(f"{leg}_{group}" for group in ACTIVE_GROUPS for leg in LEGS)
PASSIVE_JOINT_NAMES = tuple(f"{leg}_{group}" for group in ("tibia_pitch", "tibia_rod_pivot") for leg in LEGS)
TREE_JOINT_NAMES = ACTIVE_JOINT_NAMES + PASSIVE_JOINT_NAMES
BODY_NAMES = ("body",) + tuple(f"{leg}_{part}" for leg in LEGS for part in (
    "coxa", "femur", "tibia", "tibia_push_lever", "tibia_pushrod"))
COMMAND_FRAME = "anatomical_navigation"
PHYSICS_DT_S = .005
DECIMATION = 4
POLICY_DT_S = PHYSICS_DT_S * DECIMATION
ACTION_SCALE_RAD = .30
SLEW_RAD_PER_20MS = .040
SOFT_LIMIT_FACTOR = .95
OBSERVATION_FIELDS = (
    ("root_linear_velocity_navigation", 3), ("root_angular_velocity_navigation", 3),
    ("projected_gravity_navigation", 3), ("velocity_command_navigation", 3),
    ("active_motor_position_error", 18), ("active_motor_velocity", 18),
    ("previous_clipped_action", 18), ("estimated_motor_burst_headroom", 18),
)
OBSERVATION_DIM = sum(width for _, width in OBSERVATION_FIELDS)


def finite_vector(values, width, label):
    result = [float(value) for value in values]
    if len(result) != width or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label}: expected {width} finite values")
    return result


def validate_tree_names(names):
    names = tuple(names)
    if len(names) != 30 or len(set(names)) != 30 or set(names) != set(TREE_JOINT_NAMES):
        raise ValueError("Expected the 30 distinct physical four-bar tree joint names")
    return names


def active_indices(names):
    names = validate_tree_names(names)
    return tuple(names.index(name) for name in ACTIVE_JOINT_NAMES)


def passive_relations():
    return {f"{leg}_{kind}": {"source_joint": f"{leg}_tibia_lever_pivot",
            "multiplier": multiplier, "offset_rad": 0.0}
            for leg in LEGS for kind, multiplier in (("tibia_pitch", 1.0), ("tibia_rod_pivot", -1.0))}


def validate_kinematics(value):
    """Validate the coordinate interface; the asset gate validates CAD geometry."""
    if value.get("schema") != KINEMATICS_SCHEMA:
        raise ValueError("Wrong physical four-bar kinematics schema")
    if tuple(value.get("active_joint_names", ())) != ACTIVE_JOINT_NAMES:
        raise ValueError("Wrong physical motor action names/order")
    validate_tree_names(value["tree_joint_names"])
    if value.get("passive_relations") != passive_relations():
        raise ValueError("Unqualified passive-joint closure relation")
    defaults, limits = value["default_joint_positions_rad"], value["joint_limits_rad"]
    if set(defaults) != set(TREE_JOINT_NAMES) or set(limits) != set(TREE_JOINT_NAMES):
        raise ValueError("Default/limit maps must cover every physical tree coordinate")
    for name in TREE_JOINT_NAMES:
        lower, upper = finite_vector(limits[name], 2, name)
        position = finite_vector([defaults[name]], 1, name)[0]
        if not lower < upper or not lower <= position <= upper:
            raise ValueError(f"Invalid default/limits for {name}")
    for name, relation in passive_relations().items():
        if not math.isclose(defaults[name], defaults[relation["source_joint"]] * relation["multiplier"], abs_tol=1e-10):
            raise ValueError(f"Default pose violates closure: {name}")
    for key in ("nominal_height_m", "reset_root_height_m"):
        if not math.isfinite(value[key]) or value[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if value["reset_root_height_m"] < value["nominal_height_m"]:
        raise ValueError("Reset height is below geometric contact height")
    if set(value["body_paths"]) != set(BODY_NAMES):
        raise ValueError("Expected paths for all 31 physical bodies")
    return value


def load_kinematics(path):
    return validate_kinematics(json.loads(Path(path).read_text()))


def soft_limits(kinematics):
    result = []
    for name in ACTIVE_JOINT_NAMES:
        lower, upper = kinematics["joint_limits_rad"][name]
        center, half = (lower + upper) / 2, (upper - lower) * SOFT_LIMIT_FACTOR / 2
        result.append((center-half, center+half))
    return result


def runtime_manifest(kinematics, motor_manifest, *, kinematics_sha256, usd_sha256):
    validate_kinematics(kinematics)
    for name, digest in (("kinematics", kinematics_sha256), ("USD", usd_sha256)):
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Expected SHA-256 for {name}")
    motor_json = json.dumps(motor_manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "schema": "hexapod.physical_fourbar_runtime.v1", "task_id": TASK_ID, "model_id": MODEL_ID,
        "kinematics_sha256": kinematics_sha256, "usd_root_sha256": usd_sha256,
        "motor_contract": motor_manifest, "motor_contract_sha256": hashlib.sha256(motor_json.encode()).hexdigest(),
        "active_motor_names": list(ACTIVE_JOINT_NAMES), "tree_joint_names": list(kinematics["tree_joint_names"]),
        "default_motor_positions_rad": [kinematics["default_joint_positions_rad"][name] for name in ACTIVE_JOINT_NAMES],
        "soft_motor_limits_rad": [list(bounds) for bounds in soft_limits(kinematics)],
        "command_frame": COMMAND_FRAME, "body_to_navigation": [[0,-1,0],[1,0,0],[0,0,1]],
        "observation_fields": [list(field) for field in OBSERVATION_FIELDS], "observation_dim": OBSERVATION_DIM,
        "action_scale_rad": ACTION_SCALE_RAD, "action_clip": [-1., 1.], "slew_rad_per_20ms": SLEW_RAD_PER_20MS,
        "physics_dt_s": PHYSICS_DT_S, "decimation": DECIMATION, "policy_dt_s": POLICY_DT_S,
        "scope": "physical-model simulation contract; hardware calibration not admitted",
    }
