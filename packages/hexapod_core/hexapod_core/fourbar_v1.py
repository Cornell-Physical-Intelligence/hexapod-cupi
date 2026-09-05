"""Physical four-bar motor-coordinate contract, independent of Isaac and NumPy.

The knee is passive. The third commanded coordinate is the motor's pushlever
angle in the new CAD hinge frame, not the archived serial knee coordinate.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

DEFAULT_ASSET_MODEL_ID = "mkii_fourbar_v3"
MODEL_ID = DEFAULT_ASSET_MODEL_ID  # Historical default, never the resolved run identity.
ASSET_BUNDLES = MappingProxyType({
    f"mkii_fourbar_v{version}": MappingProxyType({
        "model_id": f"mkii_fourbar_v{version}",
        "usd_path_relative": f"robot/hexapod_mkii_assy/usd/hexapod_mkii_fourbar_v{version}/hexapod_mkii_fourbar_v{version}.usda",
        "closure_constraint_variant": variant,
    })
    for version, variant in ((3, "revolute_5row_v3"), (4, "planar_d6_xy_v4"), (5, "physical_mimic_v5"))
})
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
PHYSICS_DT_S = .00125
DECIMATION = 16
POLICY_DT_S = PHYSICS_DT_S * DECIMATION
NUMERICAL_RECIPE_ID = "mkii_fourbar_tgs_external_forces_800hz_v3"
SOLVER_TYPE = 1  # PhysX TGS
SOLVER_POSITION_ITERATIONS = 64
SOLVER_VELOCITY_ITERATIONS = 1
ENABLE_EXTERNAL_FORCES_EVERY_ITERATION = True
ACTION_SCALE_RAD = .30
SLEW_RAD_PER_20MS = .040
MOTOR_TARGET_SCHEDULE_ID = "linear_physics_substeps_v1"
MOTOR_CONTROLLER_PROFILE = "mkii_pd_damping_030_v1"
SOFT_LIMIT_FACTOR = .95
OBSERVATION_FIELDS = (
    ("root_linear_velocity_navigation", 3), ("root_angular_velocity_navigation", 3),
    ("projected_gravity_navigation", 3), ("velocity_command_navigation", 3),
    ("active_motor_position_error", 18), ("active_motor_velocity", 18),
    ("previous_clipped_action", 18), ("estimated_motor_burst_headroom", 18),
)
OBSERVATION_DIM = sum(width for _, width in OBSERVATION_FIELDS)


def _file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_asset_bundle(value):
    """Validate a resolved identity; live USD audits still establish its physics."""
    fields = {"model_id", "usd_path_relative", "closure_constraint_variant", "usd_root_sha256",
              "kinematics_sha256", "bundle_files_sha256"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Incomplete physical asset bundle identity")
    model_id = value["model_id"]
    if not isinstance(model_id, str) or model_id not in ASSET_BUNDLES:
        raise ValueError("Unknown physical asset model")
    descriptor = ASSET_BUNDLES[model_id]
    if any(value[key] != expected for key, expected in descriptor.items()):
        raise ValueError("Physical asset path/model/closure identity mismatch")
    usd = Path(descriptor["usd_path_relative"])
    expected_files = {usd.as_posix(), *(str(usd.parent/name) for name in ("geometry.usdc", "kinematics.json", "manifest.json"))}
    files = value["bundle_files_sha256"]
    if (not isinstance(files, dict) or set(files) != expected_files
            or not all(_is_sha256(digest) for digest in files.values())
            or value["usd_root_sha256"] != files[usd.as_posix()]
            or value["kinematics_sha256"] != files[str(usd.parent/"kinematics.json")]):
        raise ValueError("Physical bundle file hashes are incomplete or inconsistent")
    return dict(value, bundle_files_sha256=dict(files))


def resolve_asset_bundle(usd_path, *, repo_root):
    """Select a known root USD and hash its actual portable bundle with stdlib.

    A relocated checkout is supported. Arbitrary USD roots and redirected bundle
    dependencies are rejected. The original kinematics JSON describes CAD frames; its stored
    USD path is historical provenance, not the selected physical formulation.
    """
    root = Path(repo_root).resolve()
    supplied = Path(usd_path)
    actual = (supplied if supplied.is_absolute() else root/supplied).resolve()
    matches = [value for value in ASSET_BUNDLES.values() if actual == root/value["usd_path_relative"]]
    if len(matches) != 1:
        raise ValueError("USD must resolve to one of the registered physical bundle roots")
    descriptor = dict(matches[0])
    relative = Path(descriptor["usd_path_relative"])
    records = {}
    for name in (relative.name, "geometry.usdc", "kinematics.json", "manifest.json"):
        path = actual.parent/name
        if path.resolve() != path or not path.is_file():
            raise ValueError(f"Physical bundle file is missing or redirected: {name}")
        records[path.relative_to(root).as_posix()] = _file_sha256(path)
    def reject_nonfinite(value):
        raise ValueError(f"Nonfinite physical bundle manifest: {value}")
    manifest = json.loads((actual.parent/"manifest.json").read_text(), parse_constant=reject_nonfinite)
    if (not isinstance(manifest, dict) or manifest.get("pass") is not True
            or manifest.get("schema") != f"hexapod.{descriptor['model_id']}.cpu_audit.v1"
            or manifest.get("closure_constraint_variant", "revolute_5row_v3") != descriptor["closure_constraint_variant"]):
        raise ValueError("Bundle creation manifest has a different model or closure variant")
    root_hash = records[relative.as_posix()]
    kin_hash = records[str(relative.parent/"kinematics.json")]
    dependencies = manifest.get("dependencies")
    expected_dependencies = [{"path": name, "sha256": records[str(relative.parent/name)]}
                             for name in sorted((relative.name, "geometry.usdc"))]
    if (manifest.get("usd_root_sha256") != root_hash
            or manifest.get("kinematic_contract_sha256") != kin_hash
            or dependencies != expected_dependencies):
        raise ValueError("Physical bundle bytes differ from its creation manifest")
    return validate_asset_bundle(dict(descriptor, usd_root_sha256=root_hash,
        kinematics_sha256=kin_hash, bundle_files_sha256=records))


def select_asset_bundle(*, repo_root, environ):
    """Resolve explicit environment selection, retaining v3 when unspecified."""
    paths = [environ[name] for name in ("HEXAPOD_USD_PATH", "HEXAPOD_MKII_FOURBAR_USD_PATH") if name in environ]
    if any(not isinstance(path, str) or not path.strip() for path in paths):
        raise ValueError("Explicit physical USD path cannot be empty")
    if not paths:
        paths = [ASSET_BUNDLES[DEFAULT_ASSET_MODEL_ID]["usd_path_relative"]]
    selected = [resolve_asset_bundle(path, repo_root=repo_root) for path in paths]
    if any(value != selected[0] for value in selected[1:]):
        raise ValueError("Conflicting physical USD environment overrides")
    return selected[0]


def numerical_recipe(multiplier=1):
    if type(multiplier) is not int or multiplier not in (1, 2):
        raise ValueError("Numerical recipe requires nominal=1 or refined=2")
    return {"recipe_id": NUMERICAL_RECIPE_ID, "solver_type": SOLVER_TYPE,
            "solver_position_iterations": SOLVER_POSITION_ITERATIONS * multiplier,
            "solver_velocity_iterations": SOLVER_VELOCITY_ITERATIONS,
            "enable_external_forces_every_iteration": ENABLE_EXTERNAL_FORCES_EVERY_ITERATION,
            "physics_dt_s": PHYSICS_DT_S, "decimation": DECIMATION}


def validate_numerical_recipe_report(report, multiplier):
    """Require the selected recipe and actual resolved environment settings."""
    expected = numerical_recipe(multiplier)
    actual = report.get("numerical_recipe")
    if (not isinstance(actual, dict) or actual != expected
            or any(type(actual[key]) is not type(value) for key, value in expected.items())):
        raise ValueError("Numerical recipe report differs from the selected TGS contract")
    iterations = report.get("solver_iterations")
    if (type(report.get("solver_multiplier")) is not int or report["solver_multiplier"] != multiplier
            or not isinstance(iterations, list) or any(type(value) is not int for value in iterations)
            or iterations != [expected["solver_position_iterations"], expected["solver_velocity_iterations"]]):
        raise ValueError("Solver iteration report differs from the selected numerical recipe")
    runtime = report.get("runtime_manifest")
    resolved = runtime.get("resolved_simulation") if isinstance(runtime, dict) else None
    if not isinstance(resolved, dict):
        raise ValueError("Numerical recipe requires actual resolved simulation metadata")
    for key, value in expected.items():
        if key == "recipe_id":
            continue
        if resolved.get(key) != value or type(resolved.get(key)) is not type(value):
            raise ValueError(f"Actual resolved simulation differs from numerical recipe: {key}")
    return expected


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


def runtime_manifest(kinematics, motor_manifest, *, kinematics_sha256, usd_sha256, asset_bundle):
    validate_kinematics(kinematics)
    asset_bundle = validate_asset_bundle(asset_bundle)
    for name, digest in (("kinematics", kinematics_sha256), ("USD", usd_sha256)):
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Expected SHA-256 for {name}")
    if (asset_bundle["kinematics_sha256"] != kinematics_sha256
            or asset_bundle["usd_root_sha256"] != usd_sha256):
        raise ValueError("Executed USD/kinematics hashes differ from the selected physical bundle")
    motor_json = json.dumps(motor_manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "schema": "hexapod.physical_fourbar_runtime.v1", "task_id": TASK_ID, "model_id": asset_bundle["model_id"],
        "asset_bundle": asset_bundle, "usd_path_relative": asset_bundle["usd_path_relative"],
        "closure_constraint_variant": asset_bundle["closure_constraint_variant"],
        "kinematics_sha256": kinematics_sha256, "usd_root_sha256": usd_sha256,
        "motor_contract": motor_manifest, "motor_contract_sha256": hashlib.sha256(motor_json.encode()).hexdigest(),
        "active_motor_names": list(ACTIVE_JOINT_NAMES), "tree_joint_names": list(kinematics["tree_joint_names"]),
        "default_motor_positions_rad": [kinematics["default_joint_positions_rad"][name] for name in ACTIVE_JOINT_NAMES],
        "soft_motor_limits_rad": [list(bounds) for bounds in soft_limits(kinematics)],
        "command_frame": COMMAND_FRAME, "body_to_navigation": [[0,-1,0],[1,0,0],[0,0,1]],
        "observation_fields": [list(field) for field in OBSERVATION_FIELDS], "observation_dim": OBSERVATION_DIM,
        "action_scale_rad": ACTION_SCALE_RAD, "action_clip": [-1., 1.], "slew_rad_per_20ms": SLEW_RAD_PER_20MS,
        "physics_dt_s": PHYSICS_DT_S, "decimation": DECIMATION, "policy_dt_s": POLICY_DT_S,
        "numerical_recipe_id": NUMERICAL_RECIPE_ID,
        "motor_target_schedule": {"id": MOTOR_TARGET_SCHEDULE_ID, "substeps": DECIMATION,
            "endpoint_semantics": "existing 50 Hz clipped and slew-limited motor position target",
            "position_fractions": "1/substeps through 1, exact endpoint on final substep",
            "velocity_feedforward_rad_s": 0.},
        "scope": "physical-model simulation contract; hardware calibration not admitted",
    }
