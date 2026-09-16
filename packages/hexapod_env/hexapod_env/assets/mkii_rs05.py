"""Asset spec for the approved mass-corrected direct-drive robot.

``robot/active_model.json`` selects this model: 19 bodies, 18 direct-drive
joints, no four-bar, 7.466088235225788 kg. The spec pins the URDF, the model
JSON, the generated USD and the kinematic prior metadata by SHA-256, and the
contract test compares every pin with the file on disk.

Three values differ from the earlier CAD assets and are kept explicit here.
First, the six coxa joints do not share one travel, so the exact per-joint
table lives in ``joint_limits_rad`` and the three group fields carry the
envelope every leg shares. Second, the stance is the declared walking neutral
``[0, -0.30, 0.40]`` radians per leg from the pinned prior metadata, not the
CAD inspection pose. Third, the reset height is that metadata's
``reset_root_height_m``.

This module stays stdlib-only, so a contract test loads it without Isaac Lab.
The environment config that consumes it lives in ``tasks/mkii_rs05``. No
historical task, checkpoint or asset is redirected by adding this spec.
"""

from __future__ import annotations

from .spec import RUNTIME_ORDER_PROVISIONAL, HexapodAssetSpec

__all__ = [
    "MKII_RS05_ASSET",
    "MKII_RS05_CANONICAL_JOINT_NAMES",
    "MKII_RS05_LEGS",
    "MKII_RS05_METADATA_PATH",
    "MKII_RS05_METADATA_SHA256",
    "MKII_RS05_MODEL_PATH",
    "MKII_RS05_MODEL_SHA256",
    "MKII_RS05_URDF_SHA256",
    "MKII_RS05_USD_ARTIFACT_PATH",
    "MKII_RS05_USD_SHA256",
]

MKII_RS05_LEGS = ("lf", "lm", "lr", "rf", "rm", "rr")
#: Per-leg joint order, the order the prototype's prior metadata and its
#: telemetry use. The articulation order in the spec below is the block order
#: PhysX reports; the environment maps between the two by name.
MKII_RS05_CANONICAL_JOINT_NAMES = tuple(
    f"{leg}_{joint}"
    for leg in MKII_RS05_LEGS
    for joint in ("coxa_yaw", "femur_pitch", "tibia_pitch")
)

MKII_RS05_URDF_PATH = "robot/hexapod_mkii_updated_v1/urdf/hexapod_updated_rs05_mass_corrected.urdf"
MKII_RS05_MODEL_PATH = "robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json"
MKII_RS05_USD_ARTIFACT_PATH = "artifacts/mkii_updated_2026-09-10/usd_002/rs05_mass_corrected/robot.usda"
#: Kinematic prior metadata: the declared neutral stance, the reset height and
#: the toe points. ``experiments/paper_walk/train.py`` reads the same file
#: through ``--prior-metadata``. It is a replay input, never imported as code.
MKII_RS05_METADATA_PATH = (
    "artifacts/restart_2026-09-14/paper_walk_execution_001/prior_001/prior_metadata.json"
)

MKII_RS05_URDF_SHA256 = "9492fde54c50170e940b49ccb3037d7b413743a5b77bc1d8707a539d44349e78"
MKII_RS05_MODEL_SHA256 = "7126935b35398d2a7f03271335a8a1b8f2059354af2b19d9474f22b8e73e4881"
MKII_RS05_USD_SHA256 = "3be98420adc7923923757d6557d8492d5930f84128c1b1366b9c6db1bb03207c"
MKII_RS05_METADATA_SHA256 = "830cb07c0fdb3d80af82476d8e6e88f25440cfd2e9255ac1e16327ac826d259c"

MKII_RS05_MASS_KG = 7.466088235225788

#: Exact URDF travel per joint, read from the model JSON. The front and rear
#: coxa joints reach about 1.30 rad; the middle pair reaches about 0.82 rad.
MKII_RS05_JOINT_LIMITS_RAD = (
    ("lf_coxa_yaw", -1.2966923677691873, 1.2966923677691873),
    ("lm_coxa_yaw", -0.8153305600691511, 0.8153305600691511),
    ("lr_coxa_yaw", -1.3046336158657614, 1.3046336158657614),
    ("rf_coxa_yaw", -1.3046336158657614, 1.3046336158657614),
    ("rm_coxa_yaw", -0.815330560069151, 0.815330560069151),
    ("rr_coxa_yaw", -1.2966923677691873, 1.2966923677691873),
    ("lf_femur_pitch", -2.0943951023931953, 1.3962634015954636),
    ("lm_femur_pitch", -2.0943951023931953, 1.3962634015954636),
    ("lr_femur_pitch", -2.0943951023931953, 1.3962634015954636),
    ("rf_femur_pitch", -2.0943951023931953, 1.3962634015954636),
    ("rm_femur_pitch", -2.0943951023931953, 1.3962634015954636),
    ("rr_femur_pitch", -2.0943951023931953, 1.3962634015954636),
    ("lf_tibia_pitch", -0.08726646259971647, 3.141592653589793),
    ("lm_tibia_pitch", -0.08726646259971647, 3.141592653589793),
    ("lr_tibia_pitch", -0.08726646259971647, 3.141592653589793),
    ("rf_tibia_pitch", -0.08726646259971647, 3.141592653589793),
    ("rm_tibia_pitch", -0.08726646259971647, 3.141592653589793),
    ("rr_tibia_pitch", -0.08726646259971647, 3.141592653589793),
)

#: Toe point per leg in its tibia link frame, from the pinned metadata. The
#: corrected tibia runs along link +X, so the pad sits 0.1225 m out that axis.
MKII_RS05_TOE_LOCAL_POINTS_M = (
    (0.12249960783347934, -1.279745107786495e-07, -4.269342937671501e-08),
    (0.12250027947967954, 1.9952780009965557e-07, -1.354552844181148e-07),
    (0.12250003138363123, -5.680045064176779e-08, -5.077808664637763e-07),
    (0.12250019990076949, 1.0883751586458235e-08, -6.738747879805752e-07),
    (0.1225003097598866, 3.188230138686082e-08, -2.412101555242121e-07),
    (0.12250009102186771, -2.736829703048315e-09, 2.56848792860696e-08),
)

MKII_RS05_ASSET = HexapodAssetSpec(
    name="mkii_rs05_v1",
    urdf_path=MKII_RS05_URDF_PATH,
    usd_path_container="/workspace/hexapod/" + MKII_RS05_USD_ARTIFACT_PATH,
    usd_env_var="HEXAPOD_MKII_RS05_USD_PATH",
    root_link="body",
    leg_names=MKII_RS05_LEGS,
    coxa_joints=tuple(f"{leg}_coxa_yaw" for leg in MKII_RS05_LEGS),
    femur_joints=tuple(f"{leg}_femur_pitch" for leg in MKII_RS05_LEGS),
    tibia_joints=tuple(f"{leg}_tibia_pitch" for leg in MKII_RS05_LEGS),
    leg_link_names=tuple(
        (f"{leg}_coxa", f"{leg}_femur", f"{leg}_tibia") for leg in MKII_RS05_LEGS
    ),
    coxa_link_regex=".*_coxa",
    # The envelope every leg shares. ``joint_limits_rad`` below holds the
    # exact per-joint travel the task clamps against.
    coxa_limits=(-0.815330560069151, 0.815330560069151),
    femur_limits=(-2.0943951023931953, 1.3962634015954636),
    tibia_limits=(-0.08726646259971647, 3.141592653589793),
    stance_coxa_rad=0.0,
    stance_femur_rad=-0.3,
    stance_tibia_rad=0.4,
    reset_root_height_m=0.10280231400684256,
    nominal_height_m=0.09780231400684256,
    # The two fields below describe a tibia whose pad runs along link +Y. This
    # model's pad runs along link +X, so the RS05 task reads
    # ``MKII_RS05_TOE_LOCAL_POINTS_M`` and never these two values. They keep
    # the shared dataclass contract and carry the same 0.1225 m pad distance.
    distal_foot_min_y_m=0.1225,
    swing_clearance_pad_offset_y_m=0.1225,
    # Block order: six coxa joints in leg order, then the femurs, then the
    # tibias. The recorded native readback of this exact USD reports that
    # order. The Isaac Lab spawn path has not been read back, so the status
    # stays provisional and the environment checks the order at construction.
    runtime_joint_names=(
        tuple(f"{leg}_coxa_yaw" for leg in MKII_RS05_LEGS)
        + tuple(f"{leg}_femur_pitch" for leg in MKII_RS05_LEGS)
        + tuple(f"{leg}_tibia_pitch" for leg in MKII_RS05_LEGS)
    ),
    runtime_order_status=RUNTIME_ORDER_PROVISIONAL,
    mass_kg=MKII_RS05_MASS_KG,
    mass_note=(
        "Original CAD plus the recorded nominal RS05 missing-mass correction "
        "(model_rs05_mass_corrected.json); not measured inertial identification"
    ),
    joint_limits_rad=MKII_RS05_JOINT_LIMITS_RAD,
)
