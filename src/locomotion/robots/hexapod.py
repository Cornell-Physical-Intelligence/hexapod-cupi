"""Name the robot's joints and bodies, pin its asset hashes and resolve motor damping."""

import hashlib
import json
from pathlib import Path

LEGS = ("lf", "lm", "lr", "rf", "rm", "rr")
JOINT_NAMES = tuple(
    f"{leg}_{joint}" for leg in LEGS for joint in ("coxa_yaw", "femur_pitch", "tibia_pitch")
)
BODY_NAMES = ("body",) + tuple(
    f"{leg}_{body}" for leg in LEGS for body in ("coxa", "femur", "tibia")
)
URDF_SHA256 = "9492fde54c50170e940b49ccb3037d7b413743a5b77bc1d8707a539d44349e78"
MODEL_SHA256 = "7126935b35398d2a7f03271335a8a1b8f2059354af2b19d9474f22b8e73e4881"
USD_SHA256 = "3be98420adc7923923757d6557d8492d5930f84128c1b1366b9c6db1bb03207c"
MASS_KG = 7.466088235225788
KD = (
    0.44197696391810243,
    0.2456657243160222,
    0.10552064613953134,
    0.4419789704596685,
    0.24566676283633715,
    0.10552131615792909,
    0.441977634689341,
    0.24566456721764315,
    0.10552100524837532,
    0.4419765017453795,
    0.245667092769639,
    0.10552130774083751,
    0.44197471269634825,
    0.24566700905274597,
    0.10552141334046235,
    0.4419772543481437,
    0.24566767616660323,
    0.10552122889741367,
)
ISAACLAB_REVISION = "ffff603eafc6b74264a5261cc0183d6a65390d78"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_assets(asset, model_path):
    asset, model_path = Path(asset), Path(model_path)
    if sha(asset / "robot.usda") != USD_SHA256 or sha(model_path) != MODEL_SHA256:
        raise ValueError("Canonical USD/model identity differs")
    model = json.loads(model_path.read_text())
    # The JSON retains its earlier CAD import's source hash; bind the actual
    # corrected URDF shipped with this exact USD asset instead.
    if sha(asset / "source" / "source.urdf") != URDF_SHA256:
        raise ValueError("Canonical URDF identity differs")
    if {x["name"] for x in model["links"]} != set(BODY_NAMES) or {
        x["name"] for x in model["joints"]
    } != set(JOINT_NAMES):
        raise ValueError("Canonical 19-body/18-joint topology differs")
    if abs(sum(x["mass"] for x in model["links"]) - MASS_KG) > 1e-9:
        raise ValueError("Canonical mass ledger differs")
    return model


def motor_damping(joint_names):
    """Resolve exact damping gains in the native joint order."""
    if len(joint_names) != 18 or set(joint_names) != set(JOINT_NAMES):
        raise ValueError("Motor requires the canonical 18 named joints")
    return [KD[JOINT_NAMES.index(name)] for name in joint_names]
