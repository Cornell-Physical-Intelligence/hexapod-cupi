"""v2 runtime joint/action order: asset v1, the CAD assembly (ADR-0001).

This module stands beside :mod:`hexapod_core.joints` (v1, the Phase-0 mock);
it does not replace it. A checkpoint's meaning is frozen against the schema it
was trained under, so v1 is never edited and v2 is never applied to a v1
checkpoint.

WHERE THIS ORDER COMES FROM
---------------------------
The runtime order is the articulation's own joint order: Isaac Lab reads it
from PhysX when the USD is parsed and the environment consumes it positionally.
PhysX orders articulation links breadth-first from the root, children in USD
order, and Isaac Sim's URDF importer emits children in URDF declaration order.
For ``hexapod_mkii_serial.urdf`` (root ``body``, legs declared lf lm lr rf rm
rr) that predicts: the six ``coxa_yaw`` joints in leg order, then the six
``femur_pitch``, then the six ``tibia_pitch``. The mock's v1 order has the same
depth-first-by-group shape, which is the corroboration for the prediction.

STATUS: PROVISIONAL. The order below has not yet been read back from an
imported articulation. ``HexapodEnv`` compares it against
``Articulation.joint_names`` at construction and raises if they differ, so a
wrong prediction cannot silently train a checkpoint. When the Spark import
confirms it, set ``RUNTIME_ORDER_STATUS = "confirmed"`` in the same commit that
records the evidence (the ``validate.py`` transcript printing the joint names).
If it is wrong, correct the tuple here and in
``hexapod_env.assets.spec.MKII_V1_ASSET`` together; the contract test binds
them.

Unlike v1, these names do encode leg and side: ``lf_`` left-front, ``rm_``
right-middle, etc. (left = body +x, forward = body -y). The name-to-physical-
leg mapping still has to be verified on the robot before deployment.
"""

from __future__ import annotations


__all__ = [
    "ASSET_NAME",
    "COXA_SLICE",
    "DEFAULT_JOINT_POSITIONS_RAD",
    "DEFAULT_RESET_ROOT_HEIGHT_M",
    "FEMUR_SLICE",
    "JOINT_COUNT",
    "JOINTS_PER_GROUP",
    "LEG_NAMES",
    "RUNTIME_JOINT_NAMES",
    "RUNTIME_ORDER_STATUS",
    "SCHEMA_VERSION",
    "TIBIA_SLICE",
    "group_of",
    "index_of",
    "leg_of",
]

SCHEMA_VERSION = 2
ASSET_NAME = "mkii_v1"
RUNTIME_ORDER_STATUS = "provisional"

JOINT_COUNT = 18
JOINTS_PER_GROUP = 6

COXA_SLICE = slice(0, 6)
FEMUR_SLICE = slice(6, 12)
TIBIA_SLICE = slice(12, 18)

#: Leg order within each group: left/right x front/middle/rear.
LEG_NAMES: tuple[str, ...] = ("lf", "lm", "lr", "rf", "rm", "rr")

#: Action index -> joint name. Positional; index is the contract, not the name.
RUNTIME_JOINT_NAMES: tuple[str, ...] = (
    # coxa, indices 0-5
    "lf_coxa_yaw",
    "lm_coxa_yaw",
    "lr_coxa_yaw",
    "rf_coxa_yaw",
    "rm_coxa_yaw",
    "rr_coxa_yaw",
    # femur, indices 6-11
    "lf_femur_pitch",
    "lm_femur_pitch",
    "lr_femur_pitch",
    "rf_femur_pitch",
    "rm_femur_pitch",
    "rr_femur_pitch",
    # tibia, indices 12-17
    "lf_tibia_pitch",
    "lm_tibia_pitch",
    "lr_tibia_pitch",
    "rf_tibia_pitch",
    "rm_tibia_pitch",
    "rr_tibia_pitch",
)

# Reset/default joint positions in runtime order, radians. The action pipeline
# offsets from these. Source: hexapod_env.assets.spec.MKII_V1_ASSET (mirrors
# robot/hexapod_mkii_assy/stance.json): coxa 0.0, femur -0.25, tibia -0.55,
# released at a root height of 0.130 m and settling at 0.124 m.
DEFAULT_JOINT_POSITIONS_RAD: tuple[float, ...] = (
    (0.0,) * JOINTS_PER_GROUP + (-0.25,) * JOINTS_PER_GROUP + (-0.55,) * JOINTS_PER_GROUP
)
DEFAULT_RESET_ROOT_HEIGHT_M = 0.130

_GROUP_SLICES: tuple[tuple[str, slice], ...] = (
    ("coxa", COXA_SLICE),
    ("femur", FEMUR_SLICE),
    ("tibia", TIBIA_SLICE),
)


def group_of(index: int) -> str:
    """Return ``"coxa"``, ``"femur"`` or ``"tibia"`` for an action index."""

    for name, group in _GROUP_SLICES:
        if group.start <= index < group.stop:
            return name
    raise IndexError(f"Action index out of range [0, {JOINT_COUNT}): {index!r}")


def leg_of(index: int) -> str:
    """Return the leg name (``"lf"`` ... ``"rr"``) for an action index."""

    if not 0 <= index < JOINT_COUNT:
        raise IndexError(f"Action index out of range [0, {JOINT_COUNT}): {index!r}")
    return LEG_NAMES[index % JOINTS_PER_GROUP]


def index_of(joint_name: str) -> int:
    """Return the action index of a runtime joint name."""

    try:
        return RUNTIME_JOINT_NAMES.index(joint_name)
    except ValueError:
        raise KeyError(f"Unknown runtime joint name {joint_name!r}") from None
