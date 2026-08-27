"""Frozen v1 runtime joint/action order for the 18 leg joints.

WHERE THIS ORDER COMES FROM, AND WHERE IT DOES NOT
--------------------------------------------------
The runtime order is the articulation's own joint order: Isaac Lab builds it
when it parses the USD, and the environment consumes it positionally
(``joint_pos``, ``joint_vel``, the action vector, the joint targets). It is
therefore *not* declared anywhere in the training Python. In particular it is
**not** ``asset_cfg.py``'s ``COXA_JOINTS`` / ``FEMUR_JOINTS`` /
``TIBIA_JOINTS``: those tuples are used only as *name sets* to build the
``init_state.joint_pos`` dictionary, and their declaration order differs from
the runtime order for the femur and tibia groups (coxa happens to coincide).

So the authoritative order is only written down in documentation --
``docs/TRAINING.md`` §2 and the archived ``docs/archive/HANDOFF-2026-08-26.md``
§7 -- and that table is what is encoded below. It is corroborated by runtime
telemetry: every ``per_joint`` array under ``artifacts/`` (308 evaluation and
video ``metrics.json`` files at the time of writing) is emitted in
``self.raw_env._robot.joint_names`` order by ``isaaclab/evaluate_checkpoint.py``,
and all of them agree with this table exactly.

Indices 0-5 are coxa, 6-11 femur, 12-17 tibia.

WARNING, carried over verbatim in substance from the handoff: the joint names
do not encode physical leg order or left/right sides. ``revolute_1_2`` is not
"leg 2", and adjacent indices are not adjacent legs. The environment's own gait
code resolves the tripod split geometrically at runtime for exactly this
reason. Do not infer physical leg order from these names, and do not deploy to
hardware without a separately verified name-to-leg mapping measured on the
robot.
"""

from __future__ import annotations


__all__ = [
    "ASSET_CFG_COXA_JOINTS",
    "ASSET_CFG_FEMUR_JOINTS",
    "ASSET_CFG_TIBIA_JOINTS",
    "BASE_DEFAULT_JOINT_POSITIONS_RAD",
    "COXA_SLICE",
    "FEMUR_SLICE",
    "JOINT_COUNT",
    "JOINTS_PER_GROUP",
    "RUNTIME_JOINT_NAMES",
    "SCHEMA_VERSION",
    "STAGE2C_DEFAULT_JOINT_POSITIONS_RAD",
    "TIBIA_SLICE",
    "group_of",
    "index_of",
]

SCHEMA_VERSION = 1

JOINT_COUNT = 18
JOINTS_PER_GROUP = 6

COXA_SLICE = slice(0, 6)
FEMUR_SLICE = slice(6, 12)
TIBIA_SLICE = slice(12, 18)

#: Action index -> joint name. Positional; index is the contract, not the name.
RUNTIME_JOINT_NAMES: tuple[str, ...] = (
    # coxa, indices 0-5
    "revolute_1_1",
    "revolute_1_7",
    "revolute_2_5",
    "revolute_3",
    "revolute_4",
    "revolute_5",
    # femur, indices 6-11
    "revolute_1",
    "revolute_1_6",
    "revolute_1_5",
    "revolute_1_3",
    "revolute_1_4",
    "revolute_1_2",
    # tibia, indices 12-17
    "revolute_2",
    "revolute_2_6",
    "revolute_2_4",
    "revolute_2_2",
    "revolute_2_3",
    "revolute_2_1",
)

# asset_cfg.py declaration order, kept only so the contract test can prove the
# runtime groups are permutations of these name sets. Never index with these.
ASSET_CFG_COXA_JOINTS: tuple[str, ...] = (
    "revolute_1_1",
    "revolute_1_7",
    "revolute_2_5",
    "revolute_3",
    "revolute_4",
    "revolute_5",
)
ASSET_CFG_FEMUR_JOINTS: tuple[str, ...] = (
    "revolute_1",
    "revolute_1_2",
    "revolute_1_3",
    "revolute_1_4",
    "revolute_1_5",
    "revolute_1_6",
)
ASSET_CFG_TIBIA_JOINTS: tuple[str, ...] = (
    "revolute_2",
    "revolute_2_1",
    "revolute_2_2",
    "revolute_2_3",
    "revolute_2_4",
    "revolute_2_6",
)

# Reset/default joint positions, in runtime order. The action pipeline offsets
# from these, so a wrong default is a wrong absolute pose even with a correct
# policy. Every joint in a group shares one value, so the group permutation
# above does not affect these vectors.
#
# Base default: asset_cfg.py :: HEXAPOD_CFG.init_state.joint_pos
# (coxa 0.0, femur 0.40, tibia 2.10) at root height 0.210 m.
BASE_DEFAULT_JOINT_POSITIONS_RAD: tuple[float, ...] = (
    (0.0,) * JOINTS_PER_GROUP + (0.40,) * JOINTS_PER_GROUP + (2.10,) * JOINTS_PER_GROUP
)
# Stage2C override: phase2_cfg.py ::
# HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg.robot init_state
# (coxa 0.0 inherited, femur 0.60, tibia 2.2335) at root height 0.185 m.
STAGE2C_DEFAULT_JOINT_POSITIONS_RAD: tuple[float, ...] = (
    (0.0,) * JOINTS_PER_GROUP
    + (0.60,) * JOINTS_PER_GROUP
    + (2.2335,) * JOINTS_PER_GROUP
)

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


def index_of(joint_name: str) -> int:
    """Return the action index of a runtime joint name."""

    try:
        return RUNTIME_JOINT_NAMES.index(joint_name)
    except ValueError:
        raise KeyError(f"Unknown runtime joint name {joint_name!r}") from None
