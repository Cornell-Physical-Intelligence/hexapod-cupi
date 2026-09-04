"""Per-asset description of a hexapod articulation, in one place.

Every robot model the task can load is described by one
:class:`HexapodAssetSpec`: where its USD lives, what its root link and joints
are called, the joint limits, the reset stance, and the link geometry the
environment needs (foot-pad extent along the tibia). Environment configs read
these values instead of carrying their own literals, so adding a robot means
adding a spec here and one env config that points at it; nothing else moves.

This module is deliberately stdlib-only and free of relative imports so a
contract test can load it by file path on a machine without Isaac Lab. The
Isaac Lab ``ArticulationCfg`` is built from a spec in :mod:`.articulation`.

Two specs exist today:

``MOCK_ASSET``
    ``robot/hexapod_mkii_mock_assy``, the Onshape mock every Phase-0 checkpoint
    was trained on (1.5 kg body / 0.8 kg legs, user-specified targets). Its
    values are the frozen literals in ``asset_cfg.py`` / ``env_cfg.py``; the
    contract test binds them and they never change.

``MKII_V1_ASSET``
    ``robot/hexapod_mkii_assy``, the CAD assembly imported from the 2026-09-03
    onshape-to-robot export (ADR-0001 asset v1): per-part Onshape inertials,
    every RS05 hard-set to 191 g, 8.261 kg total, no sensor payload yet.
"""

from __future__ import annotations

from dataclasses import dataclass


__all__ = [
    "HexapodAssetSpec",
    "MKII_V1_ASSET",
    "MOCK_ASSET",
    "RUNTIME_ORDER_CONFIRMED",
    "RUNTIME_ORDER_PROVISIONAL",
]

#: The runtime joint order has been read back from an imported articulation and
#: matches the spec; checkpoints may be trained against it.
RUNTIME_ORDER_CONFIRMED = "confirmed"
#: The runtime joint order is a prediction (PhysX orders articulation links
#: breadth-first from the root, children in USD order). The environment
#: refuses to run if the imported articulation disagrees; confirm it, then flip
#: this to ``RUNTIME_ORDER_CONFIRMED`` in the same commit as the evidence.
RUNTIME_ORDER_PROVISIONAL = "provisional"


@dataclass(frozen=True)
class HexapodAssetSpec:
    """Everything the task needs to know about one robot model."""

    #: Short stable identifier, used in task IDs and log labels.
    name: str
    #: Repo-relative URDF the USD was imported from (documentation and tests).
    urdf_path: str
    #: USD path inside the Spark container, the default when the environment
    #: variable below is unset.
    usd_path_container: str
    #: Environment variable that overrides the USD path. One per asset, so an
    #: override for one robot never redirects another.
    usd_env_var: str
    #: Root (base) link name; also the prim under ``Robot/Geometry`` that the
    #: URDF importer nests every other link beneath.
    root_link: str
    #: Leg names in the order the leg tuples below are declared.
    leg_names: tuple[str, ...]
    coxa_joints: tuple[str, ...]
    femur_joints: tuple[str, ...]
    tibia_joints: tuple[str, ...]
    #: ``(coxa, femur, tibia)`` link names per leg, same leg order as above.
    leg_link_names: tuple[tuple[str, str, str], ...]
    #: Regex (relative to the geometry root) matching every coxa link at once.
    coxa_link_regex: str
    #: URDF joint limits, radians.
    coxa_limits: tuple[float, float]
    femur_limits: tuple[float, float]
    tibia_limits: tuple[float, float]
    #: Reset/default joint positions shared by every leg, radians.
    stance_coxa_rad: float
    stance_femur_rad: float
    stance_tibia_rad: float
    #: Root height the articulation is released at on reset.
    reset_root_height_m: float
    #: Base-height target the flat task tracks while moving.
    nominal_height_m: float
    #: Tibia link-frame +Y runs knee -> foot pad. Contact points above this
    #: are pad contact; below it is the shaft (an undesired contact).
    distal_foot_min_y_m: float
    #: Distal pad offset along tibia +Y used by the swing-clearance reward.
    swing_clearance_pad_offset_y_m: float
    #: Expected articulation joint order (index = action index), 18 names,
    #: coxa 0-5, femur 6-11, tibia 12-17.
    runtime_joint_names: tuple[str, ...]
    #: ``RUNTIME_ORDER_CONFIRMED`` or ``RUNTIME_ORDER_PROVISIONAL``.
    runtime_order_status: str
    #: Documentary total mass and where it comes from; not consumed by physics.
    mass_kg: float
    mass_note: str

    def __post_init__(self) -> None:
        groups = (self.coxa_joints, self.femur_joints, self.tibia_joints)
        if any(len(group) != 6 for group in groups) or len(self.leg_link_names) != 6:
            raise ValueError(f"{self.name}: a hexapod spec needs six of everything")
        if len(self.runtime_joint_names) != 18 or len(set(self.runtime_joint_names)) != 18:
            raise ValueError(f"{self.name}: runtime joint order needs 18 distinct names")
        if set(self.runtime_joint_names) != set(self.all_joints):
            raise ValueError(f"{self.name}: runtime joint names are not the declared joints")
        for group, group_slice in zip(groups, (slice(0, 6), slice(6, 12), slice(12, 18))):
            if set(self.runtime_joint_names[group_slice]) != set(group):
                raise ValueError(f"{self.name}: runtime order breaks the coxa/femur/tibia blocks")
        if self.runtime_order_status not in (RUNTIME_ORDER_CONFIRMED, RUNTIME_ORDER_PROVISIONAL):
            raise ValueError(f"{self.name}: unknown runtime order status {self.runtime_order_status!r}")
        for value, (low, high), label in (
            (self.stance_coxa_rad, self.coxa_limits, "coxa"),
            (self.stance_femur_rad, self.femur_limits, "femur"),
            (self.stance_tibia_rad, self.tibia_limits, "tibia"),
        ):
            if not low <= value <= high:
                raise ValueError(f"{self.name}: {label} stance {value} outside limits {low}..{high}")

    @property
    def all_joints(self) -> tuple[str, ...]:
        return self.coxa_joints + self.femur_joints + self.tibia_joints

    def geometry_root_prim(self, robot_prim_path: str) -> str:
        """Prim path of the root link's geometry under a spawned robot."""

        return f"{robot_prim_path}/Geometry/{self.root_link}"

    def default_joint_positions(self) -> dict[str, float]:
        """``init_state.joint_pos`` dictionary for this asset's stance."""

        return {
            **{name: self.stance_coxa_rad for name in self.coxa_joints},
            **{name: self.stance_femur_rad for name in self.femur_joints},
            **{name: self.stance_tibia_rad for name in self.tibia_joints},
        }


# ---------------------------------------------------------------------------
# Phase-0 mock asset. These literals mirror asset_cfg.py / env_cfg.py and the
# v1 runtime order in hexapod_core.joints; the contract test binds them.
# ---------------------------------------------------------------------------
MOCK_ASSET = HexapodAssetSpec(
    name="mock",
    urdf_path="robot/hexapod_mkii_mock_assy/urdf/hexapod_mkii_robstride.urdf",
    usd_path_container=(
        "/workspace/hexapod/robot/hexapod_mkii_mock_assy/usd/"
        "hexapod_mkii_robstride/hexapod_mkii_robstride.usda"
    ),
    usd_env_var="HEXAPOD_USD_PATH",
    root_link="root",
    leg_names=("leg0", "leg1", "leg2", "leg3", "leg4", "leg5"),
    coxa_joints=(
        "revolute_1_1",
        "revolute_1_7",
        "revolute_2_5",
        "revolute_3",
        "revolute_4",
        "revolute_5",
    ),
    femur_joints=(
        "revolute_1",
        "revolute_1_2",
        "revolute_1_3",
        "revolute_1_4",
        "revolute_1_5",
        "revolute_1_6",
    ),
    tibia_joints=(
        "revolute_2",
        "revolute_2_1",
        "revolute_2_2",
        "revolute_2_3",
        "revolute_2_4",
        "revolute_2_6",
    ),
    leg_link_names=(
        ("coxa", "femur", "tibia"),
        ("coxa_1", "femur_1", "tibia_1"),
        ("coxa_2", "femur_2", "tibia_2"),
        ("coxa_3", "femur_3", "tibia_3"),
        ("coxa_4", "femur_4", "tibia_4"),
        ("coxa_5", "femur_5", "tibia_5"),
    ),
    coxa_link_regex="coxa.*",
    coxa_limits=(-0.872665, 0.872665),
    femur_limits=(0.0, 1.74533),
    tibia_limits=(0.0, 2.53073),
    stance_coxa_rad=0.0,
    stance_femur_rad=0.40,
    stance_tibia_rad=2.10,
    reset_root_height_m=0.210,
    nominal_height_m=0.205,
    distal_foot_min_y_m=0.18,
    swing_clearance_pad_offset_y_m=0.21,
    runtime_joint_names=(
        "revolute_1_1",
        "revolute_1_7",
        "revolute_2_5",
        "revolute_3",
        "revolute_4",
        "revolute_5",
        "revolute_1",
        "revolute_1_6",
        "revolute_1_5",
        "revolute_1_3",
        "revolute_1_4",
        "revolute_1_2",
        "revolute_2",
        "revolute_2_6",
        "revolute_2_4",
        "revolute_2_2",
        "revolute_2_3",
        "revolute_2_1",
    ),
    runtime_order_status=RUNTIME_ORDER_CONFIRMED,
    mass_kg=6.3,
    mass_note="user-specified targets: 1.5 kg body + six 0.8 kg complete legs",
)


# ---------------------------------------------------------------------------
# Asset v1: the CAD assembly (robot/hexapod_mkii_assy, hexapod_mkii_serial.urdf).
# Body frame: z up, forward = -y, left = +x. coxa_yaw zero points the leg
# straight out of the body (positive = counter-clockwise about body +z);
# femur/tibia zero is the CAD pose (femur 46 deg above horizontal, knee
# interior angle 74 deg; positive femur raises, positive tibia opens). Limits
# and stance come from robot/hexapod_mkii_assy/joint_limits.json and
# stance.json; the femur/tibia ranges are derived from the CAD kinematics and
# must be replaced by measured hardware stops.
# ---------------------------------------------------------------------------
_MKII_LEGS = ("lf", "lm", "lr", "rf", "rm", "rr")

MKII_V1_ASSET = HexapodAssetSpec(
    name="mkii_v1",
    urdf_path="robot/hexapod_mkii_assy/urdf/hexapod_mkii_serial.urdf",
    usd_path_container=(
        "/workspace/hexapod/robot/hexapod_mkii_assy/usd/"
        "hexapod_mkii_serial/hexapod_mkii_serial.usda"
    ),
    usd_env_var="HEXAPOD_MKII_V1_USD_PATH",
    root_link="body",
    leg_names=_MKII_LEGS,
    coxa_joints=tuple(f"{leg}_coxa_yaw" for leg in _MKII_LEGS),
    femur_joints=tuple(f"{leg}_femur_pitch" for leg in _MKII_LEGS),
    tibia_joints=tuple(f"{leg}_tibia_pitch" for leg in _MKII_LEGS),
    leg_link_names=tuple(
        (f"{leg}_coxa", f"{leg}_femur", f"{leg}_tibia") for leg in _MKII_LEGS
    ),
    coxa_link_regex=".*_coxa",
    coxa_limits=(-0.872665, 0.872665),
    femur_limits=(-1.745329, 0.55),
    tibia_limits=(-0.95, 1.75),
    # Feet 0.11 m outboard of the yaw axes, knee interior angle 42 deg, femur
    # 32 deg above horizontal; static hip and knee torque 0.9 N*m each at
    # 8.26 kg. Bottom plate settles 0.124 m above the ground; released 6 mm
    # higher.
    stance_coxa_rad=0.0,
    stance_femur_rad=-0.25,
    stance_tibia_rad=-0.55,
    reset_root_height_m=0.130,
    nominal_height_m=0.124,
    # Foot pad collision spheres sit at tibia +Y 0.177 and 0.206 m with a
    # 0.016 m radius: pad contact lies above y = 0.161 m, the machined shaft
    # ends below it.
    distal_foot_min_y_m=0.155,
    swing_clearance_pad_offset_y_m=0.206,
    # Predicted PhysX order: breadth-first from ``body`` (all six coxa_yaw in
    # URDF leg order, then the femurs, then the tibias). Provisional until the
    # imported articulation has been read back; the environment enforces it.
    runtime_joint_names=(
        tuple(f"{leg}_coxa_yaw" for leg in _MKII_LEGS)
        + tuple(f"{leg}_femur_pitch" for leg in _MKII_LEGS)
        + tuple(f"{leg}_tibia_pitch" for leg in _MKII_LEGS)
    ),
    runtime_order_status=RUNTIME_ORDER_PROVISIONAL,
    mass_kg=8.26081,
    mass_note=(
        "Onshape per-part masses with every RS05 hard-set to 191 g "
        "(assembly_report.md); no sensor payload yet"
    ),
)
