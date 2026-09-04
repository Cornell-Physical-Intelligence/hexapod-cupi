"""Build an Isaac Lab ``ArticulationCfg`` from a :class:`HexapodAssetSpec`.

The physics properties (rigid-body caps, solver iterations, soft limit factor)
and the RS05 actuator model are shared by every asset; only the USD, the joint
names, the stance and the reset height come from the spec. ``HEXAPOD_CFG`` in
``asset_cfg.py`` stays a literal for the Phase-0 mock so its frozen values are
visible to the contract tests; new assets are built here.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

from ..asset_cfg import ROBSTRIDE_RS05_CFG
from .spec import MKII_V1_ASSET, HexapodAssetSpec


__all__ = [
    "HEXAPOD_MKII_V1_CFG",
    "MKII_V1_USD_PATH",
    "articulation_cfg_from_spec",
    "resolve_usd_path",
]


def resolve_usd_path(spec: HexapodAssetSpec) -> str:
    """The asset's USD path: its environment override, else the container path."""

    return os.environ.get(spec.usd_env_var, spec.usd_path_container)


def articulation_cfg_from_spec(
    spec: HexapodAssetSpec,
    *,
    actuator_cfg: DCMotorCfg = ROBSTRIDE_RS05_CFG,
    usd_path: str | None = None,
) -> ArticulationCfg:
    """Articulation config for ``spec`` with the shared physics settings.

    Mirrors ``asset_cfg.HEXAPOD_CFG`` property for property; a change to the
    shared physics settings belongs in both places until the mock is retired.
    """

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path if usd_path is not None else resolve_usd_path(spec),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=20.0,
                max_angular_velocity=50.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # The standing validator (isaaclab/validate.py), not this stance,
            # is the torque acceptance gate.
            pos=(0.0, 0.0, spec.reset_root_height_m),
            joint_pos=spec.default_joint_positions(),
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.95,
        actuators={"legs": actuator_cfg},
    )


MKII_V1_USD_PATH = resolve_usd_path(MKII_V1_ASSET)
HEXAPOD_MKII_V1_CFG = articulation_cfg_from_spec(MKII_V1_ASSET, usd_path=MKII_V1_USD_PATH)
