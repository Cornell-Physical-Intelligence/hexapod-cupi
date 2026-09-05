"""Opt-in CAD v2 integrity baseline; no historical task is redirected.

This config corrects import identity, reset clearance and anatomical commands.
It is a flat-ground baseline for the next simulator acceptance step; command
curricula/reward tuning and actuator identification remain separate work.
"""
from isaaclab.utils.configclass import configclass

from hexapod_core.cad_manifest_v2 import (
    ACTION_SCALE_RAD,
    COMMAND_ACTIVE_THRESHOLD,
    COMMAND_FRAME,
    DECIMATION,
    PHYSICS_DT_S,
    SLEW_LIMIT_RAD_PER_20MS,
    SOFT_JOINT_POS_LIMIT_FACTOR,
    STAND_ACTION_SCALE,
)

from ...assets.articulation import articulation_cfg_from_spec
from ...assets.mkii_v2 import MKII_V2_ASSET
from ...env_cfg import HexapodMkiiV1FlatEnvCfg
from ...ppo_cfg import HexapodPPORunnerCfg

HEXAPOD_MKII_V2_CFG = articulation_cfg_from_spec(MKII_V2_ASSET)
HEXAPOD_MKII_V2_CFG.soft_joint_pos_limit_factor = SOFT_JOINT_POS_LIMIT_FACTOR


@configclass
class HexapodMkiiV2FlatEnvCfg(HexapodMkiiV1FlatEnvCfg):
    """Same CAD links, corrected dynamics artifact/reset and body-to-nav frame."""

    robot = HEXAPOD_MKII_V2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    command_frame = COMMAND_FRAME
    nominal_height_m = MKII_V2_ASSET.nominal_height_m
    expected_runtime_joint_names = MKII_V2_ASSET.runtime_joint_names
    action_scale = ACTION_SCALE_RAD
    stand_action_scale = STAND_ACTION_SCALE
    axis_command_active_threshold = COMMAND_ACTIVE_THRESHOLD
    processed_joint_target_slew_limit_rad_per_20ms = SLEW_LIMIT_RAD_PER_20MS
    decimation = DECIMATION
    # Isaac Lab's configclass materializes inherited fields on instances;
    # the decorated subclass need not expose a class-level ``sim`` value.
    sim = HexapodMkiiV1FlatEnvCfg().sim.replace(dt=PHYSICS_DT_S, render_interval=DECIMATION)


@configclass
class HexapodMkiiV2PPORunnerCfg(HexapodPPORunnerCfg):
    """Separate provenance/output directory; this definition launches nothing."""

    experiment_name = "hexapod_mkii_v2_flat_direct"


__all__ = ["HexapodMkiiV2FlatEnvCfg", "HexapodMkiiV2PPORunnerCfg", "HEXAPOD_MKII_V2_CFG"]
