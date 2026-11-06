# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Blind rough-terrain task on terrain v2.1, with the settings of the stage1v21 runs in hexapod-locomotion."""

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from ..hexapod_env_cfg import HexapodFlatEnvCfg
from .terrains_cfg import HEXAPOD_TERRAINS_CFG


@configclass
class HexapodTerrainEnvCfg(HexapodFlatEnvCfg):
    """Blind (66 values + clock, no height scanner), 10-level terrain curriculum, everyone starts at level 0."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=HEXAPOD_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    # env origins come from the 2 m terrain tiles; env_spacing only places the cloned prims before the first reset
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    # trained settings: tripod gait clock, forward/backward walking at 0.1-0.35 m/s, small yaw commands
    gait_clock_enabled = True
    gait_period_s = 0.4
    command_forward_only = True
    command_lin_vel_min = 0.1
    command_lin_vel_max = 0.35
    command_yaw_rate_max = 0.1
    feet_air_time_threshold = 0.0
    flat_orientation_reward_scale = 0.0

    # Swing clearance: for every foot not in contact while the robot is commanded to move, reward
    # min(toe height above the mean height of the feet in contact, target) / target. Blind proxy for lifting the
    # feet over an unseen riser. Target 0.06 m: just above the 0.052 m step cap.
    swing_clearance_reward_scale: float = 0.05
    swing_clearance_target: float = 0.06

    # Success, logged as Metrics/success_eval: the episode reached time-out without a fall AND
    # | episode-mean speed along the command - |command| | < success_speed_tol_ratio * |command|.
    success_speed_tol_ratio: float = 0.25
    success_min_cmd_ratio: float = 0.1
    """Commands below this fraction of command_lin_vel_max use that floor as the tolerance base."""

    # Terrain curriculum: up a level on a success, down on a fall or an episode below this fraction of |command|.
    terrain_curriculum: bool = True
    curriculum_move_down_speed_ratio: float = 0.5

    # Command-speed curriculum (top-speed sweep), off by default: command_lin_vel_max starts at
    # command_curriculum_start and rises by command_curriculum_step whenever, over the last
    # command_curriculum_window_episodes finished episodes whose command was in the top bucket [max - step, max],
    # the success rate is >= command_curriculum_success. Capped at command_curriculum_max.
    # The top-speed runs were on flat ground (terrain_type "plane") with step 0.05 and max 0.8; the extension run
    # resumed at start 0.55 and reached ~0.65 m/s.
    command_curriculum: bool = False
    command_curriculum_start: float = 0.15
    command_curriculum_step: float = 0.05
    command_curriculum_max: float = 0.4
    command_curriculum_success: float = 0.8
    command_curriculum_window_episodes: int = 256
    """Counted in finished top-bucket episodes: 20 iterations x 24 steps is shorter than one 20 s episode."""
