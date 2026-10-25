# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from ..agents.rsl_rl_ppo_cfg import HexapodFlatPPORunnerCfg


@configclass
class HexapodTerrainPPORunnerCfg(HexapodFlatPPORunnerCfg):
    """The flat 128/128/128 actor and critic, so a flat BC + PPO checkpoint resumes here, at the fine-tuning rate
    the stage1v21 runs used."""

    max_iterations = 2000
    experiment_name = "hexapod_rough_v21_direct"
    algorithm = HexapodFlatPPORunnerCfg().algorithm.replace(learning_rate=2.0e-4)
