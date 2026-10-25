# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hexapod rough-terrain task on terrain v2.1 (stairs, pits, boxes, slopes) with the terrain and speed curricula.

Import this module (callback ``locomotion.hexapod.terrain.register``) to register the task beside the original two.
"""

import gymnasium as gym

gym.register(
    id="Hexapod-Rough-v21-Direct-v0",
    entry_point=f"{__name__}.env:HexapodTerrainEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:HexapodTerrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:HexapodTerrainPPORunnerCfg",
    },
)


def register():
    pass
