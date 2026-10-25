# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scaled rough terrain for the 0.1 m hexapod ("terrain v2.1").

Ported from github.com/ShauryaSen/hexapod-locomotion (``hexapod_terrains_cfg.py``, version 2). The robot stands
0.10 m tall (ANYmal: 0.6 m), so the Isaac Lab rough terrain is scaled down: 2 m tiles, 10 rows = 10 curriculum
levels, two columns per type. Six types: random rough, slope up, slope down, pyramid stairs up, inverted pyramid
stairs (a pit), random boxes up and down. The largest single edge at level 9 equals ``STEP_CAP``.
"""

import numpy as np

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field import hf_terrains, hf_terrains_cfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh

STEP_CAP = 0.052
"""Largest stair riser / box edge [m] = 0.8 x the 0.065 m belly clearance of the standing robot."""
STAIR_TREAD = 0.25
"""Stair tread [m]: two strides of the learned gait at 0.28 m/s (0.124 m median stride)."""
STAIR_PLATFORM = 0.8
"""Flat top of the stairs / bottom of the pit / centre of the box tile [m]: the 0.62 m foot span plus one stride."""
BOX_GRID_WIDTH = 0.12
"""Random-box grid width [m], one stride."""
TILE_SIZE = 2.0
NUM_LEVELS = 10
SLOPE_PLATFORM = 0.5
"""Flat platform at the centre of the slope tiles [m] (25% of the tile, ANYmal's ratio)."""
HF_BORDER = 0.04
"""Flat border of each height-field tile [m] (must be >= horizontal_scale)."""


@height_field_to_mesh
def random_uniform_terrain_leveled(difficulty: float, cfg: hf_terrains_cfg.HfRandomUniformTerrainCfg) -> np.ndarray:
    """Random uniform noise whose amplitude grows with the curriculum level.

    Isaac Lab's ``random_uniform_terrain`` ignores ``difficulty``. Here the upper end of the noise range is
    interpolated: level 0 uses (lo, lo + step), level 9 uses the full (lo, hi).
    """
    lo, hi = cfg.noise_range
    hi_d = max(lo + difficulty * (hi - lo), lo + cfg.noise_step)
    return hf_terrains.random_uniform_terrain.__wrapped__(difficulty, cfg.replace(noise_range=(lo, hi_d)))


def make_hexapod_terrains_cfg(
    step_cap: float = STEP_CAP,
    tread: float = STAIR_TREAD,
    stair_platform: float = STAIR_PLATFORM,
    grid_width: float = BOX_GRID_WIDTH,
    tile_size: float = TILE_SIZE,
) -> TerrainGeneratorCfg:
    """Build the generator config. Stair risers span (0.005, step_cap) over the levels; box heights span
    (0.0025, step_cap / 2), so an up-box next to a down-box is one step_cap edge."""
    rough = terrain_gen.HfRandomUniformTerrainCfg(
        proportion=1.0, noise_range=(0.005, 0.02), noise_step=0.005, border_width=HF_BORDER
    )
    rough.function = random_uniform_terrain_leveled  # set on the instance so it is never bound as a method
    sub_terrains = {
        "random_rough": rough,
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=1.0, slope_range=(0.0, 0.27), platform_width=SLOPE_PLATFORM, border_width=HF_BORDER
        ),
        "slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=1.0, slope_range=(0.0, 0.27), platform_width=SLOPE_PLATFORM, border_width=HF_BORDER
        ),
        "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.005, step_cap),
            step_width=tread,
            platform_width=stair_platform,
            border_width=HF_BORDER,
            holes=False,
        ),
        "stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.005, step_cap),
            step_width=tread,
            platform_width=stair_platform,
            border_width=HF_BORDER,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0, grid_width=grid_width, grid_height_range=(0.0025, step_cap / 2), platform_width=stair_platform
        ),
    }
    return TerrainGeneratorCfg(
        size=(tile_size, tile_size),
        # flat ground around the tiles: robots walk off the edge columns within an episode; 5 m covers 20 s at 0.25 m/s
        border_width=5.0,
        num_rows=NUM_LEVELS,
        num_cols=2 * len(sub_terrains),
        horizontal_scale=0.02,
        vertical_scale=0.001,
        slope_threshold=0.75,
        curriculum=True,
        use_cache=False,
        sub_terrains=sub_terrains,
    )


HEXAPOD_TERRAINS_CFG = make_hexapod_terrains_cfg()
"""Terrain v2.1: 10 levels x 12 columns (two per type)."""


def terrain_type_names(cfg: TerrainGeneratorCfg) -> list[str]:
    """Sub-terrain name of each column, with the rule ``TerrainGenerator._generate_curriculum_terrains`` uses."""
    names = list(cfg.sub_terrains.keys())
    proportions = np.array([sub.proportion for sub in cfg.sub_terrains.values()], dtype=float)
    proportions /= proportions.sum()
    return [
        names[int(np.min(np.where(col / cfg.num_cols + 0.001 < np.cumsum(proportions))[0]))]
        for col in range(cfg.num_cols)
    ]
