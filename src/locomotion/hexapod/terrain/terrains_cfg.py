# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scaled rough terrain for the 0.1 m hexapod ("terrain v2.2").

Ported from github.com/ShauryaSen/hexapod-locomotion (``hexapod_terrains_cfg.py``, version 2). The robot stands
0.10 m tall (ANYmal: 0.6 m), so the Isaac Lab rough terrain is scaled down: 2 m tiles, 10 rows = 10 curriculum
levels, two columns per type. Six types: random rough, slope up, slope down, pyramid stairs up, inverted pyramid
stairs (a pit), random boxes up and down. The largest single edge at level 9 equals ``STEP_CAP``.

v2.2 differs from the v2.1 geometry the stage1v21 runs trained on in three ways, each so that every reset puts all
six toes on level ground at the origin (``geometry.spawn_half_extent``):

- Stairs and pits request a 1.0 m platform, which the pinned generator builds as 0.92 m with 2 risers per side. v2.1
  requested 0.8 m, which builds 0.42 m with 3 risers, inside the 0.511 m x 0.610 m toe footprint.
- Slope tiles have a 0.8 m flat platform (v2.1: 0.5 m) and their origin height is the highest point under the spawn
  area. Isaac Lab's height-field origin is the maximum over a 2 m window, which on a 2 m tile includes the zero-height
  border: a level-9 inverted slope spawned 0.135 m above its floor.
- The flat border around the course is 9.5 m (v2.1: 5 m), enough for a successful episode at the largest supported
  command from an edge tile (``geometry.required_course_border``).

``make_hexapod_terrains_cfg(obstacle_corridor=True)`` adds a seventh type, a walled corridor of static boxes and pillars
(``corridor.py``), in two more columns. The default course is v2.2 unchanged. The task is blind, so the corridor tests
recovery from walking into walls and obstacles, not navigation.
"""

import numpy as np

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field import hf_terrains, hf_terrains_cfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

from . import corridor
from .geometry import mesh_stairs_platform, patch_max_height, spawn_half_extent

STEP_CAP = 0.052
"""Largest stair riser / box edge [m] = 0.8 x the 0.065 m belly clearance of the standing robot."""
STAIR_TREAD = 0.25
"""Stair tread [m]: two strides of the learned gait at 0.28 m/s (0.124 m median stride)."""
STAIR_PLATFORM = 1.0
"""Requested flat top of the stairs / bottom of the pit [m]. The pinned generator builds 0.92 m (2 risers per side),
the 0.61 m toe footprint plus 0.15 m on each side; see ``geometry.mesh_stairs_platform``."""
BOX_PLATFORM = 0.8
"""Flat centre of the box tile [m]; the box generator builds exactly this width."""
BOX_GRID_WIDTH = 0.12
"""Random-box grid width [m], one stride."""
TILE_SIZE = 2.0
NUM_LEVELS = 10
SLOPE_PLATFORM = 0.8
"""Flat platform at the centre of the slope tiles [m], wide enough for the toe footprint."""
HF_BORDER = 0.04
"""Flat border of each height-field tile [m] (must be >= horizontal_scale)."""
COURSE_BORDER = 9.5
"""Flat ground around the course [m]: >= ``geometry.required_course_border`` for the task's largest command."""
HORIZONTAL_SCALE = 0.02


@height_field_to_mesh
def random_uniform_terrain_leveled(difficulty: float, cfg: hf_terrains_cfg.HfRandomUniformTerrainCfg) -> np.ndarray:
    """Random uniform noise whose amplitude grows with the curriculum level.

    Isaac Lab's ``random_uniform_terrain`` ignores ``difficulty``. Here the upper end of the noise range is
    interpolated: level 0 uses (lo, lo + step), level 9 uses the full (lo, hi).
    """
    lo, hi = cfg.noise_range
    hi_d = max(lo + difficulty * (hi - lo), lo + cfg.noise_step)
    return hf_terrains.random_uniform_terrain.__wrapped__(difficulty, cfg.replace(noise_range=(lo, hi_d)))


def _spawn_area_origin(generated, cfg):
    """Replace the origin height of a height-field tile ``([mesh], origin)`` with the highest point under the spawn
    area, in tile coordinates."""
    meshes, origin = generated
    hx, hy = spawn_half_extent()
    pad = cfg.horizontal_scale
    vertices = np.vstack([mesh.vertices for mesh in meshes])
    origin = np.array(origin, dtype=float)
    origin[2] = patch_max_height(vertices, origin[:2], (hx + pad, hy + pad))
    return meshes, origin


# The sub-terrain functions are plain module-level functions: Isaac Lab serializes ``cfg.function`` as
# ``module:__name__`` (``callable_to_string``) and Hydra overrides rebuild it from that string, so a decorated wrapper
# carrying the wrapped function's name would silently resolve back to the stock function.
def random_rough_terrain(difficulty: float, cfg: hf_terrains_cfg.HfRandomUniformTerrainCfg):
    """``random_uniform_terrain_leveled`` with its origin height from the spawn area."""
    return _spawn_area_origin(random_uniform_terrain_leveled(difficulty, cfg), cfg)


def sloped_terrain(difficulty: float, cfg: hf_terrains_cfg.HfPyramidSlopedTerrainCfg):
    """Isaac Lab's ``pyramid_sloped_terrain`` (upright or inverted) with its origin height from the spawn area."""
    return _spawn_area_origin(hf_terrains.pyramid_sloped_terrain(difficulty, cfg), cfg)


def obstacle_corridor_terrain(difficulty: float, cfg: "ObstacleCorridorTerrainCfg"):
    """Walled corridor of static obstacles whose gap shrinks with ``difficulty``; origin at the tile centre on the
    ground. Returns ``([mesh, ...], origin)`` like Isaac Lab's mesh sub-terrains."""
    sx, sy = cfg.size
    wide_gap, narrow_gap = cfg.gap_range
    low_height, high_height = cfg.obstacle_height_range
    spawn_half = spawn_half_extent()
    width = sx - 2 * cfg.wall_thickness
    dimensions = (*cfg.size, *cfg.gap_range, *cfg.obstacle_height_range,
                  cfg.wall_thickness, cfg.wall_height, cfg.spawn_clearance)
    if not np.isfinite(dimensions).all():
        raise ValueError("Corridor dimensions and ranges must be finite")
    if not 0 < narrow_gap <= wide_gap <= width:
        raise ValueError("Corridor gap_range must be positive, nonincreasing and fit between the walls")
    if not (cfg.wall_thickness > 0 and cfg.wall_height > 0 and 0 < low_height <= high_height):
        raise ValueError("Corridor wall dimensions and ordered obstacle heights must be positive")
    if (
        cfg.spawn_clearance < 0
        or width < 2 * (spawn_half[0] + cfg.spawn_clearance)
        or sy < 2 * (spawn_half[1] + cfg.spawn_clearance)
    ):
        raise ValueError("Corridor spawn area and clearance must fit inside the tile and walls")
    gap = corridor.corridor_gap(difficulty, cfg.gap_range)
    obstacles = corridor.place_obstacles(
        corridor.corridor_rng(cfg.seed if cfg.layout_seed is None else cfg.layout_seed, difficulty),
        cfg.size,
        gap,
        spawn_half,
        num_obstacles=cfg.num_obstacles,
        wall_thickness=cfg.wall_thickness,
        spawn_clearance=cfg.spawn_clearance,
        height_range=cfg.obstacle_height_range,
    )
    meshes = corridor.corridor_meshes(obstacles, cfg.size, cfg.wall_thickness, cfg.wall_height)
    return meshes, np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])


@configclass
class ObstacleCorridorTerrainCfg(terrain_gen.SubTerrainBaseCfg):
    """Config of ``obstacle_corridor_terrain``. ``TerrainGenerator`` overwrites ``seed`` with its own, so the layout uses
    ``layout_seed`` when set, else that generator seed, else NumPy's global state."""

    function = obstacle_corridor_terrain
    layout_seed: int | None = None
    num_obstacles: int = corridor.NUM_OBSTACLES
    gap_range: tuple[float, float] = corridor.GAP_RANGE
    wall_thickness: float = corridor.WALL_THICKNESS
    wall_height: float = corridor.WALL_HEIGHT
    obstacle_height_range: tuple[float, float] = corridor.OBSTACLE_HEIGHT_RANGE
    spawn_clearance: float = corridor.SPAWN_CLEARANCE


def add_obstacle_corridor(cfg: TerrainGeneratorCfg) -> None:
    """Add corridor terrain with the weight of two columns in the default curriculum.

    With the curriculum disabled, Isaac Lab samples tile types by weight.
    """
    if "obstacle_corridor" not in cfg.sub_terrains:
        total = sum(terrain.proportion for terrain in cfg.sub_terrains.values())
        if cfg.num_cols < 1 or not np.isfinite(total) or total <= 0:
            raise ValueError("The terrain generator needs a positive column count and total proportion")
        proportion = 2 * total / cfg.num_cols
        cfg.sub_terrains["obstacle_corridor"] = ObstacleCorridorTerrainCfg(proportion=proportion)
        cfg.num_cols += 2


def make_hexapod_terrains_cfg(
    step_cap: float = STEP_CAP,
    tread: float = STAIR_TREAD,
    stair_platform: float = STAIR_PLATFORM,
    box_platform: float = BOX_PLATFORM,
    grid_width: float = BOX_GRID_WIDTH,
    tile_size: float = TILE_SIZE,
    course_border: float = COURSE_BORDER,
    obstacle_corridor: bool = False,
) -> TerrainGeneratorCfg:
    """Build the generator config. Stair risers span (0.005, step_cap) over the levels; box heights span
    (0.0025, step_cap / 2), so an up-box next to a down-box is one step_cap edge. ``obstacle_corridor`` adds the
    ``obstacle_corridor`` type after the six v2.2 types.

    Raises ``ValueError`` when a spawn platform would not hold the toe footprint.
    """
    need = 2 * max(spawn_half_extent())
    built = mesh_stairs_platform(tile_size, HF_BORDER, stair_platform, tread)
    for name, width in (("stairs", built), ("boxes", box_platform), ("slope", SLOPE_PLATFORM)):
        if width < need:
            raise ValueError(f"{name} platform {width:.3f} m is narrower than the {need:.3f} m spawn area")
    rough = terrain_gen.HfRandomUniformTerrainCfg(
        proportion=1.0, noise_range=(0.005, 0.02), noise_step=0.005, border_width=HF_BORDER
    )
    rough.function = random_rough_terrain  # instance attribute: never bound as a method
    slope = terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=1.0, slope_range=(0.0, 0.27), platform_width=SLOPE_PLATFORM, border_width=HF_BORDER
    )
    slope.function = sloped_terrain
    slope_inv = terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        proportion=1.0, slope_range=(0.0, 0.27), platform_width=SLOPE_PLATFORM, border_width=HF_BORDER
    )
    slope_inv.function = sloped_terrain
    sub_terrains = {
        "random_rough": rough,
        "slope": slope,
        "slope_inv": slope_inv,
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
            proportion=1.0, grid_width=grid_width, grid_height_range=(0.0025, step_cap / 2), platform_width=box_platform
        ),
    }
    cfg = TerrainGeneratorCfg(
        size=(tile_size, tile_size),
        border_width=course_border,
        num_rows=NUM_LEVELS,
        num_cols=2 * len(sub_terrains),
        horizontal_scale=HORIZONTAL_SCALE,
        vertical_scale=0.001,
        slope_threshold=0.75,
        curriculum=True,
        use_cache=False,
        sub_terrains=sub_terrains,
    )
    if obstacle_corridor:
        add_obstacle_corridor(cfg)
    return cfg


HEXAPOD_TERRAINS_CFG = make_hexapod_terrains_cfg()
"""Terrain v2.2: 10 levels x 12 columns (two per type)."""


def terrain_type_names(cfg: TerrainGeneratorCfg) -> list[str]:
    """Sub-terrain name of each column, with the rule ``TerrainGenerator._generate_curriculum_terrains`` uses."""
    names = list(cfg.sub_terrains.keys())
    proportions = np.array([sub.proportion for sub in cfg.sub_terrains.values()], dtype=float)
    proportions /= proportions.sum()
    return [
        names[int(np.min(np.where(col / cfg.num_cols + 0.001 < np.cumsum(proportions))[0]))]
        for col in range(cfg.num_cols)
    ]
