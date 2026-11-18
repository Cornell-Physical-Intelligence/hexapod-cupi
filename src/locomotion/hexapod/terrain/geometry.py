"""Reset-footprint geometry of the terrain v2.2 course, in NumPy only so the CPU tests can load this file directly.

The robot resets at its tile origin with identity yaw, the root ``ROOT_HEIGHT`` above the origin and the joints at
their defaults. Every spawn area must hold the six toe tips plus ``SPAWN_MARGIN`` on each side, level with the origin.
"""

from __future__ import annotations

import numpy as np

TOE_FOOTPRINT = (0.511, 0.610)
"""x and y extent [m] of the six toe tips at the default joint angles (coxa 0, femur -0.3, tibia 0.4 rad), from the
joint frames of the approved robot.usda (sha256 3be98420...) and the 0.130 m tibia toe offset."""
TOE_HALF_DIAGONAL = 0.348
"""Largest horizontal distance [m] from the root to a toe tip in the default pose (lf, rr: 0.166, 0.305)."""
TOE_DROP = 0.0975
"""Toe tips below the root [m] in the default pose."""
ROOT_HEIGHT = 0.103
"""Root height above the origin at reset [m] (``HEXAPOD_CFG.init_state.pos``)."""
SPAWN_MARGIN = 0.05
"""Clearance [m] between the outermost toe and the edge of a spawn platform, on each side."""


def spawn_half_extent(margin: float = SPAWN_MARGIN) -> tuple[float, float]:
    """Half-size [m] of the area around the origin that must be level with it at reset."""
    return TOE_FOOTPRINT[0] / 2 + margin, TOE_FOOTPRINT[1] / 2 + margin


def mesh_stairs_platform(size: float, border_width: float, platform_width: float, step_width: float) -> float:
    """Central platform width [m] that the pinned Isaac Lab (ffff603e) mesh pyramid stairs, upright or inverted, build.

    ``num_steps = (size - 2 border - platform) // (2 step) + 1``, and the platform is what the steps leave. The ``+ 1``
    makes it narrower than ``platform_width`` whenever the remainder is non-zero: 0.8 m requested on a 2 m tile with
    0.04 m borders and 0.25 m treads gives 3 steps and a 0.42 m platform.
    """
    num_steps = int((size - 2 * border_width - platform_width) // (2 * step_width) + 1)
    return size - 2 * border_width - 2 * num_steps * step_width


def mesh_stairs_steps(size: float, border_width: float, platform_width: float, step_width: float) -> int:
    """Number of risers per side of the same pyramid stairs."""
    return int((size - 2 * border_width - platform_width) // (2 * step_width) + 1)


def patch_max_height(vertices: np.ndarray, center_xy, half_extent) -> float:
    """Highest vertex [m] within the axis-aligned patch around ``center_xy``.

    For a height field, pad ``half_extent`` by one horizontal cell so the patch includes the vertices of every cell a
    toe can stand in; the surface between vertices is a linear interpolation, so it never exceeds this value.
    """
    v = np.asarray(vertices, dtype=float)
    inside = (np.abs(v[:, 0] - center_xy[0]) <= half_extent[0]) & (np.abs(v[:, 1] - center_xy[1]) <= half_extent[1])
    if not inside.any():
        raise ValueError(f"no vertex within {half_extent} of {center_xy}")
    return float(v[inside, 2].max())


def required_course_border(
    max_command: float, episode_length_s: float, speed_tol_ratio: float, tile_size: float
) -> float:
    """Flat border [m] that keeps a successful episode on the mesh from an edge tile's origin.

    A success has a mean speed along the command up to ``(1 + speed_tol_ratio) * command``, so it can travel that far
    in one episode. From the origin of an edge tile, ``tile_size / 2`` of the tile lies ahead; the border supplies the
    rest plus the toe reach. Episodes that leave the course anyway terminate as failures (``off_course``).
    """
    travel = (1.0 + speed_tol_ratio) * max_command * episode_length_s
    return travel + TOE_HALF_DIAGONAL - tile_size / 2


def course_bounds(tile_origins_xy: np.ndarray, tile_size: float, border_width: float, margin: float = TOE_HALF_DIAGONAL):
    """(lo, hi) xy limits [m] for the root inside which every toe stays on the generated mesh."""
    xy = np.asarray(tile_origins_xy, dtype=float).reshape(-1, 2)
    reach = tile_size / 2 + border_width - margin
    return xy.min(axis=0) - reach, xy.max(axis=0) + reach
