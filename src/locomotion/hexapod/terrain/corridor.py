"""Walled obstacle corridor sub-terrain: static obstacle layout and meshes, in NumPy only so the CPU tests can load
this file directly (``trimesh`` is imported only to build meshes).

Tile coordinates have the tile corner at (0, 0), as in Isaac Lab's mesh sub-terrains. The corridor runs along y, the
direction of the task's forward-only commands at reset yaw. Side walls fill the x edges; the y ends are open, so the
two corridor columns of one curriculum level join into one corridor.

Each obstacle is a static box or a pillar with a circular footprint bound (a box's half diagonal). The layout keeps:

- at least ``gap`` between the footprint bounds of any two obstacles;
- at least ``gap`` between each obstacle and one of the two side walls;
- at least ``gap / 2`` between each obstacle and the open y ends, so neighbouring tiles keep the smaller of their gaps;
- the spawn area (``geometry.spawn_half_extent``) plus ``spawn_clearance`` free of obstacles.

With narrow links only between an obstacle and a wall, no chain of narrow links joins the two walls, so a lane at
least ``gap`` wide stays open from one y end to the other. ``gap`` shrinks from ``gap_range[0]`` at difficulty 0 to
``gap_range[1]`` at difficulty 1; the smallest gap is wider than the toe footprint plus its spawn margin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GAP_RANGE = (0.90, 0.62)
"""Smallest free distance [m] between obstacles at difficulty 0 and 1. 0.62 m > 0.511 m toe width + 2 x 0.05 m."""
NUM_OBSTACLES = 6
"""Obstacles requested per tile; a draw that finds no free place after ``MAX_TRIES`` is skipped."""
WALL_THICKNESS = 0.05
WALL_HEIGHT = 0.25
"""Side-wall height [m], 2.5 x the 0.10 m standing robot."""
OBSTACLE_HEIGHT_RANGE = (0.15, 0.30)
"""Obstacle height [m]: taller than the robot, so it walks around them, not over them."""
PILLAR_RADIUS_RANGE = (0.03, 0.07)
BOX_SIZE_RANGE = ((0.06, 0.14), (0.03, 0.06))
"""Box (length, thickness) ranges [m]: short wall segments with a random yaw."""
PILLAR_FRACTION = 0.5
SPAWN_CLEARANCE = 0.05
"""Free distance [m] around the spawn area."""
MAX_TRIES = 200


@dataclass(frozen=True)
class Obstacle:
    """A static obstacle in tile coordinates. ``size`` is (length, thickness) of a box or (diameter, diameter) of a
    pillar."""

    kind: str
    x: float
    y: float
    yaw: float
    size: tuple[float, float]
    height: float

    @property
    def radius(self) -> float:
        """Radius [m] of a circle around the footprint."""
        if self.kind == "pillar":
            return self.size[0] / 2
        return float(np.hypot(*self.size)) / 2


def corridor_gap(difficulty: float, gap_range: tuple[float, float] = GAP_RANGE) -> float:
    """Gap [m] at ``difficulty`` in [0, 1], linear from ``gap_range[0]`` to ``gap_range[1]``."""
    d = float(np.clip(difficulty, 0.0, 1.0))
    return gap_range[0] + d * (gap_range[1] - gap_range[0])


def corridor_rng(seed: int | None, difficulty: float) -> np.random.Generator:
    """Layout generator: ``seed`` and ``difficulty`` fix the layout. With no seed, draw one from NumPy's global state,
    as Isaac Lab's ``TerrainGenerator`` does when its own ``seed`` is unset."""
    if seed is None:
        return np.random.default_rng(np.random.randint(0, 2**31 - 1))
    return np.random.default_rng([int(seed), int(round(float(difficulty) * 1e6))])


def _rect_distance(x, y, center, half):
    """Distance [m] from a point to an axis-aligned rectangle (0 inside)."""
    dx = max(abs(x - center[0]) - half[0], 0.0)
    dy = max(abs(y - center[1]) - half[1], 0.0)
    return float(np.hypot(dx, dy))


def place_obstacles(
    rng: np.random.Generator,
    size: tuple[float, float],
    gap: float,
    spawn_half: tuple[float, float],
    num_obstacles: int = NUM_OBSTACLES,
    wall_thickness: float = WALL_THICKNESS,
    spawn_clearance: float = SPAWN_CLEARANCE,
    height_range: tuple[float, float] = OBSTACLE_HEIGHT_RANGE,
    pillar_radius_range: tuple[float, float] = PILLAR_RADIUS_RANGE,
    box_size_range: tuple[tuple[float, float], tuple[float, float]] = BOX_SIZE_RANGE,
    pillar_fraction: float = PILLAR_FRACTION,
    max_tries: int = MAX_TRIES,
) -> list[Obstacle]:
    """Rejection-sample up to ``num_obstacles`` obstacles that keep the gaps in the module docstring."""
    sx, sy = size
    spawn_center = (sx / 2, sy / 2)
    spawn_free = (spawn_half[0] + spawn_clearance, spawn_half[1] + spawn_clearance)
    lo_x, hi_x = wall_thickness, sx - wall_thickness
    placed: list[Obstacle] = []
    for _ in range(num_obstacles):
        if rng.uniform() < pillar_fraction:
            d = 2 * rng.uniform(*pillar_radius_range)
            kind, dims = "pillar", (d, d)
        else:
            kind, dims = "box", (rng.uniform(*box_size_range[0]), rng.uniform(*box_size_range[1]))
        yaw, height = rng.uniform(0.0, np.pi), rng.uniform(*height_range)
        r = Obstacle(kind, 0.0, 0.0, yaw, dims, height).radius
        x_min, x_max = lo_x + r, hi_x - r
        y_min, y_max = gap / 2 + r, sy - gap / 2 - r
        if x_min >= x_max or y_min >= y_max:
            continue
        for _ in range(max_tries):
            x, y = rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)
            if _rect_distance(x, y, spawn_center, spawn_free) < r:
                continue
            if max(x - r - lo_x, hi_x - x - r) < gap:
                continue
            if any(np.hypot(x - o.x, y - o.y) < r + o.radius + gap for o in placed):
                continue
            placed.append(Obstacle(kind, float(x), float(y), float(yaw), (float(dims[0]), float(dims[1])), height))
            break
    return placed


def corridor_meshes(
    obstacles: list[Obstacle],
    size: tuple[float, float],
    wall_thickness: float = WALL_THICKNESS,
    wall_height: float = WALL_HEIGHT,
    pillar_sections: int = 16,
) -> list:
    """Ground, two side walls and the obstacles as ``trimesh.Trimesh`` meshes in tile coordinates."""
    import trimesh

    sx, sy = size
    ground = trimesh.Trimesh(
        vertices=[[0.0, 0.0, 0.0], [sx, 0.0, 0.0], [sx, sy, 0.0], [0.0, sy, 0.0]], faces=[[0, 1, 2], [0, 2, 3]]
    )
    meshes = [ground]
    for x in (wall_thickness / 2, sx - wall_thickness / 2):
        meshes.append(
            trimesh.creation.box(
                extents=(wall_thickness, sy, wall_height),
                transform=trimesh.transformations.translation_matrix((x, sy / 2, wall_height / 2)),
            )
        )
    for o in obstacles:
        transform = trimesh.transformations.rotation_matrix(o.yaw, (0.0, 0.0, 1.0))
        transform[:3, 3] = (o.x, o.y, o.height / 2)
        if o.kind == "pillar":
            mesh = trimesh.creation.cylinder(
                radius=o.size[0] / 2, height=o.height, sections=pillar_sections, transform=transform
            )
        else:
            mesh = trimesh.creation.box(extents=(o.size[0], o.size[1], o.height), transform=transform)
        meshes.append(mesh)
    return meshes
