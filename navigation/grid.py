"""A rasterised occupancy grid over an approved survey region."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import shapely

from .region import PlanMargin, Point, SurveyRegion

__all__ = ["Cell", "OccupancyGrid"]

#: An ``(east_index, north_index)`` cell address. Row-major over north.
Cell = tuple[int, int]


@dataclass(frozen=True)
class OccupancyGrid:
    """Which cells a route may enter, for one region at one margin.

    A grid is immutable and specific to a given margin (e.g. see `PlanMargin` in
    region.py).

    ``passable`` is row-major over north: cell ``(east, north)`` lives at index
    ``north * width_cells + east``, with ``(0,0)`` representing the southwestern most cell.
    """

    origin_m: Point
    cell_size_m: float
    width_cells: int
    height_cells: int
    passable: tuple[bool, ...]

    def __post_init__(self) -> None:
        origin = self.origin_m
        if isinstance(origin, (str, bytes)) or not isinstance(origin, Sequence):
            raise TypeError(
                f"origin_m must be an (east, north) pair, got {type(origin).__name__}"
            )
        pair = tuple(float(value) for value in origin)
        if len(pair) != 2 or not all(math.isfinite(value) for value in pair):
            raise ValueError(
                f"origin_m must be a finite (east, north) pair, got {origin!r}"
            )
        cell_size = float(self.cell_size_m)
        if not math.isfinite(cell_size) or cell_size <= 0.0:
            raise ValueError(
                f"cell_size_m must be finite and positive, got {self.cell_size_m!r}"
            )
        for name in ("width_cells", "height_cells"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(
                    f"{name} must be a positive int, got {value!r}"
                )
        expected = self.width_cells * self.height_cells
        values = tuple(bool(value) for value in self.passable)
        if len(values) != expected:
            raise ValueError(
                f"passable must hold width_cells * height_cells = {expected} values, "
                f"got {len(values)}"
            )
        object.__setattr__(self, "origin_m", (pair[0], pair[1]))
        object.__setattr__(self, "cell_size_m", cell_size)
        object.__setattr__(self, "passable", values)

    @property
    def cell_count(self) -> int:
        """Total cells, passable or not."""

        return self.width_cells * self.height_cells

    @property
    def passable_cell_count(self) -> int:
        """How many cells a route may enter. Zero means an empty feasible region."""

        return sum(self.passable)

    def in_bounds(self, cell: Cell) -> bool:
        """True when ``cell`` addresses a cell of this grid."""

        east, north = cell
        return 0 <= east < self.width_cells and 0 <= north < self.height_cells

    def is_passable(self, cell: Cell) -> bool:
        """True when ``cell`` is on the grid and a route may enter it."""

        if not self.in_bounds(cell):
            return False
        east, north = cell
        return self.passable[north * self.width_cells + east]

    def cell_center_m(self, cell: Cell) -> Point:
        """World position of the centre of ``cell``."""

        east, north = cell
        return (
            self.origin_m[0] + (east + 0.5) * self.cell_size_m,
            self.origin_m[1] + (north + 0.5) * self.cell_size_m,
        )

    def cell_of(self, point: Point) -> Cell:
        """Cell containing ``point``. May be off-grid; check with :meth:`in_bounds`."""

        east = math.floor(
            (float(point[0]) - self.origin_m[0]) / self.cell_size_m
        )
        north = math.floor(
            (float(point[1]) - self.origin_m[1]) / self.cell_size_m
        )
        return (int(east), int(north))

    @classmethod
    def from_region(
        cls,
        region: SurveyRegion,
        margin: PlanMargin,
        cell_size_m: float,
    ) -> "OccupancyGrid":
        """Rasterise ``region`` at ``cell_size_m``, shrunk by ``margin``.

        The grid covers ``region.bounds_m`` from its lower-left corner, rounding up so a
        partial cell is still included. A cell is passable when its centre lies inside
        ``region.feasible(margin)``; when that is empty, every cell is impassable and
        ``passable_cell_count`` is 0.
        """

        cell_size = float(cell_size_m)
        if not math.isfinite(cell_size) or cell_size <= 0.0:
            raise ValueError(
                f"cell_size_m must be finite and positive, got {cell_size_m!r}"
            )
        min_east, min_north, max_east, max_north = region.bounds_m
        width = math.ceil((max_east - min_east) / cell_size)
        height = math.ceil((max_north - min_north) / cell_size)
        easts = min_east + (np.arange(width) + 0.5) * cell_size
        norths = min_north + (np.arange(height) + 0.5) * cell_size
        # meshgrid's (height, width) layout ravels north-major, matching ``passable``.
        east_grid, north_grid = np.meshgrid(easts, norths)
        inside = shapely.contains_xy(
            region.feasible(margin), east_grid.ravel(), north_grid.ravel()
        )
        return cls(
            origin_m=(min_east, min_north),
            cell_size_m=cell_size,
            width_cells=width,
            height_cells=height,
            passable=tuple(inside.tolist()),
        )
