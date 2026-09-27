"""A rasterised cost grid over an approved survey region."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .region import PlanMargin, Point, SurveyRegion


__all__ = ["Cell", "CostGrid", "PASSABLE_CELL_COST", "RISK_WEIGHT"]

#: An ``(east_index, north_index)`` cell address. Row-major over north.
Cell = tuple[int, int]

#: Cost of a cell with clearance to spare. Every passable cell costs at least this,
#: which is what makes ``min_cell_cost`` a valid heuristic scale factor.
PASSABLE_CELL_COST = 1.0

#: Extra cost at the point of bare-minimum clearance. A cell that only just clears
#: the margin costs ``PASSABLE_CELL_COST * (1 + RISK_WEIGHT)``, so a route prefers
#: the middle of the free space when the detour is short -- which is what makes it
#: survive a GPS fix that jumps toward an edge.
RISK_WEIGHT = 1.0


@dataclass(frozen=True)
class CostGrid:
    """Per-cell traversal cost for one region at one margin.

    A grid is **immutable and margin-specific**: a different
    ``position_error_bound_m`` is a different grid, not a mutation of this one.
    That is why there is no change-tracking here -- nothing mutates, so nothing
    needs a delta. Incremental replanning arrives with its own graph type when a
    consumer exists for it.

    ``costs`` is row-major over north: cell ``(east, north)`` lives at index
    ``north * width_cells + east``. ``math.inf`` marks an impassable cell.
    """

    origin_m: Point
    cell_size_m: float
    width_cells: int
    height_cells: int
    costs: tuple[float, ...]

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
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        expected = self.width_cells * self.height_cells
        values = tuple(float(value) for value in self.costs)
        if len(values) != expected:
            raise ValueError(
                f"costs must hold width_cells * height_cells = {expected} values, "
                f"got {len(values)}"
            )
        for index, value in enumerate(values):
            # inf is the impassable marker and is legal; nan never is, and a cost
            # below the structural floor would silently break heuristic admissibility.
            if math.isnan(value) or value < PASSABLE_CELL_COST:
                raise ValueError(
                    f"costs[{index}] must be inf or at least {PASSABLE_CELL_COST}, "
                    f"got {value!r}"
                )
        object.__setattr__(self, "origin_m", (pair[0], pair[1]))
        object.__setattr__(self, "cell_size_m", cell_size)
        object.__setattr__(self, "costs", values)

    @property
    def min_cell_cost(self) -> float:
        """Structural lower bound on any cell's cost, for scaling the heuristic.

        This is the constant floor, **not** the smallest cost actually present. A
        heuristic scaled by an observed minimum stops being a lower bound the moment
        the grid changes, which is exactly the kind of bug that produces paths that
        look fine and are not optimal.
        """

        return PASSABLE_CELL_COST

    @property
    def cell_count(self) -> int:
        """Total cells, passable or not."""

        return self.width_cells * self.height_cells

    @property
    def passable_cell_count(self) -> int:
        """How many cells a route may enter. Zero means an empty feasible region."""

        return sum(1 for cost in self.costs if math.isfinite(cost))

    def in_bounds(self, cell: Cell) -> bool:
        """True when ``cell`` addresses a cell of this grid."""

        east, north = cell
        return 0 <= east < self.width_cells and 0 <= north < self.height_cells

    def cost(self, cell: Cell) -> float:
        """Traversal cost of ``cell``; ``math.inf`` off-grid or impassable."""

        if not self.in_bounds(cell):
            return math.inf
        east, north = cell
        return self.costs[north * self.width_cells + east]

    def is_passable(self, cell: Cell) -> bool:
        """True when ``cell`` is on the grid and has finite cost."""

        return math.isfinite(self.cost(cell))

    def cell_center_m(self, cell: Cell) -> Point:
        """World position of the centre of ``cell``."""

        east, north = cell
        return (
            self.origin_m[0] + (east + 0.5) * self.cell_size_m,
            self.origin_m[1] + (north + 0.5) * self.cell_size_m,
        )

    def cell_of(self, point: Point) -> Cell:
        """Cell containing ``point``. May be off-grid; check with :meth:`in_bounds`."""

        east = math.floor((float(point[0]) - self.origin_m[0]) / self.cell_size_m)
        north = math.floor((float(point[1]) - self.origin_m[1]) / self.cell_size_m)
        return (int(east), int(north))

    @classmethod
    def from_region(
        cls,
        region: SurveyRegion,
        margin: PlanMargin,
        cell_size_m: float,
    ) -> "CostGrid":
        """Rasterise ``region`` at ``cell_size_m``, inflated by ``margin``.

        # CHALLENGE 6 + 7 -- pinned by
        #   ``test_margin_larger_than_area_leaves_no_feasible_region`` and
        #   ``test_inflation_matches_a_brute_force_distance_check``.
        #
        # Required result, cell by cell, using ``region.clearance_m(centre)``:
        #   clearance <  margin.total_m                  -> math.inf (impassable)
        #   otherwise, with span = margin.position_error_bound_m:
        #     span <= 0                                  -> PASSABLE_CELL_COST
        #     excess = clearance - margin.total_m
        #     excess >= span                             -> PASSABLE_CELL_COST
        #     else PASSABLE_CELL_COST * (1 + RISK_WEIGHT * (1 - excess / span))
        #
        # Note what falls out of this rather than being coded separately: there is
        # no polygon-offsetting step. Inflating an obstacle by the footprint radius
        # and shrinking the boundary by it are the same operation as requiring
        # clearance, and requiring clearance is a per-cell test. That is why the
        # feasible region can come back empty with no special case -- every cell
        # simply fails the test, and ``passable_cell_count`` is 0.
        #
        # Grid extent: cover ``region.bounds_m``, with the origin at the bounds'
        # lower-left corner and enough cells to reach the upper-right. Use
        # ``math.ceil`` on the span so a partial cell is still included.
        #
        # The direct form is O(cells x edges) and takes well under a second at the
        # sizes here, so write that first. If it is ever optimised, the oracle test
        # pins the result and not the method.
        """

        raise NotImplementedError("CHALLENGE 6+7: rasterise and inflate the region")
