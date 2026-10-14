"""A searchable cost graph and its eight-connected grid implementation."""

from __future__ import annotations

import math
from typing import Iterable, Protocol, runtime_checkable

from .grid import Cell, OccupancyGrid

__all__ = ["CostGraph", "GridGraph", "STEPS"]

#: Eight-connected steps as ``(d_east, d_north)``, orthogonals first so that a
#: deterministic tie-break prefers a straight step over a diagonal of equal cost.
STEPS = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


@runtime_checkable
class CostGraph(Protocol):
    """Anything a route planner can search.

    Three methods, deliberately. Nothing here tracks changes: under the GPS-only
    architecture a new error bound or a revised polygon produces a new graph, so a
    delta method would have no consumer. When D* Lite arrives it brings a
    ``MutableCostGraph`` extending this one, which is why this stays narrow -- an
    implementation satisfies a Protocol structurally, so a fourth method added here
    later would break every implementation that lacks it.
    """

    def successors(self, node: Cell) -> Iterable[tuple[Cell, float]]:
        """Yield ``(b, edge_cost(node, b))`` for each node ``b`` one step from ``node``.

        Exactly the nodes with a finite cost: nothing untraversable is yielded, and
        nothing traversable is left out. The cost travels with the node so a search
        never prices the same step twice.
        """
        ...

    def edge_cost(self, a: Cell, b: Cell) -> float:
        """Cost of stepping from ``a`` to ``b``; ``math.inf`` when not traversable."""
        ...

    def heuristic(self, a: Cell, b: Cell) -> float:
        """Admissible, consistent underestimate of the cost from ``a`` to ``b``.

        Admissible means it never exceeds the true remaining cost; consistent means
        it never drops by more than the cost of the step taken. A\\* returns optimal
        paths only while both hold.
        """
        ...


class GridGraph:
    """An eight-connected :class:`CostGraph` over an :class:`OccupancyGrid`.

    Costs are distances in metres: every passable cell is equally good to cross.
    """

    def __init__(self, grid: OccupancyGrid) -> None:
        if not isinstance(grid, OccupancyGrid):
            raise TypeError(
                f"grid must be an OccupancyGrid, got {type(grid).__name__}"
            )
        self.grid = grid

    def successors(self, node: Cell) -> Iterable[tuple[Cell, float]]:
        """Yield ``(cell, cost)`` for each finite step from ``node``, in :data:`STEPS` order.

        Traversability is decided only by :meth:`edge_cost`; this asks it, so
        off-grid cells, blocked cells and corner-cutting diagonals are all dropped.
        """

        east, north = node
        for d_east, d_north in STEPS:
            candidate = (east + d_east, north + d_north)
            cost = self.edge_cost(node, candidate)
            if math.isfinite(cost):
                yield candidate, cost

    def edge_cost(self, a: Cell, b: Cell) -> float:
        """Length in metres of the step ``a`` -> ``b``, or ``math.inf``.

        Infinite unless ``b`` is one of the eight steps from ``a`` and both cells are
        passable. A diagonal step is also infinite if *either* shared orthogonal
        neighbour is impassable: otherwise the robot would squeeze between two
        diagonally touching blocked cells through a gap of zero width.
        """

        d_east, d_north = b[0] - a[0], b[1] - a[1]
        if (d_east, d_north) not in STEPS:
            return math.inf
        if not (self.grid.is_passable(a) and self.grid.is_passable(b)):
            return math.inf
        if (
            d_east
            and d_north
            and not (
                self.grid.is_passable((a[0], b[1]))
                and self.grid.is_passable((b[0], a[1]))
            )
        ):
            return math.inf
        return math.hypot(d_east, d_north) * self.grid.cell_size_m

    def heuristic(self, a: Cell, b: Cell) -> float:
        """Octile distance in metres: the shortest eight-connected path, ignoring walls."""

        d_east, d_north = abs(a[0] - b[0]), abs(a[1] - b[1])
        octile_cells = max(d_east, d_north) + (math.sqrt(2.0) - 1.0) * min(
            d_east, d_north
        )
        return octile_cells * self.grid.cell_size_m
