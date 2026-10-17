"""A searchable cost graph and its eight-connected grid implementation."""

from __future__ import annotations

import math
from typing import Iterable, Protocol, runtime_checkable

from .grid import Cell, OccupancyGrid

__all__ = ["CostGraph", "GridGraph", "STEPS"]

#: Eight-connected steps as ``(d_east, d_north)``, orthogonals first so that a
#: deterministic tie-break prefers a straight step to a diagonal of equal cost.
STEPS = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


@runtime_checkable
class CostGraph(Protocol):
    """A weighted directed graph that a route planner can search."""

    def successors(self, node: Cell) -> Iterable[tuple[Cell, float]]:
        """Yields the traversable steps out of a cell.

        Args:
            node: The cell to step from.

        Yields:
            ``(successor, cost)`` for every cell one step from ``node`` with a
            finite ``edge_cost(node, successor)``, where ``cost`` is that value.
        """
        ...

    def edge_cost(self, a: Cell, b: Cell) -> float:
        """Returns the cost of the step from ``a`` to ``b``.

        Args:
            a: The cell stepped from.
            b: The cell stepped to.

        Returns:
            The step cost, or ``math.inf`` if the step is not traversable.
        """
        ...

    def heuristic(self, a: Cell, b: Cell) -> float:
        """Returns a lower bound on the cost of a path from ``a`` to ``b``.

        The bound must be admissible, never exceeding the true cost, and
        consistent: ``heuristic(a, b) <= edge_cost(a, n) + heuristic(n, b)`` for
        every successor ``n`` of ``a``. A\\* returns optimal paths only if both
        hold.

        Args:
            a: The cell to start from.
            b: The goal cell.

        Returns:
            The lower bound, in the units of :meth:`edge_cost`.
        """
        ...


class GridGraph:
    """An eight-connected :class:`CostGraph` over an :class:`OccupancyGrid`.

    Every passab    are distances in meters.

    Args:
        grid: The occupancy grid to search, kept as ``self.grid``.

    Raises:
        TypeError: ``grid`` is not an :class:`OccupancyGrid`.
    """

    def __init__(self, grid: OccupancyGrid) -> None:
        if not isinstance(grid, OccupancyGrid):
            raise TypeError(
                f"grid must be an OccupancyGrid, got {type(grid).__name__}"
            )
        self.grid = grid

    def successors(self, node: Cell) -> Iterable[tuple[Cell, float]]:
        """Yields the traversable steps out of a cell, in :data:`STEPS` order.

        A step is traversable exactly when :meth:`edge_cost` is finite, so
        off-grid cells, blocked cells and corner-cutting diagonals are skipped.

        Args:
            node: The cell to step from.

        Yields:
            ``(successor, cost)``, where ``cost`` is the step length in meters.
        """

        east, north = node
        for d_east, d_north in STEPS:
            candidate = (east + d_east, north + d_north)
            cost = self.edge_cost(node, candidate)
            if math.isfinite(cost):
                yield candidate, cost

    def edge_cost(self, a: Cell, b: Cell) -> float:
        """Returns the length in meters of the step from ``a`` to ``b``.

        A diagonal step is allowed only when both orthogonal cells it passes are
        passable. Passability (in ``OccupancyGrid``) tests only cell centers,
        and a diagonal passes through the shared corner; given a cell length of s,
        this corner is s/sqrt(2) from each of those centers. If either center is infeasible,
        the corner may be too.

        Args:
            a: The cell stepped from.
            b: The cell stepped to.

        Returns:
            The step length, or ``math.inf`` if ``b`` is not one of the eight
            steps from ``a``, either cell is impassable, or the step is a
            blocked diagonal.
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
        """Returns the octile distance in meters from ``a`` to ``b``.

        Octile distance is the shortest eight-connected path length on an empty
        grid, so it is admissible and consistent.

        Args:
            a: The cell to start from.
            b: The goal cell.

        Returns:
            The octile distance in metres.
        """

        d_east, d_north = abs(a[0] - b[0]), abs(a[1] - b[1])
        octile_cells = max(d_east, d_north) + (math.sqrt(2.0) - 1.0) * min(
            d_east, d_north
        )
        return octile_cells * self.grid.cell_size_m
