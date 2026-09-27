"""A searchable cost graph and its eight-connected grid implementation."""
from __future__ import annotations

import math
from typing import Iterable, Protocol, runtime_checkable

from .grid import Cell, CostGrid


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

    def neighbors(self, node: Cell) -> Iterable[Cell]:
        """Yield the nodes reachable from ``node`` in one step."""
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
    """An eight-connected :class:`CostGraph` over a :class:`CostGrid`."""

    def __init__(self, grid: CostGrid) -> None:
        if not isinstance(grid, CostGrid):
            raise TypeError(f"grid must be a CostGrid, got {type(grid).__name__}")
        self.grid = grid

    def neighbors(self, node: Cell) -> Iterable[Cell]:
        """Yield on-grid neighbours of ``node`` in the fixed :data:`STEPS` order.

        Passability is not filtered here -- an impassable neighbour is reported and
        priced at ``math.inf`` by :meth:`edge_cost`. Keeping the two separate means
        there is exactly one place where traversability is decided.
        """

        east, north = node
        for d_east, d_north in STEPS:
            candidate = (east + d_east, north + d_north)
            if self.grid.in_bounds(candidate):
                yield candidate

    def edge_cost(self, a: Cell, b: Cell) -> float:
        """Cost of the step ``a`` -> ``b``.

        # CHALLENGE 2 -- pinned by ``test_diagonal_moves_do_not_cut_blocked_corners``.
        #
        # Required result:
        #   * math.inf unless ``b`` is one of the eight steps from ``a``.
        #   * math.inf if either cell is impassable.
        #   * For a diagonal step, math.inf if *either* shared orthogonal neighbour
        #     -- ``(a_east, b_north)`` and ``(b_east, a_north)`` -- is impassable.
        #     Without this the robot squeezes between two diagonally touching
        #     blocked cells through a gap of exactly zero width. It looks right on
        #     screen and it is the single most common grid-planner bug.
        #   * Otherwise the mean of the two cells' costs, times the straight-line
        #     distance between their centres:
        #       0.5 * (cost(a) + cost(b)) * hypot(d_east, d_north) * cell_size_m
        #
        # Pricing by distance is what keeps the metric heuristic comparable: cost
        # is then in metre-equivalents, and the cheapest possible metre costs
        # exactly ``grid.min_cell_cost``.
        """

        raise NotImplementedError("CHALLENGE 2: implement edge cost with corner rules")

    def heuristic(self, a: Cell, b: Cell) -> float:
        """Admissible underestimate of the remaining cost from ``a`` to ``b``.

        # CHALLENGE 1 -- pinned by
        # ``test_astar_cost_matches_dijkstra_on_weighted_grid``.
        #
        # The shortest achievable path on an eight-connected grid is the octile
        # distance, not the Euclidean one:
        #   d_east, d_north = |a - b| per axis
        #   octile_cells = max(d) + (sqrt(2) - 1) * min(d)
        #
        # Then -- and this is the whole point of the challenge -- convert to the
        # same units ``edge_cost`` returns, and scale by the *cheapest* cell:
        #   grid.min_cell_cost * octile_cells * grid.cell_size_m
        #
        # Drop the ``min_cell_cost`` factor and the heuristic stays admissible only
        # while every cell happens to cost the floor. Raise the risk weight, or set
        # a non-zero position error bound, and it starts overestimating: A\\* then
        # returns paths that are cheap-looking and not optimal, with no error and no
        # crash. That is why the test compares against Dijkstra rather than against
        # a stored expected path.
        """

        raise NotImplementedError("CHALLENGE 1: implement the scaled octile heuristic")
