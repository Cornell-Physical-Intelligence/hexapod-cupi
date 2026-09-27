"""Shortest-route search between two cells of a cost graph."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .grid import Cell, CostGrid
from .graph import CostGraph
from .region import Point


__all__ = ["Route", "a_star"]


@dataclass(frozen=True)
class Route:
    """One planned route, its cost, and what the search spent finding it."""

    #: Cells from start to goal inclusive. A start-equals-goal route holds one cell.
    cells: tuple[Cell, ...]
    #: Total ``edge_cost`` along :attr:`cells`, in metre-equivalents.
    cost: float
    #: Nodes removed from the frontier and expanded. Reported so that the value of
    #: the heuristic is measurable rather than assumed; it is not part of the route.
    expansions: int

    def __post_init__(self) -> None:
        if not self.cells:
            raise ValueError("cells must hold at least the start cell")
        cost = float(self.cost)
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError(f"cost must be finite and non-negative, got {self.cost!r}")
        if type(self.expansions) is not int or self.expansions < 0:
            raise ValueError(
                f"expansions must be a non-negative int, got {self.expansions!r}"
            )
        object.__setattr__(self, "cells", tuple((int(e), int(n)) for e, n in self.cells))
        object.__setattr__(self, "cost", cost)

    def waypoints_m(self, grid: CostGrid) -> tuple[Point, ...]:
        """Cell centres as world points, ready for ``WaypointFollower``.

        The follower takes ``(x, y)`` pairs and this returns them in order, so the
        seam to execution needs no adapter. Thinning the list is the executor's
        business, not the planner's.
        """

        return tuple(grid.cell_center_m(cell) for cell in self.cells)


def a_star(graph: CostGraph, start: Cell, goal: Cell) -> Route | None:
    """Cheapest route from ``start`` to ``goal``, or ``None`` when unreachable.

    ``None`` is the ordinary answer for a stop the robot cannot reach: #20 requires
    such a stop to be marked unreachable and left out of the route, not to raise.

    # CHALLENGE 1 + 3 + 4 -- pinned by ``test_planner.py``.
    #
    # Textbook A\\* with ``heapq``, plus two requirements the textbook leaves open:
    #
    # 3. Determinism. The operator approves this route before motion (R-01) and a
    #    changed request cancels approval, so identical inputs must give an identical
    #    route. Push ``(f, h, node)``: breaking ties on ``h`` first prefers nodes
    #    nearer the goal, which also cuts expansions, and falling back to the node
    #    coordinate makes the order *total*.
    #
    #    An insertion counter would also be deterministic per run and is still wrong
    #    here: it encodes the order nodes were discovered, so the same request
    #    resolves a tie differently if neighbours arrive in a different order.
    #    ``test_equal_cost_paths_resolve_deterministically`` plans the same tie twice
    #    with the neighbour order reversed and requires the same route, so the
    #    tie-break has to depend only on the nodes themselves. The same reasoning
    #    applies to which parent you keep when a neighbour is re-reached at equal
    #    cost -- decide it by coordinate, not by arrival order.
    #
    # 4. Count expansions. Increment once per node popped and actually expanded, and
    #    report it in :attr:`Route.expansions`.
    #
    # Also worth getting right:
    #   * Skip a popped node already closed -- a lazy-deletion heap holds stale
    #     entries, and re-expanding them is what turns A\\* quadratic.
    #   * Treat ``math.inf`` edges as absent; never push an infinite ``f``.
    #   * ``start == goal`` returns a one-cell Route of cost 0.0 -- but only if the
    #     cell is passable, otherwise ``None``.
    #   * An impassable ``start`` or ``goal`` is ``None``, not an exception.
    """

    raise NotImplementedError("CHALLENGE 1+3+4: implement A* over the CostGraph")
