"""Shortest-route search between two cells of a cost graph."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .grid import Cell, OccupancyGrid
from .graph import CostGraph
from .region import Point

__all__ = ["Route", "a_star"]


@dataclass(frozen=True)
class Route:
    """One planned route, its cost, and what the search spent finding it."""

    #: Cells from start to goal inclusive. A start-equals-goal route holds one cell.
    cells: tuple[Cell, ...]
    #: Total ``edge_cost`` along :attr:`cells`, in meters.
    cost: float
    #: Nodes removed from the frontier and expanded. Reported so that the value of
    #: the heuristic is measurable rather than assumed; it is not part of the route.
    # TODO: Read later
    expansions: int

    def __post_init__(self) -> None:
        if not self.cells:
            raise ValueError("cells must hold at least the start cell")
        cost = float(self.cost)
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError(
                f"cost must be finite and non-negative, got {self.cost!r}"
            )
        if type(self.expansions) is not int or self.expansions < 0:
            raise ValueError(
                f"expansions must be a non-negative int, got {self.expansions!r}"
            )
        object.__setattr__(
            self, "cells", tuple((int(e), int(n)) for e, n in self.cells)
        )
        object.__setattr__(self, "cost", cost)

    # NOTE: Check this spec definition
    def waypoints_m(self, grid: OccupancyGrid) -> tuple[Point, ...]:
        """Cell centers as `Point`s in the map frame, ready for ``WaypointFollower``.

        The follower takes ``(x, y)`` pairs and this returns them in order, so the
        seam to execution needs no adapter. Thinning the list is the executor's
        business, not the planner's.
        """

        return tuple(grid.cell_center_m(cell) for cell in self.cells)


def a_star(graph: CostGraph, start: Cell, goal: Cell) -> Route | None:
    """Cheapest route from ``start`` to ``goal``, or ``None`` when unreachable.

    ``None`` is the ordinary answer for a stop the robot cannot reach: #20 requires
    such a stop to be marked unreachable and left out of the route, not to raise.

    Requirements beyond textbook A\\* with ``heapq``:

    * Determinism. The operator approves this route before motion (R-01) and a
      changed request cancels approval, so identical inputs must give an identical
      route. Push ``(f, h, node)``: breaking ties on ``h`` first prefers nodes nearer
      the goal, which also cuts expansions, and falling back to the node coordinate
      makes the order *total*. An insertion counter is wrong here: it encodes the
      order nodes were discovered, so the same request resolves a tie differently if
      neighbors arrive in a different order. Likewise, when a neighbor is re-reached
      at equal cost, decide which parent to keep by coordinate, not arrival order.
    * Count expansions: once per node popped and actually expanded, reported in
      :attr:`Route.expansions`.
    * Skip a popped node already closed -- a lazy-deletion heap holds stale entries,
      and re-expanding them is what turns A\\* quadratic.
    * Take each step's cost from ``graph.successors``, which yields only finite
      steps; do not call ``edge_cost`` again. Still never push an infinite ``f``.
    * ``start == goal`` returns a one-cell Route of cost 0.0 -- but only if the cell
      is passable, otherwise ``None``.
    * An impassable or off-grid ``start`` or ``goal`` is ``None``, not an exception.
    """
    """
    # if start == goal:
        # if start is passable (see :class:`OccupancyGrid`)
            # return 1 Cell Route
        # else:
            # return None

    # Initialize Frontier
    # Add Start Node to Frontier (priority queue)

    # Initialize Settled Nodes (Set? what datatype in python)
    # Initialize expansions
    # while not (goal is settled OR frontier is empty):
        # Pop next node in frontier
        # (LAZY DELETION): if popped node is in settled:
        #   # continue
        # Add popped node to settled
        # expansions += 1
        # for every (neighbor: Cell, cost: float) of in successors
            # if in frontier and updated weight would be smaller
                # Update weight
                ### Commentary: This is A*, so be careful about the edge costs...
                ### I think for (x,y) is should be new cost = c(x,y) - h(x) + h(y)
                ### We have invariant: dist(node) = distFromDjikstra(node) + h(node)
                ### Then, update : dist(y) = dist(x) + c(x,y) - h(x) + h(y)
                ###              :         = (dist(x) - h(x)) + c(x,y) + h(y)
                ###              :         = [distFromDjikstra(x) + c(x,y)] + h(y)
                ###              :         = distFromDjikstra(y) + h(y)
            # else if not settled:
                # Add to frontier with appropriate weight (see above)
    
    # If goal is not settled:
        # return None
    # else:
    #   return Route
    """
    raise NotImplementedError("implement A* over the CostGraph")
