"""Shortest-route search between two cells of a cost graph."""

from __future__ import annotations

import math
import heapq  # TODO: Is there a cinvention for ordering of imports?
from collections import deque  # TODO: I added this manually
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

    if start == goal:
        # TODO: Arbitrary CostGraph has no notion of passability (?)
        # What should we do? see is passable (see :class:`OccupancyGrid`)
        return Route((goal,), 0.0, 0)

        # else (if not passable? see question above):
        #   return None

    # Initialize Frontier, add start
    frontier = []
    g = 0.0
    h = graph.heuristic(start, goal)
    f = g + h
    heapq.heappush(frontier, (f, h, start))

    # Value: (cost (f), prev (None if start))
    discovered = {start: (f, None)}

    settled = set()
    expansions = 0

    while goal not in settled and frontier:
        # Pop next node in frontier
        f, h, node = heapq.heappop(frontier)
        # (LAZY DELETION): if popped node is in settled:
        if node in settled:
            continue
        # Add popped node to settled
        settled.add(node)
        expansions += 1

        for neighbor, cost in graph.successors(node):
            dist = (
                f
                + cost
                - graph.heuristic(node, goal)
                + graph.heuristic(neighbor, goal)
            )
            notDiscovered = neighbor not in discovered
            if notDiscovered or dist < discovered[neighbor][0]:
                # Update weight
                ### Commentary: This is A*, so be careful about the edge costs...
                ### I think for (x,y) is should be new cost = c(x,y) - h(x) + h(y)
                ### We have invariant: dist(node) = distFromDjikstra(node) + h(node)
                ### Then, update : dist(y) = dist(x) + c(x,y) - h(x) + h(y)
                ###              :         = (dist(x) - h(x)) + c(x,y) + h(y)
                ###              :         = [distFromDjikstra(x) + c(x,y)] + h(y)
                ###              :         = distFromDjikstra(y) + h(y)
                discovered[neighbor] = (dist, node)
                heapq.heappush(
                    frontier, (dist, graph.heuristic(neighbor, goal), neighbor)
                )

    if goal not in settled:
        return None
    else:
        return _backtrack(goal, discovered, expansions)


def _backtrack(
    goal: Cell,
    discovered: dict[Cell, tuple[float, Cell | None]],
    expansions: int,
) -> Route:
    """
    Creates a route object by backtracking from ``end`` to ``start``.

    Args:
        goal: The node to backtrack from, creating a s -> ``goal`` Route. [s being defined by djikstra iteration...]
        info: [Dictionary w/ appropriate structure of data] (maybe precondition on
        djikstra's settling?)

    Returns:
        [Route, finish later]

    """
    route = deque()
    route.appendleft(goal)

    curr = discovered[goal][1]
    while curr is not None:
        # Append next
        route.appendleft(curr)

        # Update
        curr = discovered[curr][1]

    return Route(tuple(route), discovered[goal][0], expansions)
