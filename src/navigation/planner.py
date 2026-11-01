"""Shortest-route search between two cells of a cost graph."""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass

from .graph import CostGraph
from .grid import Cell, OccupancyGrid
from .region import Point

__all__ = ["Route", "a_star"]


@dataclass(frozen=True)
class Route:
    """One planned route, its cost, and what the search spent finding it."""

    #: Cells from start to goal inclusive. A start-equals-goal route holds one cell.
    cells: tuple[Cell, ...]
    #: Total ``edge_cost`` along :attr:`cells`, in meters.
    cost: float
    #: Nodes removed from the frontier and expanded. This provides a means of
    #: evaluating the performance of the heuristic.
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

    def waypoints_m(self, grid: OccupancyGrid) -> tuple[Point, ...]:
        """Cell centers as `Point`s in the map frame, in route order.

        The result can be passed directly to ``WaypointFollower``, which takes a
        sequence of ``(x, y)`` pairs.

        Note:
            It should be noted that additional processing may be worthwile before
            passing into a follower. This method returns a tuple of cells, traversing
            along an eight-connected graph. Any route at any angle other than a
            multiple of 45 degrees is a staircase of straight and diagonal steps.
            A follower that steers toward each waypoint in turn will drive it slowly and
            unevenly; simplify the list (drop waypoints on a straight line
            between their neighbors) before handing it to a follower.
        """

        return tuple(grid.cell_center_m(cell) for cell in self.cells)


def a_star(graph: CostGraph, start: Cell, goal: Cell) -> Route | None:
    """Returns the cheapest route from ``start`` to ``goal``, if one exists.

    Searches ``graph`` with A\\*, taking step costs from
    :meth:`CostGraph.successors` and cost estimates from
    :meth:`CostGraph.heuristic`.

    The resulting route is deterministic given ``CostGraph.successors`` of each
    node in ``CostGraph`` remains the same. Equal-cost ties are resolved by cell
    coordinates, never by the order in which :meth:`CostGraph.successors` yields steps.

    Args:
        graph: The graph to search. Its heuristic must be admissible and
            consistent (see :meth:`CostGraph.heuristic`); otherwise the route may
            not be the cheapest.
        start: The cell the route begins at.
        goal: The cell the route ends at.

    Returns:
        The cheapest route from ``start`` to ``goal``, with the number of cells
        expanded to find it (see :class:`Route`). ``None`` if no sequence of
        finite-cost steps leads from ``start`` to ``goal``, or if ``start`` or ``goal``
        are not in ``graph``. Otherwise, if ``start == goal``, a
        one-cell route of cost 0.0.
    """

    if start == goal:
        # TODO: Arbitrary CostGraph has no notion of passability (?)
        # What should we do? see is passable (see :class:`OccupancyGrid`)
        return Route((goal,), 0.0, 0)

        # else (if not passable? see question above):
        #   return None

    # Initialize Frontier, add start
    # Entry: (f, h, cell). heapq pops the smallest f; if there is a tie, it is broken
    # by the smaller h, then the smaller cell tuple, giving a deterministic
    # tie-breaking order.
    frontier: list[tuple[float, float, Cell]] = []
    g = 0.0
    h = graph.heuristic(start, goal)
    f = g + h
    heapq.heappush(frontier, (f, h, start))

    # Entry: cell -> (f, parent).
    # Loop Invariant: f is the cheapest cost found through settled cells
    # (see `settled` below), and parent is the cell it was reached from
    # (None for start).
    discovered: dict[Cell, tuple[float, Cell | None]] = {start: (f, None)}

    # Contains cells that have been popped from frontier.
    # Loop Invariant: a settled cell's f in `discovered` is final: its g is the
    # true shortest distance from start, so no later path can lower its f.
    # Holds because the heuristic IS CONSISTENT (see CostGraph.heuristic).
    settled: set[Cell] = set()
    expansions = 0

    while goal not in settled and frontier:
        f, h, node = heapq.heappop(frontier)

        # Lazy deletion: heapq can't update an entry, so a cell is pushed again
        # whenever a cheaper f is found. Skip the leftover copies.
        if node in settled:
            continue

        settled.add(node)
        expansions += 1

        for neighbor, cost in graph.successors(node):
            # Neighbor's f: g(node) + cost + h(neighbor), where g(node) = f - h(node).
            dist = (
                (f - graph.heuristic(node, goal))  # g(node)
                + cost
                + graph.heuristic(neighbor, goal)
            )
            not_discovered = neighbor not in discovered
            if not_discovered or dist < discovered[neighbor][0]:
                # Update if newly discovered or we found a better cost.
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
        discovered: [Dictionary w/ appropriate structure of data] (maybe precondition on
        djikstra's settling?)

    Returns:
        [Route, finish later]

    """
    route: deque[Cell] = deque()
    route.appendleft(goal)

    curr = discovered[goal][1]
    while curr is not None:
        # Append next
        route.appendleft(curr)

        # Update
        curr = discovered[curr][1]

    return Route(tuple(route), discovered[goal][0], expansions)
