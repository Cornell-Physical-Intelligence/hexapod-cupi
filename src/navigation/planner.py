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
            raise ValueError(f"cost must be finite and non-negative, got {self.cost!r}")
        if type(self.expansions) is not int or self.expansions < 0:
            raise ValueError(f"expansions must be a non-negative int, got {self.expansions!r}")
        object.__setattr__(self, "cells", tuple((int(e), int(n)) for e, n in self.cells))
        object.__setattr__(self, "cost", cost)

    def waypoints_m(self, grid: OccupancyGrid) -> tuple[Point, ...]:
        """Cell centers as `Point`s in the map frame, in route order.

        The result can be passed directly to ``WaypointFollower``, which takes a
        sequence of ``(x, y)`` pairs.

        Notes:
            It should be noted that additional processing may be worthwhile before
            passing into a follower. This method returns a tuple of cells, traversing
            along an eight-connected graph. Any route at any angle other than a
            multiple of 45 degrees is a staircase of straight and diagonal steps. A
            follower that steers toward each waypoint in turn will drive it slowly and
            unevenly; simplify the list (drop waypoints on a straight line between
            their neighbors) before handing it to a follower.
        """

        return tuple(grid.cell_center_m(cell) for cell in self.cells)


def a_star(graph: CostGraph, start: Cell, goal: Cell) -> Route | None:
    """Returns the cheapest route from ``start`` to ``goal``, if one exists.

    Searches ``graph`` with A\\*, taking step costs from
    :meth:`CostGraph.successors` and cost estimates from
    :meth:`CostGraph.heuristic`.

    The resulting route is deterministic given ``CostGraph.successors`` of each
    node in ``CostGraph`` remains the same. Ties in computed cost are resolved by cell
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
        is not in ``graph`` (see :meth:`CostGraph.__contains__`). Otherwise, if
        ``start == goal``, a one-cell route of cost 0.0.
    """

    if start not in graph or goal not in graph:
        return None

    if start == goal:
        return Route((goal,), 0.0, 0)

    # Initialize Frontier, add start
    # Entry: (f, h, cell, g). heapq pops the smallest f; if there is a tie, it is broken
    # by the smaller h, then the smaller cell tuple, giving a deterministic
    # tie-breaking order.
    frontier: list[tuple[float, float, Cell, float]] = []
    g = 0.0
    h = graph.heuristic(start, goal)
    f = g + h
    heapq.heappush(frontier, (f, h, start, g))

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
        _, _, node, g_node = heapq.heappop(frontier)

        # Lazy deletion: heapq can't update an entry, so a cell is pushed again
        # whenever a cheaper f is found. Skip the leftover copies.
        if node in settled:
            continue

        settled.add(node)
        expansions += 1

        for neighbor, cost in graph.successors(node):
            # A settled cell's f is final (see `settled`).
            if neighbor in settled:
                continue

            g_neigh = g_node + cost
            h_neigh = graph.heuristic(neighbor, goal)

            # Neighbor's f: g(neighbor) + h(neighbor).
            dist = g_neigh + h_neigh
            not_discovered = neighbor not in discovered
            if not_discovered or dist < discovered[neighbor][0]:
                # Update if newly discovered or we found a better cost.
                discovered[neighbor] = (dist, node)
                heapq.heappush(frontier, (dist, h_neigh, neighbor, g_neigh))

    if goal not in settled:
        return None
    else:
        # The goal was the last cell settled, so g_node is its cost from start.
        return _backtrack(goal, discovered, expansions, g_node)


def _backtrack(
    goal: Cell,
    discovered: dict[Cell, tuple[float, Cell | None]],
    expansions: int,
    cost: float,
) -> Route:
    """Creates a route object by backtracking from ``goal`` to the start.

    Args:
        goal: The route's last cell. Must be settled.
        discovered: The search's ``cell -> (f, parent)`` map (see :func:`a_star`),
            where ``parent`` is ``None`` only for the start cell.
        expansions: The number of cells the search expanded, stored on the route.
        cost: The route's total cost, stored on the route. See Notes.

    Returns:
        The :class:`Route` from the start to ``goal`` inclusive.

    Notes:
        A\\* orders its search by f = g + h, where g is the true cost from the start
        and h is a heuristic. The route's cost is g, not f, and ``discovered`` only
        holds f, so the caller passes g in as ``cost``.
    """

    route: deque[Cell] = deque()
    route.appendleft(goal)

    curr = discovered[goal][1]
    while curr is not None:
        # Append next
        route.appendleft(curr)

        # Update
        curr = discovered[curr][1]

    return Route(tuple(route), cost, expansions)
