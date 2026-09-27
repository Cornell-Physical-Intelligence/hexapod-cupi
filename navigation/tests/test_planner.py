"""Search correctness, corner rules, determinism and the value of the heuristic.

Grids here are built directly from ASCII art rather than from a ``SurveyRegion``, so
that a failure points at the search or the graph and never at rasterisation. The
oracle is an independent Dijkstra in this module: A\\* is checked against a different
algorithm rather than against a stored path, because the failure mode that matters --
an inadmissible heuristic -- produces a plausible path with a wrong cost and no error.

Everything is closed-form. No simulator, no randomness.
"""

from __future__ import annotations

import heapq
import math
import unittest

from navigation.graph import STEPS, CostGraph, GridGraph
from navigation.grid import CostGrid
from navigation.planner import Route, a_star


def _grid(rows: tuple[str, ...], *, cell_size_m: float = 1.0) -> CostGrid:
    """Build a grid from ASCII rows, ``rows[0]`` being the **northernmost**.

    ``.`` is a minimum-cost cell, ``#`` is impassable, and a digit is that cost.
    """

    height = len(rows)
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("rows must all be the same width")
    costs: list[float] = []
    for row in reversed(rows):  # north is the last index, so fill from the south row
        for char in row:
            costs.append(math.inf if char == "#" else 1.0 if char == "." else float(char))
    return CostGrid(
        origin_m=(0.0, 0.0),
        cell_size_m=cell_size_m,
        width_cells=width,
        height_cells=height,
        costs=tuple(costs),
    )


def _open_grid(width: int, height: int) -> CostGrid:
    return _grid(tuple("." * width for _ in range(height)))


def _dijkstra(graph, start, goal) -> tuple[float, int] | None:
    """Trusted uniform-cost oracle: ``(cost, expansions)``, or ``None``.

    Deliberately heuristic-free. Its only job is to be obviously correct, so it is
    written for clarity rather than speed and shares no code with the planner.

    Assumes both endpoints are passable; every fixture that consults the oracle
    chooses them so. Blocked-endpoint behaviour is checked against ``a_star``'s own
    contract in :class:`UnreachableTests` instead.
    """

    best = {start: 0.0}
    frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    closed: set[tuple[int, int]] = set()
    expansions = 0
    while frontier:
        cost, node = heapq.heappop(frontier)
        if node in closed:
            continue
        closed.add(node)
        expansions += 1
        if node == goal:
            return cost, expansions
        for neighbour in graph.neighbors(node):
            if neighbour in closed:
                continue
            step = graph.edge_cost(node, neighbour)
            if not math.isfinite(step):
                continue
            candidate = cost + step
            if candidate < best.get(neighbour, math.inf):
                best[neighbour] = candidate
                heapq.heappush(frontier, (candidate, neighbour))
    return None


class _ReversedNeighbors:
    """A :class:`CostGraph` that offers neighbours in the opposite order.

    Used to prove the planner's tie-break is a total order on something intrinsic to
    the nodes, not on the order they happened to be discovered in.
    """

    def __init__(self, inner) -> None:
        self.inner = inner

    def neighbors(self, node):
        return tuple(reversed(tuple(self.inner.neighbors(node))))

    def edge_cost(self, a, b) -> float:
        return self.inner.edge_cost(a, b)

    def heuristic(self, a, b) -> float:
        return self.inner.heuristic(a, b)


class SeamTests(unittest.TestCase):
    """CHALLENGE 9 -- structural conformance, mirroring ``test_producer.py``."""

    def test_grid_graph_satisfies_the_cost_graph_protocol(self) -> None:
        graph = GridGraph(_open_grid(3, 3))
        self.assertIsInstance(graph, CostGraph)

    def test_a_wrapper_also_satisfies_the_protocol(self) -> None:
        # The Protocol is satisfied structurally, so a teammate's graph needs no
        # base class. This is what keeps the seam cheap to implement against.
        graph = _ReversedNeighbors(GridGraph(_open_grid(3, 3)))
        self.assertIsInstance(graph, CostGraph)

    def test_a_graph_must_be_built_on_a_grid(self) -> None:
        with self.assertRaises(TypeError):
            GridGraph(object())

    def test_neighbours_are_on_grid_and_in_a_fixed_order(self) -> None:
        graph = GridGraph(_open_grid(3, 3))
        self.assertEqual(
            tuple(graph.neighbors((1, 1))),
            tuple((1 + de, 1 + dn) for de, dn in STEPS),
            "neighbour order is part of the contract",
        )
        # A corner has only the three in-bounds steps.
        self.assertEqual(set(graph.neighbors((0, 0))), {(1, 0), (0, 1), (1, 1)})


class RouteTests(unittest.TestCase):
    """The result type, which depends on no unimplemented method."""

    def test_a_route_needs_at_least_one_cell(self) -> None:
        with self.assertRaises(ValueError):
            Route(cells=(), cost=0.0, expansions=0)

    def test_cost_and_expansions_are_validated(self) -> None:
        for bad_cost in (float("nan"), float("inf"), -1.0):
            with self.subTest(cost=bad_cost), self.assertRaises(ValueError):
                Route(cells=((0, 0),), cost=bad_cost, expansions=0)
        for bad_count in (-1, 1.5, True):
            with self.subTest(expansions=bad_count), self.assertRaises(ValueError):
                Route(cells=((0, 0),), cost=0.0, expansions=bad_count)

    def test_waypoints_are_cell_centres_in_order(self) -> None:
        grid = _open_grid(3, 3)
        route = Route(cells=((0, 0), (1, 1)), cost=math.sqrt(2.0), expansions=2)
        self.assertEqual(route.waypoints_m(grid), ((0.5, 0.5), (1.5, 1.5)))


class CornerRuleTests(unittest.TestCase):
    """CHALLENGE 2 -- ``GridGraph.edge_cost``."""

    def test_an_open_diagonal_costs_its_euclidean_length(self) -> None:
        graph = GridGraph(_open_grid(2, 2))
        self.assertAlmostEqual(
            graph.edge_cost((0, 0), (1, 1)), math.sqrt(2.0), places=12
        )

    def test_an_orthogonal_step_costs_one_cell_size(self) -> None:
        graph = GridGraph(_open_grid(2, 2))
        self.assertAlmostEqual(graph.edge_cost((0, 0), (1, 0)), 1.0, places=12)

    def test_cost_is_the_mean_of_the_two_cells(self) -> None:
        # East cell costs 3, west cell costs 1, so the step between them costs 2.
        graph = GridGraph(_grid((".3",)))
        self.assertAlmostEqual(graph.edge_cost((0, 0), (1, 0)), 2.0, places=12)

    def test_cost_scales_with_cell_size(self) -> None:
        graph = GridGraph(_grid(("..",), cell_size_m=0.25))
        self.assertAlmostEqual(graph.edge_cost((0, 0), (1, 0)), 0.25, places=12)

    def test_non_adjacent_and_off_grid_steps_are_infinite(self) -> None:
        graph = GridGraph(_open_grid(3, 3))
        for target in ((2, 0), (0, 2), (5, 5), (-1, 0), (0, 0)):
            with self.subTest(target=target):
                self.assertEqual(graph.edge_cost((0, 0), target), math.inf)

    def test_a_step_into_or_out_of_a_blocked_cell_is_infinite(self) -> None:
        graph = GridGraph(_grid((".#",)))
        self.assertEqual(graph.edge_cost((0, 0), (1, 0)), math.inf)
        self.assertEqual(graph.edge_cost((1, 0), (0, 0)), math.inf)

    def test_diagonal_moves_do_not_cut_blocked_corners(self) -> None:
        # Two passable cells touching only at a corner. Stepping between them would
        # pass through a gap of exactly zero width, which the robot cannot do.
        graph = GridGraph(_grid((".#", "#.")))
        self.assertEqual(
            graph.edge_cost((0, 1), (1, 0)),
            math.inf,
            "a diagonal may not squeeze between two blocked cells",
        )
        self.assertIsNone(
            a_star(graph, (0, 1), (1, 0)), "so the two cells are not connected"
        )

    def test_a_single_blocked_corner_also_forbids_the_diagonal(self) -> None:
        # Only one of the two shared orthogonals is blocked, which still clips the
        # footprint. The detour through the open orthogonal is the correct answer.
        graph = GridGraph(_grid(("..", "#.")))
        self.assertEqual(graph.edge_cost((0, 1), (1, 0)), math.inf)
        route = a_star(graph, (0, 1), (1, 0))
        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.cells, ((0, 1), (1, 1), (1, 0)))
        self.assertAlmostEqual(route.cost, 2.0, places=12)


class OptimalityTests(unittest.TestCase):
    """CHALLENGE 1 -- the heuristic must stay admissible under weighted cells."""

    def test_astar_cost_matches_dijkstra_on_weighted_grid(self) -> None:
        # A cheap corridor through expensive terrain. A heuristic that forgets to
        # scale by the cheapest cell overestimates here and A* takes the expensive
        # direct route while reporting a lower cost than is achievable.
        rows = (
            "99999999",
            "91111119",
            "99999919",
            "91111119",
            "91999999",
            "91111119",
            "99999999",
        )
        graph = GridGraph(_grid(rows))
        start, goal = (1, 1), (6, 5)
        oracle = _dijkstra(graph, start, goal)
        self.assertIsNotNone(oracle, "the oracle must find a route for this fixture")
        assert oracle is not None
        route = a_star(graph, start, goal)
        self.assertIsNotNone(route)
        assert route is not None
        self.assertAlmostEqual(
            route.cost,
            oracle[0],
            places=12,
            msg="A* must return an optimal cost, not merely a plausible path",
        )

    def test_astar_matches_dijkstra_across_several_fixtures(self) -> None:
        fixtures = (
            (("...", "...", "..."), (0, 0), (2, 2)),
            ((".#.", ".#.", "..."), (0, 2), (2, 2)),
            (("1234", "2345", "3456"), (0, 0), (3, 2)),
            (("..#..", "..#..", ".....", "#####".replace("#", "."), "....."), (0, 4), (4, 4)),
        )
        for rows, start, goal in fixtures:
            with self.subTest(rows=rows):
                graph = GridGraph(_grid(rows))
                oracle = _dijkstra(graph, start, goal)
                route = a_star(graph, start, goal)
                if oracle is None:
                    self.assertIsNone(route)
                    continue
                self.assertIsNotNone(route)
                assert route is not None
                self.assertAlmostEqual(route.cost, oracle[0], places=12)

    def test_route_cells_are_contiguous_passable_and_priced_correctly(self) -> None:
        graph = GridGraph(_grid(("1234", "2345", "3456")))
        route = a_star(graph, (0, 0), (3, 2))
        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.cells[0], (0, 0))
        self.assertEqual(route.cells[-1], (3, 2))
        total = 0.0
        for previous, current in zip(route.cells, route.cells[1:]):
            step = graph.edge_cost(previous, current)
            self.assertTrue(
                math.isfinite(step), f"{previous} -> {current} is not traversable"
            )
            total += step
        self.assertAlmostEqual(route.cost, total, places=12, msg="cost must be the sum")


class DeterminismTests(unittest.TestCase):
    """CHALLENGE 3 -- approval requires the same request to yield the same route."""

    # On an open grid, (0,0) -> (2,1) has two distinct optimal paths: diagonal then
    # east, or east then diagonal. Both cost 1 + sqrt(2).
    TIE_START = (0, 0)
    TIE_GOAL = (2, 1)

    def test_equal_cost_paths_resolve_deterministically(self) -> None:
        grid = _open_grid(3, 2)
        forward = a_star(GridGraph(grid), self.TIE_START, self.TIE_GOAL)
        reversed_order = a_star(
            _ReversedNeighbors(GridGraph(grid)), self.TIE_START, self.TIE_GOAL
        )
        self.assertIsNotNone(forward)
        self.assertIsNotNone(reversed_order)
        assert forward is not None and reversed_order is not None
        self.assertAlmostEqual(forward.cost, 1.0 + math.sqrt(2.0), places=12)
        self.assertAlmostEqual(reversed_order.cost, forward.cost, places=12)
        self.assertEqual(
            forward.cells,
            reversed_order.cells,
            "the tie-break must not depend on the order neighbours are offered",
        )

    def test_repeated_planning_is_stable(self) -> None:
        graph = GridGraph(_open_grid(6, 6))
        first = a_star(graph, (0, 0), (5, 3))
        assert first is not None
        for _ in range(8):
            again = a_star(graph, (0, 0), (5, 3))
            assert again is not None
            self.assertEqual(again.cells, first.cells)


class HeuristicValueTests(unittest.TestCase):
    """CHALLENGE 4 -- a heuristic that returns zero is admissible and useless."""

    def test_heuristic_reduces_expansions_versus_dijkstra(self) -> None:
        graph = GridGraph(_open_grid(40, 40))
        start, goal = (0, 0), (39, 39)
        oracle = _dijkstra(graph, start, goal)
        assert oracle is not None
        route = a_star(graph, start, goal)
        self.assertIsNotNone(route)
        assert route is not None
        self.assertAlmostEqual(route.cost, oracle[0], places=12)
        self.assertLess(
            route.expansions,
            0.75 * oracle[1],
            msg=(
                "A* expanded nearly as many nodes as uniform-cost search; the "
                "heuristic is admissible but carries no information"
            ),
        )

    def test_octile_heuristic_never_exceeds_the_true_cost(self) -> None:
        # Admissibility, checked directly against the oracle from many sources.
        graph = GridGraph(_open_grid(8, 8))
        goal = (7, 7)
        for east in range(8):
            for north in range(8):
                start = (east, north)
                if start == goal:
                    continue
                with self.subTest(start=start):
                    oracle = _dijkstra(graph, start, goal)
                    assert oracle is not None
                    self.assertLessEqual(
                        graph.heuristic(start, goal),
                        oracle[0] + 1.0e-12,
                        "an overestimate makes A* return suboptimal routes",
                    )


class UnreachableTests(unittest.TestCase):
    """#20 requires an unreachable stop to be reported, not to raise."""

    def test_a_walled_off_goal_returns_none(self) -> None:
        graph = GridGraph(_grid((".#.", ".#.", ".#.")))
        self.assertIsNone(a_star(graph, (0, 0), (2, 0)))

    def test_an_impassable_start_or_goal_returns_none(self) -> None:
        graph = GridGraph(_grid(("#.", "..")))
        self.assertIsNone(a_star(graph, (0, 1), (1, 1)), "blocked start")
        self.assertIsNone(a_star(graph, (1, 1), (0, 1)), "blocked goal")

    def test_an_off_grid_endpoint_returns_none(self) -> None:
        graph = GridGraph(_open_grid(3, 3))
        self.assertIsNone(a_star(graph, (0, 0), (9, 9)))
        self.assertIsNone(a_star(graph, (-1, 0), (1, 1)))

    def test_start_equal_to_goal_is_a_single_cell_route(self) -> None:
        graph = GridGraph(_open_grid(3, 3))
        route = a_star(graph, (1, 1), (1, 1))
        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(route.cells, ((1, 1),))
        self.assertAlmostEqual(route.cost, 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
