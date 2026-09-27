"""Rasterisation, margin inflation, and when the feasible region is empty.

The arithmetic in ``EmptyFeasibleRegionTests`` is the reason this module exists as
its own reviewable unit: issue #20 fixes the demonstration at 6 x 4 m while R-03
sets a plain-GPS error bound of several metres, and those two facts are mutually
exclusive. The planner's correct response is an empty feasible region, reported --
not an exception, and not a route that quietly leaves the approved area.

Everything is closed-form. No search, no simulator, no randomness.
"""

from __future__ import annotations

import math
import unittest

from navigation.grid import PASSABLE_CELL_COST, RISK_WEIGHT, CostGrid
from navigation.region import PlanMargin, Polygon, SurveyRegion


DEMONSTRATION_AREA = ((0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0))

# Unfrozen placeholders. #20's "Before implementation" block still has to freeze
# these; they are written here so the arithmetic is reproducible, not to pre-empt it.
FOOTPRINT_RADIUS_M = 0.40
STOP_TOLERANCE_M = 0.15


def _margin(position_error_bound_m: float) -> PlanMargin:
    return PlanMargin(
        footprint_radius_m=FOOTPRINT_RADIUS_M,
        stop_tolerance_m=STOP_TOLERANCE_M,
        position_error_bound_m=position_error_bound_m,
    )


def _demonstration_region(with_exclusion: bool = False) -> SurveyRegion:
    exclusions = ()
    if with_exclusion:
        # #20's second map: the same area with one blocked region.
        exclusions = (Polygon(((2.5, 1.5), (3.5, 1.5), (3.5, 2.5), (2.5, 2.5))),)
    return SurveyRegion(boundary=Polygon(DEMONSTRATION_AREA), exclusions=exclusions)


def _expected_cost(region: SurveyRegion, margin: PlanMargin, centre) -> float:
    """The specification of one cell's cost, independent of how the grid builds it.

    This is the oracle: it is deliberately the slow, obvious form, so that a faster
    implementation is checked against the definition rather than against itself.
    """

    clearance = region.clearance_m(centre)
    if clearance < margin.total_m:
        return math.inf
    span = margin.position_error_bound_m
    if span <= 0.0:
        return PASSABLE_CELL_COST
    excess = clearance - margin.total_m
    if excess >= span:
        return PASSABLE_CELL_COST
    return PASSABLE_CELL_COST * (1.0 + RISK_WEIGHT * (1.0 - excess / span))


class GridStructureTests(unittest.TestCase):
    """Construction and addressing, which depend on no unimplemented method."""

    def _flat_grid(self, width: int, height: int, cost: float = 1.0) -> CostGrid:
        return CostGrid(
            origin_m=(0.0, 0.0),
            cell_size_m=1.0,
            width_cells=width,
            height_cells=height,
            costs=tuple([cost] * (width * height)),
        )

    def test_costs_must_match_the_cell_count(self) -> None:
        with self.assertRaises(ValueError):
            CostGrid(
                origin_m=(0.0, 0.0),
                cell_size_m=1.0,
                width_cells=3,
                height_cells=2,
                costs=(1.0, 1.0, 1.0),
            )

    def test_infinite_cost_is_legal_but_nan_is_not(self) -> None:
        CostGrid(
            origin_m=(0.0, 0.0),
            cell_size_m=1.0,
            width_cells=2,
            height_cells=1,
            costs=(1.0, math.inf),
        )
        with self.assertRaises(ValueError):
            CostGrid(
                origin_m=(0.0, 0.0),
                cell_size_m=1.0,
                width_cells=2,
                height_cells=1,
                costs=(1.0, float("nan")),
            )

    def test_a_cost_below_the_structural_floor_is_rejected(self) -> None:
        # A cell cheaper than the floor would break heuristic admissibility silently,
        # so it is refused at construction instead.
        with self.assertRaises(ValueError):
            CostGrid(
                origin_m=(0.0, 0.0),
                cell_size_m=1.0,
                width_cells=2,
                height_cells=1,
                costs=(PASSABLE_CELL_COST - 0.5, 1.0),
            )

    def test_cell_size_and_extent_must_be_positive(self) -> None:
        for bad_size in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(cell_size_m=bad_size), self.assertRaises(ValueError):
                CostGrid(
                    origin_m=(0.0, 0.0),
                    cell_size_m=bad_size,
                    width_cells=1,
                    height_cells=1,
                    costs=(1.0,),
                )
        for bad_extent in (0, -2, 1.5, True):
            with self.subTest(width_cells=bad_extent), self.assertRaises(ValueError):
                CostGrid(
                    origin_m=(0.0, 0.0),
                    cell_size_m=1.0,
                    width_cells=bad_extent,
                    height_cells=1,
                    costs=(1.0,),
                )

    def test_min_cell_cost_is_the_constant_floor_not_the_observed_minimum(self) -> None:
        grid = self._flat_grid(2, 2, cost=5.0)
        self.assertEqual(
            grid.min_cell_cost,
            PASSABLE_CELL_COST,
            "the heuristic scale must be a structural bound, not a sample",
        )

    def test_cell_centres_round_trip_through_cell_of(self) -> None:
        grid = self._flat_grid(6, 4)
        for east in range(6):
            for north in range(4):
                with self.subTest(cell=(east, north)):
                    centre = grid.cell_center_m((east, north))
                    self.assertEqual(grid.cell_of(centre), (east, north))

    def test_costs_are_row_major_over_north(self) -> None:
        grid = CostGrid(
            origin_m=(0.0, 0.0),
            cell_size_m=1.0,
            width_cells=2,
            height_cells=2,
            costs=(1.0, 2.0, 3.0, 4.0),
        )
        self.assertAlmostEqual(grid.cost((0, 0)), 1.0, places=12)
        self.assertAlmostEqual(grid.cost((1, 0)), 2.0, places=12)
        self.assertAlmostEqual(grid.cost((0, 1)), 3.0, places=12)
        self.assertAlmostEqual(grid.cost((1, 1)), 4.0, places=12)

    def test_off_grid_cells_are_impassable_rather_than_an_error(self) -> None:
        grid = self._flat_grid(2, 2)
        for outside in ((-1, 0), (2, 0), (0, -1), (0, 2)):
            with self.subTest(cell=outside):
                self.assertFalse(grid.in_bounds(outside))
                self.assertEqual(grid.cost(outside), math.inf)
                self.assertFalse(grid.is_passable(outside))

    def test_passable_cell_count_ignores_impassable_cells(self) -> None:
        grid = CostGrid(
            origin_m=(0.0, 0.0),
            cell_size_m=1.0,
            width_cells=2,
            height_cells=2,
            costs=(1.0, math.inf, 1.0, math.inf),
        )
        self.assertEqual(grid.cell_count, 4)
        self.assertEqual(grid.passable_cell_count, 2)


class EmptyFeasibleRegionTests(unittest.TestCase):
    """CHALLENGE 6 -- the #20 margin arithmetic, as an executable fact."""

    def test_margin_larger_than_area_leaves_no_feasible_region(self) -> None:
        # 0.40 + 0.15 + 3.00 = 3.55 m of required clearance. Shrinking 6 x 4 m by
        # that on every side leaves nothing at all: 6 - 7.1 and 4 - 7.1 are both
        # negative. This is the plain-GPS decision of #38 meeting the 6 x 4 m
        # demonstration of #20, and it is why one of the two has to move.
        grid = CostGrid.from_region(
            _demonstration_region(), _margin(3.0), cell_size_m=0.25
        )
        self.assertEqual(
            grid.passable_cell_count,
            0,
            "a several-metre GPS bound leaves no route inside a 6 x 4 m area",
        )

    def test_an_empty_region_is_reported_not_raised(self) -> None:
        # The grid still exists and still describes the area; it simply has nowhere
        # to go. Callers report that, so no exception may escape.
        grid = CostGrid.from_region(
            _demonstration_region(), _margin(3.0), cell_size_m=0.5
        )
        self.assertGreater(grid.cell_count, 0)
        self.assertEqual(grid.passable_cell_count, 0)

    def test_ideal_poses_leave_a_usable_region(self) -> None:
        # With the GPS term at zero the margin is 0.55 m, leaving a 4.9 x 2.9 m
        # interior. This is the rehearsal case #20 actually specifies.
        grid = CostGrid.from_region(
            _demonstration_region(), _margin(0.0), cell_size_m=0.25
        )
        self.assertGreater(
            grid.passable_cell_count, 0, "ideal poses must leave somewhere to go"
        )

    def test_larger_error_bound_never_adds_passable_cells(self) -> None:
        # CHALLENGE 8. Clearance is fixed by geometry, so raising the bound can only
        # raise the required clearance: feasibility shrinks monotonically and the
        # passable set of a larger bound is a subset of a smaller one's.
        region = _demonstration_region()
        previous: set[tuple[int, int]] | None = None
        for bound in (0.0, 0.25, 0.5, 1.0, 1.5):
            grid = CostGrid.from_region(region, _margin(bound), cell_size_m=0.25)
            passable = {
                (east, north)
                for north in range(grid.height_cells)
                for east in range(grid.width_cells)
                if grid.is_passable((east, north))
            }
            if previous is not None:
                self.assertTrue(
                    passable <= previous,
                    f"bound {bound} admitted cells a smaller bound refused",
                )
            previous = passable


class RasterisationTests(unittest.TestCase):
    """CHALLENGE 6 + 7 -- ``CostGrid.from_region`` against its definition."""

    def test_grid_covers_the_region_bounds(self) -> None:
        region = _demonstration_region()
        grid = CostGrid.from_region(region, _margin(0.0), cell_size_m=0.25)
        self.assertEqual(grid.origin_m, (0.0, 0.0))
        self.assertEqual((grid.width_cells, grid.height_cells), (24, 16))

    def test_a_partial_cell_is_still_covered(self) -> None:
        # 4 / 0.3 is 13.33 cells of north span, so 14 are needed to reach the top.
        region = _demonstration_region()
        grid = CostGrid.from_region(region, _margin(0.0), cell_size_m=0.3)
        self.assertEqual((grid.width_cells, grid.height_cells), (20, 14))

    def test_inflation_matches_a_brute_force_distance_check(self) -> None:
        # CHALLENGE 7. Every cell's cost must equal the definition computed directly
        # from the region's own clearance. An implementation may be optimised later;
        # this pins the result rather than the method.
        for bound in (0.0, 0.5):
            for with_exclusion in (False, True):
                region = _demonstration_region(with_exclusion=with_exclusion)
                margin = _margin(bound)
                grid = CostGrid.from_region(region, margin, cell_size_m=0.25)
                for north in range(grid.height_cells):
                    for east in range(grid.width_cells):
                        cell = (east, north)
                        expected = _expected_cost(
                            region, margin, grid.cell_center_m(cell)
                        )
                        actual = grid.cost(cell)
                        with self.subTest(
                            bound=bound, exclusion=with_exclusion, cell=cell
                        ):
                            if math.isinf(expected):
                                self.assertEqual(actual, math.inf)
                            else:
                                self.assertAlmostEqual(actual, expected, places=12)

    def test_a_drawn_exclusion_blocks_its_own_cells(self) -> None:
        region = _demonstration_region(with_exclusion=True)
        grid = CostGrid.from_region(region, _margin(0.0), cell_size_m=0.25)
        # The exclusion spans 2.5..3.5 m east, 1.5..2.5 m north; its centre must be
        # impassable, and so must a ring of cells one footprint radius around it.
        self.assertFalse(grid.is_passable(grid.cell_of((3.0, 2.0))))
        self.assertFalse(
            grid.is_passable(grid.cell_of((2.5 - FOOTPRINT_RADIUS_M / 2.0, 2.0))),
            "cells within the margin of an exclusion are inflated away",
        )


class RiskCostTests(unittest.TestCase):
    """CHALLENGE 6 -- the error-bound-scaled risk field."""

    def test_zero_error_bound_gives_uniform_cost(self) -> None:
        # With no position uncertainty there is nothing to be risk-averse about, so
        # every passable cell costs the floor and the route is purely geometric.
        grid = CostGrid.from_region(
            _demonstration_region(), _margin(0.0), cell_size_m=0.25
        )
        costs = {cost for cost in grid.costs if math.isfinite(cost)}
        self.assertEqual(costs, {PASSABLE_CELL_COST})

    def test_a_positive_error_bound_prices_bare_clearance_higher(self) -> None:
        # A cell with clearance to spare costs the floor; one that only just clears
        # the margin costs the floor times (1 + RISK_WEIGHT).
        region = _demonstration_region()
        margin = _margin(0.5)
        grid = CostGrid.from_region(region, margin, cell_size_m=0.25)
        middle = grid.cell_of((3.0, 2.0))
        self.assertAlmostEqual(grid.cost(middle), PASSABLE_CELL_COST, places=12)
        finite = [cost for cost in grid.costs if math.isfinite(cost)]
        self.assertGreater(
            max(finite),
            PASSABLE_CELL_COST,
            "cells near an edge must cost more once position error is admitted",
        )
        self.assertLessEqual(
            max(finite),
            PASSABLE_CELL_COST * (1.0 + RISK_WEIGHT) + 1.0e-12,
            "risk cost is bounded by RISK_WEIGHT",
        )


if __name__ == "__main__":
    unittest.main()
