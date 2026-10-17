"""Rasterization, margin inflation, and when the feasible region is empty.

The arithmetic in ``EmptyFeasibleRegionTests`` is the reason this module exists as
its own reviewable unit: issue #20 fixes the demonstration at 6 x 4 m while R-03
sets a plain-GPS error bound of several meters, and those two facts are mutually
exclusive. The planner's correct response is an empty feasible region, reported --
not an exception, and not a route that quietly leaves the approved area.

Everything is closed-form. No search, no simulator, no randomness.
"""

from __future__ import annotations

import unittest

from shapely import Point, box

from navigation.grid import OccupancyGrid
from navigation.region import PlanMargin, SurveyRegion


DEMONSTRATION_AREA = box(0.0, 0.0, 6.0, 4.0)

# Unfrozen placeholders. #20's "Before implementation" block still has to freeze
# these; they are written here so the arithmetic is reproducible, not to pre-empt it.
FOOTPRINT_RADIUS_M = 0.40
STOP_TOLERANCE_M = 0.15

# Shapely approximates the rounded corners of an inflated exclusion with straight
# segments, so a cell center within this distance of the exact margin may go either
# way. Well under a cell width at every margin used here.
BUFFER_TOLERANCE_M = 0.02


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
        exclusions = (box(2.5, 1.5, 3.5, 2.5),)
    return SurveyRegion(boundary=DEMONSTRATION_AREA, exclusions=exclusions)


def _clearance_m(region: SurveyRegion, center) -> float:
    """Exact distance from ``center`` to the nearest edge, 0.0 outside free space.

    This is the oracle: the slow, obvious definition, independent of ``buffer``.
    """

    point = Point(center)
    if not region.free_space.contains(point):
        return 0.0
    return region.free_space.boundary.distance(point)


class GridStructureTests(unittest.TestCase):
    """Construction and addressing, independent of any region."""

    def _open_grid(self, width: int, height: int) -> OccupancyGrid:
        return OccupancyGrid(
            origin_m=(0.0, 0.0),
            cell_size_m=1.0,
            width_cells=width,
            height_cells=height,
            passable=(True,) * (width * height),
        )

    def test_passable_must_match_the_cell_count(self) -> None:
        with self.assertRaises(ValueError):
            OccupancyGrid(
                origin_m=(0.0, 0.0),
                cell_size_m=1.0,
                width_cells=3,
                height_cells=2,
                passable=(True, True, True),
            )

    def test_cell_size_and_extent_must_be_positive(self) -> None:
        for bad_size in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(cell_size_m=bad_size), self.assertRaises(ValueError):
                OccupancyGrid(
                    origin_m=(0.0, 0.0),
                    cell_size_m=bad_size,
                    width_cells=1,
                    height_cells=1,
                    passable=(True,),
                )
        for bad_extent in (0, -2, 1.5, True):
            with self.subTest(width_cells=bad_extent), self.assertRaises(ValueError):
                OccupancyGrid(
                    origin_m=(0.0, 0.0),
                    cell_size_m=1.0,
                    width_cells=bad_extent,
                    height_cells=1,
                    passable=(True,),
                )

    def test_cell_centers_round_trip_through_cell_of(self) -> None:
        grid = self._open_grid(6, 4)
        for east in range(6):
            for north in range(4):
                with self.subTest(cell=(east, north)):
                    center = grid.cell_center_m((east, north))
                    self.assertEqual(grid.cell_of(center), (east, north))

    def test_passable_is_row_major_over_north(self) -> None:
        grid = OccupancyGrid(
            origin_m=(0.0, 0.0),
            cell_size_m=1.0,
            width_cells=2,
            height_cells=2,
            passable=(True, False, False, True),
        )
        self.assertTrue(grid.is_passable((0, 0)))
        self.assertFalse(grid.is_passable((1, 0)))
        self.assertFalse(grid.is_passable((0, 1)))
        self.assertTrue(grid.is_passable((1, 1)))

    def test_off_grid_cells_are_impassable_rather_than_an_error(self) -> None:
        grid = self._open_grid(2, 2)
        for outside in ((-1, 0), (2, 0), (0, -1), (0, 2)):
            with self.subTest(cell=outside):
                self.assertFalse(grid.in_bounds(outside))
                self.assertFalse(grid.is_passable(outside))

    def test_passable_cell_count_ignores_impassable_cells(self) -> None:
        grid = OccupancyGrid(
            origin_m=(0.0, 0.0),
            cell_size_m=1.0,
            width_cells=2,
            height_cells=2,
            passable=(True, False, True, False),
        )
        self.assertEqual(grid.cell_count, 4)
        self.assertEqual(grid.passable_cell_count, 2)


class EmptyFeasibleRegionTests(unittest.TestCase):
    """The #20 margin arithmetic, as an executable fact."""

    def test_margin_larger_than_area_leaves_no_feasible_region(self) -> None:
        # 0.40 + 0.15 + 3.00 = 3.55 m of required clearance. Shrinking 6 x 4 m by
        # that on every side leaves nothing at all: 6 - 7.1 and 4 - 7.1 are both
        # negative. This is the plain-GPS decision of #38 meeting the 6 x 4 m
        # demonstration of #20, and it is why one of the two has to move.
        grid = OccupancyGrid.from_region(
            _demonstration_region(), _margin(3.0), cell_size_m=0.25
        )
        self.assertEqual(
            grid.passable_cell_count,
            0,
            "a several-meter GPS bound leaves no route inside a 6 x 4 m area",
        )

    def test_an_empty_region_is_reported_not_raised(self) -> None:
        # The grid still exists and still describes the area; it simply has nowhere
        # to go. Callers report that, so no exception may escape.
        grid = OccupancyGrid.from_region(
            _demonstration_region(), _margin(3.0), cell_size_m=0.5
        )
        self.assertGreater(grid.cell_count, 0)
        self.assertEqual(grid.passable_cell_count, 0)

    def test_ideal_poses_leave_a_usable_region(self) -> None:
        # With the GPS term at zero the margin is 0.55 m, leaving a 4.9 x 2.9 m
        # interior. This is the rehearsal case #20 actually specifies.
        grid = OccupancyGrid.from_region(
            _demonstration_region(), _margin(0.0), cell_size_m=0.25
        )
        self.assertGreater(
            grid.passable_cell_count, 0, "ideal poses must leave somewhere to go"
        )

    def test_larger_error_bound_never_adds_passable_cells(self) -> None:
        # Clearance is fixed by geometry, so raising the bound can only raise the
        # required clearance: feasibility shrinks monotonically and the passable set
        # of a larger bound is a subset of a smaller one's.
        region = _demonstration_region(with_exclusion=True)
        previous: set[tuple[int, int]] | None = None
        for bound in (0.0, 0.25, 0.5, 1.0, 1.5):
            grid = OccupancyGrid.from_region(region, _margin(bound), cell_size_m=0.25)
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


class RasterizationTests(unittest.TestCase):
    """``OccupancyGrid.from_region`` against its definition."""

    def test_grid_covers_the_region_bounds(self) -> None:
        region = _demonstration_region()
        grid = OccupancyGrid.from_region(region, _margin(0.0), cell_size_m=0.25)
        self.assertEqual(grid.origin_m, (0.0, 0.0))
        self.assertEqual((grid.width_cells, grid.height_cells), (24, 16))

    def test_a_partial_cell_is_still_covered(self) -> None:
        # 4 / 0.3 is 13.33 cells of north span, so 14 are needed to reach the top.
        region = _demonstration_region()
        grid = OccupancyGrid.from_region(region, _margin(0.0), cell_size_m=0.3)
        self.assertEqual((grid.width_cells, grid.height_cells), (20, 14))

    def test_inflation_matches_a_brute_force_distance_check(self) -> None:
        # Every cell must agree with the exact clearance computed directly from the
        # region, except within BUFFER_TOLERANCE_M of the margin itself.
        for bound in (0.0, 0.5):
            for with_exclusion in (False, True):
                region = _demonstration_region(with_exclusion=with_exclusion)
                margin = _margin(bound)
                grid = OccupancyGrid.from_region(region, margin, cell_size_m=0.25)
                for north in range(grid.height_cells):
                    for east in range(grid.width_cells):
                        cell = (east, north)
                        clearance = _clearance_m(region, grid.cell_center_m(cell))
                        with self.subTest(
                            bound=bound, exclusion=with_exclusion, cell=cell
                        ):
                            if clearance >= margin.total_m + BUFFER_TOLERANCE_M:
                                self.assertTrue(grid.is_passable(cell))
                            elif clearance < margin.total_m - BUFFER_TOLERANCE_M:
                                self.assertFalse(grid.is_passable(cell))

    def test_a_drawn_exclusion_blocks_its_own_cells(self) -> None:
        region = _demonstration_region(with_exclusion=True)
        grid = OccupancyGrid.from_region(region, _margin(0.0), cell_size_m=0.25)
        # The exclusion spans 2.5..3.5 m east, 1.5..2.5 m north; its center must be
        # impassable, and so must a ring of cells one footprint radius around it.
        self.assertFalse(grid.is_passable(grid.cell_of((3.0, 2.0))))
        self.assertFalse(
            grid.is_passable(grid.cell_of((2.5 - FOOTPRINT_RADIUS_M / 2.0, 2.0))),
            "cells within the margin of an exclusion are inflated away",
        )


if __name__ == "__main__":
    unittest.main()
