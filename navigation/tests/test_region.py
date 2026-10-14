"""Boundary geometry and the clearance margin that decides feasibility.

What is guarded here is the *request* side of route planning: an operator's drawing
is rejected before approval rather than after motion, and the margin that R-05
requires is assembled from three independently-owned terms rather than baked into a
constant. Polygon geometry itself is Shapely's and is not re-tested here.
Everything is closed-form. No grid, no search, no randomness.
"""

from __future__ import annotations

import unittest
import warnings

from shapely import Point, Polygon, box

from navigation.region import PlanMargin, SurveyRegion


# A 6 x 4 m rectangle: the demonstration area named in issue #20.
DEMONSTRATION_AREA = box(0.0, 0.0, 6.0, 4.0)


class PlanMarginTests(unittest.TestCase):
    def test_total_is_the_sum_of_its_three_owners(self) -> None:
        margin = PlanMargin(
            footprint_radius_m=0.40,
            stop_tolerance_m=0.15,
            position_error_bound_m=3.0,
        )
        self.assertAlmostEqual(margin.total_m, 3.55, places=12)

    def test_ideal_poses_contribute_no_position_error(self) -> None:
        # The rehearsal case: #20 runs on ideal poses, so the GPS term is zero and
        # the margin reduces to the robot's own geometry.
        margin = PlanMargin(
            footprint_radius_m=0.40,
            stop_tolerance_m=0.15,
            position_error_bound_m=0.0,
        )
        self.assertAlmostEqual(margin.total_m, 0.55, places=12)

    def test_nonfinite_or_negative_terms_are_rejected(self) -> None:
        for bad in (float("nan"), float("inf"), -float("inf"), -0.1):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                PlanMargin(
                    footprint_radius_m=bad,
                    stop_tolerance_m=0.15,
                    position_error_bound_m=0.0,
                )
            with self.subTest(value=bad), self.assertRaises(ValueError):
                PlanMargin(
                    footprint_radius_m=0.40,
                    stop_tolerance_m=bad,
                    position_error_bound_m=0.0,
                )
            with self.subTest(value=bad), self.assertRaises(ValueError):
                PlanMargin(
                    footprint_radius_m=0.40,
                    stop_tolerance_m=0.15,
                    position_error_bound_m=bad,
                )

    def test_a_robot_must_have_a_positive_footprint(self) -> None:
        # Zero tolerance and zero GPS error are meaningful; a zero-radius robot is
        # not, and would make every margin check vacuous.
        with self.assertRaises(ValueError):
            PlanMargin(
                footprint_radius_m=0.0,
                stop_tolerance_m=0.15,
                position_error_bound_m=0.0,
            )


class SurveyRegionValidationTests(unittest.TestCase):
    """An operator's drawing is rejected at construction, before approval."""

    def test_components_must_be_polygons(self) -> None:
        coords = ((0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0))
        with self.assertRaises(TypeError):
            SurveyRegion(boundary=coords)
        with self.assertRaises(TypeError):
            SurveyRegion(boundary=DEMONSTRATION_AREA, exclusions=(coords,))
        with self.assertRaises(TypeError):
            SurveyRegion(boundary=DEMONSTRATION_AREA, corridor=coords)
        with self.assertRaises(TypeError):
            SurveyRegion(boundary=DEMONSTRATION_AREA, exclusions=DEMONSTRATION_AREA)

    def test_self_crossing_drawings_are_rejected(self) -> None:
        # The classic bowtie, and a collinear spike that goes out and straight back.
        bowtie = Polygon(((0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (0.0, 2.0)))
        spike = Polygon(((0.0, 0.0), (2.0, 0.0), (1.0, 0.0), (1.0, 2.0)))
        for bad in (bowtie, spike):
            with self.subTest(polygon=bad.wkt):
                with self.assertRaises(ValueError):
                    SurveyRegion(boundary=bad)
                with self.assertRaises(ValueError):
                    SurveyRegion(boundary=DEMONSTRATION_AREA, exclusions=(bad,))
                with self.assertRaises(ValueError):
                    SurveyRegion(boundary=DEMONSTRATION_AREA, corridor=bad)

    def test_nonfinite_vertices_are_rejected(self) -> None:
        for bad in (float("nan"), float("inf")):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                with warnings.catch_warnings():
                    # Shapely warns while building the ring; the rejection is ours.
                    warnings.simplefilter("ignore", RuntimeWarning)
                    polygon = Polygon(((0.0, 0.0), (1.0, 0.0), (bad, 1.0)))
                SurveyRegion(boundary=polygon)

    def test_a_concave_area_is_accepted(self) -> None:
        # Concavity is legal; only crossing is not. An L-shaped survey area is an
        # ordinary request and must not be refused.
        ell = Polygon(
            ((0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (3.0, 2.0), (3.0, 4.0), (0.0, 4.0))
        )
        SurveyRegion(boundary=ell)

    def test_exclusions_default_to_empty_and_normalise_to_a_tuple(self) -> None:
        region = SurveyRegion(boundary=DEMONSTRATION_AREA)
        self.assertEqual(region.exclusions, ())
        listed = SurveyRegion(
            boundary=DEMONSTRATION_AREA, exclusions=[box(1.0, 1.0, 2.0, 2.0)]
        )
        self.assertIsInstance(listed.exclusions, tuple)


class FreeSpaceTests(unittest.TestCase):
    def test_a_point_inside_an_exclusion_is_not_free(self) -> None:
        # The second map of #20: the same area with one blocked region drawn in it.
        region = SurveyRegion(
            boundary=DEMONSTRATION_AREA, exclusions=(box(2.0, 1.0, 4.0, 3.0),)
        )
        self.assertTrue(region.free_space.contains(Point(1.0, 2.0)))
        self.assertFalse(
            region.free_space.contains(Point(3.0, 2.0)), "drawn exclusions are no-go"
        )

    def test_a_corridor_extends_the_free_space_and_the_bounds(self) -> None:
        corridor = box(-2.0, 1.5, 0.5, 2.5)
        region = SurveyRegion(boundary=DEMONSTRATION_AREA, corridor=corridor)
        self.assertTrue(region.free_space.contains(Point(-1.0, 2.0)))
        self.assertEqual(region.bounds_m, (-2.0, 0.0, 6.0, 4.0))

    def test_bounds_without_a_corridor_are_the_boundary(self) -> None:
        region = SurveyRegion(boundary=DEMONSTRATION_AREA)
        self.assertEqual(region.bounds_m, (0.0, 0.0, 6.0, 4.0))


class FeasibleRegionTests(unittest.TestCase):
    """The margin shrinks the free space; too large a margin leaves nothing."""

    def test_feasible_region_keeps_the_margin_from_every_edge(self) -> None:
        # 0.55 m of margin leaves a 4.9 x 2.9 m interior.
        region = SurveyRegion(boundary=DEMONSTRATION_AREA)
        feasible = region.feasible(
            PlanMargin(
                footprint_radius_m=0.40,
                stop_tolerance_m=0.15,
                position_error_bound_m=0.0,
            )
        )
        min_e, min_n, max_e, max_n = feasible.bounds
        for actual, expected in zip((min_e, min_n, max_e, max_n), (0.55, 0.55, 5.45, 3.45)):
            self.assertAlmostEqual(actual, expected, places=9)

    def test_a_several_metre_gps_bound_leaves_nothing(self) -> None:
        # 0.40 + 0.15 + 3.00 = 3.55 m of clearance cannot fit inside 6 x 4 m.
        region = SurveyRegion(boundary=DEMONSTRATION_AREA)
        feasible = region.feasible(
            PlanMargin(
                footprint_radius_m=0.40,
                stop_tolerance_m=0.15,
                position_error_bound_m=3.0,
            )
        )
        self.assertTrue(feasible.is_empty)


if __name__ == "__main__":
    unittest.main()
