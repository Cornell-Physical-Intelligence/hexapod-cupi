"""Boundary geometry and the clearance margin that decides feasibility.

What is guarded here is the *request* side of route planning: an operator's drawing
is rejected before approval rather than after motion, and the margin that R-05
requires is assembled from three independently-owned terms rather than baked into a
constant. Everything is closed-form. No grid, no search, no randomness.
"""

from __future__ import annotations

import math
import unittest

from navigation.region import PlanMargin, Polygon, SurveyRegion


# A 6 x 4 m rectangle: the demonstration area named in issue #20.
DEMONSTRATION_AREA = ((0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0))


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


class PolygonStructureTests(unittest.TestCase):
    """Validation that must not depend on any unimplemented predicate."""

    def test_fewer_than_three_vertices_is_rejected(self) -> None:
        for vertices in ((), ((0.0, 0.0),), ((0.0, 0.0), (1.0, 1.0))):
            with self.subTest(vertices=vertices), self.assertRaises(ValueError):
                Polygon(vertices)

    def test_nonfinite_vertices_are_rejected(self) -> None:
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                Polygon(((0.0, 0.0), (1.0, 0.0), (bad, 1.0)))

    def test_a_vertex_must_be_a_pair(self) -> None:
        with self.assertRaises(ValueError):
            Polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0, 2.0)))
        with self.assertRaises(TypeError):
            Polygon(((0.0, 0.0), (1.0, 0.0), 3.0))

    def test_vertices_are_normalised_to_an_immutable_tuple(self) -> None:
        # Passed a list of lists, the polygon must not keep the caller's mutable
        # objects: a frozen dataclass that does is mutable through the back door.
        polygon = Polygon([[0, 0], [6, 0], [6, 4], [0, 4]])
        self.assertIsInstance(polygon.vertices, tuple)
        self.assertEqual(polygon.vertices, ((0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0)))
        for vertex in polygon.vertices:
            self.assertIsInstance(vertex, tuple)

    def test_edges_close_the_ring_exactly_once(self) -> None:
        polygon = Polygon(DEMONSTRATION_AREA)
        edges = polygon.edges
        self.assertEqual(len(edges), 4, "a quadrilateral has four edges, not five")
        self.assertEqual(edges[-1], ((0.0, 4.0), (0.0, 0.0)), "closing edge is implicit")

    def test_bounds_cover_every_vertex(self) -> None:
        polygon = Polygon(DEMONSTRATION_AREA)
        self.assertEqual(polygon.bounds_m, (0.0, 0.0, 6.0, 4.0))

    def test_signed_area_is_positive_counterclockwise(self) -> None:
        counterclockwise = Polygon(DEMONSTRATION_AREA)
        clockwise = Polygon(tuple(reversed(DEMONSTRATION_AREA)))
        self.assertAlmostEqual(counterclockwise.signed_area_m2, 24.0, places=12)
        self.assertAlmostEqual(clockwise.signed_area_m2, -24.0, places=12)


class PolygonSimplicityTests(unittest.TestCase):
    """CHALLENGE 5 -- ``Polygon.is_simple``."""

    def test_self_crossing_polygon_is_rejected(self) -> None:
        # The classic bowtie: edge (0,0)-(2,2) crosses edge (2,0)-(0,2).
        bowtie = Polygon(((0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (0.0, 2.0)))
        self.assertFalse(bowtie.is_simple(), "a bowtie crosses itself")

    def test_a_plain_rectangle_is_simple(self) -> None:
        self.assertTrue(Polygon(DEMONSTRATION_AREA).is_simple())

    def test_a_concave_polygon_is_still_simple(self) -> None:
        # Concavity is legal; only crossing is not. An L-shaped survey area is an
        # ordinary request and must not be refused.
        ell = Polygon(
            (
                (0.0, 0.0),
                (6.0, 0.0),
                (6.0, 2.0),
                (3.0, 2.0),
                (3.0, 4.0),
                (0.0, 4.0),
            )
        )
        self.assertTrue(ell.is_simple(), "an L-shaped area is simple, merely concave")

    def test_touching_at_a_shared_vertex_is_not_a_crossing(self) -> None:
        # Consecutive edges always meet at their shared vertex. An implementation
        # that reports that as an intersection rejects every polygon.
        self.assertTrue(Polygon(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))).is_simple())

    def test_collinear_backtracking_is_not_simple(self) -> None:
        # Out and straight back along the same line encloses no area and overlaps
        # itself. Orientation tests alone return zero for every triple here, so this
        # is the case a cross-product-only implementation misses.
        spike = Polygon(((0.0, 0.0), (2.0, 0.0), (1.0, 0.0), (1.0, 2.0)))
        self.assertFalse(spike.is_simple(), "collinear overlap is self-intersection")


class PolygonContainmentTests(unittest.TestCase):
    """CHALLENGE -- ``Polygon.contains`` and ``distance_to_boundary_m``."""

    def setUp(self) -> None:
        self.area = Polygon(DEMONSTRATION_AREA)

    def test_interior_and_exterior_points_are_distinguished(self) -> None:
        self.assertTrue(self.area.contains((3.0, 2.0)))
        for outside in ((-1.0, 2.0), (7.0, 2.0), (3.0, -1.0), (3.0, 5.0)):
            with self.subTest(point=outside):
                self.assertFalse(self.area.contains(outside))

    def test_a_ray_through_a_vertex_is_not_counted_twice(self) -> None:
        # A point due west of the (6, 4) corner casts a ray that leaves exactly
        # through a vertex. Counting that crossing twice flips the answer.
        diamond = Polygon(((2.0, 0.0), (4.0, 2.0), (2.0, 4.0), (0.0, 2.0)))
        self.assertTrue(diamond.contains((2.0, 2.0)))
        self.assertFalse(diamond.contains((4.5, 2.0)))

    def test_a_concave_notch_is_outside(self) -> None:
        ell = Polygon(
            (
                (0.0, 0.0),
                (6.0, 0.0),
                (6.0, 2.0),
                (3.0, 2.0),
                (3.0, 4.0),
                (0.0, 4.0),
            )
        )
        self.assertTrue(ell.contains((1.0, 3.0)))
        self.assertFalse(ell.contains((5.0, 3.0)), "the notch is not part of the area")

    def test_distance_to_boundary_uses_the_segment_not_the_line(self) -> None:
        # (8, 6) is beyond the (6, 4) corner on both axes, so the nearest point on
        # the boundary is that corner. Projecting onto the unclamped lines instead
        # gives 2.0, which is what an unclamped implementation returns.
        self.assertAlmostEqual(
            self.area.distance_to_boundary_m((8.0, 6.0)),
            math.hypot(2.0, 2.0),
            places=12,
        )

    def test_distance_to_boundary_is_the_nearest_edge(self) -> None:
        # From the centre the nearest edge is one of the long ones, 2 m away.
        self.assertAlmostEqual(
            self.area.distance_to_boundary_m((3.0, 2.0)), 2.0, places=12
        )
        # Off centre, the nearest edge is the west one.
        self.assertAlmostEqual(
            self.area.distance_to_boundary_m((0.5, 2.0)), 0.5, places=12
        )


class SurveyRegionTests(unittest.TestCase):
    def test_components_must_be_polygons(self) -> None:
        area = Polygon(DEMONSTRATION_AREA)
        with self.assertRaises(TypeError):
            SurveyRegion(boundary=DEMONSTRATION_AREA)
        with self.assertRaises(TypeError):
            SurveyRegion(boundary=area, exclusions=(DEMONSTRATION_AREA,))
        with self.assertRaises(TypeError):
            SurveyRegion(boundary=area, corridor=DEMONSTRATION_AREA)

    def test_exclusions_default_to_none_and_normalise_to_a_tuple(self) -> None:
        region = SurveyRegion(boundary=Polygon(DEMONSTRATION_AREA))
        self.assertEqual(region.exclusions, ())
        listed = SurveyRegion(
            boundary=Polygon(DEMONSTRATION_AREA),
            exclusions=[Polygon(((1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)))],
        )
        self.assertIsInstance(listed.exclusions, tuple)

    def test_a_point_inside_an_exclusion_is_outside_the_region(self) -> None:
        # The second map of #20: the same area with one blocked region drawn in it.
        exclusion = Polygon(((2.0, 1.0), (4.0, 1.0), (4.0, 3.0), (2.0, 3.0)))
        region = SurveyRegion(
            boundary=Polygon(DEMONSTRATION_AREA), exclusions=(exclusion,)
        )
        self.assertTrue(region.contains((1.0, 2.0)))
        self.assertFalse(region.contains((3.0, 2.0)), "drawn exclusions are no-go")

    def test_clearance_outside_the_region_is_zero(self) -> None:
        region = SurveyRegion(boundary=Polygon(DEMONSTRATION_AREA))
        self.assertEqual(
            region.clearance_m((-1.0, 2.0)),
            0.0,
            "outside the region there is no clearance to report",
        )

    def test_clearance_accounts_for_exclusions(self) -> None:
        exclusion = Polygon(((2.0, 1.0), (4.0, 1.0), (4.0, 3.0), (2.0, 3.0)))
        region = SurveyRegion(
            boundary=Polygon(DEMONSTRATION_AREA), exclusions=(exclusion,)
        )
        # (1, 2) is 1 m from the boundary's west edge and 1 m from the exclusion's
        # west edge, so both bind equally.
        self.assertAlmostEqual(region.clearance_m((1.0, 2.0)), 1.0, places=12)
        # (0.5, 2) is nearer the boundary than the exclusion.
        self.assertAlmostEqual(region.clearance_m((0.5, 2.0)), 0.5, places=12)


if __name__ == "__main__":
    unittest.main()
