"""Analytic geometry checks independent of the robot's desired fit."""
import unittest
import numpy as np
from scipy.spatial import ConvexHull
from build_candidate import closest_on_convex, sphere_surface_gap, support


class AnalyticGeometryTests(unittest.TestCase):
    def test_box_projection_matches_componentwise_clamp(self):
        corners = np.array([[x, y, z] for x in (-1., 1.) for y in (-2., 2.) for z in (-3., 3.)])
        points = np.array([[0, 0, 0], [2, 1, 1], [-2, -4, -6], [1, 2, 3], [.2, 5, 4]])
        expected = np.clip(points, [-1, -2, -3], [1, 2, 3])
        distances, closest = closest_on_convex(points, ConvexHull(corners))
        np.testing.assert_allclose(closest, expected, atol=1e-12)
        np.testing.assert_allclose(distances, np.linalg.norm(points-expected, axis=1), atol=1e-12)

    def test_sphere_gap_finds_bisector_edge_point_missing_from_vertices(self):
        triangles = np.array([[[-1., 0, 0], [1., 0, 0], [-1., .1, 0]]])
        result = sphere_surface_gap(triangles, np.array([[-1., 0, 0], [1., 0, 0]]), .2)
        self.assertAlmostEqual(result['maximum_outside_distance_m'], np.sqrt(1+.05**2)-.2, places=12)
        np.testing.assert_allclose(result['witness_point_mesh_m'], [0, .05, 0], atol=1e-12)

    def test_convex_projection_witness_is_a_support_certificate(self):
        tetra = np.array([[0., 0, 0], [1., 0, 0], [0, 1., 0], [0, 0, 1.]])
        point = np.array([[1., 1., 1.]])
        distance, nearest = closest_on_convex(point, ConvexHull(tetra))
        np.testing.assert_allclose(nearest, [[1/3, 1/3, 1/3]], atol=1e-12)
        normal = (point-nearest)/distance[:, None]
        self.assertAlmostEqual(float(support(point, normal)[0]-support(tetra, normal)[0]), distance[0], places=12)

    def test_translation_preserves_distance(self):
        tetra = np.array([[0., 0, 0], [1., 0, 0], [0, 1., 0], [0, 0, 1.]])
        points = np.array([[.2, .2, .2], [1., 1., 1.]])
        shift = np.array([.221, -.017, .01])
        a, qa = closest_on_convex(points, ConvexHull(tetra))
        b, qb = closest_on_convex(points+shift, ConvexHull(tetra+shift))
        np.testing.assert_allclose(a, b, atol=1e-12)
        np.testing.assert_allclose(qa+shift, qb, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
