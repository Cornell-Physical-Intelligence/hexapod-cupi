"""Analytic checks for the occupancy predicates used by this audit."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial import ConvexHull

from audit_assembly import mesh_topology, winding
from build_candidate import read_stl, write_stl


class OccupancyTests(unittest.TestCase):
    def cube(self):
        points = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'cube.stl'
            write_stl(path, ConvexHull(points))
            return read_stl(path)

    def test_watertight_cube_inside_and_outside_winding(self):
        triangles = self.cube()
        self.assertTrue(mesh_topology(triangles)['closed_consistently_wound_manifold'])
        points = np.array([[0., 0, 0], [.5, -.2, .7], [2., 0, 0], [2, 2, 2.]])
        np.testing.assert_allclose(winding(points, triangles), [1, 1, 0, 0], atol=1e-12)

    def test_reversed_closed_surface_preserves_occupancy_magnitude(self):
        triangles = self.cube()[:, [0, 2, 1], :]
        self.assertTrue(mesh_topology(triangles)['closed_consistently_wound_manifold'])
        np.testing.assert_allclose(winding(np.array([[0., 0, 0], [2., 0, 0]]), triangles), [-1, 0], atol=1e-12)

    def test_open_mesh_is_not_certified_for_winding(self):
        topology = mesh_topology(self.cube()[:-1])
        self.assertFalse(topology['closed_consistently_wound_manifold'])
        self.assertEqual(topology['boundary_edges'], 3)

    def test_inconsistent_face_winding_is_rejected(self):
        triangles = self.cube()
        triangles[0] = triangles[0, [0, 2, 1], :]
        topology = mesh_topology(triangles)
        self.assertFalse(topology['closed_consistently_wound_manifold'])
        self.assertEqual(topology['inconsistent_winding_edges'], 3)


if __name__ == '__main__':
    unittest.main()
