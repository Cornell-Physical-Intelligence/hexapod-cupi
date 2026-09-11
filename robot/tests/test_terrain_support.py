from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "isaaclab")]
from experiments.terrain.tools.terrain_readiness import terrain_mesh
from experiments.terrain.tools.terrain_fixture_checks import vertical_surface_heights
from hexapod_terrain.support_queries import TerrainSupportQueries


def records():
    result = []
    for family in ("smooth_rough", "ramp", "step", "ridge", "pit"):
        vertices, faces, entry = terrain_mesh(family, 1103)
        entry["id"] = family
        result.append((entry, Path("unused"), vertices, faces))
    return result


class TerrainSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = records()
        cls.queries = TerrainSupportQueries(cls.records, chunk_size=37, dtype=torch.float64)

    def test_indexed_heights_match_independent_triangle_intersections(self):
        xy = np.random.default_rng(4409).uniform(-1.7, 1.7, (5, 80, 2))
        query = self.queries.query_local(xy, torch.arange(5))
        for i, (_, _, vertices, faces) in enumerate(self.records):
            expected = vertical_surface_heights(vertices, faces, xy[i])
            np.testing.assert_allclose(query.height_m[i].numpy(), expected, atol=1e-8, equal_nan=True)
            np.testing.assert_array_equal(query.geometry_hit[i].numpy(), np.isfinite(expected))
        self.assertLess(self.queries.statistics["max_cell_candidates"], 100)

    def test_sharp_step_pit_and_unknown_have_distinct_semantics(self):
        step = self.records[2][0]
        edge = step["edge_x_m"]
        query = self.queries.query_local([[edge - 1e-5, .1], [edge + 1e-5, .1]], 2)
        np.testing.assert_allclose(query.height_m.numpy(), [0., step["step_height_m"]], atol=1e-10)
        query = self.queries.query_local([[0., 0.], [-1., 0.], [1.6, 0.], [np.nan, 0.]], 4)
        self.assertAlmostEqual(float(query.height_m[0]), -.12)
        self.assertEqual(query.geometry_hit.tolist(), [True, True, False, False])
        self.assertEqual(query.support_geometry.tolist(), [False, True, False, False])
        self.assertTrue(bool(query.avoidance_hazard[0]))
        self.assertTrue(torch.isnan(query.height_m[2:]).all())

    def test_boundary_margin_never_clamps_unknown_onto_terrain(self):
        query = self.queries.query_local([[1.49, 0.], [1.51, 0.]], 3, boundary_margin_m=.05)
        self.assertEqual(query.geometry_hit.tolist(), [True, False])
        self.assertEqual(query.inside_course.tolist(), [False, False])
        self.assertEqual(query.support_geometry.tolist(), [False, False])

    def test_spatial_bin_boundaries_have_no_false_holes(self):
        xy = np.array([[round(float(x), 8) + epsilon, y - epsilon] for x in np.arange(-1.5, 1.501, .05)
                       for y in (-1., -.25, .4, 1.) for epsilon in (-1e-7, 0., 1e-7)])
        query = self.queries.query_local(xy, 0)
        _, _, vertices, faces = self.records[0]
        expected = vertical_surface_heights(vertices, faces, xy)
        np.testing.assert_allclose(query.height_m.numpy(), expected, atol=1e-8, equal_nan=True)

    def test_scene_translation_and_course_yaw_preserve_clearance(self):
        # A transformed ramp must produce the same terrain-relative height.
        local = torch.tensor([[[.4, -.2, .3], [.7, .2, .3]]], dtype=torch.float64)
        initial, query = self.queries.point_clearances(local, [1], course_origins_world=[0., 0., 0.])
        yaw = .71
        rotation = torch.tensor([[np.cos(yaw), -np.sin(yaw), 0.], [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]], dtype=torch.float64)
        origin = torch.tensor([[8., -4., 2.]], dtype=torch.float64)
        moved = local @ rotation.T + origin[:, None]
        transformed, moved_query = self.queries.point_clearances(moved, [1], course_origins_world=origin, course_yaw_rad=[yaw])
        torch.testing.assert_close(transformed, initial, atol=1e-12, rtol=0)
        torch.testing.assert_close(moved_query.normal, query.normal @ rotation.T, atol=1e-12, rtol=0)
        below = local.clone()
        below[..., 2] = -.02
        clearance, _ = self.queries.point_clearances(below, [1], course_origins_world=[0., 0., 0.])
        self.assertTrue((clearance < 0).all())

    def test_required_unknown_or_stale_support_invalidates_base_reference(self):
        samples = torch.tensor([[[-1., -.2, .1], [-1., .2, .1]], [[0., 0., .1], [-1., 0., .1]]])
        root = torch.tensor([[-1., 0., .13], [0., 0., .13]])
        required = torch.ones((2, 2), dtype=torch.bool)
        result = self.queries.relative_base_height(root, samples, [1, 4], required_samples=required,
            course_origins_world=[0., 0., 0.])
        self.assertEqual(result.valid.tolist(), [True, False])
        self.assertAlmostEqual(float(result.height_m[0]), .13, places=6)
        self.assertTrue(torch.isnan(result.height_m[1]))
        observed = required.clone(); observed[0, 1] = False
        stale = self.queries.relative_base_height(root, samples, [1, 4], required_samples=required,
            observation_usable=observed, course_origins_world=[0., 0., 0.])
        self.assertFalse(stale.valid.any())
        empty = self.queries.relative_base_height(root, samples, [1, 4], required_samples=torch.zeros_like(required),
            course_origins_world=[0., 0., 0.])
        self.assertFalse(empty.valid.any())

    def test_invalid_fixture_and_nonfinite_world_pose_are_rejected_or_masked(self):
        with self.assertRaises(ValueError):
            self.queries.query_local([[0., 0.]], 99)
        with self.assertRaises(ValueError):
            self.queries.query_local([[0., 0.]], .5)
        with self.assertRaises(ValueError):
            self.queries.query_world([[0., 0., .1]], 1, course_origins_world=[0., 0., np.nan])
        query = self.queries.query_world([[0., 0., np.nan]], 1, course_origins_world=[0., 0., 0.])
        self.assertFalse(query.geometry_hit.any())


if __name__ == "__main__":
    unittest.main()
