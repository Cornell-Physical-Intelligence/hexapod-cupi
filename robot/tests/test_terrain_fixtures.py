import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "isaaclab")]
from experiments.terrain.tools.terrain_readiness import terrain_mesh, mesh_usda
from experiments.terrain.tools.terrain_fixture_checks import audit_usd, digest, load_catalog, probe_locations, vertical_surface_heights
from hexapod_terrain.fixture_adapter import MildTerrainSpec, require_matching_admission
from hexapod_terrain.robot_smoke import check_start_footprint


class TerrainFixtureTests(unittest.TestCase):
    def test_triangle_query_sees_negative_pit_not_convex_hull(self):
        vertices, faces, metadata = terrain_mesh("pit", 1103)
        heights = vertical_surface_heights(vertices, faces, [[-1., 0.], [0., 0.], [1.6, 0.]])
        self.assertAlmostEqual(heights[0], 0.)
        self.assertAlmostEqual(heights[1], -metadata["pit_depth_m"])
        self.assertTrue(np.isnan(heights[2]))

    def test_step_query_preserves_discontinuous_edge(self):
        vertices, faces, metadata = terrain_mesh("step", 2207)
        edge = metadata["edge_x_m"]
        heights = vertical_surface_heights(vertices, faces, [[edge - 1e-5, .1], [edge + 1e-5, .1]])
        np.testing.assert_allclose(heights, [0., metadata["step_height_m"]], atol=1e-12)

    def test_queries_and_contact_probes_cover_all_fixture_families(self):
        for family in ("smooth_rough", "ramp", "step", "ridge", "pit"):
            vertices, faces, metadata = terrain_mesh(family, 3301)
            points = probe_locations(metadata)
            heights = vertical_surface_heights(vertices, faces, points)
            self.assertTrue(np.isfinite(heights).all())
            self.assertEqual(heights[0], 0.)
        with self.assertRaises(ValueError):
            vertical_surface_heights(vertices, faces, [[float("nan"), 0.]])

    def test_admission_is_exact_and_cannot_be_omitted(self):
        identity = dict(variant="f050_t060", urdf_sha256="a" * 64, plan_sha256="b" * 64, stance_index=0)
        admission = dict(identity, gate={"passed": True})
        require_matching_admission(admission, identity)
        for bad in ({}, dict(identity, stance_index=1), dict(identity, variant="another")):
            with self.assertRaises(ValueError):
                require_matching_admission(admission, bad)
        with self.assertRaises(ValueError):
            require_matching_admission(dict(identity, gate={"passed": False}), identity)

    def test_start_frame_rotates_export_forward_to_course_forward(self):
        spec = MildTerrainSpec(ROOT / "unused", "train_ramp_1103")
        yaw = spec.start_body_yaw_rad
        rotated = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]) @ [0., -1.]
        np.testing.assert_allclose(rotated, [1., 0.], atol=1e-12)
        for kwargs in ({"speed_range_mps": (.05, .3)}, {"contact_offset_m": .02}, {"friction": float("nan")}):
            with self.assertRaises(ValueError):
                MildTerrainSpec(ROOT / "unused", "fixture", **kwargs)

    def test_pit_is_rejected_by_walking_adapter(self):
        spec = MildTerrainSpec(ROOT / "artifacts/terrain_readiness_2026-09-09/terrain_catalog.json", "train_pit_1103")
        with self.assertRaisesRegex(ValueError, "avoidance"):
            spec.load()

    def test_exact_selected_c_distal_collision_footprint_fits_start(self):
        import xml.etree.ElementTree as ET
        package = ROOT / "robot/hexapod_mkii_length_study"
        manifest = json.loads((package / "manifest.json").read_text())
        record = next(r for r in manifest["variants"] if r["variant"] == "f050_t060")
        plan = json.loads((ROOT / "artifacts/omni_flat_2026-09-09/inputs/training_plan.json").read_text())
        stance = plan["variants"]["f050_t060"]["stances"][0]
        catalog = ROOT / "artifacts/terrain_readiness_2026-09-09/terrain_catalog.json"
        spec = MildTerrainSpec(catalog, "train_ramp_1103")
        terrain = load_catalog(catalog, [spec.fixture_id])[0]
        rows = check_start_footprint(package, ET.parse(package / record["urdf"]).getroot(),
                                    manifest, record, stance, spec, terrain)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["passed"] for row in rows))
        displaced = dict(terrain[0], start_xy_m=[-.65, 0.])
        with self.assertRaisesRegex(ValueError, "does not fit"):
            check_start_footprint(package, ET.parse(package / record["urdf"]).getroot(),
                                 manifest, record, stance, spec, (displaced, *terrain[1:]))

    def test_catalog_hash_and_selection_are_enforced(self):
        catalog = ROOT / "artifacts/terrain_readiness_2026-09-09/terrain_catalog.json"
        records = load_catalog(catalog, ["train_pit_1103"])
        self.assertEqual(len(records), 1)
        with self.assertRaises(ValueError):
            load_catalog(catalog, ["missing"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            vertices, faces, _ = terrain_mesh("pit", 1103)
            usd = path / "fixture.usda"
            usd.write_text(mesh_usda(vertices, faces))
            np.savez(usd.with_suffix(".npz"), vertices=vertices, faces=faces)
            data = dict(fixtures=[dict(id="test", usda="fixture.usda", sha256=digest(usd),
                                      vertices=len(vertices), triangles=len(faces))])
            (path / "catalog.json").write_text(json.dumps(data))
            load_catalog(path / "catalog.json")
            usd.write_text(usd.read_text() + "\n")
            with self.assertRaises(ValueError):
                load_catalog(path / "catalog.json")

    @unittest.skipUnless(importlib.util.find_spec("pxr"), "usd-core needed for parser adversarial tests")
    def test_usd_parser_rejects_convexification_and_geometry_mismatch(self):
        vertices, faces, _ = terrain_mesh("pit", 1103)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.usda"
            source = mesh_usda(vertices, faces)
            path.write_text(source)
            np.savez(path.with_suffix(".npz"), vertices=vertices, faces=faces)
            audit_usd(path, vertices, faces)
            path.write_text(source.replace('physics:approximation = "none"', 'physics:approximation = "convexHull"'))
            with self.assertRaisesRegex(ValueError, "convexification"):
                audit_usd(path, vertices, faces)
            path.write_text(source)
            changed = vertices.copy()
            changed[0, 2] += .01
            with self.assertRaisesRegex(ValueError, "geometry differs"):
                audit_usd(path, changed, faces)


if __name__ == "__main__":
    unittest.main()
