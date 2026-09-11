import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "isaaclab")]
from experiments.terrain.tools.terrain_fixture_checks import digest, load_catalog
from hexapod_terrain.prepare_curriculum import (LEVELS, derive_mild_fixture, prepare,
    select_reset_records, surface_geometry_metrics, validate_reset_geometry)


class TerrainCurriculumTests(unittest.TestCase):
    def test_actual_local_slope_is_capped_without_changing_xy_or_triangles(self):
        records = load_catalog(ROOT / "artifacts/terrain_readiness_2026-09-09/terrain_catalog.json",
                               ["train_ramp_4409", "train_smooth_rough_1103", "train_step_1103", "train_pit_1103"])
        for entry, _, vertices, faces in records:
            if entry["family"] == "pit":
                with self.assertRaisesRegex(ValueError, "Pits"):
                    derive_mild_fixture(entry, vertices, faces, LEVELS[2])
                continue
            metadata, modified, triangles = derive_mild_fixture(entry, vertices, faces, LEVELS[2])
            np.testing.assert_array_equal(modified[:, :2], vertices[:, :2])
            np.testing.assert_array_equal(triangles, faces)
            np.testing.assert_allclose(modified[:, 2], vertices[:, 2] * metadata["applied_vertical_scale"])
            if entry["family"] in ("ramp", "smooth_rough"):
                self.assertGreater(surface_geometry_metrics(vertices, faces)["max_nonvertical_surface_slope_deg"], 5.)
                self.assertLessEqual(metadata["geometry_metrics"]["max_nonvertical_surface_slope_deg"], 5.)
            self.assertFalse(metadata["physical_validation"])

    def test_curriculum_is_reproducible_and_split_selection_cannot_leak(self):
        original = ROOT / "artifacts/terrain_readiness_2026-09-09/terrain_catalog.json"
        selected = load_catalog(original, ["train_ramp_1103", "train_smooth_rough_2207",
                                          "heldout_ramp_7103", "heldout_smooth_rough_8209"])
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            catalog = base / "terrain_catalog.json"
            entries = []
            for entry, path, _, _ in selected:
                target = base / entry["usda"]
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(path, target)
                shutil.copy2(path.with_suffix(".npz"), target.with_suffix(".npz"))
                entries.append(entry)
            catalog.write_text(json.dumps(dict(fixtures=entries)))
            kwargs = dict(variant="f050_t060", seed=42, resets_per_fixture=2)
            package = ROOT / "robot/hexapod_mkii_length_study"
            plan = ROOT / "artifacts/omni_flat_2026-09-09/inputs/training_plan.json"
            first = prepare(catalog, package, plan, base / "first", **kwargs)
            second = prepare(catalog, package, plan, base / "second", **kwargs)
            self.assertEqual(first, second)
            self.assertEqual(digest(base / "first/curriculum.json"), digest(base / "second/curriculum.json"))
            self.assertFalse(first["ready_for_training"])
            self.assertEqual(len(first["nominal_joint_positions_rad"]), 18)
            for level in range(3):
                train = select_reset_records(first, split="train", level=level)
                heldout = select_reset_records(first, split="heldout", level=level)
                self.assertFalse({r["fixture_id"] for r in train} & {r["fixture_id"] for r in heldout})
            corrupted = copy.deepcopy(first)
            corrupted["fixture_ids_by_split"]["train"].append(corrupted["fixture_ids_by_split"]["heldout"][0])
            with self.assertRaisesRegex(ValueError, "overlap"):
                select_reset_records(corrupted, split="train", level=0)
            fixture = load_catalog(base / "first/terrain_catalog.json", [first["resets"][0]["fixture_id"]])[0]
            reset = first["resets"][0]
            footprints = np.asarray(first["footprint"]["rectangles_body_xy_m"])
            for name, value in (("root_position_course_m", [1.49, 0., first["reset_height_m"]]),
                                ("root_position_course_m", [-1., 0., 0.]),
                                ("root_quaternion_course_wxyz", [1., 0., 0., 0.]),
                                ("split", "heldout")):
                invalid = dict(reset, **{name: value})
                with self.assertRaises(ValueError):
                    validate_reset_geometry(invalid, footprints, fixture, expected_root_height_m=first["reset_height_m"])


if __name__ == "__main__":
    unittest.main()
