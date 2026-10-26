"""Keep native comparison ranges fixed before candidate scoring."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path


class NativeRangeTests(unittest.TestCase):
    def suite(self, value):
        return {
            "diagnostic_screens_passed": True,
            "missing_probe_cases": [],
            "failed_probe_cases": [],
            "allocation": {"checkpoint_sha256": "weights", "evaluator_sha256": "scorer"},
            "results": [
                {
                    "case_id": str(i),
                    "pass": True,
                    "metrics": {"drift": value},
                    "checks": {
                        "drift": {"value": value, "bound": 9, "operator": "<=", "status": "pass"}
                    },
                }
                for i in range(13)
            ],
        }

    def test_replay_allows_stock_scene_but_rejects_other_physics_changes(self):
        from locomotion.tests.native_parity import require_comparison_physics

        original = {
            "floor": "80m_two_triangle_mesh_y_equals_x_seam",
            "physics_dt": 0.0025,
            "decimation": 8,
            "target_slew_rad": 0.040,
        }
        require_comparison_physics(original, original)
        stock = {**original, "floor": "isaaclab_ground_plane", "filter_collisions": True}
        require_comparison_physics(original, stock)
        for key, value in [
            ("physics_dt", 0.005),
            ("decimation", 4),
            ("target_slew_rad", 0.08),
            ("filter_collisions", False),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                require_comparison_physics(original, {**stock, key: value})
        declared = {
            **stock,
            "floor": "isaaclab_mesh",
            "floor_contact_offset_m": 0.02,
            "floor_rest_offset_m": 0.0,
        }
        require_comparison_physics(original, declared)
        require_comparison_physics(declared, declared)
        with self.assertRaises(ValueError):
            require_comparison_physics(original, {**declared, "floor": "custom_floor"})
        with self.assertRaises(ValueError):
            require_comparison_physics(original, {**declared, "material_friction": [0.5, 0.5]})

    def test_three_baselines_freeze_bounds_and_candidate_cannot_widen_them(self):
        from locomotion.tests.native_ranges import compare, freeze

        baseline = [self.suite(value) for value in (1.0, 2.0, 3.0)]
        limits = freeze(baseline)
        before = deepcopy(limits)
        self.assertTrue(compare(limits, self.suite(2.0))["pass"])
        self.assertFalse(compare(limits, self.suite(3.00001))["pass"])
        self.assertEqual(limits, before)

    def test_failed_probes_changed_gates_and_missing_baselines_reject(self):
        from locomotion.tests.native_ranges import compare, freeze

        baseline = [self.suite(1.0) for _ in range(3)]
        limits = freeze(baseline)
        with self.assertRaises(ValueError):
            freeze(baseline[:2])
        candidate = self.suite(1.0)
        candidate["results"][0]["checks"]["drift"]["bound"] = 10
        self.assertFalse(compare(limits, candidate)["pass"])
        candidate = self.suite(1.0)
        candidate["diagnostic_screens_passed"] = False
        self.assertFalse(compare(limits, candidate)["pass"])

    def test_host_receipts_survive_native_process_exit_and_reject_tampering(self):
        from locomotion.tests.native_parity import BASELINE_COMMIT, CHECKPOINT_SHA, DECLARATION_SHA
        from locomotion.tests.native_ranges import read_run, sha

        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp)
            root = pack / "run/standing"

            def save(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))

            proof = pack / "source/PARITY.json"
            save(
                proof,
                {
                    "schema": "hexapod_refactor_parity_v1",
                    "role": "baseline",
                    "baseline_commit": BASELINE_COMMIT,
                },
            )
            save(pack / "source/FREEZE_SHA256.json", {"PARITY.json": sha(proof)})
            freeze = sha(pack / "source/FREEZE_SHA256.json")
            binding = {
                "source_freeze_sha256": freeze,
                "input_files": {
                    "/retained/checkpoint.pt": CHECKPOINT_SHA,
                    "/retained/checkpoint.json": DECLARATION_SHA,
                },
            }
            save(pack / "binding.json", binding)
            save(
                pack / "PACK.json",
                {"binding_sha256": sha(pack / "binding.json"), "test_only_parity": True},
            )
            save(pack / "run/launch_binding.json", binding)
            save(
                root / "state.json",
                {
                    "status": "completed",
                    "errors": [],
                    "runtime_binding": {"runtime_tree_sha256": freeze},
                },
            )
            job = {
                "status": "completed",
                "cleanup_checked": True,
                "exit_code": 0,
                "container_id": "container",
                "source_freeze_sha256": freeze,
            }
            save(pack / "run/jobs/standing.json", job)
            cleanup = {"cleanup_checked": True, "container_id": "container"}
            save(pack / "run/cleanup.json", cleanup)
            summary = self.suite(1.0)
            summary["allocation"]["checkpoint_sha256"] = CHECKPOINT_SHA
            save(root / "evaluation/summary.json", summary)
            for index in range(13):
                save(
                    root / f"evaluation/batch_{index:03d}/force_metrics.json",
                    {"status": "available"},
                )
            # Kit can terminate Python during app.close(); the host still verifies the completed run.
            self.assertFalse((root / "parity_receipt.json").exists())
            self.assertEqual(len(read_run(root, "baseline")["results"]), 13)
            with self.assertRaises(ValueError):
                read_run(root, "candidate")
            cleanup["cleanup_checked"] = False
            save(pack / "run/cleanup.json", cleanup)
            with self.assertRaises(ValueError):
                read_run(root, "baseline")
            cleanup["cleanup_checked"] = True
            save(pack / "run/cleanup.json", cleanup)
            binding["input_files"]["/retained/checkpoint.pt"] = "changed"
            save(pack / "run/launch_binding.json", binding)
            with self.assertRaises(ValueError):
                read_run(root, "baseline")
            save(pack / "binding.json", binding)
            save(
                pack / "PACK.json",
                {"binding_sha256": sha(pack / "binding.json"), "test_only_parity": True},
            )
            with self.assertRaises(ValueError):
                read_run(root, "baseline")
            binding["input_files"]["/retained/checkpoint.pt"] = CHECKPOINT_SHA
            save(pack / "binding.json", binding)
            save(
                pack / "PACK.json",
                {"binding_sha256": sha(pack / "binding.json"), "test_only_parity": True},
            )
            save(pack / "run/launch_binding.json", binding)
            proof.write_text(proof.read_text() + "\n")
            with self.assertRaises(ValueError):
                read_run(root, "baseline")


if __name__ == "__main__":
    unittest.main()
