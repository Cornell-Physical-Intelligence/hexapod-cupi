"""Keep checkpoint provenance intact across a certified package migration."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import locomotion.evaluation.checkpoint as checkpoints

ROOT = Path(__file__).resolve().parents[3]
HISTORICAL_CAMPAIGNS = (
    "package_cutover_20261015_001",
    "floor_contact_20261016_001",
    "floor_contact_20261016_002",
    "pr84_review_20261017_001",
    "c48_receipt_renewal_20261102_001",
)


class CheckpointCompatibilityTests(unittest.TestCase):
    def write_bundle(self, directory, receipt, ranges, summary):
        for name, value in (("ranges.json", ranges), ("summary.json", summary)):
            (directory / name).write_text(json.dumps(value))
        receipt["files"] = {
            name: checkpoints.sha(directory / name) for name in ("ranges.json", "summary.json")
        }
        path = directory / "receipt.json"
        path.write_text(json.dumps(receipt))
        return path

    def bundle(self, directory):
        from locomotion.tests.test_native_ranges import NativeRangeTests

        summary = NativeRangeTests().suite(1.0)
        summary["allocation"].update(
            checkpoint_sha256=checkpoints.CHECKPOINT_SHA, judge_sha256="a" * 64
        )
        summary["loads"] = {str(index): {"status": "available"} for index in range(3)}
        ranges = checkpoints.freeze([deepcopy(summary) for _ in range(3)])
        ranges["baseline_provenance"] = [{"fixture": index} for index in range(3)]
        for name, value in (("ranges.json", ranges), ("summary.json", summary)):
            (directory / name).write_text(json.dumps(value))
        receipt = {
            "schema": checkpoints.SCHEMA,
            "mode": "evaluate",
            "eval_scope": "requirements",
            "checkpoint_sha256": checkpoints.CHECKPOINT_SHA,
            "declaration_sha256": checkpoints.DECLARATION_SHA,
            "candidate_runtime_sources": {
                "files": {"locomotion/evaluation/requirements.py": "a" * 64}
            },
            "files": {
                name: checkpoints.sha(directory / name) for name in ("ranges.json", "summary.json")
            },
        }
        path = directory / "receipt.json"
        path.write_text(json.dumps(receipt))
        return path

    def test_all_preserved_v1_receipts_validate_their_original_probe_ranges(self):
        for campaign in HISTORICAL_CAMPAIGNS:
            path = ROOT / "site/assets" / campaign / "compatibility/receipt.json"
            with self.subTest(campaign=campaign):
                receipt = checkpoints.read_compatibility_receipt(path, checkpoints.sha(path))
                self.assertEqual(receipt["schema"], checkpoints.LEGACY_SCHEMA)
                self.assertEqual(receipt["eval_scope"], "probes")
                ranges = json.loads(path.with_name("ranges.json").read_text())
                summary = json.loads(path.with_name("summary.json").read_text())
                self.assertEqual(len(ranges["ranges"]), 8088)
                self.assertEqual(len(ranges["gates"]), 210)
                self.assertTrue(checkpoints.compare(ranges, summary)["pass"])

    def test_current_measurements_reject_off_bank_route_trace(self):
        from locomotion.tests.test_native_ranges import NativeRangeTests

        summary = NativeRangeTests().suite(1.0)
        summary["cases"][0]["checks"]["command_set"] = {
            "value": 0,
            "bound": 1,
            "pass": False,
        }
        with self.assertRaisesRegex(ValueError, "complete recorded trial"):
            checkpoints.measurements(summary)

    def test_v1_receipts_still_reject_changed_probes_gates_metrics_and_provenance(self):
        source = ROOT / "site/assets" / HISTORICAL_CAMPAIGNS[-1] / "compatibility"
        original = [
            json.loads((source / name).read_text())
            for name in ("receipt.json", "ranges.json", "summary.json")
        ]
        mutations = {
            "diagnostic failure": lambda r, b, s: s.update(diagnostic_screens_passed=False),
            "missing probe": lambda r, b, s: s["missing_probe_cases"].append("missing"),
            "failed probe": lambda r, b, s: s["failed_probe_cases"].append("failed"),
            "short results": lambda r, b, s: s["results"].pop(),
            "duplicate probe": lambda r, b, s: s["results"][1].update(
                case_id=s["results"][0]["case_id"]
            ),
            "failed result": lambda r, b, s: s["results"][0].update({"pass": False}),
            "changed gate": lambda r, b, s: s["results"][10]["checks"][
                "max_applied_torque_nm"
            ].update(bound=2.0),
            "changed metric": lambda r, b, s: s["results"][0]["metrics"].update(
                base_height_std_m=1.0
            ),
            "missing loads": lambda r, b, s: s["loads"].pop("batch_000"),
            "unavailable loads": lambda r, b, s: s["loads"]["batch_000"].update(
                status="unavailable"
            ),
            "missing baseline": lambda r, b, s: b["baseline_provenance"].pop(),
            "changed scorer": lambda r, b, s: s["allocation"].update(evaluator_sha256="changed"),
            "requirements receipt": lambda r, b, s: r.update(
                schema=checkpoints.SCHEMA, eval_scope="requirements"
            ),
            "requirements ranges": lambda r, b, s: b.update(schema=checkpoints.RANGES_SCHEMA),
            "requirements summary": lambda r, b, s: s.update(
                schema=checkpoints.REQUIREMENTS_SCHEMA
            ),
            "missing scorer": lambda r, b, s: s["allocation"].pop("evaluator_sha256"),
        }
        with tempfile.TemporaryDirectory() as temporary:
            for name, mutate in mutations.items():
                with self.subTest(mutation=name):
                    receipt, ranges, summary = deepcopy(original)
                    mutate(receipt, ranges, summary)
                    path = self.write_bundle(Path(temporary), receipt, ranges, summary)
                    with self.assertRaises(ValueError):
                        checkpoints.read_compatibility_receipt(path, checkpoints.sha(path))

    def test_receipt_scope_ranges_and_summary_must_use_the_same_format(self):
        mutations = {
            "unknown receipt": lambda r, b, s: r.update(schema="unknown"),
            "legacy receipt with requirements": lambda r, b, s: r.update(
                schema=checkpoints.LEGACY_SCHEMA, eval_scope="probes"
            ),
            "legacy scope": lambda r, b, s: r.update(eval_scope="probes"),
            "missing scope": lambda r, b, s: r.pop("eval_scope"),
            "legacy ranges": lambda r, b, s: b.update(schema=checkpoints.LEGACY_RANGES_SCHEMA),
            "legacy summary": lambda r, b, s: s.update(schema=checkpoints.PROBE_SCHEMA),
            "missing allocation": lambda r, b, s: s.pop("allocation"),
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name, mutate in mutations.items():
                with self.subTest(mutation=name):
                    path = self.bundle(directory)
                    receipt, ranges, summary = [
                        json.loads((directory / filename).read_text())
                        for filename in ("receipt.json", "ranges.json", "summary.json")
                    ]
                    mutate(receipt, ranges, summary)
                    path = self.write_bundle(directory, receipt, ranges, summary)
                    with self.assertRaises(ValueError):
                        checkpoints.read_compatibility_receipt(path, checkpoints.sha(path))

    def test_receipt_pins_evidence_and_rejects_failed_or_incomplete_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self.bundle(directory)
            digest = checkpoints.sha(path)
            checkpoints.read_compatibility_receipt(path, digest)
            with self.assertRaisesRegex(ValueError, "receipt hash"):
                checkpoints.read_compatibility_receipt(path, "0" * 64)
            summary_path = directory / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["cases"][0]["stop"]["final_distance_m"] = 2.0
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "evidence hash"):
                checkpoints.read_compatibility_receipt(path, digest)
            receipt = json.loads(path.read_text())
            receipt["files"]["summary.json"] = checkpoints.sha(summary_path)
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "native parity"):
                checkpoints.read_compatibility_receipt(path, checkpoints.sha(path))
            path = self.bundle(directory)
            summary = json.loads(summary_path.read_text())
            summary["loads"].pop("0")
            summary_path.write_text(json.dumps(summary))
            receipt = json.loads(path.read_text())
            receipt["files"]["summary.json"] = checkpoints.sha(summary_path)
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "load evidence"):
                checkpoints.read_compatibility_receipt(path, checkpoints.sha(path))
            path = self.bundle(directory)
            receipt = json.loads(path.read_text())
            receipt["certified_physics_config"] = "isaaclab_mesh"
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "certified physics"):
                checkpoints.read_compatibility_receipt(path, checkpoints.sha(path))

    def test_candidate_dependency_changes_and_original_identity_changes_reject(self):
        from contracts.release import COMPATIBILITY_KEYS

        recorded = dict.fromkeys(COMPATIBILITY_KEYS, "original")
        candidate = deepcopy(recorded)
        candidate.update(
            schema="hexapod_locomotion_ppo_v2",
            adapter_sha256="new",
            entry_sha256="new",
            runtime_sources={"files": {"locomotion/robots/rs05.py": "a" * 64}},
        )
        record = {"identity": recorded, "checkpoint_sha256": checkpoints.CHECKPOINT_SHA}
        receipt = {
            "original_identity": deepcopy(recorded),
            "candidate_runtime_sources": deepcopy(candidate["runtime_sources"]),
        }
        checkpoints.require_compatibility_identity(receipt, record, candidate)
        for key in ("model_sha256", "physics_config", "seed"):
            changed = deepcopy(candidate)
            changed[key] = "different"
            with self.subTest(key=key), self.assertRaises(ValueError):
                checkpoints.require_compatibility_identity(receipt, record, changed)
        candidate["runtime_sources"]["files"]["locomotion/robots/rs05.py"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "candidate runtime"):
            checkpoints.require_compatibility_identity(receipt, record, candidate)
        record["identity"]["entry_sha256"] = "altered declaration"
        with self.assertRaisesRegex(ValueError, "original identity"):
            checkpoints.require_compatibility_identity(receipt, record, candidate)

    def test_certified_physics_replaces_the_recorded_physics_and_nothing_else(self):
        from contracts.release import COMPATIBILITY_KEYS

        recorded = dict.fromkeys(COMPATIBILITY_KEYS, "original")
        recorded["physics_config"] = {"floor": "mesh", "physics_dt": 0.0025}
        candidate = deepcopy(recorded)
        candidate.update(
            runtime_sources={"files": {}},
            physics_config={"floor": "isaaclab_mesh", "physics_dt": 0.0025},
        )
        record = {"identity": recorded, "checkpoint_sha256": checkpoints.CHECKPOINT_SHA}
        receipt = {
            "original_identity": deepcopy(recorded),
            "candidate_runtime_sources": {"files": {}},
        }
        with self.assertRaisesRegex(ValueError, "certified physics"):
            checkpoints.require_compatibility_identity(receipt, record, candidate)
        receipt["certified_physics_config"] = deepcopy(candidate["physics_config"])
        checkpoints.require_compatibility_identity(receipt, record, candidate)
        for physics in (
            {"floor": "isaaclab_mesh", "physics_dt": 0.005},
            recorded["physics_config"],
        ):
            with (
                self.subTest(physics=physics),
                self.assertRaisesRegex(ValueError, "certified physics"),
            ):
                checkpoints.require_compatibility_identity(
                    receipt, record, {**candidate, "physics_config": physics}
                )
        with self.assertRaisesRegex(ValueError, "model, physics, input or seed"):
            checkpoints.require_compatibility_identity(
                receipt, record, {**candidate, "seed": "other"}
            )

    def test_prepare_binds_receipt_and_refuses_training_or_uncertified_scope(self):
        from locomotion.spark.prepare import prepare

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = "/srv/cupi/hexapod/runs/james/checkpoint.pt"
            options = dict(
                mode="evaluate",
                checkpoint=checkpoint,
                checkpoint_sha=checkpoints.CHECKPOINT_SHA,
                checkpoint_declaration_sha=checkpoints.DECLARATION_SHA,
                checkpoint_compatibility="/srv/cupi/hexapod/runs/james/certified/receipt.json",
                checkpoint_compatibility_sha256="a" * 64,
            )
            binding = prepare(root / "pack", "/srv/cupi/hexapod/runs/james/fresh", **options)
            self.assertIn(
                ["/srv/cupi/hexapod/runs/james/certified", "/compatibility"],
                binding["extra_mounts"],
            )
            self.assertIn("/compatibility/receipt.json", binding["command_args"])
            for changed in (
                {"mode": "train"},
                {"checkpoint_compatibility_sha256": None},
            ):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    prepare(
                        root / "rejected",
                        "/srv/cupi/hexapod/runs/james/fresh",
                        **(options | changed),
                    )

    def test_translate_only_known_class_fields_without_mutation(self):
        original = {
            "class_name": "locomotion.rate_schedule:CappedRatePPO",
            "label": "locomotion.rate_schedule:CappedRatePPO",
            "nested": [{"class_name": "locomotion.action_distribution:BoundedMeanGaussian"}],
        }
        snapshot = deepcopy(original)
        translated = checkpoints.configuration(original)
        self.assertEqual(original, snapshot)
        self.assertEqual(translated["class_name"], "locomotion.rl.ppo:CappedRatePPO")
        self.assertEqual(translated["label"], original["label"])
        self.assertEqual(
            translated["nested"][0]["class_name"], "locomotion.rl.ppo:BoundedMeanGaussian"
        )

    def test_translation_keeps_configuration_changes_visible(self):
        old = {"class_name": "locomotion.rate_schedule:CappedRatePPO", "gamma": 0.99}
        candidate = {"class_name": "locomotion.rl.ppo:CappedRatePPO", "gamma": 0.98}
        self.assertNotEqual(checkpoints.configuration(old), candidate)

    def test_unknown_class_name_stays_unknown(self):
        self.assertEqual(
            checkpoints.configuration({"class_name": "unreviewed:PPO"}),
            {"class_name": "unreviewed:PPO"},
        )


if __name__ == "__main__":
    unittest.main()
