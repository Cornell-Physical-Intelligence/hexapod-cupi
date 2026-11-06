"""The destination results publish only what their reports, captures and native state are bound to."""

import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "site/assets/destination_curriculum_20261026_001/build_results.py"
SOURCE, CHECKPOINT, V9 = "s" * 64, "c" * 64, "9" * 64
CASES = [f"static:c{k}" for k in range(96)]

spec = importlib.util.spec_from_file_location("destination_build_results", BUILD)
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def write_trace(path, gap=0.0):
    root = np.zeros((4, 1, 7))
    root[..., 6] = 1.0
    root[-1, 0, 0] = gap
    np.savez(
        path,
        root_pose_xyzw=root,
        command=np.zeros((4, 1, 3)),
        terminated=np.zeros((4, 1), bool),
    )


def write_batch(folder, k, case, **report):
    batch = Path(folder, "evaluation", f"batch_{k:03d}")
    batch.mkdir(parents=True)
    write_trace(batch / "control_trace.npz")
    digest = hashlib.sha256((batch / "control_trace.npz").read_bytes()).hexdigest()
    report = {
        "checkpoint_sha256": CHECKPOINT,
        "source_sha256": SOURCE,
        "seed": build.SEED,
        "assigned_case_ids": [case],
        "results": [{"pass": True, "failed_bounds": []}],
        "files": {"control_trace.npz": digest},
        **report,
    }
    (batch / "report.json").write_text(json.dumps(report))
    return batch


class ResultsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runs, self.native = self.root / "runs", self.root / "native"
        self.output = self.root / "results.json"
        self.walk = write_batch(self.runs / "suite", 0, "speed:walk_0.05_0deg")
        write_batch(self.runs / "suite", 1, "speed:quiet_probe")
        for k, case in enumerate(CASES):
            write_batch(self.runs / "full96", k, "required:" + case.replace(":", "__", 1))
            write_batch(
                self.native, k, case, checkpoint_sha256=V9, seed=27057 + k, results=[], files={}
            )
        self.native.mkdir(exist_ok=True)
        state = {
            "identity": {"seed": build.SEED},
            "evaluation": {"results": [{"case_id": c, "pass": False} for c in CASES]},
        }
        (self.native / "state.json").write_text(json.dumps(state))
        patches = mock.patch.multiple(
            build,
            SOURCE_SHA256=SOURCE,
            V9_SHA256=V9,
            SPEED_SUITES=[("destination_v29", "suite", CHECKPOINT)],
            REQUIRED=("destination_v29", ["full96"], CHECKPOINT),
        )
        patches.start()
        self.addCleanup(patches.stop)

    def run_build(self):
        build.main(self.runs, self.native, self.output)
        return json.loads(self.output.read_text())

    def edit(self, path, **fields):
        path.write_text(json.dumps({**json.loads(path.read_text()), **fields}))

    def assert_refused(self, message):
        with self.assertRaisesRegex(ValueError, message):
            self.run_build()
        self.assertFalse(self.output.exists())

    def test_both_outputs_keep_their_bindings(self):
        result = self.run_build()
        suite = result["speed_suite"]["destination_v29"]
        required = result["required_96"]["destination_v29_surrogate"]
        native = result["required_96"]["destination_v9_native"]
        for table in (suite, required):
            self.assertEqual(
                (table["checkpoint_sha256"], table["source_sha256"], table["seed"]),
                (CHECKPOINT, SOURCE, build.SEED),
            )
            self.assertEqual(len(table["captures_sha256"]), 64)
            self.assertEqual(len(table["reports_sha256"]), 64)
        self.assertEqual(required["passed"], 96)
        self.assertEqual((native["checkpoint_sha256"], native["seed"]), (V9, build.SEED))
        self.assertEqual(len(native["state_sha256"]), 64)
        self.assertEqual(native["passed"], 0)

    def test_a_report_from_another_checkpoint_source_or_seed_is_refused(self):
        report = self.walk / "report.json"
        original = report.read_text()
        for field, value in (
            ("checkpoint_sha256", "0" * 64),
            ("source_sha256", "0" * 64),
            ("seed", 999),
        ):
            with self.subTest(field=field):
                self.edit(report, **{field: value})
                self.assert_refused(field)
                report.write_text(original)
        self.edit(self.runs / "full96/evaluation/batch_095/report.json", seed=999)
        self.assert_refused("seed")

    def test_a_changed_capture_is_refused_though_its_report_is_unchanged(self):
        write_trace(self.walk / "control_trace.npz", gap=1.0)
        self.assert_refused("differs from its declared hash")

    def test_a_report_without_its_capture_hash_is_refused(self):
        self.edit(self.walk / "report.json", files={})
        self.assert_refused("declares no control_trace.npz")

    def test_an_empty_suite_is_refused(self):
        shutil.rmtree(self.runs / "suite/evaluation")
        self.assert_refused("no evaluation batches")

    def test_the_native_run_is_held_to_its_identity_and_cases(self):
        report = self.native / "evaluation/batch_000/report.json"
        original = report.read_text()
        self.edit(report, checkpoint_sha256="0" * 64)
        self.assert_refused("checkpoint_sha256")
        report.write_text(original)
        self.edit(report, assigned_case_ids=["static:other"])
        self.assert_refused("cover the cases")
        report.write_text(original)
        state = self.native / "state.json"
        self.edit(state, identity={"seed": 999})
        self.assert_refused("seed")


if __name__ == "__main__":
    unittest.main()
