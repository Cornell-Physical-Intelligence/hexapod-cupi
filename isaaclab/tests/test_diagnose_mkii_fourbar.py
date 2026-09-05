"""Diagnostic trace integrity and motion coverage; no simulation admission."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("fourbar_diagnostic_test", ROOT/"isaaclab/diagnose_mkii_fourbar.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FourbarDiagnosticTests(unittest.TestCase):
    def test_motions_have_both_signs_and_zero_recovery(self):
        for kind, count in (("lf_tibia", 300), ("groups", 700), ("individuals", 1900)):
            rows = module.motions(kind)
            self.assertEqual(sum(row["steps"] for row in rows), count)
            self.assertEqual(rows[-1], {"motors": [], "offset_rad": 0., "steps": 100})
            for positive, negative in zip(rows[:-1:2], rows[1:-1:2]):
                self.assertEqual(positive["motors"], negative["motors"])
                self.assertEqual((positive["offset_rad"], negative["offset_rad"]), (.04, -.04))
        self.assertEqual([row["motors"][0] for row in module.motions("individuals")[:-1:2]], list(module.ACTIVE_JOINT_NAMES))
        with self.assertRaises(ValueError):
            module.motions("unreviewed")

    def test_trace_preserves_samples_columns_and_segment_ranges(self):
        with tempfile.TemporaryDirectory() as root:
            trace = module.Trace(None, None, Path(root))
            trace.columns = ["pre_q/a", "post_q/a"]
            trace.pending = [np.array([[1., 2.]], np.float32), np.array([[3., 4.]], np.float32)]
            trace.samples = 2
            trace.flush({"phase": "standing"})
            first = trace.files[0]
            self.assertEqual(first["shape"], [2, 1, 2])
            self.assertEqual(first["first_physics_sample"], 0)
            self.assertEqual(first["sha256"], module.digest(Path(root)/first["file"]))
            with np.load(Path(root)/first["file"], allow_pickle=False) as values:
                self.assertEqual(values["columns"].tolist(), trace.columns)
                self.assertEqual(values["values"].tolist(), [[[1., 2.]], [[3., 4.]]])
            trace.pending = [np.array([[5., 6.]], np.float32)]
            trace.samples = 3
            trace.flush({"phase": "driven"})
            self.assertEqual(trace.files[1]["first_physics_sample"], 2)
            self.assertEqual(trace.pending, [])

    def test_trace_rejects_nonfinite_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as root:
            trace = module.Trace(None, None, Path(root))
            trace.columns = ["q/a"]
            trace.pending = [np.array([[np.nan]], np.float32)]
            with self.assertRaises(ValueError):
                trace.flush({})
            trace.pending = [np.array([[1.]], np.float32)]
            trace.samples = 1
            target = Path(root)/"trace_000.npz"
            target.write_bytes(b"retained evidence")
            with self.assertRaises(FileExistsError):
                trace.flush({})
            self.assertEqual(target.read_bytes(), b"retained evidence")

    def test_diagnostic_cannot_supply_training_admission(self):
        from mkii_training_contract import require_admission
        contract = {"task_id": module.TASK_ID}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"report.json"
            path.write_text(json.dumps({"schema": "hexapod.fourbar_diagnostic.v1", "pass": False,
                "diagnostic_complete": True, "simulation_training_admission": False,
                "task_id": module.TASK_ID, "contract": contract, "errors": [],
                "num_envs": 32, "steps_completed": 1000}))
            with self.assertRaises(ValueError):
                require_admission(path, contract)


if __name__ == "__main__":
    unittest.main()
