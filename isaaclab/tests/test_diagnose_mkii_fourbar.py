"""Diagnostic trace integrity and motion coverage; no simulation admission."""
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("fourbar_diagnostic_test", ROOT/"isaaclab/diagnose_mkii_fourbar.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FourbarDiagnosticTests(unittest.TestCase):
    def test_body_pose_trace_keeps_named_native_xyzw_and_terrain(self):
        names = [f"body_{i}" for i in reversed(range(31))]
        pos = np.arange(2*31*3, dtype=np.float32).reshape(2, 31, 3)
        quat = np.zeros((2, 31, 4), dtype=np.float32)
        quat[..., 3] = 1.  # Native XYZW identity, preserved without reordering.
        quat[1, 4] = [.1, .2, .3, .4]
        origin = np.array([[0., 0., 0.], [5., 6., 7.]], dtype=np.float32)
        fields = module.body_pose_trace_fields(names, pos, quat, origin)
        columns = [f"{key}/{name}" for key, _, labels in fields for name in labels]
        values = np.concatenate([value for _, value, _ in fields], axis=-1)
        self.assertEqual(values.shape, (2, 220))  # 31*(3+4) + 3.
        self.assertEqual(len(columns), len(set(columns)))
        for axis_index, axis in enumerate("xyzw"):
            self.assertEqual(values[1, columns.index(f"body_link_quat_w/body_26_{axis}")], quat[1, 4, axis_index])
        self.assertEqual(values[1, columns.index("body_link_pos_w/body_30_z")], pos[1, 0, 2])
        self.assertEqual(values[1, columns.index("terrain_origin_w/z")], 7.)
        self.assertIn("XYZW", module.TRACE_COORDINATE_CONVENTIONS["body_link_quat_w"])
        self.assertIn("not centre of mass", module.TRACE_COORDINATE_CONVENTIONS["body_link_pos_w"])

    def test_body_pose_trace_rejects_misaligned_environment_or_body_rows(self):
        pos, quat, origin = np.zeros((2, 31, 3)), np.zeros((2, 31, 4)), np.zeros((2, 3))
        names = [str(i) for i in range(31)]
        for bad_names, bad_pos, bad_quat, bad_origin in (
                (names, pos, quat[:, :-1], origin), (names, pos, quat, origin[:1]),
                (["duplicate"]*31, pos, quat, origin), (names[:-1], pos, quat, origin)):
            with self.assertRaises(ValueError):
                module.body_pose_trace_fields(bad_names, bad_pos, bad_quat, bad_origin)

    def test_report_survives_nonreturning_native_teardown(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"report.json"
            program = """
import importlib.util, os, sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('diagnostic_child',sys.argv[1])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class NativeExit:
    def close(self): os._exit(0)
report={'pass':True,'diagnostic_complete':True,'errors':[],'trace_samples':8000}
m.persist_then_close(report,Path(sys.argv[2]),NativeExit())
raise AssertionError('Native teardown unexpectedly returned')
"""
            result = subprocess.run([sys.executable, "-c", program, str(ROOT/"isaaclab/diagnose_mkii_fourbar.py"), str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(path.read_text())
            self.assertFalse(report["pass"])
            self.assertTrue(report["diagnostic_complete"])
            self.assertEqual(report["trace_samples"], 8000)
            self.assertIn("FOURBAR_DIAGNOSTIC_RESULT", result.stdout)

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
        from tools.mkii_training_contract import require_admission
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
