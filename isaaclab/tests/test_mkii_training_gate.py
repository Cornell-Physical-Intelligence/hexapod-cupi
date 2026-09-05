"""Negative lineage/admission and shared-compute launcher tests; no SDK/GPU."""
import copy
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from mkii_training_contract import TASK_ID, identity, require_admission, require_checkpoint, digest
from qualify_mkii_fourbar import qualify

loader = importlib.machinery.SourceFileLoader("fourbar_host_gate", str(ROOT / "isaaclab/deploy/run-mkii-fourbar"))
spec = importlib.util.spec_from_loader(loader.name, loader)
host = importlib.util.module_from_spec(spec)
loader.exec_module(host)


class TrainingGateTests(unittest.TestCase):
    def test_identity_binds_code_and_asset_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = ["robot/hexapod_mkii_assy/usd/hexapod_mkii_fourbar_v3/hexapod_mkii_fourbar_v3.usda",
                     "robot/hexapod_mkii_assy/urdf/hexapod_mkii_linkage.urdf",
                     "artifacts/mkii_step2_2026-09-04/physical_fourbar_reference/cad_pin_frames.json",
                     "packages/test/controller.py"]
            for relative in paths:
                p = root / relative
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("initial")
            first = identity(root)
            (root / paths[-1]).write_text("changed controller")
            self.assertNotEqual(first["sha256"], identity(root)["sha256"])
            (root / paths[-1]).write_text("initial")
            self.assertEqual(first, identity(root))
            (root / paths[0]).write_text("changed asset")
            self.assertNotEqual(first["sha256"], identity(root)["sha256"])

    def test_checkpoint_requires_exact_bytes_and_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "checkpoint.pt"
            p.write_bytes(b"checkpoint fixture; never unpickle")
            contract = {"sha256": "a"*64}
            sidecar = {"contract": contract, "checkpoint_sha256": digest(p), "next_iteration": 3}
            p.with_suffix(".pt.json").write_text(json.dumps(sidecar))
            self.assertEqual(require_checkpoint(p, contract), sidecar)
            with self.assertRaises(ValueError):
                require_checkpoint(p, {"sha256": "b"*64})
            p.write_bytes(b"different bytes")
            with self.assertRaises(ValueError):
                require_checkpoint(p, contract)

    def test_probe_and_historical_reports_cannot_admit_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "report.json"
            for report in ({"pass": True}, {"pass": True, "standing_gate_pass": True},
                           {"pass": True, "simulation_training_admission": True, "contract": {}}):
                p.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    require_admission(p, {})

    def test_solver_comparison_rejects_incomplete_or_divergent_runs(self):
        contract = {"sha256": "c"*64, "task_id": TASK_ID}
        nominal = {"pass": True, "errors": [], "contract": contract, "solver_multiplier": 1,
                   "task_id": TASK_ID, "solver_iterations": [32, 4], "steps_requested": 1000,
                   "num_envs": 32, "steps_completed": 1000, "driven_steps": 2400,
                   "driven_coordinate_pass": True, "windows": {window: {"mean_height_m": .13,
                   "max_applied_nm": 1.} for window in ("settled", "driven")}}
        refined = copy.deepcopy(nominal)
        refined["solver_multiplier"] = 2
        refined["solver_iterations"] = [64, 8]
        self.assertTrue(qualify(nominal, refined, contract)["pass"])
        refined["windows"]["driven"]["max_applied_nm"] = 2.
        self.assertFalse(qualify(nominal, refined, contract)["pass"])
        refined = copy.deepcopy(nominal)
        refined["solver_multiplier"] = 2
        refined["solver_iterations"] = [64, 8]
        refined["driven_coordinate_pass"] = False
        self.assertFalse(qualify(nominal, refined, contract)["pass"])

    def test_coordination_requires_one_explicit_none_state(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "coordination.md"
            with patch.object(host, "COORDINATION", p):
                p.write_text("Instructions mention SHARING REQUESTED\nHEXAPOD_SHARE_STATUS=NONE\n")
                self.assertEqual(host.coordination_snapshot(), digest(p))
                for text in ("", "HEXAPOD_SHARE_STATUS=REQUESTED\n",
                             "HEXAPOD_SHARE_STATUS=NONE\nHEXAPOD_SHARE_STATUS=REQUESTED\n",
                             "HEXAPOD_SHARE_STATUS=NONE\nHEXAPOD_SHARE_STATUS=NONE\n"):
                    p.write_text(text)
                    with self.assertRaises(host.Blocked):
                        host.coordination_snapshot()


if __name__ == "__main__":
    unittest.main()
