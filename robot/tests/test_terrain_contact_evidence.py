from pathlib import Path
import importlib.util
import json
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "tools"
sys.path[:0] = [str(HERE), str(ROOT / "isaaclab")]
from terrain_contact_evidence import audit_contact_log


class ContactEvidenceTests(unittest.TestCase):
    def test_recorded_004_contact_warning_is_rejected(self):
        # Verbatim SDK diagnostic from attempt004; full log tested separately.
        warning = ("2026-09-10T01:36:47Z [21,246ms] [Warning] [omni.physx.tensors.plugin] "
                   "Incomplete contact data is reported in GpuRigidContactView::getContactData "
                   "because there are more contact data points than specified maxContactDataCount = 8.\n")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "log"
            path.write_text(warning)
            result = audit_contact_log(path)
            self.assertFalse(result["passed"])
            self.assertEqual(result["incomplete_data_warning_count"], 1)
            self.assertEqual(result["reported_capacities"], [8])

    def test_friction_truncation_rejects_but_capacity_configuration_does_not(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "log"
            path.write_text("maxContactDataCount = 128\nInitialization complete\n")
            self.assertTrue(audit_contact_log(path)["passed"])
            path.write_text("Incomplete friction data is reported; maxContactDataCount = 128\n")
            self.assertFalse(audit_contact_log(path)["passed"])

    def test_missing_log_does_not_admit(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                audit_contact_log(Path(directory) / "absent")

    def test_host_rejects_overflow_even_with_success_exit_and_passed_raw_gate(self):
        spec = importlib.util.spec_from_file_location("capacity_host_launcher", HERE / "launch_terrain_robot_smoke_spark.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        with TemporaryDirectory() as directory:
            output = Path(directory)
            for folder in ("jobs", "logs", "terrain"):
                (output / folder).mkdir()
            (output / "terrain/state.json").write_text(json.dumps(dict(status="completed",
                gate={"passed": True}, runtime_binding={"runtime_tree_sha256": launcher.RUNTIME_TREE})))
            process = Mock(returncode=0)
            process.poll.return_value = 0
            def launch(*args, **kwargs):
                kwargs["stdout"].write("Incomplete contact data is reported; maxContactDataCount = 128\n")
                return process
            args = SimpleNamespace(source=Path("/frozen"), output=output,
                isaaclab=Path("/isaaclab"), coordination_sha256="unchanged")
            with patch.object(launcher.os, "open", return_value=77), patch.object(launcher.os, "close"), \
                 patch.object(launcher.fcntl, "flock"), patch.object(launcher, "preflight", return_value={}), \
                 patch.object(launcher, "verified_source"), patch.object(launcher.subprocess, "Popen", side_effect=launch), \
                 patch.object(launcher, "digest", return_value="unchanged"), \
                 patch.object(launcher, "owned_container", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "incomplete contact/friction data"):
                    launcher.run_owned(args, "terrain")
            audit = json.loads((output / "jobs/terrain_contact_data_audit.json").read_text())
            self.assertFalse(audit["passed"])
            job = json.loads((output / "jobs/terrain.json").read_text())
            self.assertEqual(job["status"], "failed")
            self.assertTrue(job["cleanup_checked"])

    def test_adapter_preserves_larger_capacity_and_untracked_sensors(self):
        spec = importlib.util.spec_from_file_location("capacity_fixture_adapter", ROOT / "isaaclab/hexapod_terrain/fixture_adapter.py")
        adapter = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = adapter
        spec.loader.exec_module(adapter)
        def sensor(points=False, friction=False, capacity=None):
            return SimpleNamespace(track_contact_points=points, track_friction_forces=friction,
                max_contact_data_count_per_prim=capacity, filter_prim_paths_expr=["old"])
        cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=32),
            robot=SimpleNamespace(init_state=SimpleNamespace(pos=(0., 0., .13))),
            base_contact_sensor=sensor(), coxa_contact_sensor=sensor(),
            feet_contact_sensors=(sensor(True, True, 8), sensor(True, True, 256),
                sensor(True, True, None), sensor(False, True, 4), sensor(True, False, 8), sensor(True, True, 8)),
            femur_contact_sensors=tuple(sensor() for _ in range(6)))
        terrains = ModuleType("isaaclab.terrains")
        terrains.TerrainImporter = type("TerrainImporter", (), {})
        terrains.TerrainImporterCfg = SimpleNamespace
        identity = dict(variant="f050_t060", urdf_sha256="a" * 64, plan_sha256="b" * 64, stance_index=0)
        with patch.dict(sys.modules, {"isaaclab.terrains": terrains}), \
             patch.object(adapter.MildTerrainSpec, "load", return_value=({"start_xy_m": [-1., 0.]}, Path("fixture.usda"))):
            actual = adapter.adapt_flat_cfg_for_fixture_smoke(cfg,
                adapter.MildTerrainSpec(Path("catalog"), "ramp"),
                admission=dict(identity, gate={"passed": True}), asset_identity=identity)
        self.assertEqual([s.max_contact_data_count_per_prim for s in actual.feet_contact_sensors],
                         [128, 256, 128, 128, 128, 128])
        self.assertEqual(cfg.feet_contact_sensors[0].max_contact_data_count_per_prim, 8)
        self.assertIsNone(actual.base_contact_sensor.max_contact_data_count_per_prim)
        self.assertEqual(actual.base_contact_sensor.filter_prim_paths_expr, ["old"])
        self.assertTrue(all(s.filter_prim_paths_expr == ["/World/ground/terrain"] for s in actual.feet_contact_sensors))


if __name__ == "__main__":
    unittest.main()
