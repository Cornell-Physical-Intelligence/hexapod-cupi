import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("terrain_robot_launcher", ROOT / "tools/launch_terrain_robot_smoke_spark.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def test_only_flat_asset_copy_is_writable_and_both_commands_are_standing(self):
        source, output = Path("/frozen"), Path("/run")
        flat = launcher.command(source, output, "owned", "flat")
        terrain = launcher.command(source, output, "owned", "terrain")
        self.assertIn("/frozen:/workspace/hexapod:ro", flat)
        self.assertIn("/run/inputs/study:/study:rw", flat)
        self.assertIn("/run/inputs/study:/study:ro", terrain)
        self.assertEqual(flat[flat.index("--mode") + 1], "validate")
        self.assertIn("/outputs/flat/admission.json", terrain)
        self.assertIn("/workspace/hexapod/" + launcher.FIXTURE_ADMISSION, terrain)
        self.assertNotIn("--checkpoint", flat + terrain)
        with self.assertRaises(ValueError):
            launcher.command(source, output, "owned", "train")

    def test_container_lookup_rejects_another_name_or_id(self):
        for value in ("x /someone-else true", "unexpected /owned true"):
            with patch.object(launcher.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=value)):
                with self.assertRaises(RuntimeError):
                    launcher.owned_container("owned", "expected")

    def test_cleanup_recovers_container_after_docker_client_already_exited(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for d in ("jobs", "logs", "flat"):
                (output / d).mkdir()
            (output / "flat/state.json").write_text(json.dumps(dict(status="completed",
                runtime_binding=dict(runtime_tree_sha256=launcher.RUNTIME_TREE))))
            args = SimpleNamespace(source=Path("/frozen"), output=output, isaaclab=Path("/isaaclab"), coordination_sha256="unchanged")
            process = Mock(returncode=0)
            process.poll.return_value = 0
            with patch.object(launcher.os, "open", return_value=77), patch.object(launcher.os, "close"), \
                 patch.object(launcher.fcntl, "flock"), patch.object(launcher, "preflight", return_value={}), \
                 patch.object(launcher, "verified_source"), patch.object(launcher.subprocess, "Popen", return_value=process), \
                 patch.object(launcher, "digest", return_value="unchanged"), \
                 patch.object(launcher, "owned_container", side_effect=[("immutable-owned-id", True), ("immutable-owned-id", False)]), \
                 patch.object(launcher.subprocess, "run") as action:
                result = launcher.run_owned(args, "flat")
                action.assert_called_once_with(["docker", "stop", "--time", "20", "immutable-owned-id"],
                                               timeout=30, check=True, capture_output=True)
            self.assertEqual(result["status"], "completed")

    def test_stop_between_phases_prevents_new_container(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "jobs").mkdir()
            (output / "stop.request").touch()
            args = SimpleNamespace(source=Path("/frozen"), output=output, isaaclab=Path("/isaaclab"))
            with patch.object(launcher.subprocess, "Popen") as launch:
                with self.assertRaises(InterruptedError):
                    launcher.run_owned(args, "terrain")
                launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
