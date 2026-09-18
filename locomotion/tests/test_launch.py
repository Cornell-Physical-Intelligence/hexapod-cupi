"""Read-only checks of native ownership reuse; no remote or subprocess actions."""
import ast
from contextlib import ExitStack
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
from locomotion import launch, reservation

ROOT = Path(__file__).resolve().parents[2]
NEW = ROOT / 'locomotion'
GUARD_PARENT = ROOT / 'locomotion/tests/fixtures/launch_guarded_diagnostic.py'


def functions(path):
    return {x.name: x for x in ast.parse(path.read_text()).body if isinstance(x, ast.FunctionDef)}


class OwnedLaunchTests(unittest.TestCase):
    def binding(self):
        root = "/home/orionh/HEXAPOD_runs/restart_20260914/launcher_fixture"
        return {"schema": "hexapod_locomotion_launch_v1", "root_review_complete": True,
                "mode": "diagnostic", "module": "locomotion.train", "max_seconds": 7200,
                **{key: root+"/"+key for key in ("source", "output", "asset", "prior", "geometry_source")},
                "source_freeze_sha256": "a"*64, "input_files": {}, "stage2_complete": False,
                "physical_admission": False,
                "command_args": ["--mode", "diagnostic", "--source-freeze-sha256", "a"*64,
                                 "--asset", "/asset", "--model", "/asset/source/model.json",
                                 "--device", "cuda:0", "--headless"]}

    def verify_without_filesystem(self, binding):
        with ExitStack() as stack:
            # These are Linux host bindings; local macOS /home is a symlink.
            stack.enter_context(patch.object(reservation, "canonical_path", side_effect=Path))
            for name, result in (("verify_tree", None), ("pinned_file", None)):
                stack.enter_context(patch.object(reservation, name, return_value=result))
            return launch.verify(binding, Path(binding["source"]))

    def test_current_reservation_helpers_preserved(self):
        old, new = functions(GUARD_PARENT), functions(NEW / "reservation.py")
        for name in new:
            self.assertIn(name, old)
            self.assertEqual(ast.dump(new[name], include_attributes=False),
                             ast.dump(old[name], include_attributes=False), name)

    def test_output_and_readonly_inputs_match_supervisor(self):
        paths = {k: Path("/home/orionh/HEXAPOD_runs/restart_20260914") / k
                 for k in ("source", "output", "asset", "prior", "geometry_source")}
        binding = {"module": "locomotion.train", "command_args": ["--mode", "diagnostic", "--headless"], "extra_mounts": []}
        args = launch.command(binding, paths, "hexapod-reference-physics-" + "a"*32)
        self.assertEqual(args[-7:], ["-m", "locomotion.train", "--output", "/output/standing", "--mode", "diagnostic", "--headless"])
        mounts = [args[i+1] for i, value in enumerate(args) if value == "-v"]
        self.assertEqual(sum(x.endswith(":rw") for x in mounts), 1)
        self.assertIn(str(paths["output"])+":/output:rw", mounts)
        self.assertTrue(all(x.endswith(":ro") for x in mounts if not x.endswith(":rw")))

    def test_extra_mount_rejects_container_control_socket(self):
        paths = {k: Path("/home/orionh/HEXAPOD_runs/restart_20260914") / k
                 for k in ("source", "output", "asset", "prior", "geometry_source")}
        with self.assertRaises(ValueError):
            launch.command({"command_args": ["--mode", "diagnostic"],
                            "extra_mounts": [["/var/run/docker.sock", "/var/run/docker.sock"]]}, paths, "owned")

    def test_exact_mode_source_and_output_arguments_are_bound(self):
        base = self.binding()
        self.verify_without_filesystem(base)
        for suffix in (["--output", "/unreviewed"], ["--output=/unreviewed"],
                       ["--mode", "train"], ["--mode=train"], ["--source-freeze-sha256", "b"*64]):
            b = copy.deepcopy(base); b["command_args"] += suffix
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                self.verify_without_filesystem(b)

    def test_missing_or_mismatched_native_freeze_rejected_before_launch(self):
        for action in ("remove", "change"):
            b = self.binding(); i = b["command_args"].index("--source-freeze-sha256")
            if action == "remove":
                del b["command_args"][i:i+2]
            else:
                b["command_args"][i+1] = "b"*64
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.verify_without_filesystem(b)

    def test_wall_bound_is_an_integer_at_most_two_hours(self):
        for seconds in (True, 119, 7201, float("inf")):
            b = self.binding(); b["max_seconds"] = seconds
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                self.verify_without_filesystem(b)


if __name__ == "__main__":
    unittest.main()
