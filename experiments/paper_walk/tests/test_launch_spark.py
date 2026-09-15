"""Read-only checks of native ownership reuse; no remote or subprocess actions."""
import ast
import builtins
from contextlib import ExitStack
import copy
import dis
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
NEW = ROOT / "experiments/paper_walk"
PARENT = ROOT / "artifacts/omni_diagnostics_2026-09-09/reference_physics_008/preparation/source_overlays/tools/launch_reference_physics_spark.py"
GUARD_PARENT = ROOT / "artifacts/restart_2026-09-14/standing_translation_guard_001/launch_guarded_diagnostic.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reservation = load("_test_paper_reservation", NEW / "reservation.py")
with patch.dict(sys.modules, {"reservation": reservation}):
    launch = load("_test_paper_launch", NEW / "launch_spark.py")


def functions(path):
    return {x.name: x for x in ast.parse(path.read_text()).body if isinstance(x, ast.FunctionDef)}


class OwnedLaunchTests(unittest.TestCase):
    def binding(self):
        root = "/home/orionh/HEXAPOD_runs/restart_20260914/launcher_fixture"
        return {"schema": "canonical_paper_walk_launch_v1", "root_review_complete": True,
                "mode": "diagnostic", "max_seconds": 7200,
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
            for name, result in (("verify_tree", None), ("pinned_file", None),
                                 ("sha", launch.SUPERVISOR_MAP), ("read", {})):
                stack.enter_context(patch.object(reservation, name, return_value=result))
            return launch.verify(binding, Path(binding["source"]))

    def test_actual_parent_is_hash_bound(self):
        self.assertEqual(reservation.sha(PARENT), launch.SUPERVISOR_SHA)

    def test_deadline_changes_only_three_constants(self):
        parent = types.SimpleNamespace(__builtins__=vars(builtins))
        launch.adapt_deadline(parent, PARENT, 7200)
        original = functions(PARENT)["run_owned"]
        tree = ast.Module(body=[original], type_ignores=[])
        compiled = compile(ast.fix_missing_locations(tree), str(PARENT), "exec")
        code = next(x for x in compiled.co_consts if isinstance(x, types.CodeType))
        old = list(dis.get_instructions(code)); new = list(dis.get_instructions(parent.run_owned))
        self.assertEqual(len(old), len(new))
        changed = []
        for before, after in zip(old, new):
            self.assertEqual(before.opname, after.opname)
            if before.argval != after.argval:
                changed.append((before.opname, before.argval, after.argval))
        self.assertEqual(changed, [
            ("LOAD_CONST", 600, 7200),
            ("LOAD_CONST", 600, 7200),
            ("LOAD_CONST", "Standing phase exceeded ten-minute bound", "Paper-walk phase exceeded 7200-second bound")])
        self.assertIn(90, parent.run_owned.__code__.co_consts)

    def test_current_reservation_helpers_preserved(self):
        old, new = functions(GUARD_PARENT), functions(NEW / "reservation.py")
        for name in new:
            self.assertIn(name, old)
            self.assertEqual(ast.dump(new[name], include_attributes=False),
                             ast.dump(old[name], include_attributes=False), name)

    def test_output_and_readonly_inputs_match_supervisor(self):
        paths = {k: Path("/home/orionh/HEXAPOD_runs/restart_20260914") / k
                 for k in ("source", "output", "asset", "prior", "geometry_source")}
        binding = {"command_args": ["--mode", "diagnostic", "--headless"], "extra_mounts": []}
        args = launch.command(binding, paths, "hexapod-reference-physics-" + "a"*32)
        self.assertEqual(args[-6:], ["/source/train.py", "--output", "/output/standing", "--mode", "diagnostic", "--headless"])
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
