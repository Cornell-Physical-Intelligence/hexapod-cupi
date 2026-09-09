"""CPU-only checks for versioned task selection and Spark source-path bootstrap."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "isaaclab/train_mkii_v2.py"
_spec = importlib.util.spec_from_file_location("_train_mkii_v2_wrapper_test", WRAPPER)
wrapper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wrapper)


class TrainMkiiV2WrapperTests(unittest.TestCase):
    def test_defaults_new_task_without_changing_downstream_flags(self):
        self.assertEqual(
            wrapper.prepare_training_args(["--num_envs", "32", "--seed", "17"]),
            ["--task", wrapper.MKII_V2_TASK_ID, "--num_envs", "32", "--seed", "17"],
        )
        for value in (["--task", wrapper.MKII_V2_TASK_ID], [f"--task={wrapper.MKII_V2_TASK_ID}"]):
            self.assertEqual(wrapper.prepare_training_args(value), value)

    def test_incompatible_duplicate_missing_and_hydra_task_are_rejected(self):
        for arguments in (
            ["--task", "Isaac-Velocity-Flat-Hexapod-MKII-V1-Direct-v0"],
            ["--task=Isaac-Velocity-Flat-Hexapod-RobStride-Direct-v0"],
            ["--task"], ["--task", "--num_envs", "32"],
            ["--task", wrapper.MKII_V2_TASK_ID, f"--task={wrapper.MKII_V2_TASK_ID}"],
            ["task=wrong"], ["++env.task=wrong"],
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                wrapper.prepare_training_args(arguments)

    def test_old_checkpoint_cannot_be_adopted_through_cli_or_hydra(self):
        for arguments in (
            ["--resume"], ["--resume=true"], ["--checkpoint", "old.pt"],
            ["--load_run", "mock"], ["--load-checkpoint=model_2.pt"],
            ["agent.resume=true"], ["+agent.load_checkpoint=old.pt"],
            ["++agent.load_run=mock"],
        ):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "scratch-only"):
                wrapper.prepare_training_args(arguments)

    def test_all_abbreviated_protected_options_are_rejected_before_downstream(self):
        for option in ("task", "resume", "checkpoint", "load_run", "load_checkpoint",
                       "load-run", "load-checkpoint"):
            for length in range(1, len(option)):
                prefix = "--" + option[:length]
                for arguments in ([prefix, "wrong"], [prefix + "=wrong"]):
                    with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "Abbreviated protected"):
                        wrapper.prepare_training_args(arguments)

    def test_actual_argparse_abbreviation_cannot_override_task_or_enable_resume(self):
        # Reproduce the downstream parser behavior, not just our string match:
        # argparse accepts these abbreviated names and last task occurrence wins.
        downstream = argparse.ArgumentParser()
        downstream.add_argument("--task")
        downstream.add_argument("--resume", action="store_true")
        downstream.add_argument("--checkpoint")
        downstream.add_argument("--load_run")
        downstream.add_argument("--load_checkpoint")
        attacks = (
            (["--tas", "wrong"], "task", "wrong"),
            (["--tas=wrong"], "task", "wrong"),
            (["--resu"], "resume", True),
            (["--checkp", "old.pt"], "checkpoint", "old.pt"),
            (["--load_r", "mock"], "load_run", "mock"),
            (["--load_c=old.pt"], "load_checkpoint", "old.pt"),
        )
        for arguments, field, value in attacks:
            with self.subTest(arguments=arguments):
                vulnerable = downstream.parse_args(["--task", wrapper.MKII_V2_TASK_ID, *arguments])
                self.assertEqual(getattr(vulnerable, field), value)
                with self.assertRaises(ValueError):
                    downstream.parse_args(wrapper.prepare_training_args(arguments))

    def test_complete_unrelated_downstream_flags_remain_unchanged(self):
        legitimate = ["--num_envs", "32", "--headless", "--seed", "17",
                      "--rendering_mode", "performance", "--run_name", "cad-v2",
                      "--video", "--video_length", "100", "--max_iterations", "5",
                      "--device", "cuda:0", "--logger", "tensorboard"]
        self.assertEqual(wrapper.prepare_training_args(legitimate),
                         ["--task", wrapper.MKII_V2_TASK_ID, *legitimate])

    def test_dry_run_needs_no_site_packages_and_does_not_execute(self):
        # -S disables site/venv/editable installs and optional simulator libs.
        result = subprocess.run(
            [sys.executable, "-S", str(WRAPPER), "--dry-run", "--num_envs", "32"],
            cwd=tempfile.gettempdir(), text=True, capture_output=True, check=True,
            env={**os.environ, "PYTHONPATH": str(ROOT / "isaaclab")},
        )
        plan = json.loads(result.stdout)
        self.assertEqual(plan["execution"], "not_started")
        self.assertEqual(plan["task_id"], wrapper.MKII_V2_TASK_ID)
        self.assertIn("hexapod_mkii_serial_v2", plan["usd_path_container"])
        self.assertEqual(plan["argv"][-2:], ["--num_envs", "32"])
        self.assertIn(str(ROOT / "packages/hexapod_core"), plan["source_paths"])

    def test_noninstalled_spark_layout_registers_then_delegates_without_sim_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "sibling_module.py").write_text("VALUE = 19\n")
            upstream = path / "train.py"
            upstream.write_text('''import json, sys
import gymnasium
import sibling_module
from hexapod_core.cad_manifest_v2 import COMMAND_FRAME
from hexapod_env.assets.mkii_v2 import MKII_V2_ASSET
assert not any(key == "torch" or key.startswith("isaacsim") or key.startswith("isaaclab.") for key in sys.modules)
print(json.dumps({"argv": sys.argv, "registered": gymnasium.registry, "frame": COMMAND_FRAME, "height": MKII_V2_ASSET.reset_root_height_m, "sibling": sibling_module.VALUE}))
''')
            # Inject only a tiny gym registry; no editable installation, torch,
            # Isaac Sim or real trainer is available in this process.
            bootstrap = '''import sys, types, runpy
module = types.ModuleType("gymnasium")
module.registry = {}
def register(**kwargs):
    module.registry[kwargs["id"]] = kwargs
module.register = register
sys.modules["gymnasium"] = module
sys.argv = [sys.argv[1], "--train-script", sys.argv[2], "--num_envs", "7"]
runpy.run_path(sys.argv[0], run_name="__main__")
'''
            result = subprocess.run(
                [sys.executable, "-S", "-c", bootstrap, str(WRAPPER), str(upstream)],
                cwd=directory, text=True, capture_output=True, check=True,
                env={**os.environ, "PYTHONPATH": str(ROOT / "isaaclab")},
            )
            report = json.loads(result.stdout)
            self.assertEqual(report["argv"], [str(upstream.resolve()), "--task", wrapper.MKII_V2_TASK_ID, "--num_envs", "7"])
            self.assertEqual(list(report["registered"]), [wrapper.MKII_V2_TASK_ID])
            config = report["registered"][wrapper.MKII_V2_TASK_ID]["kwargs"]
            self.assertEqual(config["env_cfg_entry_point"], "hexapod_rl.tasks.mkii_v2:HexapodMkiiV2FlatEnvCfg")
            self.assertEqual(report["frame"], "navigation")
            self.assertEqual(report["height"], .142964)
            self.assertEqual(report["sibling"], 19)

    def test_missing_or_recursive_upstream_fails_before_optional_imports(self):
        with self.assertRaises(FileNotFoundError):
            wrapper.run_training_script(Path("/nonexistent/isaaclab/train.py"), [])
        with self.assertRaisesRegex(ValueError, "cannot be this wrapper"):
            wrapper.run_training_script(WRAPPER, [])


if __name__ == "__main__":
    unittest.main()
