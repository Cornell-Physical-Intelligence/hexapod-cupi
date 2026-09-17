"""CPU checks of the RS05 scratch-only trainer wrapper.

The wrapper selects the new task, supplies the replica and headless defaults,
and refuses every checkpoint or resume option. It imports no simulator while
parsing arguments. The corrected CAD v2 wrapper and its checks stay unchanged.
"""

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
WRAPPER = ROOT / "isaaclab/train_mkii_rs05.py"
_spec = importlib.util.spec_from_file_location("_train_mkii_rs05_wrapper_test", WRAPPER)
wrapper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wrapper)

DEFAULTS = ["--num_envs", "4096", "--headless"]


class TaskSelectionTests(unittest.TestCase):
    def test_the_defaults_select_the_new_task_and_the_declared_scale(self):
        self.assertEqual(wrapper.MKII_RS05_TASK_ID, "Isaac-Velocity-Flat-Hexapod-MKII-RS05-Direct-v0")
        self.assertEqual(wrapper.DEFAULT_NUM_ENVS, 4096)
        self.assertEqual(
            wrapper.prepare_training_args(["--seed", "17"]),
            ["--task", wrapper.MKII_RS05_TASK_ID, *DEFAULTS, "--seed", "17"],
        )

    def test_an_explicit_replica_count_or_headless_flag_is_kept_once(self):
        self.assertEqual(
            wrapper.prepare_training_args(["--num_envs", "1024"]),
            ["--task", wrapper.MKII_RS05_TASK_ID, "--headless", "--num_envs", "1024"],
        )
        self.assertEqual(
            wrapper.prepare_training_args(["--num_envs=1024", "--headless"]),
            ["--task", wrapper.MKII_RS05_TASK_ID, "--num_envs=1024", "--headless"],
        )

    def test_another_task_a_duplicate_or_a_missing_value_is_rejected(self):
        for arguments in (
            ["--task", "Isaac-Velocity-Flat-Hexapod-MKII-V2-Direct-v0"],
            ["--task=Isaac-Velocity-Flat-Hexapod-RobStride-Direct-v0"],
            ["--task"], ["--task", "--num_envs", "32"],
            ["--task", wrapper.MKII_RS05_TASK_ID, f"--task={wrapper.MKII_RS05_TASK_ID}"],
            ["task=wrong"], ["++env.task=wrong"],
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                wrapper.prepare_training_args(arguments)

    def test_the_selected_task_passes_through_unchanged(self):
        for value in (["--task", wrapper.MKII_RS05_TASK_ID], [f"--task={wrapper.MKII_RS05_TASK_ID}"]):
            self.assertEqual(wrapper.prepare_training_args(value), [*DEFAULTS, *value])


class ScratchOnlyTests(unittest.TestCase):
    def test_no_checkpoint_can_be_adopted_through_the_cli_or_hydra(self):
        for arguments in (
            ["--resume"], ["--resume=true"], ["--checkpoint", "old.pt"],
            ["--load_run", "paper_walk"], ["--load-checkpoint=model_2.pt"],
            ["agent.resume=true"], ["+agent.load_checkpoint=old.pt"],
            ["++agent.load_run=mock"],
        ):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "scratch-only"):
                wrapper.prepare_training_args(arguments)

    def test_every_abbreviated_protected_option_is_rejected(self):
        for option in ("task", "resume", "checkpoint", "load_run", "load_checkpoint",
                       "load-run", "load-checkpoint"):
            for length in range(1, len(option)):
                prefix = "--" + option[:length]
                for arguments in ([prefix, "wrong"], [prefix + "=wrong"]):
                    with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "Abbreviated protected"):
                        wrapper.prepare_training_args(arguments)

    def test_argparse_abbreviation_cannot_override_the_task_or_resume(self):
        downstream = argparse.ArgumentParser()
        downstream.add_argument("--task")
        downstream.add_argument("--resume", action="store_true")
        downstream.add_argument("--checkpoint")
        downstream.add_argument("--load_run")
        downstream.add_argument("--load_checkpoint")
        downstream.add_argument("--num_envs")
        downstream.add_argument("--headless", action="store_true")
        attacks = (
            (["--tas", "wrong"], "task", "wrong"),
            (["--tas=wrong"], "task", "wrong"),
            (["--resu"], "resume", True),
            (["--checkp", "old.pt"], "checkpoint", "old.pt"),
            (["--load_r", "paper_walk"], "load_run", "paper_walk"),
            (["--load_c=old.pt"], "load_checkpoint", "old.pt"),
        )
        for arguments, field, value in attacks:
            with self.subTest(arguments=arguments):
                vulnerable = downstream.parse_args(["--task", wrapper.MKII_RS05_TASK_ID, *arguments])
                self.assertEqual(getattr(vulnerable, field), value)
                with self.assertRaises(ValueError):
                    downstream.parse_args(wrapper.prepare_training_args(arguments))

    def test_unrelated_downstream_flags_stay_unchanged(self):
        legitimate = ["--seed", "17", "--rendering_mode", "performance", "--run_name", "rs05-scratch",
                      "--video", "--video_length", "100", "--max_iterations", "5",
                      "--device", "cuda:0", "--logger", "tensorboard"]
        self.assertEqual(wrapper.prepare_training_args(legitimate),
                         ["--task", wrapper.MKII_RS05_TASK_ID, *DEFAULTS, *legitimate])


class LaunchPlanTests(unittest.TestCase):
    def test_the_dry_run_needs_no_site_packages_and_starts_nothing(self):
        result = subprocess.run(
            [sys.executable, "-S", str(WRAPPER), "--dry-run", "--seed", "3"],
            cwd=tempfile.gettempdir(), text=True, capture_output=True, check=True,
            env={**os.environ, "PYTHONPATH": str(ROOT / "isaaclab")},
        )
        plan = json.loads(result.stdout)
        self.assertEqual(plan["execution"], "not_started")
        self.assertEqual(plan["mode"], "scratch_only")
        self.assertEqual(plan["task_id"], wrapper.MKII_RS05_TASK_ID)
        self.assertEqual(plan["usd_path_container"], wrapper.MKII_RS05_USD_PATH_CONTAINER)
        self.assertEqual(plan["usd_override_env_var"], "HEXAPOD_MKII_RS05_USD_PATH")
        self.assertIn("1024", " ".join(plan["required_before_execution"]))
        self.assertEqual(plan["argv"][-2:], ["--seed", "3"])
        self.assertIn(str(ROOT / "packages/hexapod_core"), plan["source_paths"])

    def test_the_wrapper_registers_then_delegates_without_simulator_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            upstream = path / "train.py"
            upstream.write_text('''import json, sys
import gymnasium
from hexapod_env.assets.mkii_rs05 import MKII_RS05_ASSET
assert not any(key == "torch" or key.startswith("isaacsim") or key.startswith("isaaclab.") for key in sys.modules)
print(json.dumps({"argv": sys.argv, "registered": gymnasium.registry, "mass": MKII_RS05_ASSET.mass_kg}))
''')
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
            self.assertEqual(
                report["argv"],
                [str(upstream.resolve()), "--task", wrapper.MKII_RS05_TASK_ID, "--headless", "--num_envs", "7"],
            )
            self.assertEqual(list(report["registered"]), [wrapper.MKII_RS05_TASK_ID])
            config = report["registered"][wrapper.MKII_RS05_TASK_ID]["kwargs"]
            self.assertEqual(config["env_cfg_entry_point"], "hexapod_rl.tasks.mkii_rs05:HexapodMkiiRs05FlatEnvCfg")
            self.assertEqual(report["mass"], 7.466088235225788)

    def test_a_missing_or_recursive_upstream_script_fails_early(self):
        with self.assertRaises(FileNotFoundError):
            wrapper.run_training_script(Path("/nonexistent/isaaclab/train.py"), [])
        with self.assertRaisesRegex(ValueError, "cannot be this wrapper"):
            wrapper.run_training_script(WRAPPER, [])

    def test_the_module_docstring_requires_the_1024_throughput_measurement(self):
        self.assertIn("1024", wrapper.__doc__)
        self.assertIn("transitions", wrapper.__doc__.lower())


class ExistingWrapperTests(unittest.TestCase):
    def test_the_cad_v2_wrapper_keeps_its_own_task(self):
        v2_path = ROOT / "isaaclab/train_mkii_v2.py"
        spec = importlib.util.spec_from_file_location("_train_mkii_v2_for_rs05_test", v2_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.MKII_V2_TASK_ID, "Isaac-Velocity-Flat-Hexapod-MKII-V2-Direct-v0")
        self.assertNotEqual(module.MKII_V2_TASK_ID, wrapper.MKII_RS05_TASK_ID)
        self.assertEqual(
            module.prepare_training_args(["--num_envs", "32"]),
            ["--task", module.MKII_V2_TASK_ID, "--num_envs", "32"],
        )


if __name__ == "__main__":
    unittest.main()
