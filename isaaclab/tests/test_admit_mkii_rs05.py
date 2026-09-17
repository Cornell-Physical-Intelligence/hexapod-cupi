"""CPU checks of the RS05 standing-capture runner's arguments and plan.

The runner writes a capture for the frozen scorer. It imports no simulator
while parsing arguments, so these checks run on a machine without Isaac Lab.
Nothing here starts a run or grades one.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "isaaclab/admit_mkii_rs05.py"
_spec = importlib.util.spec_from_file_location("_admit_mkii_rs05_test", RUNNER)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

GEOMETRY_DIR = ROOT / "artifacts/restart_2026-09-14/standing_translation_preparation_001/source/geometry"


def arguments(directory, **changes):
    values = {
        "--num-envs": "1",
        "--geometry": str(GEOMETRY_DIR / "geometry.json"),
        "--geometry-extrema": str(GEOMETRY_DIR / "geometry_extrema.npz"),
        "--output": str(Path(directory) / "capture_001"),
    }
    values.update(changes)
    return [item for key, value in values.items() for item in ((key, value) if value is not None else (key,))]


class ArgumentTests(unittest.TestCase):
    def test_the_defaults_bind_the_frozen_scorer_window(self):
        with tempfile.TemporaryDirectory() as directory:
            args = runner.prepare_arguments(arguments(directory))
            self.assertEqual(args.num_envs, 1)
            self.assertEqual(args.controls, runner.SCORER_CONTROLS)
            self.assertEqual(args.settle_controls, runner.SCORER_SETTLE_CONTROLS)
            self.assertEqual(args.device, "cuda:0")
            self.assertIs(args.headless, True)

    def test_replica_counts_are_positive_integers(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in ("0", "-4", "1.5", "many"):
                with self.subTest(value=value), self.assertRaises((ValueError, SystemExit)):
                    runner.prepare_arguments(arguments(directory, **{"--num-envs": value}))
            args = runner.prepare_arguments(arguments(directory, **{"--num-envs": "128"}))
            self.assertEqual(args.num_envs, 128)

    def test_a_window_the_frozen_scorer_refuses_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in ("999", "1001", "0"):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "1000"):
                    runner.prepare_arguments(arguments(directory, **{"--controls": value}))

    def test_missing_geometry_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for key in ("--geometry", "--geometry-extrema"):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "does not exist"):
                    runner.prepare_arguments(arguments(directory, **{key: str(Path(directory) / "absent")}))

    def test_an_existing_output_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "capture_001"
            existing.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                runner.prepare_arguments(arguments(directory))

    def test_the_plan_records_the_model_and_starts_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = runner.launch_plan(runner.prepare_arguments(arguments(directory, **{"--num-envs": "128"})))
            self.assertEqual(plan["execution"], "not_started")
            self.assertEqual(plan["task_id"], runner.MKII_RS05_TASK_ID)
            self.assertEqual(plan["num_envs"], 128)
            self.assertEqual(plan["asset_usd_sha256"], runner.MKII_RS05_USD_SHA256)
            self.assertIs(plan["standing_admission"], False)
            self.assertIn("score_diagnostic", plan["grading_command"])
            self.assertIn("packages/hexapod_env", " ".join(plan["source_paths"]))

    def test_the_capture_episode_outlasts_the_scored_window(self):
        with tempfile.TemporaryDirectory() as directory:
            args = runner.prepare_arguments(arguments(directory))
            seconds = runner.capture_episode_seconds(args)
            self.assertGreater(seconds, args.controls * 0.02)


class DryRunTests(unittest.TestCase):
    def test_the_dry_run_prints_a_plan_without_site_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-S", str(RUNNER), "--dry-run", *arguments(directory)],
                cwd=tempfile.gettempdir(), text=True, capture_output=True, check=True,
                env={**os.environ, "PYTHONPATH": str(ROOT / "isaaclab")},
            )
            plan = json.loads(result.stdout)
            self.assertEqual(plan["execution"], "not_started")
            self.assertEqual(plan["num_envs"], 1)
            self.assertFalse(Path(plan["output"]).exists())


if __name__ == "__main__":
    unittest.main()
