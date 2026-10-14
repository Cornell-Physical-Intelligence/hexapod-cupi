"""The standard-library reward option check that pack preparation uses on hosts without Torch."""
from dataclasses import fields
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from locomotion import reward_options, task_v4, task_v4_yaw_gate
from locomotion.tests.test_task_v4 import REMOTE

ROOT = Path(__file__).resolve().parents[2]


class RewardOptionTests(unittest.TestCase):
    def test_schema_matches_each_task_configuration(self):
        for task, config in (("v4", task_v4.RewardV4Config), ("v4_yaw_gate", task_v4_yaw_gate.RewardV4YawGateConfig)):
            types, choices = reward_options.schema(task)
            self.assertEqual(types, {f.name: f.type for f in fields(config)}, task)
            self.assertEqual(choices, {} if task == "v4" else {"yaw_gate_on": {"turn", "all", "none"}})

    def test_check_refuses_what_the_task_parser_refuses_for_keys_types_and_words(self):
        cases = (("v4_yaw_gate", "yaw_gate_on=none,yaw_gate_bonus=0.25", True), ("v4_yaw_gate", "gate_window_controls=120", True),
                 ("v4", "schedule_weight=none", False), ("v4", "yaw_gate_on=none", False),
                 ("v4_yaw_gate", "yaw_gate_on=sometimes", False), ("v4_yaw_gate", "gate_window_controls=1.5", False),
                 ("v4_yaw_gate", "yaw_gate_bonus=0.25,yaw_gate_bonus=0.5", False), ("v4", "not_a_field=1", False))
        parsers = {"v4": task_v4.reward_config, "v4_yaw_gate": task_v4_yaw_gate.reward_config}
        for task, spec, valid in cases:
            with self.subTest(task=task, spec=spec):
                if valid:
                    reward_options.check(task, spec)
                    parsers[task](spec)
                else:
                    with self.assertRaises(ValueError):
                        reward_options.check(task, spec)
                    with self.assertRaises(ValueError):
                        parsers[task](spec)

    def test_preparation_runs_without_site_packages(self):
        # python -S leaves out site-packages, so Torch cannot load; preparation must still build the pack.
        with tempfile.TemporaryDirectory() as directory:
            code = ("import json, sys; from locomotion.prepare import prepare; "
                    f"args = prepare({str(Path(directory) / 'pack')!r}, {REMOTE!r}, mode='train', reward_version='4', "
                    "gait_clock=60, reward_task='v4_yaw_gate', reward_options='yaw_gate_on=none,turn_draw_fraction=0.3')"
                    "['command_args']; print(json.dumps({'torch': 'torch' in sys.modules, "
                    "'options': args[args.index('--reward-options') + 1]}))")
            done = subprocess.run([sys.executable, "-S", "-B", "-c", code], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(json.loads(done.stdout.splitlines()[-1]),
                             {"torch": False, "options": "yaw_gate_on=none,turn_draw_fraction=0.3"})


if __name__ == "__main__":
    unittest.main()
