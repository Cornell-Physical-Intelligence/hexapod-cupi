"""Check immediate reward feedback and preserve the version 2 comparison."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from locomotion import task_v2, task_v3, train
from locomotion.prepare import prepare
from locomotion.tests.test_task_v2 import RewardDouble


REMOTE = "/srv/cupi/hexapod/runs/test/reward_v3_fixture"


class ImmediateTrackingTests(unittest.TestCase):
    def test_tracking_uses_the_endpoint_after_different_velocity_histories(self):
        tasks = [task_v3.TrainingTaskV3(RewardDouble(1)) for _ in range(2)]
        for task, speed in zip(tasks, (.05, -.05)):
            task.reset()
            task.env.commands[:] = torch.tensor([[.05, 0., .15]])
            task.remaining_controls[:] = 1000
            task.env.velocity = torch.tensor([[speed, 0., 0., speed]])
            for _ in range(60):
                task.step(torch.zeros(1, 18))
            task.env.velocity = torch.tensor([[.005, 0., 0., .015]])
            task.step(torch.zeros(1, 18))
            for key in ("linear_tracking", "yaw_tracking"):
                weight = 1. if key == "linear_tracking" else .5
                self.assertAlmostEqual(float(task.last_components[key][0]), weight * math.exp(-2.25), places=6)
            self.assertEqual(int(task.stride.count[0]), 0)
        for key in ("linear_tracking", "yaw_tracking"):
            torch.testing.assert_close(tasks[0].last_components[key], tasks[1].last_components[key], rtol=0, atol=0)

    def test_penalties_commands_and_observations_match_version_two(self):
        old = task_v2.TrainingTaskV2(RewardDouble(2))
        new = task_v3.TrainingTaskV3(RewardDouble(2))
        old.reset()
        new.reset()
        generator = torch.Generator().manual_seed(20260917)
        tracking_changed = False
        for control in range(130):
            velocity = torch.tensor([[.05 if control % 2 else 0., 0., 0., .15], [0., -.025, 0., -.15]])
            for task in (old, new):
                task.env.velocity = velocity
                task.env.rate[:] = .02 * (control % 3)
            action = torch.randn((2, 18), generator=generator) * .15
            a, b = old.step(action), new.step(action)
            for key in ("obs", "critic", "terminated", "truncated"):
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            for key in old.last_components:
                if key in ("linear_tracking", "yaw_tracking"):
                    tracking_changed |= not torch.equal(old.last_components[key], new.last_components[key])
                else:
                    torch.testing.assert_close(old.last_components[key], new.last_components[key], rtol=0, atol=0)
        self.assertTrue(tracking_changed)
        self.assertEqual(old.command_draws, new.command_draws)

    def test_configuration_changes_only_the_tracking_kernel(self):
        old, new = asdict(task_v2.REWARD_V2_CONFIG), asdict(task_v3.REWARD_V3_CONFIG)
        self.assertEqual(old.pop("tracking_kernel"), "stride")
        self.assertEqual(new.pop("tracking_kernel"), "scaled")
        self.assertEqual(old, new)
        self.assertEqual(task_v3.scorer_default.config, task_v3.REWARD_V3_CONFIG)

    def test_declaration_and_status_identify_an_unaccepted_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            task = task_v3.TrainingTaskV3(RewardDouble(1), output_dir=directory)
            declaration = json.loads((Path(directory)/"task_definition.json").read_text())
        self.assertEqual(declaration["reward_version"], task_v3.REWARD_VERSION)
        self.assertEqual(task.status()["reward_version"], task_v3.REWARD_VERSION)
        reward = declaration["reward"]
        self.assertEqual(reward["version"], task_v3.REWARD_VERSION)
        self.assertEqual(reward["kernel_audit_label"], "B")
        self.assertEqual(reward["review"]["status"], "experimental")
        self.assertNotIn("record", reward["review"])
        self.assertFalse(declaration["physics_changed"])

    def test_prepare_binds_version_three_and_evaluation_uses_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"train"
            binding = prepare(output, REMOTE, reward_version="3", action_mean="tanh",
                              observation_normalization="none", updates=2000,
                              allocation_profile="flat_pilot_v1", max_wall_seconds=21600)
            args = binding["command_args"]
            self.assertEqual(args[args.index("--reward-version")+1], "3")
            self.assertEqual(json.loads((output/"PACK.json").read_text())["reward_version"], "3")
            self.assertTrue((output/"source/locomotion/task_v3.py").is_file())
            with self.assertRaisesRegex(ValueError, "training only"):
                prepare(Path(directory)/"evaluate", REMOTE, mode="evaluate", reward_version="3")
        self.assertEqual(train.checkpoint_reward_version({"identity": {"reward_version": "3"}}), "3")


if __name__ == "__main__":
    unittest.main()
