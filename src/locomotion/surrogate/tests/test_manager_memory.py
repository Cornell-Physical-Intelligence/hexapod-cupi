"""Check the manager-based command schedule and reward memory against the training task on surrogate rollouts."""

import unittest

import torch

from locomotion.omniwalk.mdp.commands import TaskConfig
from locomotion.omniwalk.mdp.memory import CommandSchedule, RewardMemory
from locomotion.omniwalk.task import TrainingTask
from locomotion.surrogate.env import make_env


class ManagerMemoryTests(unittest.TestCase):
    def test_schedule_draws_equal_the_training_task_resample(self):
        env = make_env(5, threads=1)
        task = TrainingTask(env, TaskConfig(seed=17))
        schedule = CommandSchedule(TaskConfig(seed=17), task.reward_config.turn_draw_fraction)
        for rows in ([0, 1, 2, 3, 4], [2], [4, 0], [1, 3, 2], [0, 1, 2, 3, 4], [3]):
            rows = torch.tensor(rows)
            task._resample(rows)
            values, durations = schedule.draw(len(rows))
            self.assertTrue(torch.equal(env.commands[rows], values))
            self.assertTrue(torch.equal(task.remaining_controls[rows], durations))
        self.assertEqual(
            (schedule.command_draws, schedule.zero_command_draws),
            (task.command_draws, task.zero_command_draws),
        )
        env.close()

    def test_reward_memory_equals_the_training_task_across_resets_and_command_changes(self):
        # The manager restarts the memory before it resamples the reset rows, so the restart reads the old command.
        env = make_env(3, episode_seconds=0.6, threads=1)
        task = TrainingTask(env, TaskConfig(seed=11))
        memory = RewardMemory(
            3,
            "cpu",
            neutral=env.neutral,
            lower=env.lower,
            upper=env.upper,
            action_scale=env.cfg.action_scale_rad,
            nominal_height=task.nominal_height,
        )
        before = env.commands.clone()
        task.reset()
        memory.restart(torch.arange(3), env.current, before, env.held)
        generator = torch.Generator().manual_seed(3)
        resets = 0
        for control in range(75):
            if control == 10:
                env.commands[1] = torch.tensor([0.0, 0.0, 0.2])
            if control == 20:
                env.commands[2] = 0.0
            command = env.commands.clone()
            output = task.step(0.6 * torch.randn(3, 18, generator=generator))
            reward, components, _ = memory.reward(
                env.telemetry, command, output["terminated"], env.episode_steps
            )
            self.assertTrue(torch.equal(reward, output["reward"]), control)
            self.assertEqual(set(components), set(task.last_components))
            for key, value in components.items():
                self.assertTrue(torch.equal(value, task.last_components[key]), (control, key))
            done = (output["terminated"] | output["truncated"]).nonzero().flatten()
            if len(done):
                before = env.commands.clone()
                task.reset(done)
                memory.restart(done, env.current, before, env.held)
                resets += 1
        self.assertGreaterEqual(resets, 2)
        env.close()


if __name__ == "__main__":
    unittest.main()
