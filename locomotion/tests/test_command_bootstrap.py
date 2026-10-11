"""Check command return targets through RSL-RL, including mixed episode endings."""
from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from rsl_rl.algorithms import PPO

from locomotion import ppo


class MixedTask:
    num_envs, num_actions, device = 4, 18, 'cpu'
    cfg = SimpleNamespace(episode_seconds=20., control_dt=.02, declaration=lambda: {})

    def __init__(self):
        self.commands = torch.tensor([[.05, 0., 0.]] * 4)
        self.episode_steps = torch.zeros(4, dtype=torch.long)
        self.position = torch.ones(4)
        self.resets = []

    def declaration(self):
        return {}

    def output(self):
        obs = torch.zeros(4, 231)
        obs[:, 0] = self.position
        obs[:, 210:213] = self.commands
        return {'obs': obs, 'critic': torch.cat((obs, torch.zeros(4, 3)), -1)}

    def reset(self, selected=None):
        selected = torch.arange(4) if selected is None else selected
        self.resets.append(selected.tolist())
        self.position[selected] = 1.
        self.episode_steps[selected] = 0
        return self.output()

    def step(self, action):
        self.position += 6.
        self.episode_steps += 1
        self.commands[1:] = 0.
        return {**self.output(), 'reward': torch.ones(4),
            'terminated': torch.tensor([False, False, True, False]),
            'truncated': torch.tensor([False, False, True, True])}


class CommandBootstrapTests(unittest.TestCase):
    def test_post_action_value_holds_old_command_and_does_not_double_bootstrap(self):
        for ceiling in (None, 3e-4):
            with self.subTest(ceiling=ceiling):
                task = MixedTask()
                wrapped = ppo.VanillaVecEnv(task, command_segments='bootstrap',
                    observation_scaling='fixed', gait_clock=60, velocity_noise=.5)
                config = ppo.ppo_config(7, command_segments='bootstrap', learning_rate_max=ceiling,
                    observation_normalization='none')
                config.pop('environment_wrapper')
                config.update(num_steps_per_env=1, multi_gpu=None)
                for role in ('actor', 'critic'):
                    config[role]['hidden_dims'] = [8]
                obs = wrapped.get_observations()
                with redirect_stdout(io.StringIO()):
                    alg = PPO.construct_algorithm(obs, wrapped, config, 'cpu')
                def value(observation):
                    critic = observation['critic']
                    return critic[:, :1] + 100 * critic[:, 210:211]
                with patch.object(alg.critic, 'forward', side_effect=value), torch.no_grad():
                    action = alg.act(obs)
                    next_obs, rewards, done, extras = wrapped.step(action)
                    rng = torch.random.get_rng_state().clone()
                    alg.process_env_step(next_obs, rewards, done, extras)
                torch.testing.assert_close(torch.random.get_rng_state(), rng)
                # Before: 2 + 100 = 102. After under the held command: 14 + 100 = 114.
                torch.testing.assert_close(alg.storage.rewards[0, :, 0],
                    torch.tensor([1., 1. + .99 * 114., 1., 1. + .99 * 102.]))
                torch.testing.assert_close(alg.storage.values[0, :, 0], torch.full((4,), 102.))
                self.assertEqual(done.tolist(), [0, 1, 1, 1])
                self.assertEqual(task.resets, [[0, 1, 2, 3], [2, 3]])
                self.assertEqual(float(next_obs['critic'][1, 210]), 0.)
                self.assertEqual(next_obs['critic'][1, -2:].tolist(), [0., 0.])
                terminal = extras['command_bootstrap_observation']['critic']
                torch.testing.assert_close(terminal[1, -2:],
                    ppo.clock_features(torch.tensor([1]), 60, torch.tensor([[.05, 0., 0.]]))[0])


if __name__ == '__main__':
    unittest.main()
