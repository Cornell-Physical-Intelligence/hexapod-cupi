"""Exercise current-task AMP updates and command boundaries on CPU physics."""
import contextlib
import io
import unittest

import torch
from rsl_rl.runners import OnPolicyRunner

from locomotion.priors.amp import AMPConfig, AMPVecEnv, amp_ppo_config
from locomotion.surrogate.env import make_env
from locomotion.omniwalk.mdp.commands import TaskConfig
from locomotion.omniwalk.task import TrainingTask


class AMPTests(unittest.TestCase):
    def test_clock_and_paper_models_update_across_command_boundaries(self):
        torch.set_num_threads(1)
        for networks in ('mlp', 'paper'):
            with self.subTest(networks=networks):
                env = make_env(4, threads=0, record_motion_features=True)
                try:
                    task = TrainingTask(env, TaskConfig(seed=3))
                    options = dict(command_segments='bootstrap', action_smoothing='mean2',
                                   gait_clock=60 if networks == 'mlp' else 0)
                    wrapped = AMPVecEnv(task, **options)
                    config = amp_ppo_config(3, networks=networks, learning_rate_max=0.0003,
                        amp_config=AMPConfig(discriminator_hidden=(8,), discriminator_updates=1), **options)
                    config.pop('environment_wrapper', None)
                    with contextlib.redirect_stdout(io.StringIO()):
                        runner = OnPolicyRunner(wrapped, config, None, device='cpu')
                    obs = wrapped.get_observations()
                    task.remaining_controls[:] = 1
                    boundary_seen = False
                    for _ in range(24):
                        with torch.inference_mode():
                            action = runner.alg.act(obs)
                            obs, reward, done, extras = wrapped.step(action)
                            boundary_seen |= 'command_time_outs' in extras
                            runner.alg.process_env_step(obs, reward, done, extras)
                    with torch.inference_mode():
                        runner.alg.compute_returns(obs)
                    losses = runner.alg.update()
                    self.assertTrue(boundary_seen)
                    self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in losses.values()))
                    self.assertIn('discriminator_prior', losses)
                    self.assertLessEqual(runner.alg.learning_rate, 0.0003)
                    self.assertEqual(obs['policy'].shape[-1], 233 if networks == 'mlp' else 231)
                finally:
                    env.close()
