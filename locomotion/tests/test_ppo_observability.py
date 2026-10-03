"""Check behavior-policy drift and prove that diagnostics preserve stock PPO updates."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO

from locomotion.ppo import ppo_config, policy_change_metrics, UpdateDiagnostics
from locomotion.prepare import prepare
from locomotion.train import require_learner_configuration


def observation(value):
    return TensorDict({'policy': value, 'critic': torch.cat((value, value[:, :3]), -1)},
                      batch_size=[len(value)])


def algorithm(normalization):
    torch.manual_seed(20260917)
    obs = observation(torch.randn(32, 231) * .1)
    config = ppo_config(20260917, action_mean='tanh', observation_normalization=normalization)
    config.update(num_steps_per_env=8, multi_gpu=None)
    for role in ('actor', 'critic'):
        config[role]['hidden_dims'] = [16, 16]
    with redirect_stdout(io.StringIO()):
        result = PPO.construct_algorithm(obs, SimpleNamespace(num_envs=32, num_actions=18), config, 'cpu')
    return result, obs


def collect(result, obs):
    with torch.inference_mode():
        for step in range(8):
            action = result.act(obs)
            nxt = observation(obs['policy'] + .15)
            reward = -(action - .2).square().mean(-1)
            result.process_env_step(nxt, reward, torch.zeros(32, dtype=torch.long), {})
            obs = nxt
        result.compute_returns(obs)


class ObservationDiagnosticsTests(unittest.TestCase):
    def test_running_normalizer_changes_behavior_before_optimizer_step(self):
        result, obs = algorithm('empirical')
        weights = {k: v.clone() for k, v in result.actor.named_parameters()}
        collect(result, obs)
        metrics = policy_change_metrics(result.actor, result.storage, result.clip_param)
        self.assertGreater(metrics['kl_mean'], .02)
        for name, value in result.actor.named_parameters():
            torch.testing.assert_close(value, weights[name], rtol=0, atol=0)

    def test_raw_observations_preserve_behavior_until_an_optimizer_step(self):
        result, obs = algorithm('none')
        collect(result, obs)
        metrics = policy_change_metrics(result.actor, result.storage, result.clip_param)
        self.assertLess(abs(metrics['kl_mean']), 1e-6)
        self.assertLess(metrics['log_ratio_abs_mean'], 1e-5)
        self.assertEqual(metrics['ratio_outside_clip_fraction'], 0)

    def test_diagnostics_preserve_rng_buffers_losses_and_parameters(self):
        for mode in ('empirical', 'none'):
            with self.subTest(mode=mode):
                result, obs = algorithm(mode)
                collect(result, obs)
                baseline = copy.deepcopy(result)
                rng = torch.get_rng_state()
                actor_state = copy.deepcopy(result.actor.state_dict())
                policy_change_metrics(result.actor, result.storage, result.clip_param)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for key, value in result.actor.state_dict().items():
                    torch.testing.assert_close(value, actor_state[key], rtol=0, atol=0)
                expected = baseline.update()
                expected_rng = torch.get_rng_state()
                torch.set_rng_state(rng)
                diagnostic = UpdateDiagnostics(result)
                observed = diagnostic.update()
                self.assertEqual(expected, observed)
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                for role in ('actor', 'critic'):
                    for key, value in getattr(result, role).state_dict().items():
                        torch.testing.assert_close(value, getattr(baseline, role).state_dict()[key], rtol=0, atol=0)
                self.assertIs(type(result), PPO)
                self.assertEqual(result.storage.step, 0)
                self.assertEqual(diagnostic.latest['learning_rate_after'], result.learning_rate)
                self.assertGreaterEqual(diagnostic.latest['after']['ratio_outside_clip_fraction'], 0)

    def test_checkpoint_configuration_rejects_other_normalization(self):
        old = ppo_config(7)
        self.assertTrue(old['actor']['obs_normalization'])
        self.assertTrue(old['critic']['obs_normalization'])
        new = ppo_config(7, observation_normalization='none')
        self.assertFalse(new['actor']['obs_normalization'])
        record = {'identity': {'upstream_source_files': {}, 'ppo_config': new}}
        require_learner_configuration(record, {'upstream_source_files': {}}, new)
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            require_learner_configuration(record, {'upstream_source_files': {}}, old)
        with self.assertRaisesRegex(ValueError, 'Observation normalization'):
            ppo_config(7, observation_normalization='invalid')

    def test_frozen_package_declares_normalization_and_rejects_other_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            remote = '/srv/cupi/hexapod/runs/james/normalization_test'
            for mode in ('train', 'evaluate'):
                extra = {} if mode == 'train' else {'checkpoint': remote + '/checkpoint.pt',
                    'checkpoint_sha': 'a' * 64, 'checkpoint_declaration_sha': 'b' * 64}
                binding = prepare(path / mode, remote, mode=mode, observation_normalization='none', **extra)
                args = binding['command_args']
                self.assertEqual(args[args.index('--observation-normalization') + 1], 'none')
                self.assertEqual(json.loads((path / mode / 'PACK.json').read_text())['observation_normalization'], 'none')
            for extra in ({'mode': 'diagnostic'}, {'learner': 'amp'}):
                with self.assertRaisesRegex(ValueError, 'Observation normalization'):
                    prepare(path / 'invalid', remote, observation_normalization='none', **extra)


if __name__ == '__main__':
    unittest.main()
