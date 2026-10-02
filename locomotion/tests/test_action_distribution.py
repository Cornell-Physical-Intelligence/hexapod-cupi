"""Check bounded means, raw Gaussian likelihoods and the stock PPO update."""
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
from torch.distributions import Normal, kl_divergence

from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_callable

from locomotion.action_distribution import BoundedMeanGaussian
from locomotion.amp_ppo import amp_ppo_config
from locomotion import paper_networks, train
from locomotion.env_config import JOINT_NAMES
from locomotion.ppo import action_metrics, ppo_config
from locomotion.prepare import prepare


def observations(count=8):
    policy = torch.randn(count, 231)
    return TensorDict({'policy': policy, 'critic': torch.cat((policy, torch.randn(count, 3)), -1),
        **paper_networks.actor_observation(policy)}, batch_size=[count])


def actor_from_config(config, obs):
    settings = copy.deepcopy(config['actor'])
    model = resolve_callable(settings.pop('class_name'))
    return model(obs, config['obs_groups'], 'actor', 18, **settings)


class DistributionTests(unittest.TestCase):
    def test_update_inference_and_export_share_the_bounded_mean(self):
        distribution = BoundedMeanGaussian(5, init_std=.15, std_type='log')
        logits = torch.tensor([[-100., -1.6, 0., 1.6, 100.]])
        distribution.update(logits)
        expected = torch.tanh(logits)
        torch.testing.assert_close(distribution.mean, expected)
        torch.testing.assert_close(distribution.deterministic_output(logits), expected)
        exported = torch.jit.script(distribution.as_deterministic_output_module())
        torch.testing.assert_close(exported(logits), expected)
        self.assertTrue(bool((distribution.mean.abs() <= 1).all()))

    def test_sampling_keeps_raw_gaussian_values_outside_the_action_bounds(self):
        distribution = BoundedMeanGaussian(2, init_std=.15, std_type='log')
        logits = torch.full((512, 2), 3.)
        distribution.update(logits)
        torch.manual_seed(31)
        samples = distribution.sample()
        torch.manual_seed(31)
        expected = Normal(torch.tanh(logits), torch.full_like(logits, .15)).sample()
        torch.testing.assert_close(samples, expected, rtol=0, atol=1e-7)
        self.assertTrue(bool((samples > 1).any()))

    def test_likelihood_and_gradient_use_raw_samples_with_no_squash_jacobian(self):
        distribution = BoundedMeanGaussian(3, init_std=.2, std_type='log')
        logits = torch.tensor([[-1.6, .2, 1.6]], requires_grad=True)
        samples = torch.tensor([[-1.3, .1, 1.4]])
        distribution.update(logits)
        actual = distribution.log_prob(samples)
        expected = Normal(torch.tanh(logits), torch.full_like(logits, .2)).log_prob(samples).sum(-1)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        gradient = (samples - torch.tanh(logits)) / .2**2 * (1 - torch.tanh(logits).square())
        torch.testing.assert_close(logits.grad, gradient)
        self.assertTrue(bool((logits.grad.abs() > 0).all()))

    def test_kl_params_and_entropy_describe_the_bounded_mean_gaussian(self):
        distribution = BoundedMeanGaussian(2, init_std=.15, std_type='log')
        distribution.update(torch.tensor([[-2., .3]]))
        old = tuple(value.clone() for value in distribution.params)
        distribution.update(torch.tensor([[-.4, 1.8]]))
        new = distribution.params
        expected = kl_divergence(Normal(*old), Normal(*new)).sum(-1)
        torch.testing.assert_close(distribution.kl_divergence(old, new), expected)
        torch.testing.assert_close(distribution.entropy, Normal(*new).entropy().sum(-1))


class IntegrationTests(unittest.TestCase):
    def test_ppo_and_both_amp_actors_use_the_selected_distribution(self):
        obs = observations()
        configs = [ppo_config(3, action_mean='tanh')]
        configs += [amp_ppo_config(3, networks=networks, action_mean='tanh') for networks in ('mlp', 'paper')]
        for config in configs:
            with self.subTest(actor=config['actor']['class_name'], algorithm=config['algorithm']['class_name']):
                actor = actor_from_config(config, obs)
                self.assertIsInstance(actor.distribution, BoundedMeanGaussian)
                actor(obs, stochastic_output=True)
                torch.testing.assert_close(actor(obs), actor.output_mean)
                self.assertTrue(bool((actor(obs).abs() <= 1).all()))
        self.assertEqual(ppo_config(3)['actor']['distribution_cfg']['class_name'], 'GaussianDistribution')
        with self.assertRaisesRegex(ValueError, 'Action mean'):
            ppo_config(3, action_mean='other')

    def test_checkpoint_and_export_preserve_inference_and_normalization(self):
        torch.manual_seed(9)
        obs = observations()
        config = ppo_config(3, action_mean='tanh')
        actor = actor_from_config(config, obs)
        self.assertIsInstance(actor, MLPModel)
        actor.update_normalization(obs)
        with torch.no_grad():
            actor.mlp[-1].bias.fill_(2.)
        actor.eval()
        expected = actor(obs)
        stream = io.BytesIO()
        torch.save(actor.state_dict(), stream)
        stream.seek(0)
        restored = actor_from_config(config, obs)
        restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        restored.eval()
        torch.testing.assert_close(restored(obs), expected, rtol=0, atol=0)
        self.assertEqual(int(restored.obs_normalizer.count), len(obs))
        exported = torch.jit.script(restored.as_jit())
        torch.testing.assert_close(exported(obs['policy']), expected)
        record = {'identity': {'upstream_source_files': {}, 'ppo_config': config}}
        train.require_learner_configuration(record, {'upstream_source_files': {}}, config)
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            train.require_learner_configuration(record, {'upstream_source_files': {}}, ppo_config(3))

    def test_stock_ppo_keeps_raw_actions_and_updates_the_bounded_actor(self):
        torch.manual_seed(17)
        obs = observations(32)
        config = ppo_config(3, action_mean='tanh')
        config.update(multi_gpu=None, num_steps_per_env=8)
        for role in ('actor', 'critic'):
            config[role].update(hidden_dims=[16, 16], obs_normalization=False)
        with redirect_stdout(io.StringIO()):
            algorithm = PPO.construct_algorithm(obs, SimpleNamespace(num_envs=32, num_actions=18), config, 'cpu')
        self.assertIs(type(algorithm), PPO)
        with torch.no_grad():
            algorithm.actor.mlp[-1].weight.zero_()
            algorithm.actor.mlp[-1].bias.fill_(1.2)
        before = algorithm.actor.mlp[-1].bias.detach().clone()
        raw = []
        with torch.inference_mode():
            for _ in range(8):
                action = algorithm.act(obs)
                raw.append(action.clone())
                reward = -(action.clamp(-1, 1) - .2).square().mean(-1)
                algorithm.process_env_step(obs, reward, torch.zeros(32, dtype=torch.long), {})
            algorithm.compute_returns(obs)
        torch.testing.assert_close(algorithm.storage.actions, torch.stack(raw), rtol=0, atol=0)
        self.assertTrue(bool((algorithm.storage.actions > 1).any()))
        mean, std = algorithm.storage.distribution_params
        expected_log_prob = Normal(mean, std).log_prob(torch.stack(raw)).sum(-1)
        torch.testing.assert_close(algorithm.storage.actions_log_prob.squeeze(-1), expected_log_prob)
        metrics = action_metrics(algorithm.storage, JOINT_NAMES)
        losses = algorithm.update()
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in losses.values()))
        self.assertTrue(bool(torch.isfinite(algorithm.actor.mlp[-1].bias.grad).all()))
        self.assertLess(float(algorithm.actor.mlp[-1].bias.detach().mean()), float(before.mean()))
        self.assertEqual(algorithm.storage.step, 0)
        self.assertEqual(action_metrics(algorithm.storage, JOINT_NAMES), metrics)
        self.assertEqual(metrics['environment_controls'], 256)
        self.assertGreater(metrics['raw_sample_outside_bounds_fraction'], 0)
        self.assertLessEqual(metrics['mean_abs_max'], 1)

    def test_action_metrics_use_the_saved_rollout_means_and_named_joints(self):
        storage = SimpleNamespace(actions=torch.tensor([[[1.2, .1], [-.4, -1.1]]]),
            distribution_params=(torch.tensor([[[.96, .2], [-.5, -.98]]]),))
        report = action_metrics(storage, ['left', 'right'])
        self.assertEqual(report['scope'], 'last_collected_rollout_before_optimizer_update')
        self.assertEqual(report['environment_controls'], 2)
        self.assertEqual(report['raw_sample_outside_bounds_fraction'], .5)
        self.assertAlmostEqual(report['mean_abs_max'], .98)
        self.assertEqual(report['per_joint']['left']['mean_near_bound_fraction'], .5)
        with self.assertRaisesRegex(ValueError, 'matching rollout'):
            action_metrics(storage, ['one'])

    def test_packages_bind_the_action_choice_and_copy_its_source(self):
        remote = '/srv/cupi/hexapod/runs/james/test_action_mean'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ('train', 'evaluate'):
                options = {} if mode == 'train' else {'checkpoint': remote + '/checkpoint.pt',
                    'checkpoint_sha': 'a' * 64, 'checkpoint_declaration_sha': 'b' * 64}
                binding = prepare(root / mode, remote, mode=mode, action_mean='tanh', **options)
                arguments = binding['command_args']
                self.assertEqual(arguments[arguments.index('--action-mean') + 1], 'tanh')
                pack = json.loads((root / mode / 'PACK.json').read_text())
                self.assertEqual(pack['action_mean'], 'tanh')
                manifest = json.loads((root / mode / 'source/FREEZE_SHA256.json').read_text())
                self.assertIn('locomotion/action_distribution.py', manifest)
            legacy = prepare(root / 'legacy', remote)
            self.assertNotIn('--action-mean', legacy['command_args'])
            for options in ({'action_mean': 'other'}, {'mode': 'diagnostic', 'action_mean': 'tanh'},
                            {'mode': 'tripod', 'action_mean': 'tanh'}):
                with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'Bounded action means'):
                    prepare(root / 'bad', remote, **options)

    def test_entry_rejects_bounded_means_without_a_policy(self):
        args = ['--preflight-only', '--headless', '--mode', 'diagnostic', '--num-envs', '1',
                '--source-freeze-sha256', 'a' * 64, '--action-mean', 'tanh']
        for name in ('asset', 'model', 'geometry', 'geometry-extrema', 'stance', 'output'):
            args += ['--' + name, 'unused']
        with self.assertRaisesRegex(ValueError, 'Bounded action means'):
            train.main(args)


if __name__ == '__main__':
    unittest.main()
