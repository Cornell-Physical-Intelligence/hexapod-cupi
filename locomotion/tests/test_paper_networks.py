"""Table III network shapes, privileged-input isolation and the supervised velocity estimator."""
import copy
import unittest

import torch
from tensordict import TensorDict

from locomotion import paper_networks as nets

N = 5
GROUPS = {'actor': list(nets.ACTOR_GROUPS), 'critic': list(nets.CRITIC_GROUPS)}
DISTRIBUTION = {'class_name': 'GaussianDistribution', 'init_std': .15, 'std_type': 'log'}


def observations(seed=0, n=N):
    generator = torch.Generator().manual_seed(seed)
    draw = lambda width: torch.randn(n, width, generator=generator)
    policy = draw(231)
    return TensorDict({'policy': policy, 'critic': torch.cat((policy, draw(3)), -1), 'amp': draw(61),
                       'velocity_label': draw(3), 'privileged': draw(42), **nets.actor_observation(policy)},
                      batch_size=[n])


def models(seed=1):
    obs = observations()
    torch.manual_seed(seed)
    actor = nets.PaperActor(obs, GROUPS, 'actor', 18, distribution_cfg=copy.deepcopy(DISTRIBUTION))
    critic = nets.PaperCritic(obs, GROUPS, 'critic', 1)
    return obs, actor, critic


class ArchitectureTests(unittest.TestCase):
    def test_table_three_layer_widths(self):
        _, actor, critic = models()
        self.assertEqual(nets.layer_widths(actor.estimator), [64, 32, 3])
        self.assertEqual(nets.layer_widths(actor.memory), [512, 256, 128, 32])
        self.assertEqual(nets.layer_widths(actor.mlp), [256, 128, 64, 18])
        self.assertEqual(actor.mlp[0].in_features, 3 + 18 + 42 + 3 + 32)
        self.assertEqual(nets.layer_widths(critic.privileged_encoder), [64, 32, 8])
        self.assertEqual(nets.layer_widths(critic.mlp), [512, 256, 128, 1])
        self.assertEqual(critic.mlp[0].in_features, 3 + 18 + 42 + 8)

    def test_forward_shapes_and_stochastic_sampling(self):
        obs, actor, critic = models()
        self.assertEqual(actor(obs).shape, (N, 18))
        self.assertEqual(actor(obs, stochastic_output=True).shape, (N, 18))
        self.assertEqual(actor.get_output_log_prob(torch.zeros(N, 18)).shape, (N,))
        self.assertEqual(critic(obs).shape, (N, 1))
        self.assertEqual(actor.velocity_estimate.shape, (N, 3))

    def test_group_order_and_widths_are_validated(self):
        obs = observations()
        wrong_order = {'actor': ['command', 'proprio_history', 'previous_action'], 'critic': GROUPS['critic']}
        with self.assertRaisesRegex(ValueError, 'exactly'):
            nets.PaperActor(obs, wrong_order, 'actor', 18)
        with self.assertRaisesRegex(ValueError, 'exactly'):
            nets.PaperCritic(obs, {'critic': GROUPS['actor']}, 'critic', 1)
        narrow = obs.clone()
        narrow['proprio_history'] = narrow['proprio_history'][:, :200]
        with self.assertRaisesRegex(ValueError, 'widths'):
            nets.PaperActor(narrow, GROUPS, 'actor', 18)
        with self.assertRaisesRegex(ValueError, 'deterministic'):
            nets.PaperCritic(obs, GROUPS, 'critic', 1, distribution_cfg=copy.deepcopy(DISTRIBUTION))
        with self.assertRaises(ValueError):
            nets.PaperActor(obs, GROUPS, 'actor', 18, memory_latent=0)

    def test_actor_observation_splits_the_policy_vector(self):
        policy = torch.arange(231.).expand(2, -1)
        split = nets.actor_observation(policy)
        torch.testing.assert_close(split['proprio_history'], policy[:, :210])
        torch.testing.assert_close(split['command'], policy[:, 210:213])
        torch.testing.assert_close(split['previous_action'], policy[:, 213:])
        with self.assertRaises(ValueError):
            nets.actor_observation(torch.zeros(2, 230))

    def test_export_is_declared_unsupported(self):
        _, actor, critic = models()
        for model in (actor, critic):
            with self.assertRaises(NotImplementedError):
                model.as_jit()
            with self.assertRaises(NotImplementedError):
                model.as_onnx()


class IsolationTests(unittest.TestCase):
    def test_actor_ignores_privileged_state_and_velocity_label(self):
        obs, actor, _ = models()
        actor.eval()
        with torch.no_grad():
            before = actor(obs).clone()
            leaked = obs.clone()
            leaked['privileged'] = leaked['privileged'] + 100.
            leaked['velocity_label'] = leaked['velocity_label'] + 100.
            leaked['critic'] = leaked['critic'] + 100.
            torch.testing.assert_close(actor(leaked), before, rtol=0, atol=0)
        self.assertNotIn('privileged', actor.obs_groups)
        self.assertNotIn('velocity_label', actor.obs_groups)
        self.assertNotIn('critic', actor.obs_groups)

    def test_policy_gradient_does_not_reach_the_estimator(self):
        obs, actor, _ = models()
        actor(obs).sum().backward()
        self.assertTrue(all(p.grad is None for p in actor.estimator_parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.memory.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.mlp.parameters()))

    def test_critic_uses_privileged_state(self):
        obs, _, critic = models()
        with torch.no_grad():
            before = critic(obs).clone()
            changed = obs.clone()
            changed['privileged'] = changed['privileged'] + 1.
            self.assertGreater(float((critic(changed) - before).abs().max()), 0.)


class EstimatorTests(unittest.TestCase):
    def test_supervised_loss_trains_only_the_estimator_and_decreases(self):
        obs, actor, _ = models()
        obs['velocity_label'] = obs['proprio_history'][:, -42:-39]
        frozen = {name: p.detach().clone() for name, p in actor.named_parameters() if not name.startswith('estimator')}
        optimizer = torch.optim.Adam(actor.estimator_parameters(), lr=1e-2)
        first = float(actor.estimator_loss(obs).detach())
        for _ in range(60):
            loss = actor.estimator_loss(obs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess(float(actor.estimator_loss(obs).detach()), .5 * first)
        for name, p in actor.named_parameters():
            if not name.startswith('estimator'):
                torch.testing.assert_close(p, frozen[name], rtol=0, atol=0, msg=name)

    def test_label_shape_is_checked(self):
        obs, actor, _ = models()
        obs['velocity_label'] = torch.zeros(N, 2)
        with self.assertRaisesRegex(ValueError, 'label'):
            actor.estimator_loss(obs)


if __name__ == '__main__':
    unittest.main()
