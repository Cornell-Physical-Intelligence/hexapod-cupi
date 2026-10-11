"""Online AMP learner: dataset binding, transition pairing, style reward, PPO parity and checkpoints."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
import unittest.mock

import numpy as np
import torch

from rsl_rl.algorithms import PPO

from locomotion import amp, amp_discriminator as disc, amp_ppo, paper_networks as nets, train
from locomotion.amp_ppo import (AMPConfig, AMPPPO, AMPVecEnv, CollisionCapture, amp_ppo_config,
                                load_demonstrations)
from locomotion.ppo import VanillaVecEnv, ppo_config
from locomotion.prepare import prepare

ROOT = Path(__file__).resolve().parents[2]
REMOTE = Path('/home/orionh/HEXAPOD_runs/restart_20260914/test_amp_learner')
BODIES = ['body'] + [f'{leg}_{part}' for leg in ('lf', 'lm', 'lr', 'rf', 'rm', 'rr') for part in ('coxa', 'femur', 'tibia')]
BANK, BANK_IDENTITY = load_demonstrations()
BANK_STATES = BANK[:, :61]
BANK_STD = BANK_STATES.std(0)
RESET_STATE = BANK_STATES[0]


class EnvDouble:
    """Deterministic stand-in whose AMP states are bank states perturbed at the bank's own scale."""

    def __init__(self, n, seed=0, noise=.5):
        self.num_envs, self.device, self.noise = n, 'cpu', noise
        self.native_body_names = list(BODIES)
        self.reset_height = .1
        self.generator = torch.Generator().manual_seed(seed)
        self.capture = None
        self.telemetry = {}
        self.amp = RESET_STATE.expand(n, -1).clone()
        self.cfg = SimpleNamespace(episode_seconds=20., control_dt=.02, record_motion_features=True,
                                   declaration=lambda: {'double': True})

    def draw(self, *shape):
        return torch.randn(*shape, generator=self.generator)

    def bank_scale_states(self):
        rows = torch.randint(len(BANK_STATES), (self.num_envs,), generator=self.generator)
        return BANK_STATES[rows] + self.noise * BANK_STD * self.draw(self.num_envs, 61)

    def state(self):
        obs = self.draw(self.num_envs, 231)
        return {'obs': obs, 'critic': torch.cat((obs, self.draw(self.num_envs, 3)), -1), 'amp': self.amp.clone()}

    def reset(self, indices):
        self.amp[indices] = RESET_STATE
        return self.state()

    def step(self, action, terminate=None):
        self.amp = self.bank_scale_states()
        self.telemetry = {'root_pose_xyzw': torch.cat((torch.zeros(self.num_envs, 2), torch.full((self.num_envs, 1), .09),
                                                       torch.tensor([[0., 0., 0., 1.]]).expand(self.num_envs, -1)), -1),
                          'tibia_floor_force_world_n': self.draw(self.num_envs, 6, 3)}
        output = self.state()
        terminated = torch.zeros(self.num_envs, dtype=torch.bool) if terminate is None else terminate
        output.update(reward=torch.full((self.num_envs,), .5), terminated=terminated,
                      truncated=torch.zeros(self.num_envs, dtype=torch.bool))
        return output


class TaskDouble:
    def __init__(self, n=3, seed=0, noise=.5):
        self.env = EnvDouble(n, seed, noise)
        self.num_envs, self.device, self.cfg = n, 'cpu', self.env.cfg
        self.episode_steps = torch.zeros(n, dtype=torch.long)
        self.terminate_next = None
        self.reset_calls = []

    def declaration(self):
        return {'task': 'double'}

    def reset(self, indices=None):
        indices = torch.arange(self.num_envs) if indices is None else indices
        self.reset_calls.append(indices.clone())
        return self.env.reset(indices)

    def step(self, action):
        output = self.env.step(action, self.terminate_next)
        self.terminate_next = None
        return output


def synthetic_rollout(alg, env, steps=4):
    """Collect and update without touching the global generator."""
    obs = env.get_observations()
    for _ in range(steps):
        actions = alg.act(obs)
        obs, rewards, dones, extras = env.step(actions)
        alg.process_env_step(obs, rewards, dones, extras)
    alg.compute_returns(obs)
    return alg.update()


def construct(config, env):
    config = copy.deepcopy(config)
    config['multi_gpu'] = None
    config['num_steps_per_env'] = 4
    return PPO.construct_algorithm(env.get_observations(), env, config, 'cpu')


def linear_discriminator(weight, std=1.):
    discriminator = disc.Discriminator(torch.zeros(122), torch.full((122,), std), hidden=())
    with torch.no_grad():
        discriminator.net[0].weight.copy_(torch.as_tensor(weight, dtype=torch.float32).reshape(1, -1))
        discriminator.net[0].bias.zero_()
    return discriminator


def contains_tuple(value):
    if isinstance(value, tuple):
        return True
    if isinstance(value, dict):
        return any(contains_tuple(item) for item in value.values())
    return isinstance(value, list) and any(contains_tuple(item) for item in value)


class DatasetTests(unittest.TestCase):
    def test_admitted_bank_loads_under_the_feature_contract(self):
        self.assertEqual(BANK.shape, (18000, 122))
        self.assertEqual(BANK_IDENTITY['path'], 'locomotion/priors/datasets/amp_demonstrations_001')
        self.assertEqual(len(BANK_IDENTITY['transitions_sha256']), 64)
        self.assertEqual(BANK_IDENTITY['reviewer'], 'palerdr')

    def test_changed_bytes_or_contract_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            copy_dir = Path(temporary)/'bank'
            shutil.copytree(amp_ppo.DATASET, copy_dir)
            with np.load(copy_dir/'transitions.npz') as archive:
                data = {key: archive[key] for key in archive.files}
            data['states'][0, 0] += 1.
            np.savez(copy_dir/'transitions.npz', **data)
            with self.assertRaisesRegex(ValueError, 'differ from their manifest'):
                load_demonstrations(copy_dir)
            manifest = json.loads((copy_dir/'manifest.json').read_text())
            manifest['feature_contract']['fields'][-1]['slice'] = [43, 49]
            (copy_dir/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'feature contract'):
                load_demonstrations(copy_dir)


class LossTests(unittest.TestCase):
    def test_gradient_penalty_differentiates_the_standardized_prior_transition(self):
        weight = torch.linspace(-1, 1, 122)
        _, terms = amp_ppo.discriminator_loss(linear_discriminator(weight, std=2.), torch.randn(8, 122),
                                              torch.randn(8, 122), gradient_penalty=10.)
        self.assertAlmostEqual(terms['gradient_penalty'], .5 * 10 * float(weight.square().sum()), places=4)

    def test_gradient_penalty_ignores_feature_scale(self):
        weight = torch.linspace(-1, 1, 122)
        prior, policy = torch.randn(8, 122), torch.randn(8, 122)
        unit = amp_ppo.discriminator_loss(linear_discriminator(weight, 1.), prior, policy)[1]
        tiny = amp_ppo.discriminator_loss(linear_discriminator(weight, 1e-4), prior * 1e-4, policy * 1e-4)[1]
        for key in ('prior', 'policy', 'gradient_penalty'):
            self.assertAlmostEqual(unit[key], tiny[key], places=3)
        raw = disc.discriminator_loss(linear_discriminator(weight, 1e-4), prior * 1e-4, policy * 1e-4)[1]
        self.assertGreater(raw['gradient_penalty'], 1e6 * unit['gradient_penalty'])

    def test_least_squares_targets_match_the_offline_loss(self):
        discriminator = linear_discriminator(torch.linspace(-1, 1, 122))
        prior, policy = torch.randn(8, 122), torch.randn(8, 122)
        ours = amp_ppo.discriminator_loss(discriminator, prior, policy)[1]
        offline = disc.discriminator_loss(discriminator, prior, policy)[1]
        for key in ('prior', 'policy', 'gradient_penalty'):
            self.assertAlmostEqual(ours[key], offline[key], places=4)


class AdapterTests(unittest.TestCase):
    def test_privileged_state_is_42_values_with_reset_defaults(self):
        task = TaskDouble()
        collision = CollisionCapture(task.env)
        env = AMPVecEnv(task, collision=collision)
        self.assertEqual(amp_ppo.PRIVILEGED_WIDTH, 42)
        fresh = env.get_observations()['privileged']
        self.assertEqual(fresh.shape, (3, 42))
        torch.testing.assert_close(fresh[:, 3], torch.full((3,), .1))
        self.assertTrue(bool((fresh[:, 5:23] == 0).all()) and bool((fresh[:, 29:] == 0).all()))
        torch.testing.assert_close(fresh[:, 4], torch.ones(3))
        env.step(torch.zeros(3, 18))
        collision.maximum[1, 2] = 3.
        stepped = env.get_observations()['privileged']
        torch.testing.assert_close(stepped[:, 3], torch.full((3,), .09))
        torch.testing.assert_close(stepped[:, 5:23], task.env.telemetry['tibia_floor_force_world_n'].reshape(3, 18))
        self.assertEqual(stepped[1, 29 + 2], 1.)
        self.assertEqual(float(stepped[:, 29:].sum()), 1.)
        velocity = env.get_observations()['velocity_label']
        torch.testing.assert_close(stepped[:, :3], torch.stack((velocity[:, 1], -velocity[:, 0], velocity[:, 2]), -1))

    def test_terminal_pairs_use_the_state_before_reset(self):
        task = TaskDouble()
        env = AMPVecEnv(task)
        task.reset_calls.clear()
        task.terminate_next = torch.tensor([False, True, False])
        obs, rewards, dones, extras = env.step(torch.zeros(3, 18))
        self.assertEqual(dones.tolist(), [0, 1, 0])
        self.assertEqual(len(task.reset_calls), 1)
        self.assertFalse(torch.equal(extras['amp_next'][1], RESET_STATE))
        torch.testing.assert_close(obs['amp'][1], RESET_STATE)
        torch.testing.assert_close(obs['amp'][0], extras['amp_next'][0])
        self.assertTrue(bool(env.reset_rows[1]) and not bool(env.reset_rows[0]))
        self.assertEqual(obs['proprio_history'].shape, (3, 210))
        self.assertEqual(obs['command'].shape, (3, 3))
        self.assertEqual(obs['previous_action'].shape, (3, 18))

    def test_motion_features_are_required(self):
        task = TaskDouble()
        task.env.state = lambda: {'obs': torch.zeros(3, 231), 'critic': torch.zeros(3, 234)}
        with self.assertRaisesRegex(ValueError, 'record_motion_features'):
            AMPVecEnv(task)

    def test_collision_capture_forwards_to_the_inner_capture(self):
        env = EnvDouble(2)
        calls = []
        capture = CollisionCapture(env, lambda *args: calls.append(args[9]))
        forces = torch.zeros(2, 19, 3)
        forces[0, 0, 2] = 2.
        capture(env, None, None, None, None, None, None, None, forces, 0, None)
        forces = torch.zeros(2, 19, 3)
        forces[1, 5, 0] = 4.
        capture(env, None, None, None, None, None, None, None, forces, 1, None)
        self.assertEqual(calls, [0, 1])
        self.assertEqual(float(capture.maximum[0, 0]), 2.)
        self.assertEqual(float(capture.maximum[1].max()), 4.)
        self.assertEqual(capture.maximum.shape, (2, 13))
        env.native_body_names.remove('body')
        with self.assertRaises(ValueError):
            CollisionCapture(env)


class LearnerTests(unittest.TestCase):
    def test_discriminator_construction_leaves_the_global_generator_untouched(self):
        torch.manual_seed(5)
        construct(ppo_config(3), VanillaVecEnv(TaskDouble(seed=7)))
        stock_draw = torch.rand(4)
        torch.manual_seed(5)
        construct(amp_ppo_config(3), AMPVecEnv(TaskDouble(seed=7)))
        torch.testing.assert_close(torch.rand(4), stock_draw, rtol=0, atol=0)

    def test_discriminator_construction_leaves_cuda_generators_untouched(self):
        with unittest.mock.patch('torch.cuda.manual_seed_all') as seed_all, \
                unittest.mock.patch('torch.cuda.manual_seed') as seed_one:
            construct(amp_ppo_config(3), AMPVecEnv(TaskDouble(seed=7)))
        seed_all.assert_not_called()
        seed_one.assert_not_called()

    def seeded(self, run_seed, seed=None):
        config = amp_ppo_config(run_seed, amp_config=AMPConfig(discriminator_hidden=(16,), seed=seed))
        alg = construct(config, AMPVecEnv(TaskDouble(seed=7)))
        return alg, [value.clone() for value in alg.discriminator.state_dict().values()]

    def test_run_seeds_give_different_discriminator_initializations(self):
        first, first_weights = self.seeded(3)
        second, second_weights = self.seeded(4)
        self.assertEqual(first.amp_config.seed, 3 + amp_ppo.DISCRIMINATOR_SEED_OFFSET)
        self.assertEqual(second.amp_config.seed, 4 + amp_ppo.DISCRIMINATOR_SEED_OFFSET)
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(first_weights, second_weights)))
        self.assertFalse(torch.equal(first.generator.get_state(), second.generator.get_state()))

    def test_one_run_seed_gives_identical_discriminator_initializations(self):
        first, first_weights = self.seeded(3)
        again, again_weights = self.seeded(3)
        for a, b in zip(first_weights, again_weights):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(torch.equal(first.generator.get_state(), again.generator.get_state()))
        self.assertEqual(first.declaration()['config']['seed'], 3 + amp_ppo.DISCRIMINATOR_SEED_OFFSET)

    def test_an_explicit_discriminator_seed_is_honored(self):
        derived, derived_weights = self.seeded(3)
        explicit, explicit_weights = self.seeded(4, seed=derived.amp_config.seed)
        self.assertEqual(explicit.amp_config.seed, derived.amp_config.seed)
        for a, b in zip(derived_weights, explicit_weights):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(amp_ppo_config(4, amp_config=AMPConfig(seed=0))['algorithm']['amp_cfg']['seed'], 0)
        unresolved = amp_ppo_config(4, amp_config=AMPConfig(discriminator_hidden=(16,)))
        unresolved['algorithm']['amp_cfg']['seed'] = None
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            construct(unresolved, AMPVecEnv(TaskDouble(seed=7)))

    def test_parity_with_stock_ppo_from_one_seed_without_reseeding(self):
        off = AMPConfig(style_weight=0., discriminator_updates=0)
        torch.manual_seed(11)
        stock = construct(ppo_config(3), VanillaVecEnv(TaskDouble(seed=7)))
        stock_losses = synthetic_rollout(stock, VanillaVecEnv(TaskDouble(seed=9)))
        torch.manual_seed(11)
        ours = construct(amp_ppo_config(3, amp_config=off), AMPVecEnv(TaskDouble(seed=7)))
        self.assertIsInstance(ours, AMPPPO)
        our_losses = synthetic_rollout(ours, AMPVecEnv(TaskDouble(seed=9)))
        for key in ('value', 'surrogate', 'entropy'):
            self.assertAlmostEqual(stock_losses[key], our_losses[key], places=6)
        torch.testing.assert_close(ours.storage.actions, stock.storage.actions, rtol=0, atol=0)
        for (name, a), (_, b) in zip(stock.actor.state_dict().items(), ours.actor.state_dict().items()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
        for (name, a), (_, b) in zip(stock.critic.state_dict().items(), ours.critic.state_dict().items()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)

    def test_style_weight_scales_the_reward_the_learner_stores(self):
        for weight in (0., 1., 2.5):
            with self.subTest(weight=weight):
                torch.manual_seed(2)
                env = AMPVecEnv(TaskDouble(seed=2))
                alg = construct(amp_ppo_config(1, amp_config=AMPConfig(style_weight=weight, discriminator_updates=0)), env)
                obs = env.get_observations()
                obs, rewards, dones, extras = env.step(alg.act(obs))
                alg.process_env_step(obs, rewards, dones, extras)
                # Bank-scale states keep Eq. (2) off its clamp, so the weight is what this asserts.
                self.assertGreater(float(alg.style_rewards.min()), 0.)
                self.assertLess(float(alg.style_rewards.max()), 1.)
                torch.testing.assert_close(alg.storage.rewards[0].flatten(), rewards + weight * alg.style_rewards)
                torch.testing.assert_close(alg.policy_transitions[0, :, 61:], extras['amp_next'])

    def test_discriminator_separates_the_bank_from_bank_scale_perturbations(self):
        config = AMPConfig(discriminator_updates=40, discriminator_hidden=(64,), discriminator_learning_rate=1e-3)
        torch.manual_seed(1)
        env = AMPVecEnv(TaskDouble(n=64, seed=2))
        alg = construct(amp_ppo_config(1, amp_config=config), env)
        for _ in range(5):
            losses = synthetic_rollout(alg, env)
        unseen = EnvDouble(512, seed=99)
        perturbed = disc.transitions(unseen.bank_scale_states(), unseen.bank_scale_states())
        self.assertLess(float((perturbed.std(0) / BANK.std(0)).max()), 3.)
        with torch.no_grad():
            bank_style = float(disc.style_reward(alg.discriminator(BANK[::36])).mean())
            perturbed_style = float(disc.style_reward(alg.discriminator(perturbed)).mean())
        self.assertGreater(bank_style, perturbed_style + .3)
        for key in ('discriminator_prior', 'discriminator_policy', 'discriminator_gradient_penalty', 'style_reward_mean'):
            self.assertIn(key, losses)
        self.assertNotIn('velocity_estimator', losses)

    def test_paper_networks_train_the_estimator_after_ppo(self):
        config = AMPConfig(discriminator_updates=1, discriminator_hidden=(32,))
        env = AMPVecEnv(TaskDouble(seed=4))
        alg = construct(amp_ppo_config(2, networks='paper', amp_config=config), env)
        self.assertIsInstance(alg.actor, nets.PaperActor)
        self.assertIsNotNone(alg.estimator_optimizer)
        before = [p.detach().clone() for p in alg.actor.estimator_parameters()]
        losses = synthetic_rollout(alg, env)
        self.assertIn('velocity_estimator', losses)
        self.assertTrue(any(float((a - b.detach()).abs().max()) > 0
                            for a, b in zip(before, alg.actor.estimator_parameters())))

    def test_config_and_declaration_record_decisions(self):
        config = amp_ppo_config(0, networks='paper')
        self.assertEqual(config['algorithm']['class_name'], 'locomotion.amp_ppo:AMPPPO')
        self.assertEqual(config['actor']['class_name'], 'locomotion.paper_networks:PaperActor')
        self.assertEqual(config['obs_groups']['critic'][-1], 'privileged')
        self.assertEqual(config['networks'], 'paper')
        self.assertEqual(amp_ppo_config(0)['actor']['class_name'], 'MLPModel')
        with self.assertRaises(ValueError):
            amp_ppo_config(0, networks='rnn')
        alg = construct(amp_ppo_config(0, amp_config=AMPConfig(discriminator_hidden=(8,))), AMPVecEnv(TaskDouble()))
        declaration = alg.declaration()
        self.assertEqual(declaration['feature_contract'], amp.feature_contract())
        self.assertFalse(declaration['stage2_complete'])
        self.assertTrue(declaration['learner_decisions'] and declaration['network_decisions'])
        self.assertEqual(train.json_shape(declaration), declaration)


class CheckpointTests(unittest.TestCase):
    def trained(self, seed=4):
        config = AMPConfig(discriminator_updates=1, discriminator_hidden=(16,))
        env = AMPVecEnv(TaskDouble(seed=5))
        alg = construct(amp_ppo_config(seed, networks='paper', amp_config=config), env)
        synthetic_rollout(alg, env)
        return alg, env, config

    def test_round_trip_binds_the_bank(self):
        alg, env, config = self.trained()
        saved = alg.save()
        for key in ('discriminator_state_dict', 'discriminator_optimizer_state_dict',
                    'estimator_optimizer_state_dict', 'amp_dataset_identity'):
            self.assertIn(key, saved)
        fresh = construct(amp_ppo_config(4, networks='paper', amp_config=AMPConfig(
            discriminator_updates=1, discriminator_hidden=(16,), seed=1)), env)
        self.assertTrue(fresh.load(saved, None, True))
        with torch.no_grad():
            torch.testing.assert_close(fresh.discriminator(BANK[:8]), alg.discriminator(BANK[:8]))
        foreign = copy.deepcopy(saved)
        foreign['amp_dataset_identity']['transitions_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'bank differs'):
            fresh.load(foreign, None, True)

    def test_load_restores_only_what_the_caller_names(self):
        alg, env, _ = self.trained()
        saved = alg.save()
        fresh = construct(amp_ppo_config(4, networks='paper', amp_config=AMPConfig(
            discriminator_updates=1, discriminator_hidden=(16,), seed=1)), env)
        discriminator_before = copy.deepcopy(fresh.discriminator.state_dict())
        estimator_before = copy.deepcopy(fresh.estimator_optimizer.state_dict())
        self.assertFalse(fresh.load(saved, {'actor': True}, True))
        for name, value in fresh.discriminator.state_dict().items():
            torch.testing.assert_close(value, discriminator_before[name], rtol=0, atol=0, msg=name)
        self.assertEqual(fresh.estimator_optimizer.state_dict()['state'], estimator_before['state'])
        for (name, a), (_, b) in zip(fresh.actor.state_dict().items(), alg.actor.state_dict().items()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
        fresh.load(saved, {'discriminator': True}, True)
        for name, value in fresh.discriminator.state_dict().items():
            torch.testing.assert_close(value, alg.discriminator.state_dict()[name], rtol=0, atol=0, msg=name)
        self.assertEqual(fresh.estimator_optimizer.state_dict()['state'], estimator_before['state'])


class EvaluationIdentityTests(unittest.TestCase):
    """The evaluate path compares a fresh identity with a checkpoint record that passed through JSON."""

    def identity(self, learner='amp', networks='paper', seed=20260917):
        identity = {'learner': learner, 'networks': networks, 'motion_prior': learner == 'amp',
                    'upstream_source_files': {'algorithms/ppo.py': 'a' * 64},
                    'ppo_config': amp_ppo_config(seed, networks=networks) if learner == 'amp' else ppo_config(seed)}
        if learner == 'amp':
            identity['learner_source_files'] = {name: amp_ppo.sha(ROOT/'locomotion'/name) for name in (
                'amp.py', 'amp_discriminator.py', 'amp_ppo.py', 'paper_networks.py')}
            identity['amp_dataset'] = dict(BANK_IDENTITY)
        return identity

    def record(self, identity, updates=50):
        return json.loads(json.dumps({'identity': identity, 'updates': updates, 'checkpoint_sha256': 'b' * 64}))

    def accept(self, record, identity):
        train.require_learner_identity(record, identity)
        train.require_learner_configuration(record, identity, identity['ppo_config'])

    def test_learner_configuration_is_json_shaped(self):
        for networks in ('mlp', 'paper'):
            config = amp_ppo_config(1, networks=networks)
            self.assertFalse(contains_tuple(config))
            self.assertEqual(train.json_shape(config), config)
        self.assertFalse(contains_tuple(AMPVecEnv(TaskDouble()).cfg['amp']))

    def test_amp_checkpoint_record_matches_after_a_json_round_trip(self):
        for networks in ('mlp', 'paper'):
            with self.subTest(networks=networks):
                identity = self.identity(networks=networks)
                self.accept(self.record(identity), identity)

    def test_pickled_declaration_matches_its_json_record(self):
        identity = self.identity()
        record = self.record(identity)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'checkpoint.pt'
            torch.save({'infos': {'identity': identity, 'updates': 50}}, path)
            infos = torch.load(path, weights_only=False)['infos']
        train.require_embedded_declaration(infos, record)
        with self.assertRaisesRegex(ValueError, 'Embedded'):
            train.require_embedded_declaration({**infos, 'updates': 51}, record)
        tupled = copy.deepcopy(identity)
        tupled['ppo_config']['algorithm']['amp_cfg']['discriminator_hidden'] = (1024, 512)
        train.require_embedded_declaration({'identity': tupled, 'updates': 50}, record)

    def test_mismatched_learner_networks_or_bank_are_rejected(self):
        identity = self.identity()
        changes = {'learner': 'ppo', 'networks': 'mlp', 'motion_prior': False,
                   'learner_source_files': {**identity['learner_source_files'], 'amp_ppo.py': '0' * 64},
                   'amp_dataset': {**identity['amp_dataset'], 'transitions_sha256': '0' * 64}}
        for key, value in changes.items():
            with self.subTest(key=key):
                record = self.record({**identity, key: value})
                with self.assertRaisesRegex(ValueError, 'differs: ' + key):
                    train.require_learner_identity(record, identity)
        missing = self.record({key: value for key, value in identity.items() if key != 'amp_dataset'})
        with self.assertRaisesRegex(ValueError, 'differs: amp_dataset'):
            train.require_learner_identity(missing, identity)

    def test_changed_configuration_or_upstream_source_is_rejected(self):
        identity = self.identity()
        other = copy.deepcopy(identity)
        other['ppo_config']['algorithm']['amp_cfg']['style_weight'] = 2.
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            train.require_learner_configuration(self.record(other), identity, identity['ppo_config'])
        other = copy.deepcopy(identity)
        other['upstream_source_files']['algorithms/ppo.py'] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            train.require_learner_configuration(self.record(other), identity, identity['ppo_config'])
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            train.require_learner_configuration(self.record(self.identity(networks='mlp')), identity, identity['ppo_config'])

    def test_legacy_ppo_record_without_learner_keys(self):
        current = self.identity(learner='ppo', networks='mlp')
        legacy = {key: value for key, value in current.items() if key not in ('learner', 'networks')}
        self.accept(self.record(legacy), current)
        bare = {key: value for key, value in legacy.items() if key != 'motion_prior'}
        self.accept(self.record(bare), current)
        with self.assertRaisesRegex(ValueError, 'differs: learner'):
            train.require_learner_identity(self.record(legacy), self.identity())


class LaunchTests(unittest.TestCase):
    def test_prepare_binds_learner_networks_and_copies_the_bank(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = prepare(Path(temporary)/'amp', REMOTE, mode='train', reward_version='2',
                           learner='amp', networks='paper')['command_args']
            self.assertEqual(args[args.index('--learner')+1], 'amp')
            self.assertEqual(args[args.index('--networks')+1], 'paper')
            pack = json.loads((Path(temporary)/'amp/PACK.json').read_text())
            self.assertEqual((pack['learner'], pack['networks']), ('amp', 'paper'))
            bank = Path(temporary)/'amp/source/locomotion/priors/datasets/amp_demonstrations_001'
            self.assertEqual(amp_ppo.sha(bank/'transitions.npz'), amp_ppo.sha(amp_ppo.DATASET/'transitions.npz'))
            for name in ('amp.py', 'amp_discriminator.py', 'amp_ppo.py', 'paper_networks.py'):
                self.assertTrue((Path(temporary)/'amp/source/locomotion'/name).exists(), name)
            plain = prepare(Path(temporary)/'ppo', REMOTE, mode='train')
            self.assertNotIn('--learner', plain['command_args'])
            self.assertFalse((Path(temporary)/'ppo/source/locomotion/priors/datasets').exists())
            for kwargs in ({'learner': 'amp', 'mode': 'diagnostic'}, {'networks': 'paper'},
                           {'learner': 'sac'}, {'learner': 'amp', 'networks': 'rnn'}):
                with self.subTest(**kwargs), self.assertRaises(ValueError):
                    prepare(Path(temporary)/'bad', REMOTE, **{'mode': 'train', **kwargs})

    def test_train_rejects_paper_networks_without_amp_and_amp_diagnostics(self):
        base = ['--asset', 'a', '--model', 'm', '--geometry', 'g', '--geometry-extrema', 'e', '--stance', 's',
                '--output', 'o', '--source-freeze-sha256', 'f'*64, '--num-envs', '1', '--preflight-only']
        with self.assertRaisesRegex(ValueError, 'AMP learner'):
            train.main(base + ['--mode', 'train', '--networks', 'paper'])
        with self.assertRaisesRegex(ValueError, 'AMP learner'):
            train.main(base + ['--mode', 'diagnostic', '--learner', 'amp'])


if __name__ == '__main__':
    unittest.main()
