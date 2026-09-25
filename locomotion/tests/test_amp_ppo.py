"""Online AMP learner: dataset binding, transition pairing, style reward, PPO parity and checkpoints."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO

from locomotion import amp, amp_ppo, paper_networks as nets, train
from locomotion.amp_ppo import (AMPConfig, AMPPPO, AMPVecEnv, CollisionCapture, amp_ppo_config,
                                load_demonstrations)
from locomotion.ppo import VanillaVecEnv, ppo_config
from locomotion.prepare import prepare

ROOT = Path(__file__).resolve().parents[2]
REMOTE = Path('/home/orionh/HEXAPOD_runs/restart_20260914/test_amp_learner')
BODIES = ['body'] + [f'{leg}_{part}' for leg in ('lf', 'lm', 'lr', 'rf', 'rm', 'rr') for part in ('coxa', 'femur', 'tibia')]


class EnvDouble:
    """Deterministic stand-in for the native environment's outputs and telemetry."""

    def __init__(self, n, seed=0):
        self.num_envs, self.device = n, 'cpu'
        self.native_body_names = list(BODIES)
        self.reset_height = .1
        self.generator = torch.Generator().manual_seed(seed)
        self.capture = None
        self.telemetry = {}
        self.amp = torch.zeros(n, 61)
        self.cfg = SimpleNamespace(episode_seconds=20., control_dt=.02, record_motion_features=True,
                                   declaration=lambda: {'double': True})

    def draw(self, *shape):
        return torch.randn(*shape, generator=self.generator)

    def state(self):
        obs = self.draw(self.num_envs, 231)
        return {'obs': obs, 'critic': torch.cat((obs, self.draw(self.num_envs, 3)), -1), 'amp': self.amp.clone()}

    def reset(self, indices):
        self.amp[indices] = -5.
        return self.state()

    def step(self, action, terminate=None):
        self.amp = self.draw(self.num_envs, 61)
        self.telemetry = {'root_pose_xyzw': torch.cat((torch.zeros(self.num_envs, 2), torch.full((self.num_envs, 1), .09),
                                                       torch.tensor([[0., 0., 0., 1.]]).expand(self.num_envs, -1)), -1),
                          'tibia_floor_force_world_n': self.draw(self.num_envs, 6, 3)}
        output = self.state()
        terminated = torch.zeros(self.num_envs, dtype=torch.bool) if terminate is None else terminate
        output.update(reward=torch.full((self.num_envs,), .5), terminated=terminated,
                      truncated=torch.zeros(self.num_envs, dtype=torch.bool))
        return output


class TaskDouble:
    def __init__(self, n=3, seed=0):
        self.env = EnvDouble(n, seed)
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


def synthetic_rollout(alg, env, steps, seed):
    torch.manual_seed(seed)
    obs = env.get_observations()
    for _ in range(steps):
        actions = alg.act(obs)
        obs, rewards, dones, extras = env.step(actions)
        alg.process_env_step(obs, rewards, dones, extras)
    alg.compute_returns(obs)
    return alg.update()


def construct(config, env, seed):
    torch.manual_seed(seed)
    config = copy.deepcopy(config)
    config['multi_gpu'] = None
    config['num_steps_per_env'] = 4
    obs = env.get_observations()
    return PPO.construct_algorithm(obs, env, config, 'cpu')


class DatasetTests(unittest.TestCase):
    def test_admitted_bank_loads_under_the_feature_contract(self):
        transitions, identity = load_demonstrations()
        self.assertEqual(transitions.shape, (18000, 122))
        self.assertEqual(identity['path'], 'locomotion/priors/datasets/amp_demonstrations_001')
        self.assertEqual(len(identity['transitions_sha256']), 64)
        self.assertEqual(identity['reviewer'], 'palerdr')

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
        self.assertNotEqual(float(extras['amp_next'][1, 0]), -5.)
        self.assertEqual(float(obs['amp'][1, 0]), -5.)
        torch.testing.assert_close(obs['amp'][0], extras['amp_next'][0])
        self.assertTrue(bool(env.reset_rows[1]) and not bool(env.reset_rows[0]))
        self.assertEqual(obs['proprio_history'].shape, (3, 210))
        self.assertEqual(obs['command'].shape, (3, 3))
        self.assertEqual(obs['previous_action'].shape, (3, 18))

    def test_motion_features_are_required(self):
        task = TaskDouble()
        task.env.cfg.record_motion_features = False
        task.env.state = lambda: {'obs': torch.zeros(3, 231), 'critic': torch.zeros(3, 234)}
        with self.assertRaisesRegex(ValueError, 'record_motion_features'):
            AMPVecEnv(task)

    def test_collision_capture_forwards_to_the_inner_capture(self):
        env = EnvDouble(2)
        calls = []
        inner = lambda *args: calls.append(args[9])
        capture = CollisionCapture(env, inner)
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
    def test_parity_with_stock_ppo_when_style_and_discriminator_are_off(self):
        off = AMPConfig(style_weight=0., discriminator_updates=0)
        stock = construct(ppo_config(3), VanillaVecEnv(TaskDouble(seed=7)), 11)
        ours = construct(amp_ppo_config(3, amp_config=off), AMPVecEnv(TaskDouble(seed=7)), 11)
        self.assertIsInstance(ours, AMPPPO)
        stock_losses = synthetic_rollout(stock, VanillaVecEnv(TaskDouble(seed=9)), 4, 5)
        our_losses = synthetic_rollout(ours, AMPVecEnv(TaskDouble(seed=9)), 4, 5)
        for key in ('value', 'surrogate', 'entropy'):
            self.assertAlmostEqual(stock_losses[key], our_losses[key], places=6)
        for (name, a), (_, b) in zip(stock.actor.state_dict().items(), ours.actor.state_dict().items()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
        for (name, a), (_, b) in zip(stock.critic.state_dict().items(), ours.critic.state_dict().items()):
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
        self.assertEqual(our_losses['style_reward_mean'], 0.)

    def test_style_reward_is_added_and_discriminator_learns_the_bank(self):
        config = AMPConfig(discriminator_updates=8, discriminator_hidden=(64,), discriminator_learning_rate=1e-3)
        env = AMPVecEnv(TaskDouble(seed=2))
        alg = construct(amp_ppo_config(1, amp_config=config), env, 1)
        obs = env.get_observations()
        actions = alg.act(obs)
        obs, rewards, dones, extras = env.step(actions)
        alg.process_env_step(obs, rewards, dones, extras)
        torch.testing.assert_close(alg.storage.rewards[0].flatten(), rewards + alg.style_rewards)
        self.assertTrue(bool(((alg.style_rewards >= 0) & (alg.style_rewards <= 1)).all()))
        bank = alg.demonstrations[:512]
        alg.storage.clear()
        for _ in range(3):
            losses = synthetic_rollout(alg, env, 4, 3)
        with torch.no_grad():
            bank_style = float(amp.style_reward(alg.discriminator(bank)).mean())
            policy_style = float(amp.style_reward(alg.discriminator(alg.policy_transitions.reshape(-1, 122))).mean())
        self.assertGreater(bank_style, policy_style)
        self.assertGreater(bank_style, .5)
        for key in ('discriminator_prior', 'discriminator_policy', 'discriminator_gradient_penalty', 'style_reward_mean'):
            self.assertIn(key, losses)
        self.assertNotIn('velocity_estimator', losses)

    def test_paper_networks_train_the_estimator_after_ppo(self):
        config = AMPConfig(discriminator_updates=1, discriminator_hidden=(32,))
        env = AMPVecEnv(TaskDouble(seed=4))
        alg = construct(amp_ppo_config(2, networks='paper', amp_config=config), env, 2)
        self.assertIsInstance(alg.actor, nets.PaperActor)
        self.assertIsNotNone(alg.estimator_optimizer)
        before = [p.detach().clone() for p in alg.actor.estimator_parameters()]
        losses = synthetic_rollout(alg, env, 4, 6)
        self.assertIn('velocity_estimator', losses)
        self.assertTrue(any(float((a - b).abs().max()) > 0 for a, b in zip(before, alg.actor.estimator_parameters())))

    def test_checkpoint_round_trip_binds_the_bank(self):
        config = AMPConfig(discriminator_updates=1, discriminator_hidden=(16,))
        env = AMPVecEnv(TaskDouble(seed=5))
        alg = construct(amp_ppo_config(4, networks='paper', amp_config=config), env, 4)
        synthetic_rollout(alg, env, 4, 8)
        saved = alg.save()
        for key in ('discriminator_state_dict', 'discriminator_optimizer_state_dict',
                    'estimator_optimizer_state_dict', 'amp_dataset_identity'):
            self.assertIn(key, saved)
        fresh = construct(amp_ppo_config(4, networks='paper', amp_config=config), env, 99)
        fresh.load(saved, None, True)
        with torch.no_grad():
            torch.testing.assert_close(fresh.discriminator(alg.demonstrations[:8]), alg.discriminator(alg.demonstrations[:8]))
        foreign = copy.deepcopy(saved)
        foreign['amp_dataset_identity']['transitions_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'bank differs'):
            fresh.load(foreign, None, True)

    def test_config_and_declaration_record_decisions(self):
        config = amp_ppo_config(0, networks='paper')
        self.assertEqual(config['algorithm']['class_name'], 'locomotion.amp_ppo:AMPPPO')
        self.assertEqual(config['actor']['class_name'], 'locomotion.paper_networks:PaperActor')
        self.assertEqual(config['obs_groups']['critic'][-1], 'privileged')
        self.assertEqual(config['networks'], 'paper')
        self.assertEqual(amp_ppo_config(0)['actor']['class_name'], 'MLPModel')
        with self.assertRaises(ValueError):
            amp_ppo_config(0, networks='rnn')
        alg = construct(amp_ppo_config(0, amp_config=AMPConfig(discriminator_hidden=(8,))), AMPVecEnv(TaskDouble()), 0)
        declaration = alg.declaration()
        self.assertEqual(declaration['feature_contract'], amp.feature_contract())
        self.assertFalse(declaration['stage2_complete'])
        self.assertTrue(declaration['learner_decisions'] and declaration['network_decisions'])


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
            self.assertTrue((Path(temporary)/'amp/source/locomotion/amp_ppo.py').exists())
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
