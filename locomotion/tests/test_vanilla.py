"""Check terminal transitions, load averaging and immutable baseline packaging."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from locomotion.prepare import ROOT
from locomotion.env_config import sha as digest
from locomotion.prepare import prepare
from locomotion.ppo import TrainingLoads, VanillaVecEnv


class Task:
    num_envs = 3
    device = 'cpu'
    cfg = SimpleNamespace(episode_seconds=20., control_dt=.02, declaration=lambda: {})
    episode_steps = torch.zeros(3, dtype=torch.long)

    def __init__(self):
        self.value = torch.zeros(3, 2)
        self.resets = []

    def declaration(self):
        return {}

    def output(self):
        return {'obs': self.value.clone(), 'critic': self.value.clone()+10}

    def reset(self, selected=None):
        selected = torch.arange(3) if selected is None else selected
        self.resets.append(selected.tolist())
        self.value[selected] = 0
        return self.output()

    def step(self, action):
        self.value += 1
        return {**self.output(), 'reward': torch.tensor([2., -7., 5.]),
            'terminated': torch.tensor([False, True, False]),
            'truncated': torch.tensor([False, True, True])}


class VanillaTests(unittest.TestCase):
    def test_selected_resets_preserve_terminal_rewards_and_running_observation(self):
        task = Task()
        wrapped = VanillaVecEnv(task, tensor_dict=lambda value, **kw: value)
        obs, reward, done, extras = wrapped.step(torch.zeros(3, 18))
        self.assertEqual(task.resets, [[0, 1, 2], [1, 2]])
        self.assertEqual(obs['policy'].tolist(), [[1., 1.], [0., 0.], [0., 0.]])
        self.assertEqual(reward.tolist(), [2., -7., 5.])
        self.assertEqual(done.tolist(), [0, 1, 1])
        self.assertEqual(extras['time_outs'].tolist(), [False, False, True])

    def test_substep_loads_keep_swing_zeros_torque_signs_and_episode_windows(self):
        env = SimpleNamespace(device='cpu', native_body_names=['body', 'lf_tibia'],
            joint_names=['joint'], commands=torch.tensor([[.05, 0, 0], [0, 0, 0.]]),
            episode_steps=torch.tensor([99, 100]))
        loads = TrainingLoads(env)
        forces = torch.tensor([[[0., 0, 0], [0., 0, 20]], [[0., 0, 0], [0., 0, 0]]])
        for sign in (-1, 1):
            loads(env, None, None, None, torch.tensor([[4.], [0.]]), None, None, None,
                  forces, 0, torch.tensor([[2.*sign], [0.]]))
            env.episode_steps += 1
        report = loads.report()
        full = report['windows']['full_training']
        motion = report['windows']['commanded_locomotion_after_settle']
        self.assertEqual(full['samples_across_replicas'], 4)
        self.assertEqual(full['metrics']['total_vertical_support_n']['mean'], 10.)
        self.assertEqual(full['metrics']['applied_abs_torque_nm:joint']['mean'], 1.)
        self.assertAlmostEqual(full['metrics']['applied_abs_torque_nm:joint']['rms'], 2**.5)
        self.assertEqual(motion['samples_across_replicas'], 1)
        self.assertEqual(motion['metrics']['total_vertical_support_n']['mean'], 20.)

    def test_fresh_pack_keeps_admitted_physics_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'pack'
            remote = '/home/orionh/HEXAPOD_runs/restart_20260914/test_vanilla'
            binding = prepare(output, remote, updates=2)
            source = output/'source'
            manifest = json.loads((source/'FREEZE_SHA256.json').read_text())
            for name, expected in manifest.items():
                self.assertEqual(digest(source/name), expected)
            for name in ('env.py', 'env_config.py', 'task.py', 'evaluate.py', 'launch.py', 'reservation.py'):
                self.assertEqual(digest(source/'locomotion'/name), digest(ROOT/'locomotion'/name))
            self.assertEqual(binding['mode'], 'train')
            self.assertNotIn('--checkpoint', binding['command_args'])
            with self.assertRaises(FileExistsError):
                prepare(output, remote)
            with self.assertRaises(ValueError):
                prepare(Path(tmp)/'bad', remote, checkpoint_sha='a'*64)


if __name__ == '__main__':
    unittest.main()
