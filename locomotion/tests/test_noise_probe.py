"""Check the fixed-mean noise probe plan, record and packaging without a simulator."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from locomotion import noise_probe
from locomotion.prepare import prepare
from locomotion.tests.test_task_v2 import RewardDouble

REMOTE = '/srv/cupi/hexapod/runs/james/noise_probe_fixture'


class ProbeDouble(RewardDouble):
    """Records each received action; replica 5 terminates at its third control."""

    def __init__(self, n=128):
        super().__init__(n)
        self.current['linear'] = torch.zeros(n, 3)
        self.actions, self.resets = [], []

    def reset(self, indices=None):
        indices = torch.arange(self.num_envs) if indices is None else indices
        self.resets.append(indices.tolist())
        return super().reset(indices)

    def step(self, action):
        output = super().step(action)
        self.actions.append(action.clone())
        self.telemetry.update(tibia_floor_force_world_n=torch.zeros(self.num_envs, 6, 3),
                              tibia_floor_force_min_norm_400hz=torch.zeros(self.num_envs, 6))
        output['terminated'] = torch.arange(self.num_envs) == 5 if len(self.actions) == 3 else output['terminated']
        return output


class NoiseProbeTests(unittest.TestCase):
    def test_plan_crosses_each_pose_with_each_deviation(self):
        rows = noise_probe.plan(128)
        cells = {(row['pose'], row['standard_deviation']) for row in rows}
        self.assertEqual(len(cells), len(noise_probe.POSES) * len(noise_probe.STANDARD_DEVIATIONS))
        for cell in cells:
            self.assertEqual(sum((row['pose'], row['standard_deviation']) == cell for row in rows), 8)
        self.assertEqual(rows[0]['mean'], [0.]*18)
        self.assertEqual(rows[-1]['mean'], [0., -1.5, 0.]*6)
        self.assertEqual(set(noise_probe.POSE_SCOPE), set(noise_probe.POSES))
        with self.assertRaisesRegex(ValueError, '128 replicas'):
            noise_probe.plan(32)

    def test_run_holds_means_scales_noise_and_keeps_fallen_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            env = ProbeDouble()
            summary = noise_probe.run(env, Path(directory)/'probe', seed=11)
            actions = torch.stack(env.actions)
            self.assertEqual(actions.shape, (noise_probe.CONTROLS, 128, 18))
            # Zero-deviation cells receive their exact mean on every control.
            for row in noise_probe.plan(128):
                column = actions[:, row['replica']]
                if row['standard_deviation'] == 0:
                    torch.testing.assert_close(column, torch.tensor(row['mean']).expand_as(column), rtol=0, atol=0)
                else:
                    self.assertAlmostEqual(float((column - torch.tensor(row['mean'])).std()), row['standard_deviation'], delta=.01)
            self.assertEqual(env.resets, [list(range(128)), [5]])
            self.assertEqual(float((env.commands != 0).sum()), 0.)
            with np.load(Path(directory)/'probe/telemetry.npz') as saved:
                self.assertEqual(set(saved.files), set(noise_probe.FIELDS) | {'terminated', 'truncated'})
                self.assertEqual(saved['action'].shape, (noise_probe.CONTROLS, 128, 18))
                self.assertEqual(int(saved['terminated'].sum()), 1)
                np.testing.assert_array_equal(saved['action'], actions.numpy())
            self.assertEqual((summary['terminations'], len(summary['cells'])), (1, 16))
            record = json.loads((Path(directory)/'probe/plan.json').read_text())
            self.assertIsNone(record['optimizer'])
            self.assertFalse(record['stage2_complete'])
            repeat = ProbeDouble()
            noise_probe.run(repeat, Path(directory)/'repeat', seed=11)
            torch.testing.assert_close(torch.stack(repeat.actions), actions, rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                noise_probe.run(ProbeDouble(), Path(directory)/'probe', seed=11)

    def test_pack_runs_the_training_entry_with_128_replicas_and_no_learner_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            binding = prepare(Path(directory)/'probe', REMOTE, mode='probe', seed=7)
            args = binding['command_args']
            self.assertEqual((binding['mode'], binding['module']), ('probe', 'locomotion.train'))
            self.assertEqual(args[args.index('--num-envs')+1], '128')
            self.assertEqual(args[args.index('--seed')+1], '7')
            self.assertNotIn('--updates', args)
            self.assertIn('locomotion/noise_probe.py', json.loads((Path(directory)/'probe/source/FREEZE_SHA256.json').read_text()))
            for index, bad in enumerate((dict(num_envs=32), dict(reward_version='4'), dict(action_mean='tanh'),
                                         dict(observation_scaling='fixed'))):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    prepare(Path(directory)/f'bad{index}', REMOTE, mode='probe', **bad)


if __name__ == '__main__':
    unittest.main()
