"""Check the extended-update opt-in and the exploration decay length without native simulation."""
import json
from pathlib import Path
import tempfile
import unittest

from locomotion import ppo, train
from locomotion.prepare import prepare
from locomotion.tests.test_learner_options import construct

REMOTE = '/srv/cupi/hexapod/runs/james/training_budget_fixture'
REPO = Path(__file__).resolve().parents[2]
ASSET = REPO/'robot/hexapod_mkii_updated_v1/usd'
MODEL = REPO/'robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json'
# The retained reward v4 full-command-bank learner options, as recorded in the reference packs.
REFERENCE = dict(reward_version='4', action_mean='tanh', observation_normalization='none',
                 observation_scaling='fixed', command_segments='bootstrap', learning_rate_max=3e-4,
                 action_std_final=.05, gait_clock=60, action_smoothing='mean2', velocity_noise=.5)


def entry(*extra):
    return ['--asset', str(ASSET), '--model', str(MODEL), '--geometry', 'g', '--geometry-extrema', 'e',
            '--stance', 's', '--output', 'o', '--source-freeze-sha256', 'f'*64, '--num-envs', '128',
            '--mode', 'train', '--preflight-only', *extra]


def schedule_values(start, final, decay, updates):
    """The deviation each update trains with, read through DeviationSchedule."""
    algorithm = construct(ppo.ppo_config(3, action_mean='tanh', observation_normalization='none'))
    seen = []
    algorithm.update = lambda: seen.append(float(algorithm.actor.distribution.log_std_param.exp()[0])) or {}
    schedule = ppo.DeviationSchedule(algorithm, start, final, decay)
    for _ in range(updates):
        schedule.update()
    return seen


class UpdateLimitTests(unittest.TestCase):
    def test_training_keeps_2000_updates_without_the_opt_in(self):
        self.assertEqual((train.UPDATE_LIMIT, train.EXTENDED_UPDATE_LIMIT), (2000, 5000))
        self.assertEqual(train.update_limit('train', False), 2000)
        self.assertEqual(train.update_limit('evaluate', False), 2000)
        self.assertEqual(train.update_limit('train', True), 5000)
        for mode in ('evaluate', 'diagnostic', 'probe', 'throughput', 'tripod'):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'training only'):
                train.update_limit(mode, True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plain = prepare(root/'plain', REMOTE, mode='train', updates=2000)
            self.assertNotIn('--extended-updates', plain['command_args'])
            self.assertNotIn('extended_updates', json.loads((root/'plain/PACK.json').read_text()))
            for updates in (2001, 5000):
                with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, 'Invalid native allocation'):
                    prepare(root/f'over{updates}', REMOTE, mode='train', updates=updates)
                self.assertFalse((root/f'over{updates}').exists())
            extended = prepare(root/'extended', REMOTE, mode='train', updates=5000, extended_updates=True)
            args = extended['command_args']
            self.assertEqual(args[args.index('--updates')+1], '5000')
            self.assertEqual(args.count('--extended-updates'), 1)
            pack = json.loads((root/'extended/PACK.json').read_text())
            self.assertEqual((pack['updates'], pack['extended_updates']), (5000, True))
            for index, bad in enumerate((dict(mode='train', updates=5001, extended_updates=True),
                                         dict(mode='train', updates=10, extended_updates=1),
                                         dict(mode='diagnostic', updates=10, extended_updates=True))):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    prepare(root/f'bad{index}', REMOTE, **bad)
            checkpoint = dict(checkpoint=REMOTE+'/checkpoint.pt', checkpoint_sha='a'*64, checkpoint_declaration_sha='b'*64)
            with self.assertRaisesRegex(ValueError, 'training only'):
                prepare(root/'evaluate', REMOTE, mode='evaluate', updates=2000, extended_updates=True, **checkpoint)

    def test_entry_rejects_5000_updates_without_the_opt_in(self):
        with self.assertRaisesRegex(ValueError, 'bounded scratch training'):
            train.main(entry('--updates', '5000', '--device', 'cuda:0', '--headless'))
        with self.assertRaisesRegex(ValueError, 'bounded scratch training'):
            train.main(entry('--updates', '5001', '--extended-updates', '--device', 'cuda:0', '--headless'))
        # With the opt-in the update check passes; the preflight then stops at the absent frozen source record.
        with self.assertRaises(FileNotFoundError) as caught:
            train.main(entry('--updates', '5000', '--extended-updates', '--device', 'cuda:0', '--headless'))
        self.assertIn('FREEZE_SHA256.json', str(caught.exception))
        evaluate = [x if x != 'train' else 'evaluate' for x in entry('--updates', '2000', '--extended-updates')]
        with self.assertRaisesRegex(ValueError, 'training only'):
            train.main(evaluate)


class DecayLengthTests(unittest.TestCase):
    def test_default_decay_spans_the_run_and_configuration_is_unchanged(self):
        self.assertEqual(train.decay_updates(2000, None), 2000)
        self.assertEqual(train.decay_updates(5000, 2000), 2000)
        base = ppo.ppo_config(3, action_mean='tanh', observation_normalization='none', action_std_final=.05)
        self.assertEqual(base['exploration'], {'action_std_final': .05})
        decoupled = ppo.ppo_config(3, action_mean='tanh', observation_normalization='none', action_std_final=.05,
                                   action_std_decay_updates=2000)
        self.assertEqual(decoupled['exploration'], {'action_std_final': .05, 'action_std_decay_updates': 2000})
        self.assertEqual({k: v for k, v in decoupled.items() if k != 'exploration'},
                         {k: v for k, v in base.items() if k != 'exploration'})
        for bad in (dict(action_std_decay_updates=2000), dict(action_std_final=.05, action_std_decay_updates=0),
                    dict(action_std_final=.05, action_std_decay_updates=2000.)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ppo.ppo_config(3, **bad)

    def test_decoupled_schedule_matches_the_original_through_its_decay_then_holds(self):
        original = schedule_values(.15, .05, 20, 20)
        decoupled = schedule_values(.15, .05, 20, 50)
        self.assertEqual(decoupled[:20], original)
        self.assertAlmostEqual(original[0], .15, places=6)
        self.assertAlmostEqual(original[10], .10, places=6)
        self.assertTrue(all(abs(value - .05) < 1e-6 for value in decoupled[20:]))
        # A decay across the whole longer run would leave a larger deviation at the original decay end.
        coupled = schedule_values(.15, .05, 50, 50)
        self.assertGreater(coupled[20], decoupled[20] + .02)

    def test_packs_record_the_decay_for_training_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = prepare(root/'reference', REMOTE, mode='train', updates=2000, **REFERENCE)
            self.assertNotIn('--action-std-decay-updates', reference['command_args'])
            self.assertNotIn('action_std_decay_updates', json.loads((root/'reference/PACK.json').read_text()))
            binding = prepare(root/'long', REMOTE, mode='train', updates=5000, extended_updates=True,
                              action_std_decay_updates=2000, **REFERENCE)
            args = binding['command_args']
            self.assertEqual(args[args.index('--action-std-decay-updates')+1], '2000')
            self.assertEqual(args[args.index('--action-std-final')+1], '0.05')
            pack = json.loads((root/'long/PACK.json').read_text())
            self.assertEqual((pack['updates'], pack['action_std_decay_updates'], pack['action_std_final']), (5000, 2000, .05))
            # The decoupled pack differs from the reference only by the two declared options and the update count.
            stripped = [x for x in args if x not in ('--extended-updates',)]
            position = stripped.index('--action-std-decay-updates')
            del stripped[position:position+2]
            reference_args = list(reference['command_args'])
            reference_args[reference_args.index('--updates')+1] = '5000'
            freeze = reference_args.index('--source-freeze-sha256')+1
            reference_args[freeze] = stripped[freeze]
            self.assertEqual(stripped, reference_args)
            evaluation = {key: value for key, value in REFERENCE.items() if key != 'reward_version'}
            evaluated = prepare(root/'evaluate', REMOTE, mode='evaluate', updates=2000, action_std_decay_updates=2000,
                                checkpoint=REMOTE+'/checkpoint.pt', checkpoint_sha='a'*64,
                                checkpoint_declaration_sha='b'*64, **evaluation)['command_args']
            self.assertEqual(evaluated[evaluated.index('--action-std-decay-updates')+1], '2000')
            for index, bad in enumerate((dict(mode='train', updates=2000, action_std_decay_updates=2000),
                                         dict(mode='train', updates=1000, action_std_final=.05, action_std_decay_updates=2000),
                                         dict(mode='train', updates=2000, action_std_final=.05, action_std_decay_updates=0))):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    prepare(root/f'bad{index}', REMOTE, **bad)

    def test_entry_requires_a_final_deviation_and_a_decay_inside_the_run(self):
        with self.assertRaisesRegex(ValueError, 'final action deviation'):
            train.main(entry('--updates', '2000', '--action-std-decay-updates', '2000'))
        with self.assertRaisesRegex(ValueError, 'decay length'):
            train.main(entry('--updates', '1000', '--action-std-final', '0.05', '--action-std-decay-updates', '2000'))


if __name__ == '__main__':
    unittest.main()
