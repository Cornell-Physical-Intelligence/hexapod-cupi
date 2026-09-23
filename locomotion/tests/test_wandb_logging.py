"""Check opt-in W&B training arguments without starting Isaac, Docker or W&B."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from locomotion import train
from locomotion.prepare import prepare

ARGS = ['--asset', 'a', '--model', 'm', '--geometry', 'g', '--geometry-extrema', 'e', '--stance', 's',
        '--output', 'o', '--source-freeze-sha256', 'f'*64, '--num-envs', '128', '--preflight-only']
ROOT = '/home/orionh/HEXAPOD_runs/restart_20260914/wandb_fixture'


class WandbLoggingTests(unittest.TestCase):
    def test_train_rejects_incomplete_or_misplaced_wandb_options(self):
        for extra in (['--mode', 'evaluate', '--logger', 'wandb', '--wandb-project', 'hexapod'],
                      ['--mode', 'train', '--logger', 'wandb'],
                      ['--mode', 'train', '--logger', 'wandb', '--wandb-project', 'bad name'],
                      ['--mode', 'train', '--wandb-project', 'hexapod'],
                      ['--mode', 'train', '--wandb-mode', 'online']):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, 'W&B'):
                train.main(ARGS+extra)
        online = ARGS+['--mode', 'train', '--logger', 'wandb', '--wandb-project', 'hexapod', '--wandb-mode', 'online']
        with patch('importlib.util.find_spec', return_value=object()), patch.dict(os.environ, {'WANDB_API_KEY': ''}):
            with self.assertRaisesRegex(ValueError, 'W&B'):
                train.main(online)

    def test_valid_wandb_options_pass_to_asset_verification(self):
        with patch('importlib.util.find_spec', return_value=object()), patch.dict(os.environ, {'WANDB_API_KEY': 'k'}):
            for mode in ('offline', 'online'):
                with self.subTest(mode=mode), self.assertRaises(Exception) as caught:
                    train.main(ARGS+['--mode', 'train', '--logger', 'wandb', '--wandb-project', 'hexapod',
                                     '--wandb-mode', mode])
                self.assertNotIn('W&B', str(caught.exception))

    def test_task_scalars_keep_finite_numbers_only(self):
        status = {'a': 1, 'b': {'c': .5, 'd': None, 'e': 'text', 'f': [1., 2.], 'g': float('nan'), 'h': True}}
        self.assertEqual(train.scalars(status, 'Task'), {'Task/a': 1, 'Task/b/c': .5})
        self.assertEqual(train.ConfigRecord({'x': 1}).to_dict(), {'x': 1})

    def test_prepare_binds_wandb_arguments_for_training_only(self):
        with tempfile.TemporaryDirectory() as temp:
            binding = prepare(Path(temp)/'train', ROOT, mode='train', logger='wandb', wandb_project='hexapod-amp')
            args = binding['command_args']
            self.assertEqual(args[args.index('--logger'):args.index('--logger')+6],
                             ['--logger', 'wandb', '--wandb-project', 'hexapod-amp', '--wandb-mode', 'offline'])
            self.assertNotIn('--logger', prepare(Path(temp)/'plain', ROOT, mode='train')['command_args'])
            for bad, options in (('diagnostic', {'logger': 'wandb', 'wandb_project': 'hexapod-amp'}),
                                 ('train', {'logger': 'wandb'}),
                                 ('train', {'wandb_project': 'hexapod-amp'}),
                                 ('train', {'logger': 'wandb', 'wandb_project': 'hexapod-amp', 'wandb_mode': 'disabled'})):
                with self.subTest(mode=bad, options=options), self.assertRaises(ValueError):
                    prepare(Path(temp)/'bad', ROOT, mode=bad, **options)


if __name__ == '__main__':
    unittest.main()
