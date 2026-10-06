"""Check resume compatibility and checkpoint lineage with short CPU training runs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from locomotion.surrogate import train
from locomotion.surrogate.tests.test_learner import run, train_sha


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.code, cls.output, cls.state = run(cls.temporary.name, 'parent', '--action-std-final', '0.05')
        cls.checkpoint = cls.output / 'checkpoint_update000001.pt'

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_resume_pins_parent_and_keeps_counts_and_original_deviation_schedule(self):
        self.assertEqual(self.code, 0, self.state.get('traceback'))
        code, output, state = run(self.temporary.name, 'child', '--resume', str(self.checkpoint),
                                  '--action-std-final', '0.05')
        self.assertEqual(code, 0, state.get('traceback'))
        self.assertEqual((state['updates'], state['transitions']), (2, 192))
        lineage = state['identity']['resume']
        self.assertEqual(lineage['checkpoint_sha256'], train_sha(self.checkpoint))
        self.assertEqual(lineage['record_sha256'], train_sha(self.checkpoint.with_suffix('.json')))
        self.assertEqual(lineage['identity'], self.state['identity'])
        self.assertEqual((lineage['updates'], lineage['transitions']), (1, 96))
        row = json.loads((output / 'metrics.jsonl').read_text())
        self.assertAlmostEqual(row['mean_action_std'], .05, places=6)
        previous = json.loads((self.output / 'metrics.jsonl').read_text())
        self.assertEqual(row['policy_update']['learning_rate_before'], previous['policy_update']['learning_rate_after'])
        code, _, state = run(self.temporary.name, 'grandchild', '--resume', str(output / 'checkpoint_update000002.pt'),
                              '--action-std-final', '0.05')
        self.assertEqual(code, 0, state.get('traceback'))
        self.assertEqual((state['updates'], state['transitions']), (3, 288))

    def test_changed_training_contract_fails_before_an_update(self):
        variants = [('--action-mean', 'tanh'), ('--observation-scaling', 'fixed'), ('--num-envs', '8'),
                    ('--task', 'locomotion.task_v3:TrainingTaskV3'), ('--contact', 'noslip_iterations=0'),
                    ('--ppo-config-overrides', '{"num_steps_per_env":12}'), ('--episode-seconds', '10')]
        for index, arguments in enumerate(variants):
            with self.subTest(arguments=arguments):
                code, output, state = run(self.temporary.name, f'mismatch{index}', '--resume', str(self.checkpoint),
                                          '--action-std-final', '0.05', *arguments)
                self.assertEqual(code, 1)
                self.assertIn('Resume configuration differs', state['errors'][0])
                self.assertFalse((output / 'metrics.jsonl').exists())
                self.assertEqual(list(output.glob('checkpoint*.pt')), [])

    def test_source_mismatch_and_tampered_records_are_rejected(self):
        identity = copy.deepcopy(self.state['identity'])
        identity['source_files']['ppo.py'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'source_files'):
            train.resume_record(self.checkpoint, identity)
        identity = self.state['identity']
        for field, value, message in (('checkpoint_sha256', '0' * 64, 'hash'),
                                      ('transitions', 999, 'transition'), ('updates', 2, 'metadata')):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'checkpoint.pt'
                path.write_bytes(self.checkpoint.read_bytes())
                record = json.loads(self.checkpoint.with_suffix('.json').read_text())
                record[field] = value
                if field == 'updates':
                    record['transitions'] = 192
                path.with_suffix('.json').write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, message):
                    train.resume_record(path, identity)


if __name__ == '__main__':
    unittest.main()
