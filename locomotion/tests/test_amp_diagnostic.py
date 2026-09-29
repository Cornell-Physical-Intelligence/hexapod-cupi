"""The offline discriminator diagnostic binds its numbers to inputs and source, and is repeatable."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from locomotion import amp_diagnostic, amp_discriminator as disc, amp_ppo

ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT/'locomotion/tests/fixtures/control_trace.npz'
SMALL = amp_diagnostic.DiagnosticConfig(steps=20, batch=64, hidden=(16,), evaluation_rows=200)
SETS = {'bank_training', 'bank_held_out', 'policy_trace_held_out', 'standing_held_out'}


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = amp_diagnostic.report(TRACE, SMALL, command='test')

    def test_report_binds_inputs_source_and_scope(self):
        report = self.report
        self.assertEqual(report['schema'], amp_diagnostic.SCHEMA)
        self.assertFalse(report['native_evidence'] or report['stage2_complete'])
        self.assertEqual(report['inputs']['policy_trace'],
                         {'path': 'locomotion/tests/fixtures/control_trace.npz', 'sha256': amp_ppo.sha(TRACE),
                          'transitions': 1000})
        self.assertEqual(report['inputs']['demonstration_bank'], amp_ppo.load_demonstrations()[1])
        for name, digest in report['source_files'].items():
            self.assertEqual(digest, amp_ppo.sha(ROOT/name), name)
        self.assertEqual(set(report['source_files']), {'locomotion/'+name for name in amp_diagnostic.SOURCE_FILES})
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False)), report)

    def test_three_variants_score_the_same_held_out_sets(self):
        variants = self.report['results']['variants']
        self.assertEqual(list(variants), ['raw_input_penalty', 'standardized_input_penalty',
                                          'standardized_input_penalty_with_standing_negatives'])
        for variant in variants.values():
            self.assertEqual(set(variant['mean_style_reward']), SETS)
            self.assertTrue(all(0. <= value <= 1. for value in variant['mean_style_reward'].values()))
        rows = self.report['results']['rows']
        self.assertEqual(rows['bank_training'] + rows['bank_held_out'], 18000)
        self.assertEqual(rows['policy_trace_training'] + rows['policy_trace_held_out'], 1000)
        self.assertEqual(variants['standardized_input_penalty_with_standing_negatives']['negatives'],
                         rows['policy_trace_training'] + rows['bank_training'])

    def test_raw_input_penalty_starts_larger_on_the_bank(self):
        variants = self.report['results']['variants']
        raw = variants['raw_input_penalty']['first_update_loss_terms']['gradient_penalty']
        standardized = variants['standardized_input_penalty']['first_update_loss_terms']['gradient_penalty']
        self.assertGreater(raw, 100 * standardized)
        spread = self.report['results']['bank_feature_spread']
        self.assertEqual(spread['features'], 61)
        self.assertLess(spread['min_std'], spread['std_floor'])
        self.assertEqual(set(spread['fields']), {'joint_position', 'joint_velocity', 'linear_velocity',
                                                 'angular_velocity', 'height', 'toe_position'})

    def test_report_is_repeatable(self):
        again = amp_diagnostic.report(TRACE, SMALL, command='test')
        self.assertEqual(again['results'], self.report['results'])

    def test_standing_transitions_hold_pose_with_zero_velocity(self):
        states = torch.arange(122.).reshape(2, 61) + 1
        pairs = amp_diagnostic.standing(states)
        torch.testing.assert_close(pairs[:, :61], pairs[:, 61:])
        self.assertTrue(bool((pairs[:, 18:42] == 0).all()))
        torch.testing.assert_close(pairs[:, :18], states[:, :18])
        torch.testing.assert_close(pairs[:, 42:61], states[:, 42:])
        self.assertEqual(pairs.shape, (2, 2 * disc.AMP_WIDTH))

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'result.json'
            output.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'fresh output'):
                amp_diagnostic.main(['--policy-trace', str(TRACE), '--output', str(output)])


if __name__ == '__main__':
    unittest.main()
