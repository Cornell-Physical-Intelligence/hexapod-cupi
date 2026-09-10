import copy
import json
from pathlib import Path
import unittest

from analyze import summarize

RAW = Path(__file__).resolve().parent.parent / 'direct_omni_train_smoke003_terminal_001/run/train/training_receipt.json'

class ReviewChecks(unittest.TestCase):
    def test_actual40_lr_rows_and_two_gradient_rows(self):
        report = summarize(json.loads(RAW.read_text()))
        self.assertEqual(len(report['sparse_gradients']), 2)
        self.assertEqual(report['kl_above_0_02_all40'], 35)
        self.assertEqual(sum(u['lr_after_exact_literal_floor'] for u in report['updates']), 31)
        self.assertEqual(sum(u['lr_after_floor_with_1e_minus12_relative_tolerance'] for u in report['updates']), 32)
        self.assertGreater(report['sparse_gradients'][0]['quiet_dot_total_gradient_interval'][0], 0)
        self.assertLess(report['sparse_gradients'][1]['quiet_dot_total_gradient_interval'][1], 0)

    def test_wrong_learning_rate_or_missing_sparse_sample_rejected(self):
        raw = json.loads(RAW.read_text())
        wrong = copy.deepcopy(raw)
        wrong['optimizer_updates'][0]['minibatches'][1]['learning_rate_after'] = 5e-5
        with self.assertRaises(AssertionError):
            summarize(wrong)
        missing = copy.deepcopy(raw)
        missing['optimizer_updates'][0]['minibatches'][19]['gradient'] = None
        with self.assertRaises(AssertionError):
            summarize(missing)

    def test_mode_counts_and_order_are_checked(self):
        raw = json.loads(RAW.read_text())
        wrong = copy.deepcopy(raw)
        wrong['optimizer_updates'][0]['minibatches'][1]['pair_counts']['quiet_pairs'] += 1
        with self.assertRaises(AssertionError):
            summarize(wrong)
        swapped = copy.deepcopy(raw)
        rows = swapped['optimizer_updates'][1]['minibatches']
        rows[1], rows[2] = rows[2], rows[1]
        with self.assertRaises(AssertionError):
            summarize(swapped)

if __name__ == '__main__':
    unittest.main()
