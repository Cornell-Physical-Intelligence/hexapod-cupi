import copy
import json
from pathlib import Path
import unittest

from analyze import summarize

RAW = Path(__file__).resolve().parent.parent / 'direct_omni_quiet_pilot_training_observation_001/training_receipt.json'

class PilotReview(unittest.TestCase):
    def test_actual50_updates1000_rows_eight_gradients(self):
        d = summarize(json.loads(RAW.read_text()))
        self.assertEqual(len(d['updates']), 50)
        self.assertEqual(len(d['sparse_gradients']), 8)
        self.assertEqual(d['kl_above_0_02_all1000'], 108)
        negative = [(g['update'], g['minibatch']) for g in d['sparse_gradients'] if g['quiet_dot_total_gradient_interval'][1] < 0]
        self.assertEqual(negative, [(10, 20), (25, 20), (50, 1)])

    def test_missing_late_gradient_or_changed_lr_rejects(self):
        raw = json.loads(RAW.read_text())
        d = copy.deepcopy(raw)
        d['optimizer_updates'][49]['minibatches'][19]['gradient'] = None
        with self.assertRaises(AssertionError):
            summarize(d)
        d = copy.deepcopy(raw)
        d['optimizer_updates'][20]['minibatches'][2]['learning_rate_after'] = .001
        with self.assertRaises(AssertionError):
            summarize(d)

    def test_mode_count_corruption_rejects(self):
        d = json.loads(RAW.read_text())
        d['optimizer_updates'][24]['minibatches'][10]['pair_counts']['moving_pairs'] += 1
        with self.assertRaises(AssertionError):
            summarize(d)

if __name__ == '__main__':
    unittest.main()
