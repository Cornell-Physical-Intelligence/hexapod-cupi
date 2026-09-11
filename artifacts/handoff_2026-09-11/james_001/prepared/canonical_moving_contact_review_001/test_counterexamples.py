import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
from proposed_bridge_rule import ContactHoldFailure, enforce_hold


class Counterexamples(unittest.TestCase):
    def fixtures(self, n=1):
        return np.ones((n, 6), bool), np.ones((n, 8, 6), bool), np.ones((n, 8), bool), np.arange(8001, 8009)

    def check(self, data, phase='moving_or_stopping'):
        return enforce_hold(*data, prior_counter=8000, phase=phase)

    def test_tripod_and_stop_return_remain_allowed_but_not_quiet_pass(self):
        a, b, v, c = self.fixtures()
        b[:, :4, [1, 3, 5]] = False
        b[:, 4:6, [0, 2, 4]] = False
        # Last2 samples return all feet; no command-dependent phase switch exists.
        r = self.check((a, b, v, c))
        self.assertTrue(r['contact_rule_allows_commit'])
        self.assertFalse(r['formal_quality_pass'])
        self.assertIsNone(r['qualified_lift'])
        with self.assertRaises(ContactHoldFailure):
            self.check((a, b, v, c), 'neutral_actor_boundary')

    def test_endpoint_all_six_does_not_hide_midhold_support_violation(self):
        a, b, v, c = self.fixtures(2)
        b[0, 6, :4] = False
        b[1, 2, :5] = False
        self.assertTrue(b[:, -1].all())
        with self.assertRaises(ContactHoldFailure) as caught:
            self.check((a, b, v, c))
        record = caught.exception.record
        self.assertEqual((record['first_violation']['env'], record['first_violation']['substep']), (1, 2))
        self.assertEqual(record['first_violation']['explicit_counter'], 8003)
        self.assertEqual(record['support_counts'][0, 6], 2)

    def test_invalid_evidence_and_counter_gap_are_never_contact_dropout(self):
        a, b, v, c = self.fixtures()
        v[0, 2] = False
        with self.assertRaisesRegex(ValueError, 'invalid'):
            self.check((a, b, v, c))
        v[:] = True; c[2] += 1
        with self.assertRaisesRegex(ValueError, 'chronology'):
            self.check((a, b, v, c))

    def test_force_zero_is_telemetry_not_qualified_geometric_flight(self):
        a, b, v, c = self.fixtures()
        b[0, 2, 4] = False  # Could be zero reported resultant while geometry remains nearby.
        r = self.check((a, b, v, c))
        self.assertEqual(r['force_contact_departures'].sum(), 1)
        self.assertEqual(r['force_contact_returns'].sum(), 1)
        self.assertIsNone(r['qualified_lift']); self.assertIsNone(r['qualified_landing'])

    def test_single_sample_tripod_dropout_is_a_real_rule_failure_not_proven_collapse(self):
        a, b, v, c = self.fixtures()
        b[:, :, 3:] = False
        b[:, 4, 2] = False
        with self.assertRaises(ContactHoldFailure) as caught:
            self.check((a, b, v, c))
        self.assertEqual(caught.exception.record['first_violation']['count'], 2)
        self.assertEqual(caught.exception.record['first_violation']['substep'], 4)
        # No geometric/velocity information was supplied: a count violation is
        # proven, but neither benign chatter nor a physical fall is proven.
        self.assertIsNone(caught.exception.record['qualified_lift'])

    def test_count_three_cannot_prove_support_polygon_contains_com(self):
        a, b, v, c = self.fixtures()
        b[:, :, 3:] = False
        self.assertTrue(self.check((a, b, v, c))['contact_rule_allows_commit'])
        # Synthetic three support points allx>=1, COMx0 lies outside their hull.
        # The count rule intentionally makes no static/dynamic stability claim.
        points = np.array([[1., -1.], [2., 0.], [1., 1.]])
        self.assertGreater(points[:, 0].min(), 0.)

    def test_existing_draft_false_result_requires_explicit_enforcement(self):
        path = Path(__file__).parent / 'PARENT_contact_contract.py'
        spec = importlib.util.spec_from_file_location('_review_parent_contact', path)
        parent = importlib.util.module_from_spec(spec); sys.modules[spec.name] = parent
        spec.loader.exec_module(parent)
        a, b, v, c = self.fixtures(); b[:, 3] = False
        record = parent.contact_facts(a, b, v, phase='moving_or_stopping', moving_rule=parent.ProposedMovingRule('example_only', 3))
        self.assertFalse(record['proposed_count_check'][0])  # Current facts helper does not raise.
        with self.assertRaises(ContactHoldFailure):
            self.check((a, b, v, c))


if __name__ == '__main__':
    unittest.main()
