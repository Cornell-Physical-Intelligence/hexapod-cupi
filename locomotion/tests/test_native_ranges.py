"""Keep native comparison ranges fixed before candidate scoring."""
from copy import deepcopy
import unittest


class NativeRangeTests(unittest.TestCase):
    def suite(self, value):
        return {'diagnostic_screens_passed': True, 'missing_probe_cases': [], 'failed_probe_cases': [],
                'allocation': {'checkpoint_sha256': 'weights', 'evaluator_sha256': 'scorer'},
                'results': [{'case_id': str(i), 'pass': True, 'metrics': {'drift': value},
                             'checks': {'drift': {'value': value, 'bound': 9, 'operator': '<=', 'status': 'pass'}}}
                            for i in range(13)]}

    def test_three_baselines_freeze_bounds_and_candidate_cannot_widen_them(self):
        from locomotion.tests.native_ranges import freeze, compare
        baseline = [self.suite(value) for value in (1., 2., 3.)]
        limits = freeze(baseline)
        before = deepcopy(limits)
        self.assertTrue(compare(limits, self.suite(2.))['pass'])
        self.assertFalse(compare(limits, self.suite(3.00001))['pass'])
        self.assertEqual(limits, before)

    def test_failed_probes_changed_gates_and_missing_baselines_reject(self):
        from locomotion.tests.native_ranges import freeze, compare
        baseline = [self.suite(1.) for _ in range(3)]
        limits = freeze(baseline)
        with self.assertRaises(ValueError):
            freeze(baseline[:2])
        candidate = self.suite(1.)
        candidate['results'][0]['checks']['drift']['bound'] = 10
        self.assertFalse(compare(limits, candidate)['pass'])
        candidate = self.suite(1.)
        candidate['diagnostic_screens_passed'] = False
        self.assertFalse(compare(limits, candidate)['pass'])


if __name__ == '__main__':
    unittest.main()
