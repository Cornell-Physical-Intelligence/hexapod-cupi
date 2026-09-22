"""Check compatible support motion and the forward swing workspace."""
from dataclasses import replace
import unittest

import numpy as np

from locomotion.tests.test_tripod import feet, measured_fixture
from locomotion.tests.test_tripod_geometry import make_geometry
from locomotion.tripod import overlap_sweep
from locomotion.tripod_config import SWEEPS


class ForwardOverlapTests(unittest.TestCase):
    def test_ground_overlap_has_the_support_velocity_and_continuous_boundaries(self):
        for p in (.02, .10, .85, .97):
            before = np.array(overlap_sweep(p-1e-6, True))
            after = np.array(overlap_sweep(p+1e-6, True))
            np.testing.assert_allclose((after-before)/2e-6, [-2., -2.], atol=1e-8)
        np.testing.assert_allclose(overlap_sweep(0., True), [-1., 1.])
        np.testing.assert_allclose(overlap_sweep(1., True), [1., -1.])
        for p in (.15, .215, .735, .80):
            epsilon = 1e-6
            left = np.array(overlap_sweep(p-epsilon, True))
            center = np.array(overlap_sweep(p, True))
            right = np.array(overlap_sweep(p+epsilon, True))
            np.testing.assert_allclose((center-left)/epsilon, (right-center)/epsilon, atol=1e-7)

    def test_forward_lift_fits_the_envelope_and_retains_clearance(self):
        c = make_geometry(SWEEPS['forward_overlap'][0])
        for mode, minimum in (('low', .012), ('raised', .016)):
            base = c.stance(mode); lifted = base.copy()
            lifted[:, 1:] += c.lift(mode, forward=True)
            c._validate_target(lifted)
            self.assertTrue(np.all(feet(lifted)[:, 2]-feet(base)[:, 2] >= minimum))
        with self.assertRaises(ValueError):
            replace(SWEEPS['overlap'][0], forward_support_overlap=True)

    def test_early_touchdown_commands_modes_and_reset_keep_targets_bounded(self):
        for touchdown in (.51, .70, .95):
            c = make_geometry(SWEEPS['forward_overlap'][0]); targets = [c.target.reshape(18).copy()]
            for command, mode in (([.1, 0., 0.], 'low'), ([.1, 0., 0.], 'raised'),
                                  ([0., 0., .2], 'raised'), ([0., 0., -.2], 'raised'),
                                  ([.05, 0., 0.], 'low'), ([0., 0., 0.], 'low')):
                for _ in range(300):
                    q = c.step(command, measured_fixture(c, touchdown=touchdown), mode)
                    c._validate_target(q.reshape(6, 3)); targets.append(q)
                self.assertIsNone(c.fault)
            self.assertTrue(c.stopped)
            self.assertLessEqual(np.max(abs(np.diff(targets, axis=0))), .040000001)
            c.reset(); np.testing.assert_array_equal(c.target, c.neutral)

    def test_yaw_targets_match_the_preceding_candidate(self):
        for sign in (-1., 1.):
            a = make_geometry(SWEEPS['startup'][0])
            b = make_geometry(SWEEPS['forward_overlap'][0])
            for _ in range(250):
                force = measured_fixture(a)
                np.testing.assert_array_equal(a.step([0., 0., sign*.2], force),
                                              b.step([0., 0., sign*.2], force))


if __name__ == '__main__':
    unittest.main()
