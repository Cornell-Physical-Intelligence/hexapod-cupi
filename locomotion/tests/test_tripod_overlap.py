"""Check the declared yaw support overlap and preserve forward targets."""
from dataclasses import replace
import unittest

import numpy as np

from locomotion.tripod import overlap_sweep
from locomotion.tripod_config import SWEEPS
from locomotion.tests.test_tripod import measured_fixture
from locomotion.tests.test_tripod_geometry import make_geometry


class TripodOverlapTests(unittest.TestCase):
    def test_horizontal_velocity_matches_support_at_contact_overlap(self):
        eps = 1e-7
        for phase in (.03, .10, .15, .80, .90, .97):
            speed = (np.array(overlap_sweep(phase+eps))
                     - np.array(overlap_sweep(phase-eps)))/(2*eps)
            np.testing.assert_allclose(speed, [-2., -2.], atol=4e-6)
        np.testing.assert_allclose(overlap_sweep(0.), [-1., 1.])
        np.testing.assert_allclose(overlap_sweep(1.), [1., -1.])
        for phase in (-.01, 1.01, np.nan):
            with self.assertRaises(ValueError):
                overlap_sweep(phase)
        with self.assertRaises(ValueError):
            replace(SWEEPS['retimed'][0], support_overlap=True)

    def test_forward_targets_match_and_turn_transitions_keep_bounds(self):
        for touchdown in (.51, .7, .95):
            current = make_geometry(SWEEPS['overlap'][0])
            previous = make_geometry(SWEEPS['pd_filtered'][0])
            for command, mode in [([.1, 0., 0.], 'low'), ([.1, 0., 0.], 'raised'),
                                  ([0., 0., 0.], 'low')]:
                for _ in range(300):
                    a = current.step(command, measured_fixture(current, touchdown=touchdown), mode)
                    b = previous.step(command, measured_fixture(previous, touchdown=touchdown), mode)
                    np.testing.assert_array_equal(a, b)
            targets = [current.target.reshape(18).copy()]
            for command, mode in [([0., 0., .2], 'low'), ([0., 0., -.2], 'raised'),
                                  ([0., 0., 0.], 'low')]:
                for _ in range(300):
                    q = current.step(command, measured_fixture(current, touchdown=touchdown), mode)
                    current._validate_target(q.reshape(6, 3));targets.append(q)
                self.assertIsNone(current.fault)
            self.assertTrue(current.stopped)
            np.testing.assert_allclose(current.target, current.neutral, atol=1e-12)
            self.assertLessEqual(np.max(abs(np.diff(targets, axis=0))), .040000001)

    def test_landed_pitch_follows_the_horizontal_baseline(self):
        for sign in (-1., 1.):
            c = make_geometry(SWEEPS['overlap'][0]);c.command = np.array([0., 0., sign*.2])
            c.state = 'walk';c.progress = .86;c.contacts[:] = True;c.seen_off[c.swing] = True
            coefficient = c._sweep();old_shape = np.where(c.swing, *overlap_sweep(c.progress))
            c.target = c.stance('low')+coefficient*old_shape[:, None]
            c.target[c.swing, 1:] += [.008, -.008]
            old = c.target.copy();q = c.step(c.command, np.full(6, 12.)).reshape(6, 3)
            new_shape = np.where(c.swing, *overlap_sweep(c.progress))
            np.testing.assert_allclose(q-old, coefficient*(new_shape-old_shape)[:, None], atol=1e-12)
            self.assertTrue(c.landed[c.swing].all())


if __name__ == '__main__':
    unittest.main()
