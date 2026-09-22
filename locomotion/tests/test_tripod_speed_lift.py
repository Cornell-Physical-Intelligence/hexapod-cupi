"""Check reduced radial swing motion with unchanged accepted endpoints."""
from dataclasses import replace
import unittest

import numpy as np

from locomotion.tests.test_tripod import feet, measured_fixture
from locomotion.tests.test_tripod_geometry import make_geometry
from locomotion.tripod_config import SWEEPS


class SpeedLiftTests(unittest.TestCase):
    def test_low_speed_reduces_radial_excursion_and_retains_toe_lift(self):
        c = make_geometry(SWEEPS['speed_lift'][0]); c.command = np.array([.05, 0., 0.])
        base = c.stance('low'); q = base.copy(); q[:, 1:] += c.lift('low', forward=True)
        p = feet(base); delta = feet(q)-p
        radial = np.sum(delta[:, :2]*p[:, :2], axis=1)/np.linalg.norm(p[:, :2], axis=1)
        self.assertTrue(np.all((radial > 0.) & (radial < .004)))
        self.assertTrue(np.all(delta[:, 2] > .012))
        c._validate_target(q)
        with self.assertRaises(ValueError):
            replace(SWEEPS['startup'][0], speed_adapted_lift=True)

    def test_high_speed_yaw_and_raised_targets_match_the_preceding_candidate(self):
        for command, mode in (([.1, 0., 0.], 'low'), ([.05, 0., 0.], 'raised'),
                              ([0., 0., .2], 'low'), ([0., 0., -.2], 'low'),
                              ([0., 0., .2], 'raised'), ([0., 0., -.2], 'raised')):
            a = make_geometry(SWEEPS['forward_overlap'][0])
            b = make_geometry(SWEEPS['speed_lift'][0])
            for _ in range(300):
                force = measured_fixture(a)
                np.testing.assert_array_equal(a.step(command, force, mode), b.step(command, force, mode))

    def test_speed_interval_touchdown_and_stops_keep_bounded_targets(self):
        for speed in np.linspace(.01, .10, 10):
            for touchdown in (.51, .90):
                c = make_geometry(SWEEPS['speed_lift'][0]); previous = c.target.reshape(18).copy()
                for control in range(400):
                    command = [float(speed), 0., 0.] if control < 250 else [0., 0., 0.]
                    q = c.step(command, measured_fixture(c, touchdown=touchdown))
                    c._validate_target(q.reshape(6, 3))
                    self.assertLessEqual(np.max(abs(q-previous)), .040000001); previous = q
                self.assertIsNone(c.fault); self.assertTrue(c.stopped)
                c.reset(); np.testing.assert_array_equal(c.target, c.neutral)
        for speed in (.05, .10):
            values = []
            for v in (speed-1e-8, speed):
                c.command = np.array([v, 0., 0.]); values.append(c.lift('low', forward=True))
            self.assertLess(np.max(abs(values[1]-values[0])), 1e-7)


if __name__ == '__main__':
    unittest.main()
