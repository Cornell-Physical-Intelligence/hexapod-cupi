"""Check the declared time law and its bounded native-adapter outputs."""
import io
import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from locomotion.tripod import displacement_clock, wave
from locomotion.tripod_config import SWEEPS
from locomotion.tripod_evaluate import NativeController
from locomotion.tests.test_tripod import MODEL, STANCE, measured_fixture
from locomotion.tests.test_tripod_geometry import make_geometry


class TripodClockTests(unittest.TestCase):
    def test_clock_endpoints_symmetry_and_rate_continuity(self):
        ramp = .1; epsilon = 1e-6
        clock = lambda s: displacement_clock(s, ramp)
        self.assertEqual(clock(0.), 0.); self.assertEqual(clock(1.), 1.)
        for s in np.linspace(0., 1., 501):
            self.assertAlmostEqual(clock(s)+clock(1.-s), 1., places=14)
        for s in (ramp, 1.-ramp):
            left = (clock(s)-clock(s-epsilon))/epsilon
            right = (clock(s+epsilon)-clock(s))/epsilon
            self.assertAlmostEqual(left, 1/(1-ramp), places=7)
            self.assertAlmostEqual(right, left, places=7)
        self.assertLess(clock(epsilon)/epsilon, 1e-8)
        self.assertLess((1-clock(1-epsilon))/epsilon, 1e-8)

    def test_clock_changes_timing_and_preserves_the_sine_lift_path(self):
        for s in np.linspace(.001, .999, 501):
            h = displacement_clock(s, .1)
            phase = -math.pi/2+math.acos(1-2*h)
            sine, lift = wave(phase)
            self.assertAlmostEqual(float(sine), 2*h-1, places=12)
            self.assertAlmostEqual(float(lift), 4*h*(1-h), places=12)
        self.assertGreater(displacement_clock(.2, .1), .5*(1-math.cos(.2*math.pi)))
        s = np.linspace(0., 1., 10001)
        velocity = np.diff([displacement_clock(x, .1) for x in s])/np.diff(s)
        self.assertAlmostEqual(float(velocity.mean()), 1., places=10)
        self.assertAlmostEqual(float(np.mean(abs(velocity-1))), .18380438, places=6)

    def test_native_targets_keep_bounds_through_contacts_stops_and_modes(self):
        config = SWEEPS['retimed'][0]
        for touchdown in (.51, .7, .95):
            c = make_geometry(config)
            env = SimpleNamespace(neutral=torch.tensor(c.neutral.reshape(18), dtype=torch.float32),
                lower=torch.tensor(c.lower.reshape(18)), upper=torch.tensor(c.upper.reshape(18)),
                commands=torch.zeros((1, 3)), device='cpu', cfg=SimpleNamespace(action_scale_rad=.35),
                model=MODEL, reference_metadata=STANCE, capture=SimpleNamespace(last_control=[]))
            policy = NativeController(env, {'clearance': 'low'}, config, io.StringIO())
            targets = [policy.motor_target.copy()]
            for command, mode in [([.1, 0., 0.], 'low'), ([0., 0., .2], 'raised'),
                                  ([0., 0., -.2], 'raised'), ([.1, 0., 0.], 'raised'),
                                  ([0., 0., 0.], 'low')]:
                env.commands = torch.tensor([command]); policy.case['clearance'] = mode
                for _ in range(300):
                    force = measured_fixture(policy.controller, touchdown=touchdown)
                    sample = np.zeros((1, 6, 3)); sample[0, :, 2] = force
                    env.capture.last_control = [{'distal_force_world_n': sample} for _ in range(8)]
                    policy(None); targets.append(policy.motor_target.copy())
            targets = np.array(targets)
            self.assertTrue(np.all(targets >= c.lower.reshape(18)))
            self.assertTrue(np.all(targets <= c.upper.reshape(18)))
            self.assertLessEqual(np.max(abs(targets-c.neutral.reshape(18))), .35000001)
            self.assertLessEqual(np.max(abs(np.diff(targets, axis=0))), .040000001)
            self.assertTrue(policy.controller.stopped)
            np.testing.assert_allclose(policy.motor_target, c.neutral.reshape(18), atol=1e-7)


if __name__ == '__main__':
    unittest.main()
