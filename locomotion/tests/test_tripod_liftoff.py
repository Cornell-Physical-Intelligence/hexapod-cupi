"""Check lift timing and motor bounds for the contact-timing adaptation."""
from dataclasses import replace
import io
import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from locomotion.tripod import displacement_clock, wave
from locomotion.tripod_config import SWEEPS, TripodConfig
from locomotion.tripod_evaluate import NativeController
from locomotion.tests.test_tripod import MODEL, STANCE, measured_fixture
from locomotion.tests.test_tripod_geometry import make_geometry


class TripodLiftoffTests(unittest.TestCase):
    def test_lift_retains_peak_and_support_while_extending_clearance(self):
        phase = np.linspace(-math.pi/2, 3*math.pi/2, 1001)
        hip, lift = wave(phase, .75)
        paper_hip, paper_lift = wave(phase)
        np.testing.assert_array_equal(hip, paper_hip)
        np.testing.assert_allclose(lift, np.maximum(np.cos(phase), 0.)**1.5, atol=1e-12)
        self.assertAlmostEqual(float(lift.max()), 1.)
        self.assertTrue(np.all(lift >= paper_lift))
        np.testing.assert_array_equal(lift[np.cos(phase) <= 0.], 0.)
        self.assertGreater(float(wave(-math.pi/3, .75)[1]), float(wave(-math.pi/3)[1]))
        for epsilon in (1e-3, 1e-4, 1e-5):
            h = displacement_clock(epsilon, .1)
            start_rate = (4*h*(1-h))**.75/epsilon
            self.assertLess(start_rate, 150*epsilon**1.25)
        with self.assertRaises(ValueError):
            wave(phase, .7)
        with self.assertRaises(ValueError):
            TripodConfig(swing_lift_power=.75)
        with self.assertRaises(ValueError):
            replace(SWEEPS['liftoff'][0], joint_feedback_gain=.5)

    def test_native_targets_keep_bounds_through_contacts_stops_and_modes(self):
        config = SWEEPS['liftoff'][0]
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
                    action = policy(None)
                    np.testing.assert_allclose((env.neutral+.35*action[0]).numpy(), policy.motor_target, atol=5e-8)
                    policy.controller._validate_target(policy.motor_target.reshape(6, 3))
                    targets.append(policy.motor_target.copy())
            self.assertTrue(policy.controller.stopped)
            np.testing.assert_allclose(policy.motor_target, c.neutral.reshape(18), atol=1e-7)
            self.assertLessEqual(np.max(abs(np.diff(targets, axis=0))), .040000001)


if __name__ == '__main__':
    unittest.main()
