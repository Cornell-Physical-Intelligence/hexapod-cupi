"""Exercise sampled feedback attenuation and filter reset in the native adapter."""
import io
import json
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from locomotion.tripod_config import SWEEPS, TripodConfig
from locomotion.tripod_evaluate import NativeController
from locomotion.tests.test_tripod import MODEL, STANCE, measured_fixture
from locomotion.tests.test_tripod_geometry import make_geometry


class TripodVelocityFilterTests(unittest.TestCase):
    def test_filter_reduces_alternating_feedback_and_resets_after_stop(self):
        components = []
        for variant in ('velocity', 'velocity_filtered'):
            config = SWEEPS[variant][0];c = make_geometry(config)
            env = SimpleNamespace(neutral=torch.tensor(c.neutral.reshape(18), dtype=torch.float32),
                lower=torch.tensor(c.lower.reshape(18)), upper=torch.tensor(c.upper.reshape(18)),
                commands=torch.zeros((1, 3)), device='cpu', cfg=SimpleNamespace(action_scale_rad=.35),
                model=MODEL, reference_metadata=STANCE, capture=SimpleNamespace(last_control=[]))
            stream = io.StringIO();policy = NativeController(env, {'clearance': 'low'}, config, stream)
            targets = [policy.motor_target.copy()]
            for command, mode in [([.1, 0., 0.], 'low'), ([0., 0., .2], 'raised'),
                                  ([0., 0., -.2], 'raised'), ([.1, 0., 0.], 'raised'),
                                  ([0., 0., 0.], 'low')]:
                env.commands = torch.tensor([command]);policy.case['clearance'] = mode
                for _ in range(300):
                    force = measured_fixture(policy.controller, touchdown=.7)
                    sample = np.zeros((1, 6, 3));sample[0, :, 2] = force
                    env.capture.last_control = [{'distal_force_world_n': sample,
                        'joint_velocity_rad_s': np.full((1, 18), (-1.)**policy.control)} for _ in range(8)]
                    policy(None);policy.controller._validate_target(policy.motor_target.reshape(6, 3))
                    targets.append(policy.motor_target.copy())
            rows = [json.loads(line) for line in stream.getvalue().splitlines()]
            alternating = np.array([r['joint_velocity_feedback_offset_rad'][0] for r in rows[100:300]])
            components.append(abs(np.mean(alternating*(-1.)**np.arange(100, 300))))
            self.assertTrue(policy.controller.stopped)
            np.testing.assert_allclose(policy.motor_target, c.neutral.reshape(18), atol=1e-7)
            self.assertLessEqual(np.max(abs(np.diff(targets, axis=0))), .040000001)
            if variant == 'velocity_filtered':
                np.testing.assert_array_equal(policy.filtered_reference_velocity, 0.)
                np.testing.assert_array_equal(policy.filtered_measured_velocity, 0.)
                self.assertTrue(any(max(abs(np.array(r['filtered_velocity_error_rad_s']))) > .1 for r in rows))
        self.assertLess(components[1], .5*components[0])
        with self.assertRaises(ValueError):
            TripodConfig(joint_velocity_filter_hz=5.)


if __name__ == '__main__':
    unittest.main()
