"""Check the velocity-feedback motor law and bounded transition behavior."""
import io
import json
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from locomotion.env import motor_force
from locomotion.env_config import KD
from locomotion.tripod import damping_target, velocity_feedback_target
from locomotion.tripod_config import SWEEPS
from locomotion.tripod_evaluate import NativeController
from locomotion.tests.test_tripod import MODEL, STANCE, measured_fixture
from locomotion.tests.test_tripod_geometry import make_geometry


class TripodVelocityTests(unittest.TestCase):
    def test_motor_law_adds_velocity_error_damping_without_position_gain(self):
        c = make_geometry(SWEEPS['velocity'][0])
        neutral, lower, upper = [x.reshape(18) for x in (c.neutral, c.lower, c.upper)]
        reference = neutral+.02; position = reference-.01
        desired_velocity = np.linspace(-.2, .2, 18)
        actual_velocity = desired_velocity-.1
        servo = damping_target(reference, reference-.02*desired_velocity)
        target, offset, clipped = velocity_feedback_target(servo, desired_velocity,
            actual_velocity, neutral, lower, upper, 2., 1.)
        self.assertFalse(clipped)
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32)
        requested, _, _ = motor_force(tensor(position), tensor(actual_velocity), tensor(target), tensor(KD))
        np.testing.assert_allclose(requested.numpy(), 12*.01+3*np.array(KD)*.1, atol=5e-7)
        target, offset, clipped = velocity_feedback_target(neutral+.34, np.zeros(18),
            np.full(18, -100.), neutral, lower, upper, 2., 1.)
        self.assertTrue(clipped)
        self.assertLessEqual(abs(offset).max(), .070)
        self.assertLessEqual(abs(target-neutral).max(), .350000001)
        with self.assertRaisesRegex(ValueError, 'finite joint velocities'):
            velocity_feedback_target(servo, desired_velocity, np.full(18, np.nan),
                                     neutral, lower, upper, 2., 1.)

    def test_native_targets_stay_bounded_and_return_to_unbiased_standing(self):
        config = SWEEPS['velocity'][0];c = make_geometry(config)
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
                velocity = np.full((1, 18), .6*np.sin(policy.control*.1))
                env.capture.last_control = [{'distal_force_world_n': sample,
                    'joint_velocity_rad_s': velocity} for _ in range(8)]
                action = policy(None)
                np.testing.assert_allclose((env.neutral+.35*action[0]).numpy(), policy.motor_target, atol=5e-8)
                policy.controller._validate_target(policy.motor_target.reshape(6, 3))
                targets.append(policy.motor_target.copy())
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertTrue(any(max(abs(np.array(r['joint_velocity_feedback_offset_rad']))) > .01 for r in rows))
        self.assertTrue(all(max(abs(np.array(r['joint_velocity_feedback_offset_rad']))) <= .070 for r in rows))
        self.assertEqual(rows[-1]['joint_velocity_feedback_blend'], 0.)
        self.assertTrue(policy.controller.stopped)
        np.testing.assert_allclose(policy.motor_target, c.neutral.reshape(18), atol=1e-7)
        self.assertLessEqual(np.max(abs(np.diff(targets, axis=0))), .040000001)


if __name__ == '__main__':
    unittest.main()
