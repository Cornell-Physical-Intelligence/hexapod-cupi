"""Check bounded encoder feedback through the unchanged motor interface."""
import io
import json
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from locomotion.env import motor_force
from locomotion.env_config import KD
from locomotion.tripod import feedback_blend, feedback_target
from locomotion.tripod_config import SWEEPS
from locomotion.tripod_evaluate import NativeController
from locomotion.tests.test_tripod import MODEL, STANCE, measured_fixture
from locomotion.tests.test_tripod_geometry import make_geometry


class TripodFeedbackTests(unittest.TestCase):
    def test_encoder_feedback_changes_requested_torque_with_fixed_motor_gains(self):
        c = make_geometry(SWEEPS['feedback'][0])
        neutral, lower, upper = [x.reshape(18) for x in (c.neutral, c.lower, c.upper)]
        reference = neutral+.03; measured = reference-.02
        for gain in (.5, 1.):
            target, offset, clipped = feedback_target(reference, reference, measured,
                neutral, lower, upper, gain, 1.)
            np.testing.assert_allclose(offset, gain*.02)
            self.assertFalse(clipped)
            tensor = lambda x: torch.as_tensor(x, dtype=torch.float32)
            requested, _, _ = motor_force(tensor(measured), tensor(np.zeros(18)),
                                           tensor(target), tensor(KD))
            np.testing.assert_allclose(requested.numpy(), 12*(1+gain)*.02, atol=5e-7)
        target, offset, clipped = feedback_target(neutral+.34, neutral+.34, neutral-1.,
            neutral, lower, upper, 1., 1.)
        self.assertTrue(clipped)
        self.assertLessEqual(abs(offset).max(), .070)
        self.assertLessEqual(abs(target-neutral).max(), .350000001)
        self.assertTrue(np.all((target >= lower) & (target <= upper)))
        with self.assertRaisesRegex(ValueError, 'finite measured'):
            feedback_target(reference, reference, np.full(18, np.nan), neutral, lower, upper, 1., 1.)

    def test_start_and_stop_blend_has_zero_endpoint_rate(self):
        for state, first, last in (('start', 0., 1.), ('settle', 1., 0.)):
            self.assertEqual(feedback_blend(state, 0., 1.), first)
            self.assertEqual(feedback_blend(state, 1., 1.), last)
            self.assertLess(abs(feedback_blend(state, 1e-6, 1.)-first)/1e-6, 1e-5)
            self.assertLess(abs(feedback_blend(state, 1-1e-6, 1.)-last)/1e-6, 1e-5)
        for state in ('idle', 'clearance', 'fault'):
            self.assertEqual(feedback_blend(state, .5, 1.), 0.)
        self.assertEqual(feedback_blend('walk', .5, 1.), 1.)

    def test_native_adapter_keeps_limits_and_returns_to_unbiased_standing(self):
        for config in SWEEPS['feedback']:
            c = make_geometry(config)
            env = SimpleNamespace(neutral=torch.tensor(c.neutral.reshape(18), dtype=torch.float32),
                lower=torch.tensor(c.lower.reshape(18)), upper=torch.tensor(c.upper.reshape(18)),
                commands=torch.zeros((1, 3)), device='cpu', cfg=SimpleNamespace(action_scale_rad=.35),
                model=MODEL, reference_metadata=STANCE, capture=SimpleNamespace(last_control=[]))
            stream = io.StringIO(); policy = NativeController(env, {'clearance': 'low'}, config, stream)
            targets = [policy.motor_target.copy()]
            for command, mode in [([.1, 0., 0.], 'low'), ([0., 0., .2], 'raised'),
                                  ([0., 0., -.2], 'raised'), ([.1, 0., 0.], 'raised'),
                                  ([0., 0., 0.], 'low')]:
                env.commands = torch.tensor([command]); policy.case['clearance'] = mode
                for _ in range(300):
                    force = measured_fixture(policy.controller, touchdown=.7)
                    sample = np.zeros((1, 6, 3)); sample[0, :, 2] = force
                    measured = policy.controller.target.reshape(18)+np.tile([.01, .05, -.067], 6)
                    env.capture.last_control = [{'distal_force_world_n': sample,
                        'joint_position_rad': measured[None]} for _ in range(8)]
                    action = policy(None)
                    np.testing.assert_allclose((env.neutral+.35*action[0]).numpy(), policy.motor_target, atol=5e-8)
                    policy.controller._validate_target(policy.motor_target.reshape(6, 3))
                    targets.append(policy.motor_target.copy())
            records = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertTrue(any(max(abs(np.array(row['joint_feedback_offset_rad']))) > .01 for row in records))
            self.assertTrue(all(max(abs(np.array(row['joint_feedback_offset_rad']))) <= .070 for row in records))
            self.assertEqual(records[-1]['joint_feedback_blend'], 0.)
            self.assertTrue(policy.controller.stopped)
            np.testing.assert_allclose(policy.motor_target, c.neutral.reshape(18), atol=1e-7)
            self.assertLessEqual(np.max(abs(np.diff(targets, axis=0))), .040000001)


if __name__ == '__main__':
    unittest.main()
