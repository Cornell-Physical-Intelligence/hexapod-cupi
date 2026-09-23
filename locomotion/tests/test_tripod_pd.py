"""Check combined feedback through the native motor-target interface."""
from dataclasses import replace
import io
import json
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from locomotion.tripod import damping_target
from locomotion.tripod_config import SWEEPS
from locomotion.tripod_evaluate import NativeController
from locomotion.tests.test_tripod import MODEL, STANCE, measured_fixture
from locomotion.tests.test_tripod_geometry import make_geometry


class TripodCombinedFeedbackTests(unittest.TestCase):
    def test_combine_offsets_before_bounds_and_stop_without_feedback_bias(self):
        config = SWEEPS['pd_filtered'][0];c = make_geometry(config)
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
                measured = policy.controller.target.reshape(18)-.15
                env.capture.last_control = [{'distal_force_world_n': sample,
                    'joint_position_rad': measured[None],
                    'joint_velocity_rad_s': np.full((1, 18), 1.2)} for _ in range(8)]
                policy(None);policy.controller._validate_target(policy.motor_target.reshape(6, 3))
                targets.append(policy.motor_target.copy())
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        previous_reference = c.neutral.reshape(18).copy();previous_motor = previous_reference.copy()
        exposed_clip_order = False
        lower = np.maximum(c.lower.reshape(18), c.neutral.reshape(18)-.35)
        upper = np.minimum(c.upper.reshape(18), c.neutral.reshape(18)+.35)
        for row in rows:
            reference = np.array(row['nominal_target_rad']);servo = damping_target(reference, previous_reference)
            position = np.array(row['joint_feedback_offset_rad']);velocity = np.array(row['joint_velocity_feedback_offset_rad'])
            combined = np.clip(servo+position+velocity, lower, upper)
            premature = np.clip(np.clip(servo+position, lower, upper)+velocity, lower, upper)
            exposed_clip_order |= bool(np.max(abs(combined-premature)) > .001)
            expected = previous_motor+np.clip(combined-previous_motor, -.040, .040)
            np.testing.assert_allclose(row['target_rad'], expected, atol=1e-12)
            previous_reference, previous_motor = reference, expected
        self.assertTrue(exposed_clip_order)
        self.assertTrue(policy.controller.stopped)
        self.assertLessEqual(np.max(abs(np.diff(targets, axis=0))), .040000001)
        np.testing.assert_allclose(policy.motor_target, c.neutral.reshape(18), atol=1e-7)
        with self.assertRaises(ValueError):
            replace(config, joint_feedback_gain=1.)


if __name__ == '__main__':
    unittest.main()
