"""Check motor-law cancellation and emitted targets through the native adapter."""
import io
import json
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from locomotion.env import motor_force
from locomotion.env_config import KD
from locomotion.tripod import damping_target
from locomotion.tripod_config import SWEEPS
from locomotion.tripod_evaluate import NativeController
from locomotion.tests.test_tripod import MODEL, STANCE, measured_fixture
from locomotion.tests.test_tripod_geometry import make_geometry


class DampingTripodTests(unittest.TestCase):
    def test_correction_cancels_existing_damping_at_reference_velocity(self):
        q = np.linspace(-.3, .4, 18)
        velocity = np.linspace(-1.5, 1.5, 18)
        target = damping_target(q, q-.02*velocity)
        tensor = lambda value: torch.as_tensor(value, dtype=torch.float32)
        requested, _, _ = motor_force(tensor(q), tensor(velocity), tensor(target), tensor(KD))
        np.testing.assert_allclose(requested.numpy(), 0., atol=3e-7)
        np.testing.assert_array_equal(damping_target(q, q), q)

    def test_native_adapter_records_reference_and_bounded_motor_targets(self):
        config = SWEEPS['damping'][0]
        for touchdown in (.51, .7, .95):
            c = make_geometry(config)
            env = SimpleNamespace(neutral=torch.tensor(c.neutral.reshape(18), dtype=torch.float32),
                lower=torch.tensor(c.lower.reshape(18)), upper=torch.tensor(c.upper.reshape(18)),
                commands=torch.zeros((1, 3)), device='cpu', cfg=SimpleNamespace(action_scale_rad=.35),
                model=MODEL, reference_metadata=STANCE, capture=SimpleNamespace(last_control=[]))
            stream = io.StringIO(); policy = NativeController(env, {'clearance': 'low'}, config, stream)
            emitted = [policy.motor_target.copy()]
            for command, mode in [([.1, 0., 0.], 'low'), ([0., 0., .2], 'raised'),
                                  ([0., 0., -.2], 'raised'), ([.1, 0., 0.], 'raised'),
                                  ([0., 0., 0.], 'low')]:
                env.commands = torch.tensor([command]); policy.case['clearance'] = mode
                for _ in range(300):
                    force = measured_fixture(policy.controller, touchdown=touchdown)
                    sample = np.zeros((1, 6, 3)); sample[0, :, 2] = force
                    env.capture.last_control = [{'distal_force_world_n': sample} for _ in range(8)]
                    action = policy(None)
                    target = (env.neutral+.35*action[0]).numpy()
                    np.testing.assert_allclose(target, policy.motor_target, atol=5e-8)
                    policy.controller._validate_target(policy.motor_target.reshape(6, 3))
                    emitted.append(policy.motor_target.copy())
            rows = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertTrue(any(not np.allclose(r['target_rad'], r['nominal_target_rad']) for r in rows))
            self.assertTrue(policy.controller.stopped)
            np.testing.assert_allclose(policy.motor_target, c.neutral.reshape(18), atol=1e-7)
            self.assertLessEqual(np.max(abs(np.diff(emitted, axis=0))), .040000001)


if __name__ == '__main__':
    unittest.main()
