"""Check the declared geometry correction against independent model kinematics."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from locomotion.env_config import JOINT_NAMES
from locomotion.tripod import PHASE, TripodController, wave
from locomotion.tripod_config import GEOMETRY_SWEEP
from locomotion.tripod_kinematics import stance_geometry
from locomotion.tests.test_tripod import JOINTS, MODEL, STANCE, feet, measured_fixture


def make_geometry(config=GEOMETRY_SWEEP[0]):
    return TripodController(STANCE['nominal_joint_position_rad'],
        [JOINTS[n]['lower'] for n in JOINT_NAMES], [JOINTS[n]['upper'] for n in JOINT_NAMES], config,
        model=MODEL, toe_local_points=STANCE['toe_local_points_m'])


class GeometryTripodTests(unittest.TestCase):
    def test_analytic_jacobians_match_independent_finite_differences(self):
        c = make_geometry()
        for mode in ('low', 'raised'):
            q = c.stance(mode)
            points, jacobians = stance_geometry(MODEL, STANCE['toe_local_points_m'], q)
            np.testing.assert_allclose(points, feet(q), atol=1e-12)
            for leg in range(6):
                for joint in range(3):
                    plus, minus = q.copy(), q.copy()
                    plus[leg, joint] += 1e-6; minus[leg, joint] -= 1e-6
                    actual = (feet(plus)[leg]-feet(minus)[leg])/(2e-6)
                    np.testing.assert_allclose(jacobians[leg, :, joint], actual, atol=1e-9)

    def test_declared_sweep_bounds_and_stance_triangle_compatibility(self):
        for config in GEOMETRY_SWEEP:
            c = make_geometry(config)
            for mode in ('low', 'raised'):
                c.mode = mode; base = c.stance(mode)
                for command in ([.05, 0., 0.], [.1, 0., 0.], [0., 0., .2], [0., 0., -.2]):
                    c.command = np.array(command); a = c._sweep(); edges = {0: [], 1: []}
                    for phase in np.linspace(-np.pi/2, 3*np.pi/2, 401):
                        sine, lift = wave(phase+PHASE)
                        q = base+a*sine[:, None]; q[:, 1:] += lift[:, None]*c.lift(mode)
                        c._validate_target(q)
                        points = feet(q); support = np.flatnonzero(np.cos(phase+PHASE) < -1e-8)
                        if len(support) == 3:
                            edges[int(support[0])].append([np.linalg.norm(points[i]-points[j])
                                for i, j in zip(support, np.roll(support, 1))])
                    # A first-order map leaves second-order geometric error at full speed.
                    for values in edges.values():
                        self.assertLess(np.max(np.ptp(values, axis=0)), .008)
                lifted = base.copy(); lifted[:, 1:] += c.lift(mode)
                self.assertTrue(np.all(feet(lifted)[:, 2]-feet(base)[:, 2] >= (.012 if mode == 'low' else .016)))
            self.assertTrue(np.all(feet(c.stance('low'))[:, 2]-feet(c.stance('raised'))[:, 2] >= .008))

    def test_touchdown_timing_stops_and_clearance_keep_target_bounds(self):
        for config in GEOMETRY_SWEEP:
            for touchdown in (.51, .7, .95):
                c = make_geometry(config); targets = [c.target.reshape(18).copy()]
                for command, mode in [([.1, 0., 0.], 'low'), ([0., 0., .2], 'raised'),
                                      ([0., 0., -.2], 'raised'), ([.1, 0., 0.], 'raised'),
                                      ([0., 0., 0.], 'low')]:
                    for _ in range(300):
                        q = c.step(command, measured_fixture(c, touchdown=touchdown), mode)
                        c._validate_target(q.reshape(6, 3)); targets.append(q)
                    self.assertIsNone(c.fault)
                self.assertTrue(c.stopped)
                np.testing.assert_allclose(c.target, c.neutral, atol=1e-12)
                self.assertLessEqual(np.max(abs(np.diff(targets, axis=0))), .040000001)
                c.reset(); replay = c.step([.1, 0., 0.], np.full(6, 12.))
                np.testing.assert_array_equal(replay, c.neutral.reshape(18))

    def test_geometry_input_and_finite_candidate_selection_are_required(self):
        from locomotion.prepare import prepare
        c = make_geometry()
        with self.assertRaises(ValueError):
            TripodController(c.neutral, c.lower, c.upper, GEOMETRY_SWEEP[0])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'pack'
            binding = prepare(path, '/home/orionh/HEXAPOD_runs/restart_20260914/geometry_cpu_fixture',
                mode='tripod', tripod_adaptation='geometry', candidate=1)
            args = binding['command_args']
            self.assertEqual(args[args.index('--tripod-adaptation')+1], 'geometry')
            frozen = json.loads((path/'source/FREEZE_SHA256.json').read_text())
            self.assertIn('locomotion/tripod_kinematics.py', frozen)
            with self.assertRaises(ValueError):
                prepare(Path(temp)/'bad', '/home/orionh/HEXAPOD_runs/restart_20260914/geometry_cpu_fixture',
                    mode='tripod', tripod_adaptation='geometry', candidate=2)

    def test_missing_touchdown_fault_preserves_bounded_targets(self):
        for config in GEOMETRY_SWEEP:
            c = make_geometry(config)
            for _ in range(160):
                force = np.full(6, 12.)
                if c.state == 'walk':
                    force[c.swing] = 0.
                q = c.step([.1, 0., 0.], force)
                c._validate_target(q.reshape(6, 3))
                if c.fault:
                    break
            self.assertEqual(c.fault, 'touchdown_timeout')
            np.testing.assert_array_equal(c.step([0., 0., 0.], np.full(6, 12.)), q)


if __name__ == '__main__':
    unittest.main()
