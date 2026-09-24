"""Check command transforms and periodic geometry before native replay."""
import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from dataclasses import replace

import numpy as np
import casadi as ca

from locomotion.priors.commands import motion_cases, command_index
from locomotion.priors.model import RobotModel
from locomotion.priors.model import ROOT
from locomotion.priors.optimize import Config, cycle_state, initial_trajectory, planar_pose, schedule, restart_values


class OmniTrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RobotModel()

    def test_command_bank_and_all_twenty_workspace_seeds(self):
        cases = motion_cases()
        self.assertEqual(len(cases), 20)
        for i, case in enumerate(cases):
            with self.subTest(command=case['command']):
                self.assertEqual(command_index(case['command']), i)
                forward, left, yaw = case['command']
                config = Config(forward_mps=forward, left_mps=left, yaw_rate_rad_s=yaw)
                config.validate()
                q, v, a, forces, feet = initial_trajectory(self.model, config)
                self.assertEqual(q.shape, (61, 24))
                np.testing.assert_allclose(q[-1], cycle_state(config, q[0]), atol=1e-12)
                np.testing.assert_allclose(v[-1], cycle_state(config, v[0], velocity=True), atol=1e-10)
                rotation, shift = planar_pose(config, config.period_s)
                np.testing.assert_allclose(feet[-1], feet[0]@rotation.T+shift, atol=1e-12)
                phase, support = schedule(config, np.arange(61)*.02)
                for leg in range(6):
                    continuous = support[1:, leg]&support[:-1, leg]&(phase[1:, leg] >= phase[:-1, leg])
                    np.testing.assert_allclose(np.diff(feet[:, leg], axis=0)[continuous], 0., atol=1e-12)

    def test_terminal_contacts_are_implied_by_cycle_closure(self):
        config = Config(forward_mps=.04, yaw_rate_rad_s=.15)
        pair = ca.SX.sym('endpoints', 48)
        initial, terminal = pair[:24], pair[24:]
        feet = self.model.kinematics(initial)[0]
        end_feet = self.model.kinematics(terminal)[0]
        constraints = ca.vertcat(terminal-cycle_state(config, initial), ca.vec(feet), ca.vec(end_feet))
        function = ca.Function('endpoint_jac', [pair], [ca.jacobian(constraints, pair)])
        state = np.r_[0., 0., .1, .03, -.02, .04, np.tile([.03, -.28, .42], 6)]
        end = cycle_state(config, state)
        rotation, shift = planar_pose(config, config.period_s)
        np.testing.assert_allclose(np.asarray(self.model.kinematics(end)[0]),
            rotation@np.asarray(self.model.kinematics(state)[0])+shift[:, None], atol=1e-12)
        jacobian = np.asarray(function(np.r_[state, end]))
        self.assertEqual(jacobian.shape, (60, 48))
        self.assertEqual(np.linalg.matrix_rank(jacobian, tol=1e-10), 42)
        self.assertEqual(np.linalg.matrix_rank(jacobian[:-18], tol=1e-10), 42)

    def test_forward_left_and_turn_axes(self):
        np.testing.assert_allclose(planar_pose(Config(), 1.)[1], [0, -.05, 0])
        np.testing.assert_allclose(planar_pose(Config(forward_mps=0, left_mps=.05), 1.)[1], [.05, 0, 0])
        config = Config(forward_mps=.04, yaw_rate_rad_s=.15)
        rotation, position = planar_pose(config, 1.)
        self.assertGreater(position[0], 0.)
        self.assertLess(position[1], 0.)
        np.testing.assert_allclose(rotation.T@rotation, np.eye(3), atol=1e-12)

    def test_cycle_inverse_and_rotated_velocity(self):
        config = Config(forward_mps=.04, yaw_rate_rad_s=-.15)
        state = np.arange(24, dtype=float)/100
        for velocity in (False, True):
            converted = cycle_state(config, state, velocity=velocity)
            np.testing.assert_allclose(cycle_state(config, converted, reverse=True, velocity=velocity), state, atol=1e-12)
        self.assertFalse(np.allclose(cycle_state(config, state, velocity=True)[:2], state[:2]))

    def test_zero_yaw_limit_and_planar_composition(self):
        config = Config(forward_mps=.04, left_mps=.01, yaw_rate_rad_s=1e-12)
        np.testing.assert_allclose(planar_pose(config, 1.)[1], [.01, -.04, 0], atol=1e-12)
        config = Config(forward_mps=.04, yaw_rate_rad_s=.15)
        r1, p1 = planar_pose(config, .7)
        r2, p2 = planar_pose(config, .5)
        r, p = planar_pose(config, 1.2)
        np.testing.assert_allclose(r1@r2, r, atol=1e-12)
        np.testing.assert_allclose(p1+r1@p2, p, atol=1e-12)

    def test_cli_preserves_half_cycle_start_in_frozen_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'replay'
            subprocess.run([sys.executable, '-m', 'locomotion.priors.prepare',
                '--trajectory-directory', str(ROOT/'locomotion/priors/tests/fixtures/solve'),
                '--output', str(output), '--remote-root',
                '/home/orionh/HEXAPOD_runs/restart_20260914/test_amp_half_cycle',
                '--inputs', str(ROOT/'configs/locomotion_spark.json'), '--start-phase', '0.5'],
                check=True, capture_output=True, text=True)
            args = json.loads((output/'binding.json').read_text())['command_args']
            self.assertEqual(args[args.index('--start-phase')+1], '0.5')

    def test_restart_keeps_problem_and_model_identity(self):
        fixture = ROOT/'locomotion/priors/tests/fixtures/solve/trajectory.npz'
        config = Config(max_iterations=1200)
        values = restart_values(fixture, config, self.model)
        with np.load(fixture) as original:
            for name, value in values.items():
                np.testing.assert_array_equal(value, original[name])
        with self.assertRaisesRegex(ValueError, 'configuration'):
            restart_values(fixture, replace(config, left_mps=.01), self.model)
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp)/'trajectory.npz'
            copy.write_bytes(fixture.read_bytes()+b'changed')
            for name in ('INPUT.json', 'RESULT.json'):
                (copy.parent/name).write_bytes((fixture.parent/name).read_bytes())
            with self.assertRaisesRegex(ValueError, 'bytes'):
                restart_values(copy, config, self.model)


if __name__ == '__main__':
    unittest.main()
