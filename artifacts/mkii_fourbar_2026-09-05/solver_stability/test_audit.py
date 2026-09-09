"""Analytic regression cases for the diagnostic, without Isaac Sim."""
import unittest

import numpy as np

from audit import critical_timestep, effective_motor_inertia, step_matrix


class StabilityTests(unittest.TestCase):
    def test_semi_implicit_boundary(self):
        inertia, kp, kd = .000725, 30., .6
        limit = critical_timestep(inertia, kp, kd, "semi_implicit_euler")
        for multiplier, stable in ((.999, True), (1.001, False)):
            poles = np.linalg.eigvals(step_matrix(inertia, kp, kd, limit * multiplier,
                                                  "semi_implicit_euler"))
            self.assertEqual(bool(max(abs(poles)) < 1), stable)

    def test_zoh_boundary_and_constant_acceleration(self):
        inertia, kp, kd, dt = .000725, 30., .6, .00125
        state = np.array([.02, -.3])
        acceleration = (-kp * state[0] - kd * state[1]) / inertia
        expected = state + np.array([dt * state[1] + .5 * dt * dt * acceleration,
                                     dt * acceleration])
        actual = step_matrix(inertia, kp, kd, dt, "constant_acceleration_zoh") @ state
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
        limit = critical_timestep(inertia, kp, kd, "constant_acceleration_zoh")
        self.assertAlmostEqual(limit, 2 * inertia / kd)
        for multiplier, stable in ((.999, True), (1.001, False)):
            poles = np.linalg.eigvals(step_matrix(inertia, kp, kd, limit * multiplier,
                                                  "constant_acceleration_zoh"))
            self.assertEqual(bool(max(abs(poles)) < 1), stable)

    def test_free_passive_coordinate_reduces_reflected_inertia(self):
        # A motor force with second coordinate unforced has Schur inertia2-1/3.
        mass = np.array([[2., 1.], [1., 3.]])
        effective, individual = effective_motor_inertia(mass, [0])
        self.assertAlmostEqual(effective[0, 0], 5 / 3)
        self.assertAlmostEqual(individual[0], 5 / 3)
        self.assertLess(effective[0, 0], mass[0, 0])

    def test_light_lever_model_distinguishes_two_candidate_timesteps(self):
        for method in ("semi_implicit_euler", "constant_acceleration_zoh"):
            for dt, stable in ((.0025, False), (.00125, True)):
                poles = np.linalg.eigvals(step_matrix(.000725, 30., .6, dt, method))
                self.assertEqual(bool(max(abs(poles)) < 1), stable)


if __name__ == "__main__":
    unittest.main()
