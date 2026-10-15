"""Check dynamics against energy derivatives and external-wrench identities."""
import unittest

import casadi as ca
import numpy as np

from locomotion.priors.model import NEUTRAL, RobotModel


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RobotModel()
        cls.q = np.r_[.1, -.2, .11, .03, -.04, .07, NEUTRAL]

    def test_gravity_matches_potential_gradient(self):
        q, m = self.q, self.model
        h = 1e-6
        gradient = np.array([(float(m.potential(q+np.eye(24)[i]*h))
            -float(m.potential(q-np.eye(24)[i]*h)))/(2*h) for i in range(24)])
        actual = np.asarray(m.generalized_dynamics(q, np.zeros(24), np.zeros(24), np.zeros((3, 6)))).ravel()
        np.testing.assert_allclose(actual, gradient, atol=2e-8, rtol=2e-7)

    def test_free_fall_requires_no_motor_torque(self):
        a = np.zeros(24); a[2] = -9.81
        actual = np.asarray(self.model.inverse_dynamics(self.q, np.zeros(24), a, np.zeros((3, 6))))
        np.testing.assert_allclose(actual, 0., atol=1e-12)

    def test_mass_matrix_is_symmetric_positive_and_matches_energy(self):
        m, q, z = self.model, self.q, np.zeros(24)
        bias = np.asarray(m.generalized_dynamics(q, z, z, np.zeros((3, 6)))).ravel()
        mass = np.column_stack([np.asarray(m.generalized_dynamics(q, z, np.eye(24)[i],
            np.zeros((3, 6)))).ravel()-bias for i in range(24)])
        np.testing.assert_allclose(mass, mass.T, atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(mass).min(), 0.)
        np.testing.assert_allclose(np.diag(mass)[:3], m.mass, atol=1e-12)

    def test_contact_wrench_obeys_virtual_work(self):
        m, q, z = self.model, self.q, np.zeros(24)
        forces = np.arange(18).reshape(3, 6)/4
        unloaded = np.asarray(m.generalized_dynamics(q, z, z, np.zeros((3, 6)))).ravel()
        loaded = np.asarray(m.generalized_dynamics(q, z, z, forces)).ravel()
        h = 1e-6
        work = []
        for i in range(24):
            plus = np.asarray(m.kinematics(q+np.eye(24)[i]*h)[0])
            minus = np.asarray(m.kinematics(q-np.eye(24)[i]*h)[0])
            work.append(np.sum(forces*(plus-minus)/(2*h)))
        np.testing.assert_allclose(unloaded-loaded, work, atol=2e-8)

    def test_body_velocity_bias_preserves_energy(self):
        # dT/dt = v^T C for a rigid-body tree without external forces or gravity.
        m, q = self.model, self.q
        v = np.random.default_rng(42).normal(0, .1, 24)
        def mass_at(position):
            s = ca.SX.sym('a', 24)
            fn = ca.Function('mass_check', [s], [ca.jacobian(
                m.generalized_dynamics(position, np.zeros(24), s, np.zeros((3, 6))), s)])
            return np.asarray(fn(np.zeros(24)))
        h = 1e-5
        derivative = (mass_at(q+h*v)-mass_at(q-h*v))/(2*h)
        bias = np.asarray(m.generalized_dynamics(q, v, np.zeros(24), np.zeros((3, 6)))).ravel()
        gravity = np.asarray(m.generalized_dynamics(q, np.zeros(24), np.zeros(24), np.zeros((3, 6)))).ravel()
        self.assertAlmostEqual(float(v@(bias-gravity)), float(.5*v@derivative@v), places=9)


if __name__ == '__main__':
    unittest.main()
