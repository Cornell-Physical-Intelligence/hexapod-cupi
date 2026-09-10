import unittest
import numpy as np
from static_patterns import G,M,geometry,solve

class StaticTests(unittest.TestCase):
    def test_exact_kinematic_jacobians_and_leg_gravity_match_finite_differences(self):
        q=G.q0.numpy();geo=geometry(q);eps=1e-6
        for leg in range(6):
            for joint in range(3):
                plus=q.copy();minus=q.copy();plus[leg,joint]+=eps;minus[leg,joint]-=eps
                a,b=geometry(plus),geometry(minus)
                jac=(a['foot'][leg]-b['foot'][leg])/(2*eps)
                np.testing.assert_allclose(jac,geo['J'][leg,:,joint],atol=1e-9)
                gravity=-(a['potential_energy_j']-b['potential_energy_j'])/(2*eps)
                self.assertAlmostEqual(gravity,geo['tau_gravity'][leg,joint],places=7)

    def test_static_equilibrium_readback_and_cone_bracket(self):
        for support in ((0,2,4),(1,3,5),(0,2,3,5),(0,1,2,3,4)):
            lower=solve(G.q0.numpy(),support,.3,'outer');upper=solve(G.q0.numpy(),support,.3,'inner')
            self.assertTrue(lower['feasible']);self.assertTrue(upper['feasible'])
            self.assertLessEqual(lower['minimax_motor_torque_nm'],upper['minimax_motor_torque_nm']+1e-7)
            self.assertLess(upper['equilibrium_residual_max'],1e-7)
            self.assertTrue(upper['circular_cone_witness'])
            forces=np.array(upper['contact_force_n']);tau=np.array(upper['actuator_torque_nm']);grav=np.array(upper['gravity_joint_torque_nm'])
            for leg in set(range(6))-set(support):
                np.testing.assert_array_equal(forces[leg],0)
                np.testing.assert_allclose(tau[leg],-grav[leg],atol=1e-12)

    def test_unilateral_same_side_tripod_has_no_static_equilibrium(self):
        r=solve(G.q0.numpy(),(0,1,2),.6,'outer')
        self.assertFalse(r['feasible']);self.assertLess(r['support_margin_m'],0)

if __name__=='__main__':unittest.main()
