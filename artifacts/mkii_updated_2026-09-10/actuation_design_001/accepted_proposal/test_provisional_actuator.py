import json,unittest
from pathlib import Path
import numpy as np
from provisional_actuator import effort
from servo_candidate import candidate

class Proposal(unittest.TestCase):
 def test_envelope_demand_and_power_remain_distinct(self):
  speed=np.array([0,450,477,480,500,-477]*3)*2*np.pi/60
  a=effort(np.zeros(18),speed,np.ones(18)*3,np.ones(18)*12,np.zeros(18))
  np.testing.assert_allclose(a['requested_nm'],36.)
  np.testing.assert_allclose(a['software_applied_nm'],[1.6,1.6,.5,0,0,.5]*3,atol=1e-13)
  self.assertTrue(a['clipped'].all());self.assertLess(a['mechanical_power_w'][5],0)
  b=effort(np.zeros(18),-speed,np.ones(18)*-3,np.ones(18)*12,np.zeros(18))
  np.testing.assert_allclose(a['software_applied_nm'],-b['software_applied_nm'])

 def test_invalid_values_reject_without_modifying_input(self):
  q=np.zeros(18);target=np.zeros(18);copy=target.copy();target[0]=np.nan
  with self.assertRaises(ValueError):effort(q,q,target,q,q)
  self.assertTrue(np.isnan(target[0]));np.testing.assert_array_equal(q,copy)

 def test_linear_candidate_direct_time_integration(self):
  r=json.loads((Path(__file__).parent/'inertia_report.json').read_text());M=np.array(r['mass_matrix_kg_m2'])
  c=candidate(M);kp=np.array(c['stiffness_nm_per_rad']);kd=np.array(c['damping_nm_s_per_rad']);dt=c['dt_s']
  q=np.linspace(-.001,.001,18);v=np.zeros(18);initial=.5*np.dot(kp*q,q)
  for _ in range(1000):
   v+=dt*np.linalg.solve(M,-kp*q-kd*v);q+=dt*v
  final=.5*np.dot(kp*q,q)+.5*v@M@v
  self.assertLess(final,initial*1e-12)
  self.assertLess(c['linear_fixed_root_semiimplicit_euler_spectral_radius'],1.)

if __name__=='__main__':unittest.main()
