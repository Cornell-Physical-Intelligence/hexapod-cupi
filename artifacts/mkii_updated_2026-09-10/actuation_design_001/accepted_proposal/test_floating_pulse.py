import json,unittest
from pathlib import Path
import numpy as np
from floating_pulse import floating_mass,prediction,skew

class FloatingStudy(unittest.TestCase):
 def setUp(self):
  self.model=json.loads((Path(__file__).resolve().parents[2]/'robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json').read_text())

 def test_kinetic_energy_and_external_momentum(self):
  M,r=floating_mass(self.model);np.testing.assert_allclose(M,M.T,atol=1e-16);np.linalg.cholesky(M)
  x=np.linspace(-.02,.03,24);energy=0.
  for link in self.model['links']:
   d=r['details'][link['name']];v=x[:3]+np.cross(x[3:6],d['com'])+d['Jv']@x[6:];w=x[3:6]+d['Jw']@x[6:]
   energy+=.5*link['mass']*(v@v)+.5*w@(d['rotation']@link['inertia']@d['rotation'].T)@w
  self.assertAlmostEqual(energy,.5*x@M@x,places=14)
  u=np.zeros(24);u[8]=.005;dv=np.linalg.solve(M,u)*.02
  np.testing.assert_allclose((M@dv)[:6],0,atol=1e-18)

 def test_old_case_rejects_and_new_fits_before_native(self):
  old=prediction(self.model,.02,8,40);new=prediction(self.model)
  self.assertGreater(old['maximum_all_cases_joint_excursion_rad'],.05)
  self.assertLess(new['maximum_all_cases_joint_excursion_rad'],.01)
  self.assertLess(new['maximum_all_cases_joint_speed_rad_s'],.4)
  self.assertTrue(all(c['applied_positive_joint_response_rad']>0 for c in new['cases']))

if __name__=='__main__':unittest.main()
