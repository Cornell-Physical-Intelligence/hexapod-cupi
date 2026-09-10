from pathlib import Path
import unittest,json
import numpy as np
from inertia_probe import calculate,frames
MODEL=Path(__file__).resolve().parents[2]/'robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json'
class Study(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.m=json.loads(MODEL.read_text())
 def test_spd_symmetry_and_name_permutation(self):
  a=calculate(self.m);b=calculate(self.m,order=list(reversed(a['order'])))
  np.testing.assert_allclose(a['M'],a['M'].T,atol=1e-14)
  self.assertGreater(np.linalg.eigvalsh(a['M'])[0],0.)
  np.testing.assert_allclose(a['M'],b['M'][::-1,::-1],atol=1e-14)
  np.testing.assert_allclose(a['hold'],b['hold'][::-1],atol=1e-14)
 def test_gravity_gradient_independent_potential_difference(self):
  q=np.linspace(-.02,.02,18);r=calculate(self.m,q);eps=1e-6
  fd=[]
  for j in range(18):
   p=q.copy();p[j]+=eps;n=q.copy();n[j]-=eps
   fd.append((calculate(self.m,p)['potential']-calculate(self.m,n)['potential'])/(2*eps))
  np.testing.assert_allclose(fd,r['hold'],atol=5e-9)
 def test_kinetic_energy_against_finite_difference_link_motion(self):
  q=np.linspace(-.02,.02,18);v=np.sin(np.arange(18));eps=1e-6
  a=calculate(self.m,q);p=calculate(self.m,q+eps*v);n=calculate(self.m,q-eps*v);energy=0.
  for link in self.m['links']:
   d=a['details'][link['name']];dp=p['details'][link['name']];dn=n['details'][link['name']]
   cv=(dp['com']-dn['com'])/(2*eps);Rd=(dp['rotation']-dn['rotation'])/(2*eps);W=Rd@d['rotation'].T
   w=np.array([W[2,1],W[0,2],W[1,0]]);Iw=d['rotation']@link['inertia']@d['rotation'].T
   energy+=.5*link['mass']*cv@cv+.5*w@Iw@w
  self.assertAlmostEqual(energy,.5*v@a['M']@v,places=9)
 def test_wrong_order_and_nonfinite_reject(self):
  with self.assertRaises(ValueError):calculate(self.m,order=['x']*18)
  with self.assertRaises(ValueError):calculate(self.m,np.full(18,np.nan))
if __name__=='__main__':unittest.main()
