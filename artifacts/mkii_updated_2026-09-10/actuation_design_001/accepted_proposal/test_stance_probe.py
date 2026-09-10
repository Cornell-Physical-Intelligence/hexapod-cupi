"""Independent equilibrium and virtual-work checks of the neutral static study."""
from pathlib import Path
import json,unittest
import numpy as np
from pxr import Usd
from inertia_probe import calculate,frames
from stance_probe import collision_points,evaluate

ROOT=Path(__file__).resolve().parents[2]
ASSET=ROOT/'artifacts/mkii_updated_2026-09-10/usd_002/rs05_mass_corrected'

class StaticStudy(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.model=json.loads((ASSET/'source/model.json').read_text())
  cls.clouds,cls.sources=collision_points(Usd.Stage.Open(str(ASSET/'robot.usda')))
  cls.result=evaluate(cls.model,cls.clouds)

 def test_force_and_moment_equilibrium(self):
  r=self.result;self.assertTrue(r['ideal_vertical_static_lp_success']);self.assertEqual(len(self.sources),153)
  F=np.array(r['normal_forces_n']);self.assertTrue((F>=1.).all())
  calc=calculate(self.model);forces=np.c_[np.zeros((6,2)),F]
  locations=np.array([x['body_frame_position']for x in r['feet']])
  weight=sum(x['mass']for x in self.model['links'])*9.81
  np.testing.assert_allclose(forces.sum(0),[0,0,weight],atol=1e-10)
  total_moment=np.cross(locations,forces).sum(0)
  for link in self.model['links']:
   total_moment+=np.cross(calc['details'][link['name']]['com'],[0,0,-9.81*link['mass']])
  np.testing.assert_allclose(total_moment,0,atol=1e-9)
  self.assertLess(r['foot_height_spread_m'],1e-6)
  self.assertGreater(r['non_tibia_collision_clearance_m'],.07)

 def test_joint_torques_by_independent_virtual_work(self):
  r=self.result;order=r['joint_names'];F=r['normal_forces_n'];eps=1e-6
  def loaded_potential(q):
   calc=calculate(self.model,q);T=calc['frames'];v=calc['potential']
   for f,n in zip(r['feet'],F):
    transform=T[f['leg']+'_tibia'];p=transform[:3,:3]@f['tibia_local_vertex']+transform[:3,3]
    v-=n*p[2]
   return v
  derivative=[]
  for k in range(18):
   q=np.zeros(18);q[k]=eps;a=loaded_potential(q);q[k]=-eps;b=loaded_potential(q)
   derivative.append((a-b)/(2*eps))
  np.testing.assert_allclose(derivative,r['joint_hold_torque_nm'],atol=5e-9)
  self.assertAlmostEqual(max(abs(np.array(derivative))),r['minimax_abs_joint_torque_nm'],places=8)

if __name__=='__main__':unittest.main()
