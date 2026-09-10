"""Independent kinematic, gravity and force-balance checks for the CPU screen."""
import json
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools"))
import screen_length_mechanics as screen


class MechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package=ROOT/"robot/hexapod_mkii_length_study"
        cls.mapping=json.loads((cls.package/"manifest.json").read_text())["link_joint_mapping"]
        cls.xml=ET.parse(cls.package/"urdf/f070_t070.urdf").getroot()
        cls.robot=screen.Robot(cls.xml,cls.mapping,screen.mesh_clouds(cls.xml,cls.package))

    def test_analytic_ik_and_jacobian_match_exact_fk_all_legs(self):
        rng=np.random.default_rng(102)
        for leg in self.robot.legs:
            for _ in range(10):
                q=rng.uniform(leg.lower+.1,leg.upper-.1)
                row=leg.fk(q)
                recovered,error=leg.ik(row["foot"],q,leg.toe)
                self.assertLess(error,1e-6)
                np.testing.assert_allclose(recovered,q,atol=8e-6)
                for j in range(3):
                    dq=np.eye(3)[j]*1e-6
                    finite=(leg.fk(q+dq)["foot"]-leg.fk(q-dq)["foot"])/2e-6
                    np.testing.assert_allclose(finite,row["J"][:,j],atol=2e-9)

    def test_leg_gravity_matches_potential_energy_gradient(self):
        for leg in self.robot.legs:
            q=np.array([.2,.6,1.8]);row=leg.fk(q)
            def potential(q):return sum(mass*9.81*p[2] for (mass,_),p in zip(leg.properties,leg.fk(q)["com"]))
            for j in range(3):
                dq=np.eye(3)[j]*1e-6
                self.assertAlmostEqual((potential(q+dq)-potential(q-dq))/2e-6,row["gravity"][j],places=7)

    def test_contact_solution_respects_balance_friction_and_peak(self):
        rows,com=self.robot.pose(np.tile(np.radians([0,30,120]),(6,1)))
        for support in screen.SUPPORTS["tripod"]+screen.SUPPORTS["wave"]:
            load=self.robot.load(rows,com,support)
            self.assertIsNotNone(load)
            forces=np.array(load["forces_n"])
            np.testing.assert_allclose(forces.sum(axis=0),[0,0,self.robot.mass*9.81],atol=1e-8)
            moment=sum((np.cross(row["foot"],f) for row,f in zip(rows,forces)),np.zeros(3))
            np.testing.assert_allclose(moment,np.cross(com,[0,0,self.robot.mass*9.81]),atol=1e-8)
            self.assertGreaterEqual(forces[:,2].min(),-1e-9)
            self.assertTrue(np.all(np.abs(forces[:,:2]).sum(axis=1)<=.8*forces[:,2]+1e-8))
            self.assertAlmostEqual(np.abs(load["torques_nm"]).max(),load["peak_torque_nm"])
        self.assertIsNone(self.robot.load(rows,com+np.array([5.,0,0]),screen.SUPPORTS["tripod"][0]))

    def test_straight_leg_unreachable_target_is_not_silently_accepted(self):
        leg=self.robot.legs[0];q=np.array([0,.4,1.5])
        _,error=leg.ik(np.array([3.,-3.,-2.]),q,leg.toe)
        self.assertGreater(error,1.)


if __name__=="__main__":unittest.main()
