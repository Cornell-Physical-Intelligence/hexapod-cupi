"""Small numerical regression checks; full artifact audits run in the builder."""
import unittest
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
from build_updated_urdf import origin_xml, inertia_sum, frame

class BuilderNumerics(unittest.TestCase):
    def test_rpy_serialization_at_and_near_both_gimbal_poles(self):
        for pitch in [np.pi/2, -np.pi/2, np.pi/2-1e-14, -np.pi/2+1e-14, np.pi/2-1e-7]:
            for yaw,roll in [(2.30089273,-2.7536246),(-.814,-2.806),(0.,.3)]:
                T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',[roll,pitch,yaw]).as_matrix()
                # Reproduce the roundoff introduced by changing link frames.
                U=np.eye(4);U[:3,:3]=Rotation.from_euler('xyz',[.31,-.19,.44]).as_matrix()
                T=np.linalg.inv(U)@(U@T)
                parent=ET.Element('visual');origin_xml(parent,T)
                actual=Rotation.from_euler('xyz',np.fromstring(parent.find('origin').get('rpy'),sep=' ')).as_matrix()
                np.testing.assert_allclose(actual,T[:3,:3],atol=1e-12,rtol=0)

    def test_full_tensor_and_off_diagonal_parallel_axis_sum(self):
        I=np.array([[.3,.01,.02],[.01,.4,.03],[.02,.03,.5]])
        parts=[{'mass':2.,'com':[1.,2.,3.],'inertia':I.tolist()}, {'mass':2.,'com':[-1.,2.,3.],'inertia':I.tolist()}]
        mass,com,total=inertia_sum(parts)
        self.assertEqual(mass,4.);np.testing.assert_allclose(com,[0.,2.,3.])
        np.testing.assert_allclose(total,2*I+np.diag([0.,4.,4.]))

    def test_joint_frame_is_right_handed_with_requested_positive_axis(self):
        T=frame([.1,.2,.3],[1.,0.,.2],[0.,-1.,0.])
        np.testing.assert_allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(T[:3,:3]),1.)
        self.assertGreater(np.cross(T[:3,2],T[:3,0])[2],0.)

if __name__=='__main__':unittest.main()
