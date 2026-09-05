"""Small analytic and source-identity checks for the standalone fit audit."""
import json
from pathlib import Path
import unittest

import numpy as np

import audit


class ColliderFitMathTests(unittest.TestCase):
    def shape(self,kind):
        return {'kind':kind,'transform':np.eye(4),'radius':1.,'length':4.,'half_size':np.array([1.,2.,3.])}

    def test_exact_primitive_distances_including_corner_and_rim(self):
        p=np.array([[0.,0.,0.],[2.,0.,0.],[2.,3.,4.]])
        np.testing.assert_allclose(audit.signed_distance(p,self.shape('sphere')),[-1,1,np.sqrt(29)-1])
        np.testing.assert_allclose(audit.signed_distance(p,self.shape('box')),[-1,1,np.sqrt(3)])
        np.testing.assert_allclose(audit.signed_distance(np.array([[0.,0.,0.],[2.,0.,3.]]),self.shape('cylinder')),[-1,np.sqrt(2)])

    def test_oriented_box_support_uses_rotation_and_translation(self):
        shape=self.shape('box');shape['transform'][:3,:3]=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        shape['transform'][:3,3]=[4,5,6]
        np.testing.assert_allclose(audit.support([shape],np.eye(3)),[6,6,9])

    def test_two_sphere_union_keeps_waist_gap(self):
        a=self.shape('sphere');b=self.shape('sphere')
        a['transform'][0,3]=-.9;b['transform'][0,3]=.9
        actual=audit.union_distance(np.array([[0.,.8,0.]]),[a,b])[0]
        self.assertAlmostEqual(actual,np.sqrt(.9**2+.8**2)-1)
        self.assertGreater(actual,0.)

    def test_report_binds_real_inputs_and_all_shapes(self):
        report=json.loads((Path(__file__).parent/'report.json').read_text())
        self.assertEqual(report['source']['audit_script_sha256'],audit.sha(audit.__file__))
        self.assertEqual(report['source']['linkage_urdf_sha256'],audit.sha(audit.kin.URDF))
        self.assertEqual(report['source']['kinematics_sha256'],audit.sha(audit.kin.CONTRACT))
        self.assertEqual(len(report['links']),31)
        self.assertEqual(sum(x['colliders']for x in report['links']),171)
        self.assertEqual(len(report['parts']),1927)
        self.assertEqual(len(report['source']['meshes_sha256']),77)
        self.assertEqual(len(report['footpads']),6)


if __name__=='__main__':unittest.main()
