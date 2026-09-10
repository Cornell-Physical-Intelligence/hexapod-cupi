"""Exercise wide-angle audit on a temporary predecessor fixture, never a new asset."""
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from prepare_updated_usd import fk_model, gf_matrix
from audit_joint_review_usd import audit_successor

PREDECESSOR = Path(sys.argv.pop(1)).resolve()

class WideAngularAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='wide-audit-fixture-', dir=Path(__file__).parent)
        self.bundle = Path(self.tmp.name) / 'fixture'
        shutil.copytree(PREDECESSOR, self.bundle)
        model_path = self.bundle / 'source/model.json'
        d = json.loads(model_path.read_text())
        source_path = self.bundle / 'source/source.urdf'
        tree = ET.parse(source_path)
        stage = Usd.Stage.Open(str(self.bundle / 'robot.usda'))
        for j in d['joints']:
            bounds = [-120., 80.] if j['name'].endswith('_femur_pitch') else [-5., 180.] if j['name'].endswith('_tibia_pitch') else None
            if bounds is None:
                continue
            j['lower'], j['upper'] = map(math.radians, bounds)
            node = tree.getroot().find(f"joint[@name='{j['name']}']/limit")
            node.set('lower', repr(j['lower']))
            node.set('upper', repr(j['upper']))
            api = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath('/Robot/joints/' + j['name']))
            api.GetLowerLimitAttr().Set(bounds[0])
            api.GetUpperLimitAttr().Set(bounds[1])
        model_path.write_text(json.dumps(d))
        tree.write(source_path)
        stage.GetRootLayer().Save()

    def tearDown(self):
        self.tmp.cleanup()

    def test_positive180_endpoint_and_wide_ranges(self):
        r = audit_successor(self.bundle)
        self.assertTrue(r['pass'], r['errors'])
        self.assertEqual(r['kinematic_sample_count'], 45)

    def test_wrapping_positive180_to_negative180_is_rejected(self):
        stage = Usd.Stage.Open(str(self.bundle / 'robot.usda'))
        UsdPhysics.RevoluteJoint(stage.GetPrimAtPath('/Robot/joints/lf_tibia_pitch')).GetUpperLimitAttr().Set(-180.)
        stage.GetRootLayer().Save()
        r = audit_successor(self.bundle)
        self.assertFalse(r['pass'])
        self.assertIn('Positive180 endpoint was wrapped or changed: lf_tibia_pitch', r['errors'])

    def test_small_yaw_quatf_roundoff_is_not_a_false_failure(self):
        model_path = self.bundle / 'source/model.json'
        d = json.loads(model_path.read_text())
        angle = math.radians(-0.515)
        name = 'lm_coxa_yaw'
        j = next(j for j in d['joints'] if j['name'] == name)
        j['quaternion_xyzw'] = [0., 0., math.sin(angle / 2), math.cos(angle / 2)]
        model_path.write_text(json.dumps(d))
        source_path = self.bundle / 'source/source.urdf'
        tree = ET.parse(source_path)
        tree.getroot().find(f"joint[@name='{name}']/origin").set('rpy', f'0 0 {angle!r}')
        tree.write(source_path)
        stage = Usd.Stage.Open(str(self.bundle / 'robot.usda'))
        for link_name, frame in fk_model(d).items():
            UsdGeom.Xformable(stage.GetPrimAtPath('/Robot/' + link_name)).GetOrderedXformOps()[0].Set(gf_matrix(frame))
        stored = Gf.Quatf(math.cos(angle / 2), Gf.Vec3f(0., 0., math.sin(angle / 2)))
        UsdPhysics.RevoluteJoint(stage.GetPrimAtPath('/Robot/joints/' + name)).GetLocalRot0Attr().Set(stored)
        self.assertGreater(abs(Gf.Rotation(Gf.Quatd(stored)).GetAngle() - 0.515), 0.0001)
        stage.GetRootLayer().Save()
        r = audit_successor(self.bundle)
        self.assertTrue(r['pass'], r['errors'])

    def test_wrong_joint_relationship_is_rejected(self):
        stage = Usd.Stage.Open(str(self.bundle / 'robot.usda'))
        UsdPhysics.RevoluteJoint(stage.GetPrimAtPath('/Robot/joints/lf_tibia_pitch')).GetBody1Rel().SetTargets(['/Robot/lm_tibia'])
        stage.GetRootLayer().Save()
        r = audit_successor(self.bundle)
        self.assertFalse(r['pass'])
        self.assertIn('USD joint relationship mismatch: lf_tibia_pitch', r['errors'])

if __name__ == '__main__':
    unittest.main()
