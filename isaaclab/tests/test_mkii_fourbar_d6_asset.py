"""Diagnostic planar D6 must preserve CAD physics and remove only redundant rows."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools'))
try:
    import numpy as np
    from pxr import Gf, Usd, UsdPhysics
    import mkii_fourbar_kinematics as kin
    import prepare_mkii_fourbar_usd as builder
    import audit_mkii_fourbar_usd as audit
except ImportError:
    np = None


@unittest.skipIf(np is None, 'NumPy and OpenUSD are required')
class PlanarD6AssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='fourbar_d6_')
        cls.base = Path(cls.directory.name)
        cls.output = cls.base/'candidate'/'robot.usda'
        cls.historical = ROOT/kin.USD_RELATIVE
        cls.historical_bytes = {p.name: p.read_bytes() for p in cls.historical.parent.iterdir() if p.is_file()}
        # Compare the closure variants under identical host math. Archived
        # Mac-authored matrices can differ from Linux regeneration by ~1e-16;
        # the original bundle has its own independent source audit below.
        cls.control = cls.base/'revolute_control'/'robot.usda'
        cls.control_report = builder.prepare(cls.control)
        cls.report = builder.prepare(cls.output, closure_variant=builder.PLANAR_D6_CLOSURE)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def clone(self, name):
        dest = self.base/name
        shutil.copytree(self.output.parent, dest)
        return dest/self.output.name

    def test_roundtrip_and_exact_preservation_of_all_nonclosure_data(self):
        self.assertTrue(self.report['pass'], self.report['errors'])
        self.assertTrue(self.control_report['pass'], self.control_report['errors'])
        self.assertEqual(self.control_report['closure_constraint_variant'], builder.REVOLUTE_CLOSURE)
        self.assertEqual(self.report['closure_constraint_rows_per_loop'], 2)
        self.assertEqual(self.report['closure_locked_axes'], ['transX', 'transY'])
        self.assertEqual(self.report['closure_constraint_variant'], builder.PLANAR_D6_CLOSURE)
        self.assertEqual(self.report['physical_validation'], 'not_performed')
        self.assertEqual((self.report['rigid_bodies'], self.report['tree_joints'],
                          self.report['closure_joints'], self.report['active_joints']), (31, 30, 6, 18))
        self.assertEqual((self.output.parent/'kinematics.json').read_bytes(), kin.CONTRACT.read_bytes())
        old, new = Usd.Stage.Open(str(self.control)), Usd.Stage.Open(str(self.output))
        closure_names = set(json.loads(kin.CONTRACT.read_text())['closure_joint_names'])
        self.assertEqual({str(p.GetPath()) for p in old.Traverse()}, {str(p.GetPath()) for p in new.Traverse()})
        for prim in old.Traverse():
            other = new.GetPrimAtPath(prim.GetPath())
            if prim.GetName() in closure_names:
                self.assertEqual(prim.GetTypeName(), 'PhysicsRevoluteJoint')
                self.assertEqual(other.GetTypeName(), 'PhysicsJoint')
                # Both physical endpoints and the actual cut-local axes remain
                # bit-identical, so existing complete 3D residual gates apply.
                for key in ('physics:localPos0', 'physics:localPos1', 'physics:localRot0', 'physics:localRot1'):
                    self.assertEqual(prim.GetAttribute(key).Get(), other.GetAttribute(key).Get())
                for key in ('physics:body0', 'physics:body1'):
                    self.assertEqual(prim.GetRelationship(key).GetTargets(), other.GetRelationship(key).GetTargets())
                continue
            self.assertEqual(prim.GetTypeName(), other.GetTypeName(), str(prim.GetPath()))
            self.assertEqual(prim.GetAppliedSchemas(), other.GetAppliedSchemas())
            self.assertEqual({a.GetName(): a.Get() for a in prim.GetAttributes()},
                             {a.GetName(): a.Get() for a in other.GetAttributes()}, str(prim.GetPath()))
            self.assertEqual({r.GetName(): r.GetTargets() for r in prim.GetRelationships()},
                             {r.GetName(): r.GetTargets() for r in other.GetRelationships()})
        for filename, data in self.historical_bytes.items():
            self.assertEqual((self.historical.parent/filename).read_bytes(), data)

    def test_relocation_default_v3_audit_and_explicit_variant_guard(self):
        path = self.clone('portable')
        self.assertTrue(audit.validate(kin.URDF, kin.PINS, path)['pass'])
        mismatch = audit.validate(kin.URDF, kin.PINS, path, closure_variant=builder.REVOLUTE_CLOSURE)
        self.assertFalse(mismatch['pass'])
        self.assertTrue(any('Requested closure constraint variant' in e for e in mismatch['errors']))
        original = audit.validate(kin.URDF, kin.PINS, self.historical)
        self.assertTrue(original['pass'], original['errors'])
        self.assertEqual(original['closure_constraint_rows_per_loop'], 5)
        self.assertEqual(original['closure_constraint_variant'], builder.REVOLUTE_CLOSURE)

    def test_free_axis_extra_constraint_drive_and_wrong_type_are_rejected(self):
        mutations = {
            'free_x': lambda p: UsdPhysics.LimitAPI(p, 'transX').CreateLowAttr(-2.),
            'extra_z': lambda p: UsdPhysics.LimitAPI.Apply(p, 'transZ').CreateLowAttr(1.),
            'drive': lambda p: UsdPhysics.DriveAPI.Apply(p, 'rotZ').CreateStiffnessAttr(1.),
            'wrong_type': lambda p: p.SetTypeName('PhysicsSphericalJoint'),
            'axis_override': lambda p: UsdPhysics.RevoluteJoint(p).CreateAxisAttr('X'),
            'unexcluded': lambda p: UsdPhysics.Joint(p).CreateExcludeFromArticulationAttr(False),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                path = self.clone(label)
                stage = Usd.Stage.Open(str(path))
                mutate(stage.GetPrimAtPath('/Robot/Physics/lf_tibia_loop_closure'))
                stage.GetRootLayer().Save()
                result = audit.validate(kin.URDF, kin.PINS, path)
                self.assertFalse(result['pass'], label)
                self.assertTrue(any('lf_tibia_loop_closure' in e for e in result['errors']), result['errors'])

    def test_wrong_identity_and_bad_frame_are_rejected(self):
        path = self.clone('bad_frame')
        stage = Usd.Stage.Open(str(path))
        p = stage.GetPrimAtPath('/Robot/Physics/lf_tibia_loop_closure')
        UsdPhysics.Joint(p).CreateLocalPos0Attr(Gf.Vec3f(0., 0., 0.))
        metadata = dict(stage.GetRootLayer().customLayerData)
        metadata['closure_constraint_variant'] = 'unrecognized'
        stage.GetRootLayer().customLayerData = metadata
        stage.GetRootLayer().Save()
        result = audit.validate(kin.URDF, kin.PINS, path)
        self.assertFalse(result['pass'])
        self.assertTrue(any('Unknown closure constraint variant' in e for e in result['errors']))
        self.assertTrue(any('hinge frame differs' in e for e in result['errors']))

    def test_unknown_variant_and_overwrite_fail_before_mutation(self):
        path = self.base/'invalid'/'robot.usda'
        with self.assertRaisesRegex(ValueError, 'Unknown closure'):
            builder.prepare(path, closure_variant='guess')
        self.assertFalse(path.parent.exists())
        before = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, 'immutable'):
            builder.prepare(self.output, closure_variant=builder.PLANAR_D6_CLOSURE)
        self.assertEqual(self.output.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
