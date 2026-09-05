"""The v5 physical coupling must retain geometry and transmit constraint work."""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import numpy as np
from pxr import Usd, UsdPhysics

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import prepare_mkii_fourbar_usd as builder
import audit_mkii_fourbar_usd as audit
import mkii_fourbar_kinematics as kin


class PhysicalMimicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='mkii_physical_mimic_')
        cls.base=Path(cls.tmp.name)
        cls.control=cls.base/'control'/'model.usda'
        cls.path=ROOT/'robot/hexapod_mkii_assy/usd/hexapod_mkii_fourbar_v5/hexapod_mkii_fourbar_v5.usda'
        builder.prepare(cls.control)
        cls.report=audit.validate(kin.URDF,kin.PINS,cls.path,closure_variant=builder.PHYSICAL_MIMIC_CLOSURE)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_all_body_geometry_inertia_limits_and_drives_preserved(self):
        r=self.report
        self.assertTrue(r['pass'],r['errors'])
        self.assertEqual((r['rigid_bodies'],r['tree_joints'],r['active_joints'],r['closure_joints'],r['physical_mimic_constraints']), (31,30,18,0,12))
        self.assertLess(r['prescribed_closure_maxima']['point_error_m'],2e-7)
        self.assertEqual((self.path.parent/'geometry.usdc').read_bytes(),(self.control.parent/'geometry.usdc').read_bytes())
        self.assertEqual((self.path.parent/'kinematics.json').read_bytes(),kin.CONTRACT.read_bytes())
        a,b=Usd.Stage.Open(str(self.control)),Usd.Stage.Open(str(self.path))
        for p in a.Traverse():
            if p.GetName().endswith('_tibia_loop_closure'):
                self.assertFalse(b.GetPrimAtPath(p.GetPath()));continue
            q=b.GetPrimAtPath(p.GetPath());self.assertTrue(q)
            self.assertEqual(p.GetTypeName(),q.GetTypeName())
            attrs=lambda prim:{v.GetName():v.Get() for v in prim.GetAttributes() if 'mimic' not in v.GetName().lower()}
            self.assertEqual(attrs(p),attrs(q),str(p.GetPath()))
            rels=lambda prim:{v.GetName():v.GetTargets() for v in prim.GetRelationships() if 'mimic' not in v.GetName().lower()}
            self.assertEqual(rels(p),rels(q))

    def test_gearing_reference_compliance_and_extra_drive_fail(self):
        mutations={
            'wrong_gearing':lambda p:p.GetAttribute('physxMimicJoint:rotZ:gearing').Set(1.),
            'offset':lambda p:p.GetAttribute('physxMimicJoint:rotZ:offset').Set(.1),
            'compliance':lambda p:p.GetAttribute('physxMimicJoint:rotZ:naturalFrequency').Set(100.),
            'reference':lambda p:p.GetRelationship('physxMimicJoint:rotZ:referenceJoint').SetTargets(['/Robot/Physics/rf_tibia_lever_pivot']),
            'drive':lambda p:UsdPhysics.DriveAPI.Apply(p,'angular').CreateStiffnessAttr(1.),
        }
        for label,mutate in mutations.items():
            dest=self.base/label;shutil.copytree(self.path.parent,dest)
            path=dest/self.path.name;stage=Usd.Stage.Open(str(path))
            mutate(stage.GetPrimAtPath('/Robot/Physics/lf_tibia_pitch'));stage.GetRootLayer().Save()
            result=audit.validate(kin.URDF,kin.PINS,path)
            self.assertFalse(result['pass'],label)

    def test_bilateral_constraint_impulses_do_no_work_on_allowed_motion(self):
        # [lever, knee, rod] velocities on the closed parallelogram branch
        # are [v,v,-v]. Both constraint reaction vectors are orthogonal to
        # this tangent, preserving virtual work of the physical full-body tree.
        stage=Usd.Stage.Open(str(self.path))
        for leg in kin.LEGS:
            gears=[stage.GetPrimAtPath('/Robot/Physics/'+leg+'_'+suffix).GetAttribute(
                'physxMimicJoint:rotZ:gearing').Get() for suffix in ('tibia_pitch','tibia_rod_pivot')]
            constraint=np.array([[gears[0],1.,0.],[gears[1],0.,1.]])
            tangent=np.array([1.,1.,-1.])
            np.testing.assert_array_equal(constraint@tangent,np.zeros(2))
            for impulses in (np.array([3.,-7.]),np.array([-2.,5.])):
                reaction=constraint.T@impulses
                self.assertEqual(float(reaction@tangent),0.)
                self.assertNotEqual(reaction[0],0.) # reference motor receives reaction


if __name__=='__main__':unittest.main()
