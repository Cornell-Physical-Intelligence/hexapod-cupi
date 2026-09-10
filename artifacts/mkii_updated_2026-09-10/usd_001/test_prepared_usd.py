"""Meaningful negative checks against a complete saved USD bundle.
Run: python test_prepared_usd.py /path/to/rs05_mass_corrected
"""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from prepare_updated_usd import audit

BUNDLE = Path(sys.argv.pop(1)).resolve()

class PreparedAssetRegression(unittest.TestCase):
    def check_mutation_rejected(self, mutate, expected_message):
        with tempfile.TemporaryDirectory(prefix='usd-regression-', dir=BUNDLE.parent) as directory:
            copy = Path(directory) / 'bundle'
            shutil.copytree(BUNDLE, copy)
            stage = Usd.Stage.Open(str(copy / 'robot.usda'))
            mutate(stage)
            stage.GetRootLayer().Save()
            stage = None
            report = audit(copy)
            self.assertFalse(report['pass'])
            self.assertTrue(any(expected_message in e for e in report['errors']), report['errors'])

    def test_saved_asset_passes(self):
        self.assertTrue(audit(BUNDLE)['pass'])

    def test_inverse_principal_axis_bug_is_rejected(self):
        def mutate(stage):
            api = UsdPhysics.MassAPI(stage.GetPrimAtPath('/Robot/lf_coxa'))
            api.GetPrincipalAxesAttr().Set(api.GetPrincipalAxesAttr().Get().GetInverse())
        self.check_mutation_rejected(mutate, 'Body roundtrip lf_coxa')

    def test_joint_pivot_displacement_is_rejected(self):
        def mutate(stage):
            api = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath('/Robot/joints/lf_femur_pitch'))
            p = api.GetLocalPos0Attr().Get()
            api.GetLocalPos0Attr().Set(Gf.Vec3f(p[0] + 0.001, p[1], p[2]))
        self.check_mutation_rejected(mutate, 'Joint frames/limits lf_femur_pitch')

    def test_collision_omission_is_rejected(self):
        def mutate(stage):
            p = next(p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI))
            p.RemoveAPI(UsdPhysics.CollisionAPI)
        self.check_mutation_rejected(mutate, 'Selected collision membership differs')

    def test_same_count_wrong_visual_mesh_is_rejected(self):
        def mutate(stage):
            p = stage.GetPrimAtPath('/Robot/body/visuals/part_0000')
            p.GetReferences().ClearReferences()
            p.GetReferences().AddReference('./geometry.usdc', '/Geometry/mesh_016')
        self.check_mutation_rejected(mutate, 'Visual reference mismatch part_0000')

if __name__ == '__main__':
    unittest.main()
