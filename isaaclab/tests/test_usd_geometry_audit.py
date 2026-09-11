"""CPU OpenUSD geometry regressions; no Isaac Sim or GPU is required."""

from pathlib import Path
import sys
import tempfile
import unittest

from pxr import Gf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from tools.assets.usd_geometry_audit import validate_geometry  # noqa: E402


URDF = """<robot name="fixture">
  <link name="body"/>
  <link name="leg">
    <collision><origin xyz="0.01 0.05 0.04" rpy="0 0 1.5707963267948966"/>
      <geometry><box size="0.04 0.02 0.06"/></geometry></collision>
    <collision><origin xyz="-0.02 0.01 0.02" rpy="0 1.5707963267948966 0"/>
      <geometry><cylinder radius="0.015" length="0.08"/></geometry></collision>
    <collision><origin xyz="0.01 0.02 -0.3" rpy="0.3 -1 0.4"/>
      <geometry><sphere radius="0.02"/></geometry></collision>
    <collision><origin xyz="0.05 0.02 -0.3"/>
      <geometry><sphere radius="0.02"/></geometry></collision>
  </link>
  <joint name="hip" type="revolute">
    <parent link="body"/><child link="leg"/>
    <origin xyz="0.2 -0.1 0.3" rpy="0 0 1.5707963267948966"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="0.5" effort="2" velocity="3"/>
  </joint>
</robot>"""


class GeometryAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.urdf = Path(self.directory.name) / "robot.urdf"
        self.urdf.write_text(URDF)
        self.stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(self.stage, 1)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.Xform.Define(self.stage, "/Robot")
        body = UsdGeom.Xform.Define(self.stage, "/Robot/body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.ArticulationRootAPI.Apply(body.GetPrim())
        self.leg = UsdGeom.Xform.Define(self.stage, "/Robot/body/leg")
        self.leg.AddTranslateOp().Set(Gf.Vec3d(0.2, -0.1, 0.3))
        self.leg.AddRotateZOp().Set(90)
        UsdPhysics.RigidBodyAPI.Apply(self.leg.GetPrim())
        self.joint = UsdPhysics.RevoluteJoint.Define(self.stage, "/Robot/hip")
        self.joint.CreateBody0Rel().SetTargets([body.GetPath()])
        self.joint.CreateBody1Rel().SetTargets([self.leg.GetPath()])
        self.joint.CreateAxisAttr("Z")
        self.joint.CreateLocalPos0Attr(Gf.Vec3f(0.2, -0.1, 0.3))
        self.joint.CreateLocalRot0Attr(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90).GetQuat()))
        self.joint.CreateLowerLimitAttr(-57.29577951308232)
        self.joint.CreateUpperLimitAttr(28.64788975654116)
        box = UsdGeom.Cube.Define(self.stage, "/Robot/body/leg/box")
        box.CreateSizeAttr(1)
        box.AddTranslateOp().Set(Gf.Vec3d(0.01, 0.05, 0.04))
        # Swapping box X/Y dimensions and omitting its 90-degree yaw preserves
        # the same physical box, so a literal Euler-angle comparison is wrong.
        box.AddScaleOp().Set(Gf.Vec3f(0.02, 0.04, 0.06))
        cylinder = UsdGeom.Cylinder.Define(self.stage, "/Robot/body/leg/cylinder")
        cylinder.CreateAxisAttr("X")
        cylinder.CreateRadiusAttr(0.015)
        cylinder.CreateHeightAttr(0.08)
        cylinder.AddTranslateOp().Set(Gf.Vec3d(-0.02, 0.01, 0.02))
        for name, x in (("sphere", 0.01), ("sphere2", 0.05)):
            sphere = UsdGeom.Sphere.Define(self.stage, f"/Robot/body/leg/{name}")
            sphere.CreateRadiusAttr(0.02)
            sphere.AddTranslateOp().Set(Gf.Vec3d(x, 0.02, -0.3))
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
        for shape in (box, cylinder):
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())

    def audit(self):
        return validate_geometry(self.urdf, self.stage)

    def assert_failure(self, text):
        result = self.audit()
        self.assertTrue(any(text in error for error in result["errors"]), result)

    def test_equivalent_primitive_orientations_and_joint_graph_pass(self):
        result = self.audit()
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["counts"], {
            "expected_bodies": 2, "bodies": 2, "expected_joints": 1,
            "joints": 1, "expected_collisions": 4, "collisions": 4,
            "articulation_roots": 1,
        })

    def test_body_translation_is_compared_with_urdf(self):
        self.leg.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.21, -0.1, 0.3))
        self.assert_failure("body_translation_m")

    def test_global_rigid_placement_does_not_change_robot_geometry(self):
        root = UsdGeom.Xform(self.stage.GetPrimAtPath("/Robot"))
        root.AddTranslateOp().Set(Gf.Vec3d(2, -1, 0.15))
        root.AddRotateZOp().Set(30)
        self.assertEqual(self.audit()["errors"], [])

    def test_ancestor_scale_is_rejected(self):
        UsdGeom.Xform(self.stage.GetPrimAtPath("/Robot")).AddScaleOp().Set(Gf.Vec3f(1.01))
        self.assert_failure("ancestors have scale")

    def test_cancelled_ancestor_scale_is_rejected(self):
        UsdGeom.Xform(self.stage.GetPrimAtPath("/Robot")).AddScaleOp().Set(Gf.Vec3f(2))
        UsdGeom.Xform(self.stage.GetPrimAtPath("/Robot/body")).AddScaleOp().Set(Gf.Vec3f(0.5))
        self.assert_failure("ancestors have scale")

    def test_nonunit_joint_quaternion_is_rejected(self):
        self.joint.GetLocalRot1Attr().Set(Gf.Quatf(0))
        self.assert_failure("quaternion must be finite and normalized")

    def test_animated_body_is_rejected(self):
        attribute = self.leg.GetPrim().GetAttribute("xformOp:rotateZ")
        attribute.Set(90, Usd.TimeCode(0))
        attribute.Set(100, Usd.TimeCode(1))
        self.assert_failure("animated body transform")

    def test_combined_body_and_collider_world_error_is_reported(self):
        self.leg.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.20008, -0.1, 0.3))
        result = self.audit()
        self.assertGreater(result["max_errors"]["collision_world_surface_displacement_bound_m"], 0.000079)

    def test_wrong_joint_relationship_fails_despite_correct_counts(self):
        self.joint.GetBody0Rel().SetTargets([self.leg.GetPath()])
        self.assert_failure("body0/body1")

    def test_reversed_positive_axis_fails(self):
        self.joint.GetLocalRot1Attr().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), 180).GetQuat()))
        self.assert_failure("positive axis")

    def test_zero_twist_mismatch_fails_even_when_axes_match(self):
        self.joint.GetLocalRot1Attr().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), 3).GetQuat()))
        self.assert_failure("zero-pose frames")

    def test_joint_anchor_offset_fails(self):
        self.joint.GetLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0.001))
        self.assert_failure("joint_anchor_m")

    def test_joint_limit_change_fails(self):
        self.joint.GetUpperLimitAttr().Set(30)
        self.assert_failure("joint_limit_rad")

    def test_extra_fixed_joint_fails(self):
        UsdPhysics.FixedJoint.Define(self.stage, "/Robot/unexpected")
        self.assert_failure("joint names differ")

    def test_disabled_joint_fails(self):
        self.joint.CreateJointEnabledAttr(False)
        self.assert_failure("joint disabled")

    def test_missing_articulation_root_fails(self):
        self.stage.GetPrimAtPath("/Robot/body").RemoveAPI(UsdPhysics.ArticulationRootAPI)
        self.assert_failure("one floating articulation root")

    def test_kinematic_body_fails(self):
        UsdPhysics.RigidBodyAPI(self.leg.GetPrim()).CreateKinematicEnabledAttr(True)
        self.assert_failure("body must be enabled and dynamic")

    def test_disabled_body_fails(self):
        UsdPhysics.RigidBodyAPI(self.leg.GetPrim()).CreateRigidBodyEnabledAttr(False)
        self.assert_failure("body must be enabled and dynamic")

    def test_renamed_body_fails_despite_correct_count(self):
        self.stage.RemovePrim("/Robot/body/leg")
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(self.stage, "/Robot/body/other").GetPrim())
        self.assert_failure("rigid-body names differ")

    def test_missing_collision_fails(self):
        self.stage.RemovePrim("/Robot/body/leg/sphere")
        self.assert_failure("collision count")

    def test_duplicate_colliders_do_not_match_two_different_urdf_shapes(self):
        self.stage.GetPrimAtPath("/Robot/body/leg/sphere2").GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.01, 0.02, -0.3))
        self.assert_failure("no one-to-one collision match")

    def test_changed_dimensions_fail(self):
        UsdGeom.Sphere(self.stage.GetPrimAtPath("/Robot/body/leg/sphere")).GetRadiusAttr().Set(0.021)
        self.assert_failure("no one-to-one collision match")

    def test_changed_box_rotation_fails(self):
        UsdGeom.Cube(self.stage.GetPrimAtPath("/Robot/body/leg/box")).AddRotateZOp().Set(10)
        self.assert_failure("no one-to-one collision match")

    def test_ten_times_collision_angular_tolerance_fails(self):
        import math
        UsdGeom.Cylinder(self.stage.GetPrimAtPath("/Robot/body/leg/cylinder")).AddRotateZOp().Set(math.degrees(0.002))
        self.assert_failure("no one-to-one collision match")

    def test_ten_times_surface_displacement_limit_fails(self):
        self.stage.GetPrimAtPath("/Robot/body/leg/sphere").GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.011, 0.02, -0.3))
        result = self.audit()
        self.assertGreater(result["max_errors"]["collision_surface_displacement_bound_m"], 0.0009)
        self.assertTrue(result["errors"])

    def test_nonuniform_sphere_scale_fails(self):
        UsdGeom.Sphere(self.stage.GetPrimAtPath("/Robot/body/leg/sphere")).AddScaleOp().Set(Gf.Vec3f(1, 1, 1.1))
        self.assert_failure("unsupported ellipsoid")

    def test_disabled_collision_fails(self):
        UsdPhysics.CollisionAPI(self.stage.GetPrimAtPath("/Robot/body/leg/box")).CreateCollisionEnabledAttr(False)
        self.assert_failure("collision is disabled")

    def test_wrong_units_fail(self):
        UsdGeom.SetStageMetersPerUnit(self.stage, 0.01)
        self.assert_failure("metresPerUnit=1")

    def test_saved_reopened_stage_passes(self):
        path = Path(self.directory.name) / "robot.usda"
        self.stage.GetRootLayer().Export(str(path))
        self.assertEqual(validate_geometry(self.urdf, Usd.Stage.Open(str(path)))["errors"], [])


if __name__ == "__main__":
    unittest.main()
