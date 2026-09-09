"""CPU regression tests for USD inertia preparation; no Isaac application.

Minimal in-memory USD stages isolate the mass and transform gates. The saved
Spark dump is a regression fixture: its 19 bad tensors must fail before they
are regenerated from URDF, including after USD serialization.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location("prepare_mkii_usd_test_subject", ROOT / "tools/prepare_mkii_usd.py")
subject = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = subject
spec.loader.exec_module(subject)

URDF = ROOT / "robot/hexapod_mkii_assy/urdf/hexapod_mkii_serial.urdf"
SPARK_DUMP = ROOT / "artifacts/project_review_2026-09-04/evidence/spark_usd.json"
PREPARED = ROOT / "robot/hexapod_mkii_assy/usd/hexapod_mkii_serial_v2/hexapod_mkii_serial_v2.usda"


def independent_rpy(roll, pitch, yaw):
    """Explicit axis rotations, independent of the implementation's formula."""
    rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return rz @ ry @ rx


def independent_quaternion_matrix(quaternion):
    """Hamilton [w,x,y,z] vector rotation, independent of Gf matrix conventions."""
    w = float(quaternion.GetReal())
    x, y, z = map(float, quaternion.GetImaginary())
    # Normalize only to remove Quatf rounding after independently checking norm.
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if abs(norm - 1.0) > 1e-6:
        raise AssertionError(f"Non-unit principal-axis quaternion: {norm}")
    w, x, y, z = (component / norm for component in (w, x, y, z))
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def stage_with_units():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.Xform.Define(stage, "/Robot")
    return stage


def author_body(stage, name, inertial, *, path=None, moments=None, quaternion=None):
    prim = UsdGeom.Xform.Define(stage, path or f"/Robot/{name}").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr(float(inertial.mass))
    api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, inertial.com)))
    if moments is None:
        moments, quaternion = subject.principal_axes(inertial.tensor)
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, moments)))
    api.CreatePrincipalAxesAttr(quaternion)
    return api


def write_urdf(path, tensor, *, mass=2.3, xyz="0.12 -0.23 0.34", rpy="0.31 -0.47 0.63"):
    robot = ET.Element("robot", name="test")
    link = ET.SubElement(robot, "link", name="body")
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=xyz, rpy=rpy)
    ET.SubElement(inertial, "mass", value=str(mass))
    values = dict(ixx=tensor[0, 0], ixy=tensor[0, 1], ixz=tensor[0, 2],
                  iyy=tensor[1, 1], iyz=tensor[1, 2], izz=tensor[2, 2])
    ET.SubElement(inertial, "inertia", **{key: str(value) for key, value in values.items()})
    ET.ElementTree(robot).write(path)
    return robot


class InertiaConversionTests(unittest.TestCase):
    def test_nonzero_com_and_rpy_transform_tensor_at_com_without_parallel_axis_shift(self):
        principal_rotation = independent_rpy(-0.7, 0.41, 0.89)
        tensor = principal_rotation @ np.diag([0.7, 1.1, 1.5]) @ principal_rotation.T
        frame = independent_rpy(0.31, -0.47, 0.63)
        expected = frame @ tensor @ frame.T
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "body.urdf"
            write_urdf(path, tensor)
            actual = subject.read_inertials(path)["body"]
        self.assertEqual(actual.mass, 2.3)
        np.testing.assert_allclose(actual.com, [0.12, -0.23, 0.34], atol=0, rtol=0)
        np.testing.assert_allclose(actual.tensor, expected, atol=1e-14, rtol=0)
        moments, quaternion = subject.principal_axes(actual.tensor)
        rotation = independent_quaternion_matrix(quaternion)
        np.testing.assert_allclose(rotation @ np.diag(moments) @ rotation.T, expected, atol=1e-7, rtol=1e-6)
        # A COM translation must not change principal moments at the COM.
        np.testing.assert_allclose(moments, [0.7, 1.1, 1.5], atol=1e-14, rtol=0)

    def test_random_rotated_tensors_use_proper_principal_to_link_rotation(self):
        generator = np.random.default_rng(4921)
        for _ in range(30):
            rotation = independent_rpy(*generator.uniform(-math.pi, math.pi, 3))
            moments = np.sort(generator.uniform(0.8, 1.2, 3))
            expected = rotation @ np.diag(moments) @ rotation.T
            diagonal, quaternion = subject.principal_axes(expected)
            actual_axes = independent_quaternion_matrix(quaternion)
            self.assertAlmostEqual(np.linalg.det(actual_axes), 1.0, places=12)
            np.testing.assert_allclose(actual_axes @ actual_axes.T, np.eye(3), atol=1e-12, rtol=0)
            np.testing.assert_allclose(actual_axes @ np.diag(diagonal) @ actual_axes.T, expected, atol=1e-7, rtol=1e-6)

    def test_repeated_isotropic_and_triangle_equality_moments(self):
        rotation = independent_rpy(0.2, 0.9, -1.1)
        for diagonal in ([1, 1, 1], [1, 1, 1.9], [1, 2, 2], [1, 1, 2]):
            with self.subTest(moments=diagonal):
                tensor = rotation @ np.diag(diagonal) @ rotation.T
                moments, quaternion = subject.principal_axes(tensor)
                actual_axes = independent_quaternion_matrix(quaternion)
                self.assertGreater(np.linalg.det(actual_axes), 0.999999)
                np.testing.assert_allclose(actual_axes @ np.diag(moments) @ actual_axes.T, tensor, atol=1e-7, rtol=1e-6)

    def test_urdf_rejects_nonfinite_nonpositive_and_nonphysical_properties(self):
        cases = [
            (np.eye(3), {"mass": 0}), (np.eye(3), {"mass": -1}),
            (np.eye(3), {"mass": math.nan}), (np.eye(3), {"mass": math.inf}),
            (np.diag([-1, 1, 1]), {}), (np.diag([0, 1, 1]), {}),
            (np.diag([1, 1, 3]), {}), (np.diag([1, 1, math.nan]), {}),
            (np.diag([1, 1, math.inf]), {}),
            (np.eye(3), {"xyz": "0 nan 0"}), (np.eye(3), {"rpy": "0 inf 0"}),
            (np.eye(3), {"xyz": "0 0"}), (np.eye(3), {"rpy": "0 0 0 0"}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.urdf"
            for index, (tensor, kwargs) in enumerate(cases):
                with self.subTest(case=index), self.assertRaises(ValueError):
                    write_urdf(path, tensor, **kwargs)
                    subject.read_inertials(path)

    def test_urdf_rejects_duplicate_missing_names_or_missing_inertial(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.urdf"
            for kind in ("duplicate", "missing_name", "missing_inertial", "empty"):
                robot = write_urdf(path, np.eye(3))
                if kind == "duplicate":
                    robot.append(ET.fromstring(ET.tostring(robot.find("link"))))
                elif kind == "missing_name":
                    del robot.find("link").attrib["name"]
                elif kind == "missing_inertial":
                    link = robot.find("link")
                    link.remove(link.find("inertial"))
                else:
                    robot.remove(robot.find("link"))
                ET.ElementTree(robot).write(path)
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    subject.read_inertials(path)

    def test_urdf_rejects_missing_mass_tensor_components_or_duplicate_inertial(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.urdf"
            for kind in ("mass", "mass_value", "inertia", "component", "duplicate_inertial"):
                robot = write_urdf(path, np.eye(3))
                link = robot.find("link")
                inertial = link.find("inertial")
                if kind in ("mass", "inertia"):
                    inertial.remove(inertial.find(kind))
                elif kind == "mass_value":
                    del inertial.find("mass").attrib["value"]
                elif kind == "component":
                    del inertial.find("inertia").attrib["ixy"]
                else:
                    link.append(ET.fromstring(ET.tostring(inertial)))
                ET.ElementTree(robot).write(path)
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    subject.read_inertials(path)


class MassComparisonTests(unittest.TestCase):
    def setUp(self):
        self.inertial = subject.Inertial(2.3, np.array([0.12, -0.23, 0.34]), np.diag([0.7, 1.1, 1.5]))
        self.expected = {"body": self.inertial}
        self.stage = stage_with_units()
        self.api = author_body(self.stage, "body", self.inertial)

    def test_valid_mass_properties_pass(self):
        report = subject.mass_comparison(self.stage, self.expected)
        self.assertFalse(report["errors"])
        self.assertEqual(len(report["links"]), 1)

    def test_missing_mass_api_or_blocked_required_attribute_reports_rejection(self):
        for missing in ("MassAPI", "physics:mass", "physics:centerOfMass",
                        "physics:diagonalInertia", "physics:principalAxes"):
            stage = stage_with_units()
            api = author_body(stage, "body", self.inertial)
            if missing == "MassAPI":
                api.GetPrim().RemoveAPI(UsdPhysics.MassAPI)
            else:
                api.GetPrim().GetAttribute(missing).Block()
            with self.subTest(missing=missing):
                report = subject.mass_comparison(stage, self.expected)
                self.assertTrue(report["errors"])
                self.assertEqual(len(report["links"]), 1)
                self.assertFalse(report["links"][0]["inertia_ok"])

    def test_duplicate_rigid_body_name_is_rejected(self):
        author_body(self.stage, "body", self.inertial, path="/Robot/other/body")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            subject.mass_comparison(self.stage, self.expected)

    def test_missing_and_extra_rigid_bodies_are_rejected(self):
        report = subject.mass_comparison(self.stage, {"missing_body": self.inertial})
        self.assertTrue(report["errors"])
        self.assertIn("missing_body", report["errors"][0])
        self.assertIn("unexpected=['body']", report["errors"][0])

    def test_wrong_and_nonfinite_mass_or_com_are_rejected(self):
        for bad_mass in (0, -1, 2.5, math.nan, math.inf):
            self.api.GetMassAttr().Set(bad_mass)
            with self.subTest(mass=bad_mass):
                report = subject.mass_comparison(self.stage, self.expected)
                self.assertTrue(report["errors"])
                self.assertFalse(report["links"][0]["mass_ok"])
        self.api.GetMassAttr().Set(self.inertial.mass)
        for com in ((0, 0, 0), (math.nan, 0, 0), (0, math.inf, 0)):
            self.api.GetCenterOfMassAttr().Set(Gf.Vec3f(*com))
            with self.subTest(com=com):
                report = subject.mass_comparison(self.stage, self.expected)
                self.assertTrue(report["errors"])
                self.assertFalse(report["links"][0]["com_ok"])

    def test_wrong_nonunit_and_nonfinite_quaternions_are_rejected(self):
        quaternions = (
            Gf.Quatf(math.sqrt(0.5), Gf.Vec3f(0, 0, math.sqrt(0.5))),
            Gf.Quatf(0, Gf.Vec3f(0)), Gf.Quatf(2, Gf.Vec3f(0)),
            Gf.Quatf(math.nan, Gf.Vec3f(0)), Gf.Quatf(1, Gf.Vec3f(math.inf, 0, 0)),
        )
        for quaternion in quaternions:
            self.api.GetPrincipalAxesAttr().Set(quaternion)
            with self.subTest(quaternion=str(quaternion)):
                report = subject.mass_comparison(self.stage, self.expected)
                self.assertTrue(report["errors"])
                self.assertFalse(report["links"][0]["inertia_ok"])

    def test_wrong_zero_negative_and_nonfinite_diagonal_moments_are_rejected(self):
        for moments in ((0.7, 1.1, 1.8), (0, 1.1, 1.5), (-0.7, 1.1, 1.5),
                        (math.nan, 1.1, 1.5), (0.7, math.inf, 1.5)):
            self.api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*moments))
            with self.subTest(moments=moments):
                report = subject.mass_comparison(self.stage, self.expected)
                self.assertTrue(report["errors"])
                self.assertFalse(report["links"][0]["inertia_ok"])

    def test_saved_spark_19_bad_tensors_fail_then_regeneration_passes_after_serialization(self):
        expected = subject.read_inertials(URDF)
        dump = json.loads(SPARK_DUMP.read_text())
        stage = stage_with_units()
        for body in dump["bodies"]:
            attributes = body["attributes"]
            w, x, y, z = attributes["physics:principalAxes"]
            evidence_inertial = subject.Inertial(attributes["physics:mass"],
                np.array(attributes["physics:centerOfMass"]), expected[body["name"]].tensor)
            author_body(stage, body["name"], evidence_inertial,
                        moments=attributes["physics:diagonalInertia"], quaternion=Gf.Quatf(w, Gf.Vec3f(x, y, z)))
        before = subject.mass_comparison(stage, expected)
        self.assertEqual(len(before["links"]), 19)
        self.assertEqual(sum(not row["inertia_ok"] for row in before["links"]), 19)
        self.assertTrue(all(row["mass_ok"] and row["com_ok"] for row in before["links"]))
        for name, inertial in expected.items():
            api = UsdPhysics.MassAPI(stage.GetPrimAtPath(f"/Robot/{name}"))
            moments, quaternion = subject.principal_axes(inertial.tensor)
            api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*map(float, moments)))
            api.GetPrincipalAxesAttr().Set(quaternion)
            # Independent tensor reconstruction prevents implementation/helpers
            # from agreeing on the same inverse-convention error.
            rotation = independent_quaternion_matrix(quaternion)
            np.testing.assert_allclose(rotation @ np.diag(moments) @ rotation.T,
                                       inertial.tensor, rtol=1e-5, atol=1e-8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roundtrip.usda"
            stage.GetRootLayer().Export(str(path))
            reopened = Usd.Stage.Open(str(path))
            after = subject.mass_comparison(reopened, expected)
        self.assertFalse(after["errors"], after["errors"])
        self.assertEqual(len(after["links"]), 19)


class StageUnitAndScaleTests(unittest.TestCase):
    def make_stage(self):
        stage = stage_with_units()
        UsdGeom.Xform.Define(stage, "/Robot/ancestor")
        prim = UsdGeom.Xform.Define(stage, "/Robot/ancestor/body").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        return stage

    def test_rigid_rotation_and_translation_are_allowed(self):
        stage = self.make_stage()
        ancestor = UsdGeom.Xformable(stage.GetPrimAtPath("/Robot/ancestor"))
        ancestor.AddTranslateOp().Set(Gf.Vec3d(0.1, -3, 4))
        ancestor.AddRotateXYZOp().Set(Gf.Vec3f(31, -47, 63))
        subject.check_stage_units_and_scale(stage)

    def test_wrong_length_mass_and_up_axis_units_are_rejected(self):
        for kind in ("metres", "kilograms", "up_axis", "missing_units"):
            stage = self.make_stage()
            if kind == "metres":
                UsdGeom.SetStageMetersPerUnit(stage, 0.001)
            elif kind == "kilograms":
                UsdPhysics.SetStageKilogramsPerUnit(stage, 0.001)
            elif kind == "up_axis":
                UsdGeom.SetStageUpAxis(stage, "Y")
            else:
                stage.ClearMetadata("metersPerUnit")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                subject.check_stage_units_and_scale(stage)

    def test_scaled_reflected_sheared_and_nonfinite_ancestors_are_rejected(self):
        for kind in ("uniform_scale", "nonuniform_scale", "reflection", "shear", "nonfinite"):
            stage = self.make_stage()
            ancestor = UsdGeom.Xformable(stage.GetPrimAtPath("/Robot/ancestor"))
            matrix = np.eye(4)
            if kind == "uniform_scale":
                matrix[:3, :3] *= 1.01
            elif kind == "nonuniform_scale":
                matrix[0, 0] = 1.01
            elif kind == "reflection":
                matrix[0, 0] = -1
            elif kind == "shear":
                matrix[0, 1] = 0.01
            else:
                matrix[3, 0] = math.nan
            ancestor.AddTransformOp().Set(Gf.Matrix4d(*map(float, matrix.flat)))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                subject.check_stage_units_and_scale(stage)


class PreparedBundleIntegrationTests(unittest.TestCase):
    """Real 16 MB bundle checked on CPU, without starting Isaac or PhysX."""

    def assert_valid(self, path):
        report = subject.validate(URDF, path)
        self.assertTrue(report["pass"], report["errors"])
        self.assertEqual(len(report["mass_properties"]["links"]), 19)
        self.assertFalse(report["geometry"]["errors"])
        self.assertTrue(all(row["mass_ok"] and row["com_ok"] and row["inertia_ok"]
                            for row in report["mass_properties"]["links"]))
        return report

    def test_shipped_bundle_validates_from_disk(self):
        self.assertTrue(PREPARED.is_file(), "The prepared asset must ship with its integrity tests")
        self.assert_valid(PREPARED)

    def test_bundle_is_portable_after_copy_to_unrelated_directory(self):
        reference = self.assert_valid(PREPARED)
        with tempfile.TemporaryDirectory() as directory:
            relocated = Path(directory) / "different location"
            shutil.copytree(PREPARED.parent, relocated)
            report = self.assert_valid(relocated / PREPARED.name)
            self.assertEqual(report["dependencies"], reference["dependencies"])
            self.assertEqual(report["mass_properties"], reference["mass_properties"])

    def test_prepare_from_preserved_source_reproduces_physical_data(self):
        source = PREPARED.parent / "source/hexapod_mkii_serial.usda"
        reference = self.assert_valid(PREPARED)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "new_bundle/rebuilt.usda"
            result = subject.prepare(URDF, source, destination)
            self.assertTrue(result["pass"])
            self.assertEqual(len(result["source_inertia_failures"]), 19)
            report = self.assert_valid(destination)
            self.assertEqual(report["mass_properties"], reference["mass_properties"])
            # A CPU rebuild must preserve geometry and links, rather than merely
            # repairing tensors on a different imported robot.
            self.assertEqual(report["geometry"], reference["geometry"])
            self.assertTrue((destination.parent / "SHA256SUMS").is_file())
            self.assertTrue((destination.parent / "asset_validation.json").is_file())

    def test_existing_destination_and_already_prepared_source_are_refused(self):
        source = PREPARED.parent / "source/hexapod_mkii_serial.usda"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "exists/asset.usda"
            destination.parent.mkdir()
            destination.write_text("preserve this artifact\n")
            with self.assertRaisesRegex(ValueError, "never overwritten"):
                subject.prepare(URDF, source, destination)
            self.assertEqual(destination.read_text(), "preserve this artifact\n")
            nested = Path(directory) / "nested/asset.usda"
            with self.assertRaisesRegex(ValueError, "already prepared"):
                subject.prepare(URDF, PREPARED, nested)
            self.assertFalse(nested.parent.exists())

    def test_wrong_com_source_fails_without_creating_partial_output(self):
        source = PREPARED.parent / "source/hexapod_mkii_serial.usda"
        with tempfile.TemporaryDirectory() as directory:
            copied_source = Path(directory) / "source"
            shutil.copytree(source.parent, copied_source)
            edited_source = copied_source / source.name
            stage = Usd.Stage.Open(str(edited_source), load=Usd.Stage.LoadAll)
            body = subject.rigid_bodies(stage)["lf_tibia"]
            UsdPhysics.MassAPI(body).CreateCenterOfMassAttr(Gf.Vec3f(0.5, 0.0, 0.0))
            stage.GetRootLayer().Save()
            del stage
            destination = Path(directory) / "must_not_exist/prepared.usda"
            with self.assertRaisesRegex(ValueError, "mass or COM mismatch"):
                subject.prepare(URDF, edited_source, destination)
            self.assertFalse(destination.exists())
            self.assertFalse(destination.parent.exists())

if __name__ == "__main__":
    unittest.main()
