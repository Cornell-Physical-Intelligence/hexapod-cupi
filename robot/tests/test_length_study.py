"""CPU-only regression checks for the morphology experiment."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("study", ROOT / "robot/tools/generate_length_study.py")
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


class LengthStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name)
        cls.config = json.loads(study.DEFAULT_CONFIG.read_text())
        with np.errstate(all="raise"):
            cls.manifest = study.generate(output=cls.output)
        cls.base, cls.mapping, cls.transfer, cls.clouds = study.prepare(cls.config)
        cls.source = ET.parse(ROOT / cls.config["inertia_urdf"]).getroot()
        cls.mock = ET.parse(ROOT / cls.config["geometry_urdf"]).getroot()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def robot(self, name):
        return ET.parse(self.output / "urdf" / f"{name}.urdf").getroot()

    def test_all_49_combinations_and_individual_source_masses(self):
        self.assertEqual(len(self.manifest["variants"]), 49)
        self.assertEqual(sum(v["is_unchanged_length_baseline"] for v in self.manifest["variants"]), 1)
        for record in self.manifest["variants"]:
            robot = self.robot(record["variant"])
            self.assertTrue(study.validate(robot, self.output)["static_structure_pass"])
            self.assertAlmostEqual(record["audit"]["total_mass_kg"], 8.26081134, places=8)
            self.assertFalse(record["screening"]["training_ready"])
            for name, transfer in self.transfer.items():
                source = self.source.find(f"link[@name='{transfer['source_link']}']")
                target = robot.find(f"link[@name='{name}']")
                self.assertAlmostEqual(study.inertial(source)[0], study.inertial(target)[0], places=10)

    def test_baseline_joint_transforms_and_geometry_match_archive(self):
        robot = self.robot("f100_t100")
        for joint in robot.findall("joint"):
            old = self.mock.find(f"joint[@name='{joint.get('name')}']")
            np.testing.assert_allclose(study.origin(joint), study.origin(old), atol=1e-12)
            for tag in ("parent", "child", "axis", "limit"):
                self.assertEqual(joint.find(tag).attrib, old.find(tag).attrib)
        for link in robot.findall("link"):
            old = self.mock.find(f"link[@name='{link.get('name')}']")
            for tag in ("visual", "collision"):
                np.testing.assert_allclose(study.origin(link.find(tag)), study.origin(old.find(tag)), atol=1e-12)
                self.assertEqual(link.find(f"{tag}/geometry/mesh").get("scale"), "1 1 1")

    def test_coxa_and_mounts_never_change(self):
        baseline = self.robot("f100_t100")
        for record in self.manifest["variants"]:
            robot = self.robot(record["variant"])
            for leg in self.mapping.values():
                name = leg["links"]["coxa"]
                self.assertEqual(ET.tostring(robot.find(f"link[@name='{name}']")), ET.tostring(baseline.find(f"link[@name='{name}']")))
                for kind in ("coxa", "femur"):
                    name = leg["joints"][kind]
                    self.assertEqual(ET.tostring(robot.find(f"joint[@name='{name}']")), ET.tostring(baseline.find(f"joint[@name='{name}']")))

    def test_tensor_rotation_preserves_energy_and_principal_moments(self):
        for name, transfer in self.transfer.items():
            source = self.source.find(f"link[@name='{transfer['source_link']}']")
            target = self.base.find(f"link[@name='{name}']")
            _, source_com, source_tensor = study.inertial(source)
            _, target_com, target_tensor = study.inertial(target)
            rotation = np.asarray(transfer["source_to_mock_rotation"])
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=12)
            np.testing.assert_allclose(np.linalg.eigvalsh(source_tensor), np.linalg.eigvalsh(target_tensor), atol=1e-12)
            np.testing.assert_allclose(target_com, rotation @ source_com + transfer["translation_m"], atol=1e-12)
            omega = np.array([0.3, -1.2, 0.75])
            self.assertAlmostEqual(float(omega @ source_tensor @ omega), float((rotation @ omega) @ target_tensor @ (rotation @ omega)), places=12)

    def test_two_independent_lengths_move_geometry_and_com_about_joint(self):
        for sf, st, name in ((0.5, 1.1, "f050_t110"), (1.1, 0.5, "f110_t050")):
            robot = self.robot(name)
            for leg in self.mapping.values():
                knee = robot.find(f"joint[@name='{leg['joints']['tibia']}']")
                self.assertAlmostEqual(study.origin(knee)[1, 3], 0.145 * sf)
                for kind, scale in (("femur", sf), ("tibia", st)):
                    link = robot.find(f"link[@name='{leg['links'][kind]}']")
                    base = self.base.find(f"link[@name='{leg['links'][kind]}']")
                    m, com, tensor = study.inertial(link)
                    bm, bcom, btensor = study.inertial(base)
                    self.assertEqual(m, bm)
                    np.testing.assert_allclose(com, bcom * [1, scale, 1], atol=1e-12)
                    np.testing.assert_allclose(tensor, btensor, atol=1e-14)
                    for tag in ("visual", "collision"):
                        node, old = link.find(tag), base.find(tag)
                        points = study.stl_vertices(study.mesh_path(old.find("geometry/mesh").get("filename")))
                        actual = points * study.vec(node.find("geometry/mesh").get("scale")) + study.origin(node)[:3, 3]
                        expected = (points + study.origin(old)[:3, 3]) * [1, scale, 1]
                        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_static_balance_and_ground_intersection_are_reported(self):
        baseline = next(v for v in self.manifest["variants"] if v["variant"] == "f100_t100")
        self.assertTrue(baseline["six_foot_geometry_eligible"])
        self.assertLess(baseline["screening"]["balance_residual"], 1e-8)
        self.assertAlmostEqual(sum(baseline["screening"]["normal_reactions_n"]), 8.26081134 * 9.81, places=7)
        # A long femur with a very short tibia hits the chassis at this common
        # pose: a low torque alone must never mark this variant as admitted.
        short = next(v for v in self.manifest["variants"] if v["variant"] == "f110_t050")
        self.assertFalse(short["six_foot_geometry_eligible"])
        self.assertLess(short["screening"]["nonfoot_mesh_vertex_clearance_m"], 0)

    def test_motor_continuous_and_peak_are_distinct(self):
        cfg = self.manifest["actuator_config_snapshot"]
        self.assertEqual(cfg["effort_limit"], 1.6)
        self.assertEqual(cfg["saturation_effort"], 5.5)
        self.assertEqual(cfg["armature"], 0.0007)
        self.assertAlmostEqual(cfg["velocity_limit"], 480 * 2 * np.pi / 60)

    def test_generation_is_reproducible_and_sources_are_unchanged(self):
        before = {path: study.digest(ROOT / path) for path in (self.config["geometry_urdf"], self.config["inertia_urdf"])}
        again = study.generate(output=self.output)
        self.assertEqual(again, self.manifest)
        self.assertEqual(before, {path: study.digest(ROOT / path) for path in before})


if __name__ == "__main__":
    unittest.main()
