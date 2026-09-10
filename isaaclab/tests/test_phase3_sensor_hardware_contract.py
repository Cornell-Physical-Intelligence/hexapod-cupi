"""CPU/static contracts for Phase3 sensor mounts and hardware accounting."""

from __future__ import annotations

import ast
import math
import unittest
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).parents[1]
SENSOR_CFG_PATH = ROOT / "hexapod_phase3" / "sensor_cfg.py"
SMOKE_PATH = ROOT / "phase3_sensor_smoke.py"
PHASE3_README_PATH = ROOT / "hexapod_phase3" / "README.md"
SENSOR_README_PATH = ROOT.parent / "robot" / "sensors" / "README.md"

SENSOR_CFG_SOURCE = SENSOR_CFG_PATH.read_text()
SENSOR_CFG_TREE = ast.parse(SENSOR_CFG_SOURCE)
SMOKE_SOURCE = SMOKE_PATH.read_text()


def _class(name: str) -> ast.ClassDef:
    for node in SENSOR_CFG_TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"Missing class {name}")


def _function(name: str) -> ast.FunctionDef:
    for node in SENSOR_CFG_TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Missing function {name}")


def _literal(class_node: ast.ClassDef, name: str):
    for node in class_node.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and node.value is not None
        ):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing {class_node.name}.{name}")


def _compiled_class(name: str):
    node = _class(name)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"dataclass": dataclass}
    exec(compile(module, SENSOR_CFG_PATH, "exec"), namespace)
    return namespace[name]


def _rotate_wxyz(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    rotation = (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ),
        (
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ),
        (
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )
    return tuple(
        sum(rotation[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )


class D455MountQuaternionTest(unittest.TestCase):
    def test_optical_forward_is_body_minus_y_and_twenty_degrees_down(self):
        mounts = _class("Phase3MountAssumptions")
        quaternion = _literal(mounts, "camera_quat_b_wxyz")
        self.assertEqual(_literal(mounts, "camera_downward_pitch_deg"), 20.0)
        self.assertAlmostEqual(sum(value * value for value in quaternion), 1.0, places=14)

        forward_b = _rotate_wxyz(quaternion, (1.0, 0.0, 0.0))
        expected = (
            0.0,
            -math.cos(math.radians(20.0)),
            -math.sin(math.radians(20.0)),
        )
        for actual, target in zip(forward_b, expected):
            self.assertAlmostEqual(actual, target, places=14)
        self.assertLess(forward_b[1], 0.0)
        self.assertLess(forward_b[2], 0.0)
        actual_pitch_deg = math.degrees(
            math.atan2(-forward_b[2], math.hypot(forward_b[0], forward_b[1]))
        )
        self.assertAlmostEqual(actual_pitch_deg, 20.0, places=12)


class HardwareAccountingTest(unittest.TestCase):
    def test_published_envelope_and_mass_sums_are_exact(self):
        accounting_type = _compiled_class("Phase3HardwareAccounting")
        accounting = accounting_type()

        self.assertEqual(accounting.mid360_dimensions_m_lwh, (0.065, 0.065, 0.060))
        self.assertEqual(accounting.mid360_vertical_fov_range_deg, (-7.0, 52.0))
        self.assertEqual(accounting.mid360_horizontal_fov_deg, 360.0)
        self.assertEqual(accounting.mid360_vertical_fov_deg, 59.0)
        self.assertEqual(accounting.mid360_min_range_m, 0.10)
        self.assertEqual(accounting.mid360_max_range_10_percent_reflectivity_m, 40.0)
        self.assertEqual(accounting.mid360_max_range_80_percent_reflectivity_m, 70.0)
        self.assertEqual(accounting.mid360_average_power_w, 6.5)
        self.assertEqual(accounting.mid360_cold_peak_power_w, 14.0)

        self.assertEqual(accounting.d455_dimensions_m_ldh, (0.124, 0.026, 0.029))
        self.assertEqual(accounting.d455_depth_fov_deg_hv, (87.0, 58.0))
        self.assertEqual(accounting.d455_min_depth_m, 0.52)
        self.assertEqual(accounting.d455_ideal_range_m, (0.60, 6.0))
        self.assertEqual(accounting.d455_max_operating_mode_power_w, 3.46147)

        self.assertAlmostEqual(accounting.combined_sensor_mass_kg, 0.381, places=12)
        self.assertAlmostEqual(
            accounting.projected_body_payload_mass_kg, 1.881, places=12
        )
        self.assertAlmostEqual(accounting.projected_robot_mass_kg, 6.681, places=12)

    def test_report_is_explicitly_accounting_only_and_keeps_identity_fork(self):
        accounting_type = _compiled_class("Phase3HardwareAccounting")
        report = accounting_type().as_report()
        self.assertIs(report["applied_to_articulation"], False)
        self.assertEqual(report["mass_budget"]["excludes"], [
            "brackets",
            "cables",
            "compute",
        ])
        self.assertEqual(report["mass_budget"]["combined_sensor_mass_kg"], 0.381)
        self.assertEqual(report["mass_budget"]["projected_body_payload_mass_kg"], 1.881)
        self.assertEqual(report["mass_budget"]["projected_robot_mass_kg"], 6.681)
        self.assertIn("near-hemispherical", report["configured_lidar"]["classification"])
        self.assertEqual(
            report["identification_fork"]["unitree_l1_fov_deg_hv"],
            (360.0, 90.0),
        )
        self.assertIs(
            report["identification_fork"][
                "requires_physical_label_or_photo_confirmation"
            ],
            True,
        )


class StaticIntegrationContractTest(unittest.TestCase):
    def test_smoke_report_exposes_mount_and_hardware_accounting(self):
        self.assertIn("PHASE3_HARDWARE_ACCOUNTING", SMOKE_SOURCE)
        self.assertIn('"mount_assumptions": asdict(PHASE3_MOUNTS)', SMOKE_SOURCE)
        self.assertIn(
            '"hardware_accounting": PHASE3_HARDWARE_ACCOUNTING.as_report()',
            SMOKE_SOURCE,
        )
        self.assertIn("near-hemispherical coverage surrogate", SMOKE_SOURCE)

    def test_accounting_does_not_modify_articulation_physics(self):
        robot_cfg_source = ast.get_source_segment(
            SENSOR_CFG_SOURCE, _function("_make_phase3_robot_cfg")
        )
        assert robot_cfg_source is not None
        self.assertIn("HEXAPOD_CFG.replace", robot_cfg_source)
        self.assertNotIn("PHASE3_HARDWARE_ACCOUNTING", robot_cfg_source)
        self.assertNotIn("mass", robot_cfg_source)

    def test_both_docs_record_confirmed_inventory_and_separate_acceptance(self):
        # The project lead confirmed Mid-360/D455 ownership and authorized
        # parallel work. Preserve sensor identity and qualification boundaries,
        # without resurrecting the superseded label/photo/permission request.
        for path in (PHASE3_README_PATH, SENSOR_README_PATH):
            with self.subTest(path=path):
                source = path.read_text().lower()
                for term in ("near-hemispherical", "360 x 59", "mid-360", "d455",
                             "confirmed", "parallel", "standing", "separate"):
                    self.assertIn(term, source)


if __name__ == "__main__":
    unittest.main()
