"""Source-defined cut endpoints, mimic kinematics and honest closure results."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import audit_mkii_linkage as audit

LINKAGE = ROOT / "robot/hexapod_mkii_assy/urdf/hexapod_mkii_linkage.urdf"


class LinkageAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = ET.parse(LINKAGE).getroot()
        cls.cut = audit.cut_reference(audit.REFERENCE / "leg_reference.json", audit.REFERENCE / "leg_parts.json")
        cls.positions = {f"{leg}_{kind}": {"coxa_yaw": 0.0, "femur_pitch": -0.25, "tibia_pitch": -0.55}[kind]
                         for leg in audit.LEGS for kind in audit.INDEPENDENT}

    def test_endpoint_comments_match_independent_source_frames(self):
        audit.verify_cut_comments(LINKAGE.read_text(), self.cut)
        changed = LINKAGE.read_text().replace("-0.053773135", "-0.050773135", 1)
        with self.assertRaisesRegex(ValueError, "differs from the leg reference"):
            audit.verify_cut_comments(changed, self.cut)

    def test_all_12_passive_joint_angles_follow_the_18_independent_motor_names(self):
        transforms, positions = audit.forward_kinematics(self.root, self.positions)
        self.assertEqual(len(transforms), 31)
        self.assertEqual(len(positions), 30)
        for leg in audit.LEGS:
            self.assertEqual(positions[f"{leg}_tibia_lever_pivot"], -0.55)
            self.assertEqual(positions[f"{leg}_tibia_rod_pivot"], 0.55)

    def test_real_asset_reports_small_axial_residual_without_claiming_exact_closure(self):
        result = audit.audit(samples=181)
        self.assertTrue(result["animated_mimic_topology_pass"])
        self.assertFalse(result["physical_closed_loop_joint_authored"])
        self.assertFalse(result["cut_point_coincidence_pass"])
        self.assertEqual(result["cut_point_coincidence_tolerance_mm"], 0.1)
        for leg in audit.LEGS:
            sweep = result["sweep"][leg]
            self.assertGreater(sweep["max_point_residual_mm"], 0.49)
            self.assertLess(sweep["max_point_residual_mm"], 0.51)
            self.assertLess(sweep["max_transverse_offset_mm"], 0.02)
            self.assertLess(sweep["max_cut_axis_misalignment_deg"], 0.31)
            # The serial merged pushrod explains the visible centimetre gap;
            # it cannot be mistaken for the sub-millimetre linkage residual.
            self.assertGreater(result["serial_merged_rod_snapshots"]["stance"][leg]["point_residual_mm"], 16.0)
        self.assertEqual(set(result["reference_cad_angles_outside_software_limits_rad"]),
                         {"lr_coxa_yaw", "rf_coxa_yaw"})

    def test_upstream_yaw_and_femur_motion_preserve_local_closure_distance(self):
        baseline, _ = audit.forward_kinematics(self.root, self.positions)
        reference = audit.closure_measurements(baseline, self.cut)
        generator = np.random.default_rng(204)
        for _ in range(12):
            positions = dict(self.positions)
            for leg in audit.LEGS:
                positions[f"{leg}_coxa_yaw"] = generator.uniform(-0.872665, 0.872665)
                positions[f"{leg}_femur_pitch"] = generator.uniform(-1.745329, 0.55)
            transforms, _ = audit.forward_kinematics(self.root, positions)
            actual = audit.closure_measurements(transforms, self.cut)
            for leg in audit.LEGS:
                self.assertAlmostEqual(actual[leg]["point_residual_mm"], reference[leg]["point_residual_mm"], places=10)
                self.assertAlmostEqual(actual[leg]["transverse_offset_mm"], reference[leg]["transverse_offset_mm"], places=10)

    def test_wrong_mimic_sign_creates_large_gap_and_is_reported(self):
        text = LINKAGE.read_text().replace('joint="lf_tibia_pitch" multiplier="-1.0"',
                                          'joint="lf_tibia_pitch" multiplier="1.0"', 1)
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "wrong_mimic.urdf"
            changed.write_text(text)
            result = audit.audit(changed, samples=3)
        self.assertFalse(result["animated_mimic_topology_pass"])
        self.assertFalse(result["cut_point_coincidence_pass"])
        self.assertGreater(result["snapshots"]["stance"]["lf"]["point_residual_mm"], 50.0)

    def test_missing_nonfinite_out_of_limit_and_cyclic_mimic_inputs_are_rejected(self):
        for fault in ("missing", "nonfinite", "out_of_limit"):
            positions = dict(self.positions)
            if fault == "missing":
                del positions["lf_tibia_pitch"]
            else:
                positions["lf_tibia_pitch"] = float("nan") if fault == "nonfinite" else 2.0
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                audit.forward_kinematics(self.root, positions)
        changed = copy.deepcopy(self.root)
        changed.find("joint[@name='lf_tibia_rod_pivot']/mimic").set("joint", "lf_tibia_rod_pivot")
        with self.assertRaisesRegex(ValueError, "cyclic"):
            audit.forward_kinematics(changed, self.positions)


if __name__ == "__main__":
    unittest.main()
