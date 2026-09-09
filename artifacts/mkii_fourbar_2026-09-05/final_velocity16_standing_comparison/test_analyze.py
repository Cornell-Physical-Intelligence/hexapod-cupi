"""Evidence-analyzer checks; synthetic records never replace primary reports."""
import copy
import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("standing_pair_analysis", Path(__file__).with_name("analyze.py"))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def velocity_window():
    return {
        "passive_velocity_relation_samples": 7680 * 32 * 12,
        "passive_velocity_relation_squared_sum": 3.,
        "rms_passive_velocity_relation_error_rad_s": math.sqrt(3. / (7680 * 32 * 12)),
        "closure_relative_point_velocity_samples": 7680 * 32 * 6,
        "closure_relative_point_velocity_squared_norm_sum": .004,
        "rms_closure_relative_point_velocity_m_s": math.sqrt(.004 / (7680 * 32 * 6)),
    }


def report_pair():
    nominal = {
        "contract": {"sha256": "synthetic_same_identity"},
        "runtime_manifest": {"resolved_simulation": {"solver_position_iterations": 64, "solver_velocity_iterations": 16}},
        "reset_root_positions_m": [[5. - 2 * (i // 6), -5. + 2 * (i % 6), .14297] for i in range(32)],
        "reset_first_env_joint_positions_rad": [0.] * 30,
        "reset_max_joint_error_rad": 0.,
        "joint_names": ["coordinate_" + str(i) for i in range(30)],
        "active_motor_names": ["motor_" + str(i) for i in range(18)],
        "body_names": ["body_" + str(i) for i in range(31)],
        "velocity_constraint_telemetry": {"acceptance_use": "observational_only_no_velocity_thresholds",
            "passive_relation_names": list(range(12)), "closure_pin_names": list(range(6))},
        "windows": {"startup": {"mean_height_m": .14, "max_applied_nm": 1.5, "max_demand_nm": 1.5},
                    "settled": {"mean_height_m": .136, "max_applied_nm": .67, "max_demand_nm": .67}},
    }
    refined = copy.deepcopy(nominal)
    refined["runtime_manifest"]["resolved_simulation"]["solver_position_iterations"] = 128
    return nominal, refined


def compare_synthetic(nominal, refined):
    identity = {"source_commit": "synthetic_same_commit", "physical_standing_pass": True}
    with patch.object(analysis, "load_report", side_effect=[(nominal, {}, identity), (refined, {}, identity)]):
        return analysis.compare("nominal", "refined")


class StandingEvidenceTests(unittest.TestCase):
    def test_rms_uses_all_relations_environments_and_samples(self):
        result = analysis.check_velocity_arithmetic(velocity_window(), 7680, 32)
        self.assertEqual(result["passive_velocity_relation"]["samples"], 2_949_120)
        self.assertEqual(result["closure_relative_point_velocity"]["samples"], 1_474_560)

    def test_wrong_population_and_wrong_rms_are_rejected(self):
        for field in ("passive_velocity_relation_samples", "rms_closure_relative_point_velocity_m_s"):
            with self.subTest(field=field):
                window = velocity_window()
                window[field] *= 2
                with self.assertRaises(ValueError):
                    analysis.check_velocity_arithmetic(window, 7680, 32)

    def test_position_solver_comparison_never_admits_short_run(self):
        result = compare_synthetic(*report_pair())
        self.assertTrue(result["standing_comparison_pass"])
        self.assertFalse(result["simulation_training_admission"])
        self.assertFalse(result["complete_full_validation"])

    def test_changed_reset_placement_is_rejected(self):
        nominal, refined = report_pair()
        refined["reset_root_positions_m"][31][0] += .001
        with self.assertRaisesRegex(ValueError, "reset"):
            compare_synthetic(nominal, refined)

    def test_changed_controller_runtime_is_rejected(self):
        nominal, refined = report_pair()
        refined["runtime_manifest"]["controller_damping"] = .6
        with self.assertRaisesRegex(ValueError, "Runtime differs"):
            compare_synthetic(nominal, refined)

    def test_existing_torque_convergence_bound_is_enforced(self):
        nominal, refined = report_pair()
        refined["windows"]["settled"]["max_applied_nm"] = .9
        result = compare_synthetic(nominal, refined)
        self.assertFalse(result["standing_comparison_pass"])
        self.assertEqual(result["standing_comparisons"]["max_applied_nm"]["unchanged_bound"], .05)

    def test_capped_torque_cannot_hide_raw_demand_difference(self):
        nominal, refined = report_pair()
        refined["windows"]["settled"]["max_demand_nm"] = 2.
        result = compare_synthetic(nominal, refined)
        self.assertTrue(result["standing_comparison_pass"])
        self.assertFalse(result["raw_demand_comparison"]["within_bound"])
        self.assertFalse(result["current_qualifier_standing_metrics_pass"])
        self.assertFalse(result["simulation_training_admission"])


if __name__ == "__main__":
    unittest.main()
