"""CPU-only tests for bounded stance-validation log grading."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


MODULE_PATH = Path(__file__).parents[1] / "summarize_stance_validation.py"
SPEC = importlib.util.spec_from_file_location("hexapod_stance_summary", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
summary_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary_module
SPEC.loader.exec_module(summary_module)


def _passing_log() -> str:
    return """\
joint_count=18
body_count=19
foot_count=6
stance_root_height_m=0.185000
stance_femur_angle_rad=0.600000
stance_tibia_angle_rad=2.233500
min_base_height_m=0.176000
mean_base_height_m=0.179000
post_settle_mean_base_height_m=0.179100
post_settle_mean_height_std_m=0.001500
post_settle_max_height_std_m=0.001800
post_settle_mean_vertical_velocity_rms_mps=0.012000
post_settle_mean_roll_pitch_rate_rms_radps=0.080000
post_settle_mean_tilt_rms_deg=1.200000
post_settle_non_foot_contact_fraction=0.00000000
max_abs_torque_nm=1.600000
max_abs_computed_torque_nm=3.200000
post_settle_mean_abs_computed_torque_nm=0.700000
post_settle_torque_saturation_fraction=0.00400000
unexpected_terminations=0
unexpected_truncations=0
VALIDATION_PASS
"""


class StanceValidationSummaryTest(unittest.TestCase):
    def _candidate(self, log_path: Path, *, exit_code: int = 0):
        return summary_module.CandidateInput(
            name="recommended",
            root_height_m=0.185,
            femur_angle_rad=0.6,
            tibia_angle_rad=2.2335,
            log_path=log_path,
            exit_code=exit_code,
        )

    def test_passing_log_is_safe_and_preserves_stability_metrics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "recommended.log"
            log_path.write_text(_passing_log(), encoding="utf-8")
            result = summary_module.summarize_candidate(self._candidate(log_path))
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["safe"])
        self.assertEqual(result["unsafe_reasons"], [])
        self.assertAlmostEqual(
            result["metrics"]["post_settle_mean_height_std_m"], 0.0015
        )

    def test_nonfoot_and_rs05_failures_are_retained_as_unsafe(self):
        failing_log = (
            _passing_log()
            .replace(
                "post_settle_non_foot_contact_fraction=0.00000000",
                "post_settle_non_foot_contact_fraction=0.01000000",
            )
            .replace("max_abs_torque_nm=1.600000", "max_abs_torque_nm=1.620000")
            .replace(
                "post_settle_torque_saturation_fraction=0.00400000",
                "post_settle_torque_saturation_fraction=0.00600000",
            )
            .replace("VALIDATION_PASS\n", "")
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "unsafe.log"
            log_path.write_text(failing_log, encoding="utf-8")
            result = summary_module.summarize_candidate(
                self._candidate(log_path, exit_code=1)
            )
        self.assertEqual(result["status"], "unsafe")
        self.assertFalse(result["safe"])
        self.assertEqual(len(result["unsafe_reasons"]), 3)
        self.assertIn(
            "coxa_femur_or_tibia_shaft_contacted_ground", result["unsafe_reasons"]
        )

    def test_timeout_is_reported_without_fabricating_metrics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "timeout.log"
            log_path.write_text("container startup\n", encoding="utf-8")
            result = summary_module.summarize_candidate(
                self._candidate(log_path, exit_code=124)
            )
        self.assertEqual(result["status"], "timeout")
        self.assertFalse(result["safe"])
        self.assertEqual(result["metrics"], {})
        self.assertTrue(any("timeout" in reason for reason in result["error_reasons"]))

    def test_raw_computed_demand_above_rs05_peak_is_unsafe(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "peak_demand.log"
            log_path.write_text(
                _passing_log()
                .replace(
                    "max_abs_computed_torque_nm=3.200000",
                    "max_abs_computed_torque_nm=5.510000",
                )
                .replace("VALIDATION_PASS\n", ""),
                encoding="utf-8",
            )
            result = summary_module.summarize_candidate(
                self._candidate(log_path, exit_code=1)
            )
        self.assertEqual(result["status"], "unsafe")
        self.assertIn(
            "computed_torque_demand_exceeded_rs05_peak_limit",
            result["unsafe_reasons"],
        )

    def test_summary_keeps_later_candidates_after_an_unsafe_result(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            unsafe_path = directory / "current.log"
            safe_path = directory / "recommended.log"
            unsafe_path.write_text(
                _passing_log()
                .replace("max_abs_torque_nm=1.600000", "max_abs_torque_nm=1.620000")
                .replace("VALIDATION_PASS\n", ""),
                encoding="utf-8",
            )
            safe_path.write_text(_passing_log(), encoding="utf-8")
            candidates = (
                summary_module.CandidateInput(
                    "current", 0.210, 0.4, 2.1, unsafe_path, 1
                ),
                summary_module.CandidateInput(
                    "recommended", 0.185, 0.6, 2.2335, safe_path, 0
                ),
            )
            result = summary_module.build_summary(
                candidates, steps=250, num_envs=32, timeout_seconds=180
            )
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["safe_candidates"], ["recommended"])
        self.assertFalse(result["all_candidates_safe"])


if __name__ == "__main__":
    unittest.main()
