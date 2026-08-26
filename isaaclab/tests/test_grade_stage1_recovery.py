"""CPU-only tests for deterministic Stage-1 recovery grading."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).parents[1] / "grade_stage1_recovery.py"
SPEC = importlib.util.spec_from_file_location("hexapod_grade_stage1_recovery", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
grader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = grader
SPEC.loader.exec_module(grader)


def _passing_row(command: tuple[float, float, float], index: int) -> dict[str, object]:
    vx, vy, yaw = command
    return {
        "index": index,
        "command": {
            "frame": "navigation",
            "body_vx_mps": vx,
            "body_vy_mps": vy,
            "yaw_rate_radps": yaw,
        },
        "mean_command_frame_linear_velocity_mps": [vx, vy, 0.0],
        "mean_command_frame_angular_velocity_radps": [0.0, 0.0, yaw],
        "rmse_command_error": [0.03, 0.02, 0.03],
        "planar_velocity_rmse_mps": 0.04,
        "yaw_rate_rmse_radps": 0.03,
        "falls": 0,
        "timeouts": 0,
        "maximum_tilt_degrees": 5.0,
        "base_height_std_m": 0.006,
        "vertical_velocity_rms_mps": 0.05,
        "roll_pitch_angular_velocity_rms_radps": 0.18,
        "tilt_rms_degrees": 2.5,
        "torque": {
            "rms_applied_nm": 0.8,
            "max_per_joint_rms_applied_nm": 1.2,
            "computed_demand_over_rating_fraction": 0.10,
            "maximum_computed_over_rating_burst_s": 0.10,
            "peak_abs_computed_nm": 4.0,
        },
    }


def _passing_report(checkpoint: str = "/run/model_50.pt") -> dict[str, object]:
    rows = [
        _passing_row(command, index)
        for index, (_, command) in enumerate(grader.COMMAND_CONTRACT)
    ]
    # Matching is by command, not fragile result-list position.
    rows.reverse()
    return {
        "checkpoint": checkpoint,
        "task": grader.TASK_ID,
        "command_frame": "navigation",
        "deterministic_policy": True,
        "results": rows,
    }


class GradeStage1RecoveryTest(unittest.TestCase):
    def test_complete_contract_passes_and_selects_checkpoint(self):
        grade = grader.grade_payload(_passing_report())
        self.assertEqual(grade["accepted_checkpoint_count"], 1)
        self.assertEqual(grade["best_accepted_checkpoint"], "/run/model_50.pt")
        self.assertTrue(grade["evaluations"][0]["accepted"])
        self.assertEqual(grade["evaluations"][0]["failed_command_count"], 0)

    def test_sign_noise_cannot_masquerade_as_axis_acquisition(self):
        report = _passing_report()
        row = next(
            row
            for row in report["results"]
            if row["command"]["body_vy_mps"] > 0.0
            and row["command"]["yaw_rate_radps"] == 0.0
        )
        row["mean_command_frame_linear_velocity_mps"][1] = 1.0e-5
        grade = grader.grade_payload(report)
        result = next(
            result
            for result in grade["evaluations"][0]["results"]
            if result["label"] == "forward_lateral_left"
        )
        self.assertFalse(result["accepted"])
        self.assertTrue(
            any("acquisition floor" in reason for reason in result["rejection_reasons"])
        )

    def test_forward_fall_and_rs05_violations_are_all_hard_failures(self):
        report = _passing_report()
        row = next(
            row
            for row in report["results"]
            if abs(row["command"]["body_vx_mps"] - 0.30) < 1.0e-9
        )
        row["mean_command_frame_linear_velocity_mps"][0] = 0.20
        row["falls"] = 1
        row["torque"]["computed_demand_over_rating_fraction"] = 0.21
        row["torque"]["maximum_computed_over_rating_burst_s"] = 0.22
        row["torque"]["peak_abs_computed_nm"] = 5.6
        grade = grader.grade_payload(report)
        result = next(
            result
            for result in grade["evaluations"][0]["results"]
            if result["label"] == "forward_0p30"
        )
        reasons = "\n".join(result["rejection_reasons"])
        self.assertIn("falls=1", reasons)
        self.assertIn("preservation floor", reasons)
        self.assertIn("computed demand over rating fraction", reasons)
        self.assertIn("computed over-rating burst", reasons)
        self.assertIn("peak absolute computed torque", reasons)

    def test_only_fully_accepted_checkpoints_are_ranked(self):
        better = _passing_report("/run/model_75.pt")
        worse = copy.deepcopy(_passing_report("/run/model_100.pt"))
        for row in worse["results"]:
            row["rmse_command_error"][0] = 0.05
            row["planar_velocity_rmse_mps"] = 0.06
        rejected = copy.deepcopy(_passing_report("/run/model_124.pt"))
        rejected["results"][0]["timeouts"] = 1
        grade = grader.grade_payload({"evaluations": [worse, rejected, better]})
        self.assertEqual(
            grade["accepted_checkpoints_ranked"],
            ["/run/model_75.pt", "/run/model_100.pt"],
        )
        self.assertEqual(grade["best_accepted_checkpoint"], "/run/model_75.pt")

    def test_float32_roundoff_at_rs05_limits_is_not_rejected(self):
        report = _passing_report()
        for row in report["results"]:
            row["torque"]["max_per_joint_rms_applied_nm"] = 1.600000023841858
            row["torque"]["computed_demand_over_rating_fraction"] = 0.20000000298
            row["torque"]["maximum_computed_over_rating_burst_s"] = 0.20000000298
            row["torque"]["peak_abs_computed_nm"] = 5.5000001
        grade = grader.grade_payload(report)
        self.assertTrue(grade["evaluations"][0]["accepted"])

    def test_incompatible_task_or_incomplete_contract_is_rejected(self):
        wrong_task = _passing_report()
        wrong_task["task"] = "wrong-task"
        with self.assertRaisesRegex(grader.ReportError, "task must be"):
            grader.grade_payload(wrong_task)

        incomplete = _passing_report()
        incomplete["results"].pop()
        with self.assertRaisesRegex(grader.ReportError, "expected exactly"):
            grader.grade_payload(incomplete)


if __name__ == "__main__":
    unittest.main()
