"""CPU-only tests for Stage-2 axis acquisition and stability grading."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).parents[1] / "grade_stage2_axis_acquisition.py"
SPEC = importlib.util.spec_from_file_location(
    "hexapod_grade_stage2_axis_acquisition", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
grader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = grader
SPEC.loader.exec_module(grader)


SEED_CHECKPOINT = (
    "/workspace/hexapod/isaaclab/logs/rsl_rl/"
    "hexapod_robstride_phase2_recovery_stage2_direct/"
    "seed_from_stage1_model25/model_25_stage1_seed.pt"
)


def _row(
    command: tuple[float, float, float],
    index: int,
    *,
    stability_scale: float = 1.0,
) -> dict[str, object]:
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
        "rmse_command_error": [0.03, 0.03, 0.05],
        "planar_velocity_rmse_mps": 0.05,
        "yaw_rate_rmse_radps": 0.05,
        "falls": 0,
        "timeouts": 0,
        "maximum_tilt_degrees": 4.0,
        "base_height_std_m": 0.008 * stability_scale,
        "vertical_velocity_rms_mps": 0.08 * stability_scale,
        "roll_pitch_angular_velocity_rms_radps": 0.30 * stability_scale,
        "tilt_rms_degrees": 2.0 * stability_scale,
        "torque": {
            "rms_applied_nm": 0.8,
            "max_per_joint_rms_applied_nm": 1.2,
            "computed_demand_over_rating_fraction": 0.10,
            "maximum_computed_over_rating_burst_s": 0.10,
            "peak_abs_computed_nm": 4.0,
        },
    }


def _report(
    checkpoint: str,
    *,
    stability_scale: float = 1.0,
) -> dict[str, object]:
    rows = [
        _row(command, index, stability_scale=stability_scale)
        for index, (_, command) in enumerate(grader.COMMAND_CONTRACT)
    ]
    rows.reverse()
    return {
        "checkpoint": checkpoint,
        "task": grader.TASK_ID,
        "command_frame": "navigation",
        "seed": grader.EXPECTED_SEED,
        "deterministic_policy": True,
        "policy_step_seconds": 0.02,
        "requested_steps": 500,
        "warmup_steps": 25,
        "commands_evaluated_in_parallel": 6,
        "reset_stance_override": None,
        "results": rows,
    }


def _payload(*candidates: dict[str, object]) -> dict[str, object]:
    return {"evaluations": [_report(SEED_CHECKPOINT), *candidates]}


class GradeStage2AxisAcquisitionTest(unittest.TestCase):
    def test_full_axis_safe_candidates_rank_by_stability_before_iteration(self):
        earlier = _report("/run/model_25.pt", stability_scale=0.95)
        later_but_calmer = _report("/run/model_100.pt", stability_scale=0.80)
        grade = grader.grade_payload(_payload(earlier, later_but_calmer))
        self.assertEqual(grade["axis_safe_candidate_count"], 2)
        self.assertEqual(grade["accepted_checkpoint_count"], 2)
        self.assertEqual(grade["best_accepted_checkpoint"], "/run/model_100.pt")
        self.assertEqual(
            grade["accepted_checkpoints_ranked"],
            ["/run/model_100.pt", "/run/model_25.pt"],
        )

    def test_wrong_sign_and_subthreshold_magnitude_fail_axis_acquisition(self):
        candidate = _report("/run/model_50.pt", stability_scale=0.9)
        row = next(
            row
            for row in candidate["results"]
            if row["command"]["body_vy_mps"] < 0.0
        )
        row["mean_command_frame_linear_velocity_mps"][1] = 0.01
        grade = grader.grade_payload(_payload(candidate))
        evaluation = grade["evaluations"][0]
        result = next(
            result
            for result in evaluation["results"]
            if result["label"] == "forward_lateral_right"
        )
        self.assertFalse(evaluation["axis_safe"])
        self.assertFalse(result["axis_safe"])
        reasons = "\n".join(result["axis_rejection_reasons"])
        self.assertIn("sign is wrong", reasons)
        self.assertIn("acquisition floor 0.0400", reasons)

    def test_one_command_stability_regression_cannot_hide_in_mean(self):
        candidate = _report("/run/model_75.pt", stability_scale=0.80)
        row = next(
            row
            for row in candidate["results"]
            if abs(row["command"]["body_vx_mps"] - 0.30) < 1.0e-9
        )
        row["base_height_std_m"] = 0.008 * 1.101
        grade = grader.grade_payload(_payload(candidate))
        evaluation = grade["evaluations"][0]
        self.assertTrue(evaluation["axis_safe"])
        self.assertFalse(evaluation["stability_within_seed_limit"])
        self.assertFalse(evaluation["accepted"])
        self.assertLess(evaluation["mean_stability_regression_fraction"], 0.0)
        result = next(
            result
            for result in evaluation["results"]
            if result["label"] == "forward_0p30"
        )
        self.assertTrue(
            any(
                "base-height std" in reason
                for reason in result["stability_rejection_reasons"]
            )
        )

    def test_forward_fall_and_rs05_gates_remain_hard(self):
        candidate = _report("/run/model_125.pt", stability_scale=0.9)
        row = next(
            row
            for row in candidate["results"]
            if abs(row["command"]["body_vx_mps"] - 0.20) < 1.0e-9
        )
        row["mean_command_frame_linear_velocity_mps"][0] = 0.15
        row["falls"] = 1
        row["torque"]["computed_demand_over_rating_fraction"] = 0.21
        row["torque"]["maximum_computed_over_rating_burst_s"] = 0.22
        row["torque"]["peak_abs_computed_nm"] = 5.6
        grade = grader.grade_payload(_payload(candidate))
        result = next(
            result
            for result in grade["evaluations"][0]["results"]
            if result["label"] == "forward_0p20"
        )
        reasons = "\n".join(result["axis_rejection_reasons"])
        self.assertIn("falls=1", reasons)
        self.assertIn("preservation floor", reasons)
        self.assertIn("computed demand over rating fraction", reasons)
        self.assertIn("computed over-rating burst", reasons)
        self.assertIn("peak absolute computed torque", reasons)

    def test_float32_roundoff_at_hard_limits_is_allowed(self):
        candidate = _report("/run/model_149.pt")
        for row in candidate["results"]:
            row["torque"]["max_per_joint_rms_applied_nm"] = 1.600000023841858
            row["torque"]["computed_demand_over_rating_fraction"] = 0.20000000298
            row["torque"]["maximum_computed_over_rating_burst_s"] = 0.20000000298
            row["torque"]["peak_abs_computed_nm"] = 5.5000001
        grade = grader.grade_payload(_payload(candidate))
        self.assertTrue(grade["evaluations"][0]["accepted"])

    def test_seed_identity_and_equal_playback_contract_are_required(self):
        bad_seed = _report("/wrong/model_25.pt")
        candidate = _report("/run/model_0.pt")
        with self.assertRaisesRegex(grader.ReportError, "immutable staged Stage-1 seed"):
            grader.grade_payload({"evaluations": [bad_seed, candidate]})

        mismatched = copy.deepcopy(candidate)
        mismatched["seed"] = 999
        with self.assertRaisesRegex(grader.ReportError, "seed must be"):
            grader.grade_payload(_payload(mismatched))


if __name__ == "__main__":
    unittest.main()
