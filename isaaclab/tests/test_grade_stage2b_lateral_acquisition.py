"""CPU-only tests for the fail-closed Stage-2B lateral batch grader."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).parents[1] / "grade_stage2b_lateral_acquisition.py"
SPEC = importlib.util.spec_from_file_location(
    "hexapod_grade_stage2b_lateral_acquisition", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
grader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = grader
SPEC.loader.exec_module(grader)


SEED_CHECKPOINT = (
    "/workspace/hexapod/isaaclab/logs/rsl_rl/"
    "hexapod_robstride_phase2_recovery_stage2b_lateral_direct/"
    "seed_from_stage2_model25/model_25_stage2_seed.pt"
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
    stability_scale: float = 0.9,
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
        "commands_evaluated_in_parallel": 8,
        "reset_stance_override": None,
        "results": rows,
    }


def _candidates() -> list[dict[str, object]]:
    return [
        _report(f"/run/model_{iteration}.pt")
        for iteration in grader.EXPECTED_CANDIDATE_ITERATIONS
    ]


def _payload(candidates: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "evaluations": [
            _report(SEED_CHECKPOINT, stability_scale=1.0),
            *(candidates if candidates is not None else _candidates()),
        ]
    }


def _candidate(
    candidates: list[dict[str, object]], iteration: int
) -> dict[str, object]:
    return next(
        report
        for report in candidates
        if report["checkpoint"] == f"/run/model_{iteration}.pt"
    )


def _result_row(report: dict[str, object], label: str) -> dict[str, object]:
    command = dict(grader.COMMAND_CONTRACT)[label]
    return next(
        row
        for row in report["results"]
        if all(
            abs(row["command"][key] - expected) < 1.0e-9
            for key, expected in zip(
                ("body_vx_mps", "body_vy_mps", "yaw_rate_radps"), command
            )
        )
    )


class GradeStage2BLateralAcquisitionTest(unittest.TestCase):
    def test_complete_safe_batch_ranks_stability_before_iteration(self):
        candidates = _candidates()
        _candidate(candidates, 0).update(
            _report("/run/model_0.pt", stability_scale=0.80)
        )
        _candidate(candidates, 90).update(
            _report("/run/model_90.pt", stability_scale=0.70)
        )
        grade = grader.grade_payload(_payload(candidates))
        self.assertEqual(grade["candidate_count"], 11)
        self.assertEqual(grade["accepted_checkpoint_count"], 11)
        self.assertEqual(grade["best_accepted_checkpoint"], "/run/model_90.pt")

    def test_wrong_sign_and_explicit_lateral_floors_are_hard(self):
        candidates = _candidates()
        report = _candidate(candidates, 20)
        pure = _result_row(report, "pure_y_negative")
        pure["mean_command_frame_linear_velocity_mps"][1] = 0.039
        bridge = _result_row(report, "bridge_y_positive")
        bridge["mean_command_frame_linear_velocity_mps"][1] = 0.031

        grade = grader.grade_payload(_payload(candidates))
        evaluation = next(
            item for item in grade["evaluations"] if item["iteration"] == 20
        )
        pure_result = next(
            item for item in evaluation["results"] if item["label"] == "pure_y_negative"
        )
        bridge_result = next(
            item
            for item in evaluation["results"]
            if item["label"] == "bridge_y_positive"
        )
        self.assertIn(
            "lateral sign is wrong",
            "\n".join(pure_result["axis_rejection_reasons"]),
        )
        self.assertIn("below 0.0400", "\n".join(pure_result["axis_rejection_reasons"]))
        self.assertIn(
            "below 0.0320", "\n".join(bridge_result["axis_rejection_reasons"])
        )

    def test_both_lateral_pairs_require_symmetric_response(self):
        candidates = _candidates()
        report = _candidate(candidates, 30)
        _result_row(report, "pure_y_positive")[
            "mean_command_frame_linear_velocity_mps"
        ][1] = 0.10
        _result_row(report, "pure_y_negative")[
            "mean_command_frame_linear_velocity_mps"
        ][1] = -0.06
        _result_row(report, "bridge_y_positive")[
            "mean_command_frame_linear_velocity_mps"
        ][1] = 0.08
        _result_row(report, "bridge_y_negative")[
            "mean_command_frame_linear_velocity_mps"
        ][1] = -0.04

        grade = grader.grade_payload(_payload(candidates))
        evaluation = next(
            item for item in grade["evaluations"] if item["iteration"] == 30
        )
        reasons = "\n".join(
            reason
            for result in evaluation["results"]
            for reason in result["axis_rejection_reasons"]
        )
        self.assertIn("pure_y lateral symmetry ratio 0.6000", reasons)
        self.assertIn("bridge_y lateral symmetry ratio 0.5000", reasons)

    def test_pure_y_leakage_and_forward_yaw_preservation_are_hard(self):
        candidates = _candidates()
        report = _candidate(candidates, 40)
        pure = _result_row(report, "pure_y_positive")
        pure["mean_command_frame_linear_velocity_mps"][0] = 0.081
        pure["mean_command_frame_angular_velocity_radps"][2] = 0.151
        forward = _result_row(report, "forward_0p20")
        forward["mean_command_frame_linear_velocity_mps"][0] = 0.159
        yaw = _result_row(report, "yaw_negative")
        yaw["mean_command_frame_angular_velocity_radps"][2] = -0.099

        grade = grader.grade_payload(_payload(candidates))
        evaluation = next(
            item for item in grade["evaluations"] if item["iteration"] == 40
        )
        reasons = "\n".join(
            reason
            for result in evaluation["results"]
            for reason in result["axis_rejection_reasons"]
        )
        self.assertIn("pure-y absolute forward leakage", reasons)
        self.assertIn("uncommanded absolute yaw rate", reasons)
        self.assertIn("forward velocity preservation", reasons)
        self.assertIn("absolute yaw-rate preservation", reasons)

    def test_rs05_falls_timeouts_and_tilt_gates_remain_hard(self):
        candidates = _candidates()
        report = _candidate(candidates, 50)
        row = _result_row(report, "forward_0p30")
        row["falls"] = 1
        row["timeouts"] = 2
        row["maximum_tilt_degrees"] = 20.01
        row["torque"]["max_per_joint_rms_applied_nm"] = 1.61
        row["torque"]["computed_demand_over_rating_fraction"] = 0.21
        row["torque"]["maximum_computed_over_rating_burst_s"] = 0.21
        row["torque"]["peak_abs_computed_nm"] = 5.51

        grade = grader.grade_payload(_payload(candidates))
        evaluation = next(
            item for item in grade["evaluations"] if item["iteration"] == 50
        )
        result = next(
            item for item in evaluation["results"] if item["label"] == "forward_0p30"
        )
        reasons = "\n".join(result["axis_rejection_reasons"])
        for expected in (
            "falls=1",
            "timeouts=2",
            "maximum tilt",
            "max per-joint RMS applied torque",
            "computed demand over rating fraction",
            "computed over-rating burst",
            "peak absolute computed torque",
        ):
            self.assertIn(expected, reasons)

    def test_each_stability_component_and_score_are_command_local(self):
        candidates = _candidates()
        report = _candidate(candidates, 60)
        row = _result_row(report, "bridge_y_negative")
        row["base_height_std_m"] = 0.008 * 1.101

        grade = grader.grade_payload(_payload(candidates))
        evaluation = next(
            item for item in grade["evaluations"] if item["iteration"] == 60
        )
        result = next(
            item
            for item in evaluation["results"]
            if item["label"] == "bridge_y_negative"
        )
        self.assertTrue(evaluation["axis_safe"])
        self.assertFalse(evaluation["stability_within_seed_limit"])
        self.assertTrue(
            any(
                "base-height std" in reason
                for reason in result["stability_rejection_reasons"]
            )
        )

    def test_float32_roundoff_at_hard_limits_is_allowed(self):
        candidates = _candidates()
        report = _candidate(candidates, 70)
        for row in report["results"]:
            row["torque"]["max_per_joint_rms_applied_nm"] = 1.600000023841858
            row["torque"]["computed_demand_over_rating_fraction"] = 0.20000000298
            row["torque"]["maximum_computed_over_rating_burst_s"] = 0.20000000298
            row["torque"]["peak_abs_computed_nm"] = 5.5000001
        grade = grader.grade_payload(_payload(candidates))
        evaluation = next(
            item for item in grade["evaluations"] if item["iteration"] == 70
        )
        self.assertTrue(evaluation["accepted"])

    def test_seed_identity_and_full_candidate_set_are_required(self):
        payload = _payload()
        payload["evaluations"][0]["checkpoint"] = "/wrong/model_25.pt"
        with self.assertRaisesRegex(grader.ReportError, "immutable staged Stage-2"):
            grader.grade_payload(payload)

        incomplete = _payload()
        incomplete["evaluations"].pop()
        with self.assertRaisesRegex(grader.ReportError, "exactly the immutable seed"):
            grader.grade_payload(incomplete)

        duplicate = _payload()
        duplicate["evaluations"][-1]["checkpoint"] = "/other/model_90.pt"
        with self.assertRaisesRegex(grader.ReportError, "duplicate Stage-2B candidate"):
            grader.grade_payload(duplicate)

        wrong_task = _payload()
        wrong_task["evaluations"][1]["task"] = "wrong"
        with self.assertRaisesRegex(grader.ReportError, "task must be"):
            grader.grade_payload(wrong_task)


if __name__ == "__main__":
    unittest.main()
