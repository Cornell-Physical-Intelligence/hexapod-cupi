"""CPU-only tests for the fail-closed Stage-2C stable-forward grader."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).parents[1] / "grade_stage2c_stable_forward.py"
SPEC = importlib.util.spec_from_file_location(
    "hexapod_grade_stage2c_stable_forward", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
grader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = grader
SPEC.loader.exec_module(grader)


SEED_CHECKPOINT = (
    "/workspace/hexapod/isaaclab/logs/rsl_rl/"
    f"{grader.EXPECTED_EXPERIMENT}/"
    "seed_from_stage2_model25/model_25_stage2_seed.pt"
)
RUN_NAME = f"2026-08-25_07-30-00_{grader.EXPECTED_RUN_LABEL}"
JOINT_NAMES = tuple(f"joint_{index:02d}" for index in range(grader.EXPECTED_JOINT_COUNT))


def _candidate_checkpoint(iteration: int, run_name: str = RUN_NAME) -> str:
    return (
        "/workspace/hexapod/isaaclab/logs/rsl_rl/"
        f"{grader.EXPECTED_EXPERIMENT}/{run_name}/model_{iteration}.pt"
    )


def _row(
    command: tuple[float, float, float],
    index: int,
    *,
    stability_scale: float = 1.0,
    stand_stability_scale: float | None = None,
    planar_rmse: float = 0.05,
) -> dict[str, object]:
    vx, vy, yaw = command
    is_stand = command == (0.0, 0.0, 0.0)
    row_stability_scale = (
        stand_stability_scale
        if is_stand and stand_stability_scale is not None
        else stability_scale
    )
    rms_targets = (
        grader.STAND_ABSOLUTE_STABILITY_TARGETS
        if is_stand
        else grader.ABSOLUTE_STABILITY_TARGETS
    )
    return {
        "index": index,
        "command": {
            "frame": "navigation",
            "body_vx_mps": vx,
            "body_vy_mps": vy,
            "yaw_rate_radps": yaw,
        },
        "samples": grader.EXPECTED_SAMPLES,
        "measured_seconds": grader.EXPECTED_MEASURED_SECONDS,
        "mean_command_frame_linear_velocity_mps": [vx, vy, 0.0],
        "mean_command_frame_angular_velocity_radps": [0.0, 0.0, yaw],
        "planar_velocity_rmse_mps": planar_rmse,
        "falls": 0,
        "timeouts": 0,
        **{
            key: target * row_stability_scale
            for key, target in rms_targets.items()
        },
        **{
            key: target * row_stability_scale
            for key, target in grader.TAIL_STABILITY_TARGETS.items()
        },
        "torque": {
            "rms_applied_nm": 0.8,
            "max_per_joint_rms_applied_nm": 1.2,
            "computed_demand_over_rating_fraction": 0.10,
            "maximum_computed_over_rating_burst_s": 0.10,
            "peak_abs_computed_nm": 4.0,
            "per_joint": [
                {
                    "name": name,
                    "rms_applied_nm": 0.8,
                    "peak_abs_applied_nm": 1.6,
                    "peak_abs_computed_nm": 4.0,
                    "applied_at_rating_fraction": 0.10,
                    "computed_demand_over_rating_fraction": 0.10,
                    "maximum_computed_over_rating_burst_s": 0.10,
                }
                for name in JOINT_NAMES
            ],
        },
    }


def _report(
    checkpoint: str,
    *,
    stability_scale: float = 0.80,
    stand_stability_scale: float | None = None,
    planar_rmse: float = 0.05,
) -> dict[str, object]:
    rows = [
        _row(
            command,
            index,
            stability_scale=stability_scale,
            stand_stability_scale=stand_stability_scale,
            planar_rmse=planar_rmse,
        )
        for index, (_, command) in enumerate(grader.COMMAND_CONTRACT)
    ]
    # Result list order is not semantically important; row index and command
    # identity remain pinned by the grader.
    rows.reverse()
    return {
        "checkpoint": checkpoint,
        "task": grader.TASK_ID,
        "command_frame": "navigation",
        "seed": grader.EXPECTED_SEED,
        "deterministic_policy": True,
        "policy_step_seconds": grader.EXPECTED_POLICY_STEP_SECONDS,
        "requested_steps": grader.EXPECTED_REQUESTED_STEPS,
        "warmup_steps": grader.EXPECTED_WARMUP_STEPS,
        "commands_evaluated_in_parallel": len(grader.COMMAND_CONTRACT),
        "reset_stance_override": None,
        "results": rows,
    }


def _candidates() -> list[dict[str, object]]:
    return [
        _report(_candidate_checkpoint(iteration))
        for iteration in grader.EXPECTED_CANDIDATE_ITERATIONS
    ]


def _payload(candidates: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "evaluations": [
            _report(
                SEED_CHECKPOINT,
                stability_scale=1.0,
                stand_stability_scale=0.80,
            ),
            *(candidates if candidates is not None else _candidates()),
        ]
    }


def _candidate(
    candidates: list[dict[str, object]], iteration: int
) -> dict[str, object]:
    return next(
        report
        for report in candidates
        if report["checkpoint"] == _candidate_checkpoint(iteration)
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


def _set_stability_scale(row: dict[str, object], scale: float) -> None:
    for key, _, _, target in grader.STABILITY_COMPONENTS:
        row[key] = target * scale


class GradeStage2CStableForwardTest(unittest.TestCase):
    def test_complete_safe_batch_ranks_stability_before_tracking(self):
        candidates = _candidates()
        # model90 is calmer than model100, so it wins despite markedly worse
        # tracking.  This directly protects the branch's primary objective.
        _candidate(candidates, 90).update(
            _report(
                _candidate_checkpoint(90),
                stability_scale=0.58,
                planar_rmse=0.095,
            )
        )
        _candidate(candidates, 100).update(
            _report(
                _candidate_checkpoint(100),
                stability_scale=0.59,
                planar_rmse=0.010,
            )
        )
        grade = grader.grade_payload(_payload(candidates))
        self.assertEqual(grade["candidate_count"], 13)
        self.assertEqual(grade["accepted_checkpoint_count"], 13)
        self.assertEqual(
            grade["best_accepted_checkpoint"], _candidate_checkpoint(90)
        )

    def test_forward_lower_upper_and_planar_rmse_are_hard(self):
        candidates = _candidates()
        report = _candidate(candidates, 20)
        low = _result_row(report, "forward_0p16")
        low["mean_command_frame_linear_velocity_mps"][0] = 0.127
        high = _result_row(report, "forward_0p20")
        high["mean_command_frame_linear_velocity_mps"][0] = 0.261
        noisy = _result_row(report, "forward_0p30")
        noisy["planar_velocity_rmse_mps"] = 0.1001

        grade = grader.grade_payload(_payload(candidates))
        evaluation = next(
            item for item in grade["evaluations"] if item["iteration"] == 20
        )
        reasons = "\n".join(
            reason
            for result in evaluation["results"]
            for reason in result["operational_rejection_reasons"]
        )
        self.assertIn("below 0.1280", reasons)
        self.assertIn("exceeds 0.2600", reasons)
        self.assertIn("planar velocity RMSE", reasons)
        self.assertFalse(evaluation["accepted"])

    def test_stand_planar_and_yaw_gates_are_hard(self):
        candidates = _candidates()
        stand = _result_row(_candidate(candidates, 30), "stand")
        stand["mean_command_frame_linear_velocity_mps"][:2] = [0.0301, 0.0]
        stand["mean_command_frame_angular_velocity_radps"][2] = -0.0801

        grade = grader.grade_payload(_payload(candidates))
        evaluation = next(
            item for item in grade["evaluations"] if item["iteration"] == 30
        )
        result = next(
            item for item in evaluation["results"] if item["label"] == "stand"
        )
        reasons = "\n".join(result["operational_rejection_reasons"])
        self.assertIn("stand planar speed", reasons)
        self.assertIn("stand absolute yaw rate", reasons)

    def test_falls_timeouts_and_all_rs05_limits_are_hard(self):
        candidates = _candidates()
        row = _result_row(_candidate(candidates, 40), "forward_0p30")
        row["falls"] = 1
        row["timeouts"] = 2
        row["torque"]["peak_abs_computed_nm"] = 4.401
        row["torque"]["computed_demand_over_rating_fraction"] = 0.151
        row["torque"]["maximum_computed_over_rating_burst_s"] = 0.141
        row["torque"]["max_per_joint_rms_applied_nm"] = 1.401

        grade = grader.grade_payload(_payload(candidates))
        evaluation = next(
            item for item in grade["evaluations"] if item["iteration"] == 40
        )
        result = next(
            item
            for item in evaluation["results"]
            if item["label"] == "forward_0p30"
        )
        reasons = "\n".join(result["operational_rejection_reasons"])
        for expected in (
            "falls=1",
            "timeouts=2",
            "peak absolute computed torque",
            "computed demand over rating fraction",
            "computed over-rating burst",
            "max per-joint RMS applied torque",
        ):
            self.assertIn(expected, reasons)

    def test_float32_roundoff_at_hard_limits_is_allowed(self):
        candidates = _candidates()
        report = _candidate(candidates, 50)
        for row in report["results"]:
            row["torque"]["peak_abs_computed_nm"] = 4.4000001
            row["torque"]["computed_demand_over_rating_fraction"] = 0.15000001
            row["torque"]["maximum_computed_over_rating_burst_s"] = 0.14000001
            row["torque"]["max_per_joint_rms_applied_nm"] = 1.40000002
        grade = grader.grade_payload(_payload(candidates))
        evaluation = next(
            item for item in grade["evaluations"] if item["iteration"] == 50
        )
        self.assertTrue(evaluation["accepted"])

    def test_tail_gates_are_hard_and_allow_gate_roundoff(self):
        candidates = _candidates()
        failed_row = _result_row(_candidate(candidates, 0), "forward_0p20")
        for key, target in grader.TAIL_STABILITY_TARGETS.items():
            failed_row[key] = target + 2.0 * grader.GATE_ABS_TOLERANCE

        boundary_row = _result_row(
            _candidate(candidates, 10), "forward_0p20"
        )
        _set_stability_scale(boundary_row, 0.60)
        for key, target in grader.TAIL_STABILITY_TARGETS.items():
            boundary_row[key] = target + 0.5 * grader.GATE_ABS_TOLERANCE

        grade = grader.grade_payload(_payload(candidates))
        failed = next(item for item in grade["evaluations"] if item["iteration"] == 0)
        failed_result = next(
            item
            for item in failed["results"]
            if item["label"] == "forward_0p20"
        )
        reasons = "\n".join(failed_result["absolute_stability_rejection_reasons"])
        for expected in (
            "base-height peak-to-peak",
            "absolute vertical-velocity p95",
            "roll/pitch-rate p95",
            "tilt p95",
            "maximum tilt",
            "yaw-rate RMSE",
        ):
            self.assertIn(expected, reasons)
        self.assertFalse(failed["accepted"])

        boundary = next(
            item for item in grade["evaluations"] if item["iteration"] == 10
        )
        boundary_result = next(
            item
            for item in boundary["results"]
            if item["label"] == "forward_0p20"
        )
        self.assertTrue(boundary_result["absolute_stability_safe"])
        self.assertTrue(boundary["accepted"])

    def test_stand_absolute_rms_targets_are_hard_with_roundoff_tolerance(self):
        candidates = _candidates()
        failed_row = _result_row(_candidate(candidates, 20), "stand")
        for key, target in grader.STAND_ABSOLUTE_STABILITY_TARGETS.items():
            failed_row[key] = target + 2.0 * grader.GATE_ABS_TOLERANCE

        boundary_row = _result_row(_candidate(candidates, 30), "stand")
        for key, target in grader.STAND_ABSOLUTE_STABILITY_TARGETS.items():
            boundary_row[key] = target + 0.5 * grader.GATE_ABS_TOLERANCE

        grade = grader.grade_payload(_payload(candidates))
        failed = next(
            item for item in grade["evaluations"] if item["iteration"] == 20
        )
        failed_result = next(
            item for item in failed["results"] if item["label"] == "stand"
        )
        self.assertEqual(
            len(failed_result["absolute_stability_rejection_reasons"]), 4
        )
        self.assertFalse(failed_result["absolute_stability_safe"])

        boundary = next(
            item for item in grade["evaluations"] if item["iteration"] == 30
        )
        boundary_result = next(
            item for item in boundary["results"] if item["label"] == "stand"
        )
        self.assertTrue(boundary_result["absolute_stability_safe"])

    def test_per_joint_rows_are_complete_unique_and_fail_closed(self):
        missing = _payload()
        missing_torque = _result_row(
            missing["evaluations"][1], "forward_0p20"
        )["torque"]
        missing_torque["per_joint"].pop()
        with self.assertRaisesRegex(grader.ReportError, "exactly 18 rows"):
            grader.grade_payload(missing)

        duplicate = _payload()
        duplicate_torque = _result_row(
            duplicate["evaluations"][1], "forward_0p20"
        )["torque"]
        duplicate_torque["per_joint"][1]["name"] = duplicate_torque["per_joint"][0][
            "name"
        ]
        with self.assertRaisesRegex(grader.ReportError, "duplicate name"):
            grader.grade_payload(duplicate)

        nonfinite = _payload()
        nonfinite_torque = _result_row(
            nonfinite["evaluations"][1], "forward_0p20"
        )["torque"]
        nonfinite_torque["per_joint"][0]["peak_abs_applied_nm"] = float("nan")
        with self.assertRaisesRegex(grader.ReportError, "must be finite"):
            grader.grade_payload(nonfinite)

        negative = _payload()
        negative_torque = _result_row(
            negative["evaluations"][1], "forward_0p20"
        )["torque"]
        negative_torque["per_joint"][0]["applied_at_rating_fraction"] = -0.001
        with self.assertRaisesRegex(grader.ReportError, "must be nonnegative"):
            grader.grade_payload(negative)

        boolean = _payload()
        boolean_torque = _result_row(
            boolean["evaluations"][1], "forward_0p20"
        )["torque"]
        boolean_torque["per_joint"][0]["rms_applied_nm"] = True
        with self.assertRaisesRegex(grader.ReportError, "must be numeric"):
            grader.grade_payload(boolean)

    def test_worst_per_joint_over_rating_fraction_is_hard(self):
        candidates = _candidates()
        boundary_row = _result_row(
            _candidate(candidates, 40), "forward_0p30"
        )
        boundary_row["torque"]["per_joint"][7][
            "computed_demand_over_rating_fraction"
        ] = 0.2500005
        failed_row = _result_row(_candidate(candidates, 50), "forward_0p30")
        failed_row["torque"]["per_joint"][7][
            "computed_demand_over_rating_fraction"
        ] = 0.250002

        grade = grader.grade_payload(_payload(candidates))
        boundary = next(
            item for item in grade["evaluations"] if item["iteration"] == 40
        )
        self.assertTrue(boundary["accepted"])
        failed = next(
            item for item in grade["evaluations"] if item["iteration"] == 50
        )
        failed_result = next(
            item
            for item in failed["results"]
            if item["label"] == "forward_0p30"
        )
        self.assertIn(
            "worst per-joint computed demand over rating fraction (joint_07)",
            "\n".join(failed_result["operational_rejection_reasons"]),
        )
        self.assertFalse(failed["accepted"])

    def test_each_moving_command_requires_fifteen_percent_improvement(self):
        candidates = _candidates()
        exact = _result_row(_candidate(candidates, 60), "forward_0p16")
        _set_stability_scale(exact, 0.85)
        short = _result_row(_candidate(candidates, 70), "forward_0p16")
        _set_stability_scale(short, 0.851)

        grade = grader.grade_payload(_payload(candidates))
        exact_eval = next(
            item for item in grade["evaluations"] if item["iteration"] == 60
        )
        short_eval = next(
            item for item in grade["evaluations"] if item["iteration"] == 70
        )
        self.assertTrue(exact_eval["accepted"])
        result = next(
            item
            for item in short_eval["results"]
            if item["label"] == "forward_0p16"
        )
        self.assertIn(
            "required-improvement limit",
            "\n".join(result["stability_rejection_reasons"]),
        )
        self.assertFalse(short_eval["accepted"])

    def test_moving_component_cannot_hide_regression_in_composite(self):
        candidates = _candidates()
        row = _result_row(_candidate(candidates, 80), "forward_0p20")
        row["base_height_std_m"] = 0.0035 * 1.051
        row["vertical_velocity_rms_mps"] = 0.070 * 0.50
        row["roll_pitch_angular_velocity_rms_radps"] = 0.28 * 0.50
        row["tilt_rms_degrees"] = 0.90 * 0.50

        grade = grader.grade_payload(_payload(candidates))
        evaluation = next(
            item for item in grade["evaluations"] if item["iteration"] == 80
        )
        result = next(
            item
            for item in evaluation["results"]
            if item["label"] == "forward_0p20"
        )
        self.assertLess(
            result["metrics"]["normalized_stability_composite"], 0.85
        )
        self.assertIn(
            "base-height std",
            "\n".join(result["stability_rejection_reasons"]),
        )
        self.assertFalse(evaluation["relative_stability_safe"])

    def test_stand_allows_ten_percent_but_not_more_per_component(self):
        candidates = _candidates()
        exact = _result_row(_candidate(candidates, 90), "stand")
        # The immutable seed stands at 80% of the 1 mm absolute limit.  A 10%
        # relative regression therefore remains below the hard absolute gate.
        exact["base_height_std_m"] = 0.001 * 0.88
        failed = _result_row(_candidate(candidates, 100), "stand")
        failed["base_height_std_m"] = 0.001 * 0.8811

        grade = grader.grade_payload(_payload(candidates))
        exact_eval = next(
            item for item in grade["evaluations"] if item["iteration"] == 90
        )
        failed_eval = next(
            item for item in grade["evaluations"] if item["iteration"] == 100
        )
        self.assertTrue(exact_eval["accepted"])
        failed_result = next(
            item
            for item in failed_eval["results"]
            if item["label"] == "stand"
        )
        self.assertIn(
            "base-height std",
            "\n".join(failed_result["stability_rejection_reasons"]),
        )

    def test_moving_absolute_rms_targets_are_hard_admission_gates(self):
        candidates = _candidates()
        row = _result_row(_candidate(candidates, 110), "forward_0p16")
        # One component is 4% above its absolute target/seed; the remaining
        # three improve enough that the composite is substantially better.
        row["base_height_std_m"] = 0.0035 * 1.04
        row["vertical_velocity_rms_mps"] = 0.070 * 0.60
        row["roll_pitch_angular_velocity_rms_radps"] = 0.28 * 0.60
        row["tilt_rms_degrees"] = 0.90 * 0.60

        grade = grader.grade_payload(_payload(candidates))
        evaluation = next(
            item for item in grade["evaluations"] if item["iteration"] == 110
        )
        result = next(
            item
            for item in evaluation["results"]
            if item["label"] == "forward_0p16"
        )
        self.assertFalse(evaluation["accepted"])
        self.assertFalse(evaluation["absolute_stability_safe"])
        self.assertFalse(result["metrics"]["absolute_stability_targets_met"])
        self.assertTrue(result["metrics"]["absolute_stability_target_reasons"])
        self.assertIn(
            "base-height std",
            "\n".join(result["absolute_stability_rejection_reasons"]),
        )
        self.assertTrue(grade["absolute_targets_are_admission_gates"])

    def test_seed_identity_complete_set_one_run_and_playback_are_required(self):
        wrong_seed = _payload()
        wrong_seed["evaluations"][0]["checkpoint"] = "/wrong/model_25.pt"
        with self.assertRaisesRegex(grader.ReportError, "immutable staged Stage-2"):
            grader.grade_payload(wrong_seed)

        incomplete = _payload()
        incomplete["evaluations"].pop()
        with self.assertRaisesRegex(grader.ReportError, "exactly the immutable seed"):
            grader.grade_payload(incomplete)

        duplicate = _payload()
        duplicate["evaluations"][-1]["checkpoint"] = _candidate_checkpoint(110)
        with self.assertRaisesRegex(grader.ReportError, "duplicate checkpoint"):
            grader.grade_payload(duplicate)

        mixed_run = _payload()
        mixed_run["evaluations"][-1]["checkpoint"] = _candidate_checkpoint(
            119, f"alternate_{grader.EXPECTED_RUN_LABEL}"
        )
        with self.assertRaisesRegex(grader.ReportError, "one exact run directory"):
            grader.grade_payload(mixed_run)

        wrong_duration = _payload()
        wrong_duration["evaluations"][1]["requested_steps"] = 499
        with self.assertRaisesRegex(grader.ReportError, "requested_steps must be 500"):
            grader.grade_payload(wrong_duration)

        missing_stance_audit = _payload()
        del missing_stance_audit["evaluations"][1]["reset_stance_override"]
        with self.assertRaisesRegex(grader.ReportError, "reset_stance_override is missing"):
            grader.grade_payload(missing_stance_audit)

    def test_nonfinite_negative_and_noninteger_metrics_fail_closed(self):
        negative = _payload()
        _result_row(negative["evaluations"][1], "forward_0p20")[
            "base_height_std_m"
        ] = -0.001
        with self.assertRaisesRegex(grader.ReportError, "must be nonnegative"):
            grader.grade_payload(negative)

        nonfinite = _payload()
        _result_row(nonfinite["evaluations"][1], "forward_0p20")[
            "vertical_velocity_rms_mps"
        ] = float("nan")
        with self.assertRaisesRegex(grader.ReportError, "must be finite"):
            grader.grade_payload(nonfinite)

        fractional_falls = _payload()
        _result_row(fractional_falls["evaluations"][1], "forward_0p20")[
            "falls"
        ] = 0.5
        with self.assertRaisesRegex(grader.ReportError, "nonnegative integer"):
            grader.grade_payload(fractional_falls)


if __name__ == "__main__":
    unittest.main()
