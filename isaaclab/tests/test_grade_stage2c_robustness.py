"""CPU-only tests for the randomized Stage-2C robustness grader."""

from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import sys
import unittest


ISAACLAB_DIR = Path(__file__).parents[1]
MODULE_PATH = ISAACLAB_DIR / "grade_stage2c_robustness.py"
SPEC = importlib.util.spec_from_file_location(
    "hexapod_grade_stage2c_robustness", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
grader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = grader
SPEC.loader.exec_module(grader)

NOMINAL_MODULE_PATH = ISAACLAB_DIR / "grade_stage2c_stable_forward.py"
NOMINAL_SPEC = importlib.util.spec_from_file_location(
    "hexapod_grade_stage2c_stable_forward_for_robust_test", NOMINAL_MODULE_PATH
)
assert NOMINAL_SPEC is not None and NOMINAL_SPEC.loader is not None
nominal = importlib.util.module_from_spec(NOMINAL_SPEC)
sys.modules[NOMINAL_SPEC.name] = nominal
NOMINAL_SPEC.loader.exec_module(nominal)


RUN_NAME = f"2026-08-25_07-30-00_{grader.EXPECTED_RUN_LABEL}"
CHECKPOINT = (
    "/workspace/hexapod/isaaclab/logs/rsl_rl/"
    f"{grader.EXPECTED_EXPERIMENT}/{RUN_NAME}/model_90.pt"
)
JOINT_NAMES = tuple(f"joint_{index:02d}" for index in range(grader.EXPECTED_JOINT_COUNT))


def _row(
    command: tuple[float, float, float], index: int, *, stability_scale: float = 0.80
) -> dict[str, object]:
    vx, vy, yaw = command
    stand = command == (0.0, 0.0, 0.0)
    rms_components = (
        grader.STAND_RMS_COMPONENTS if stand else grader.MOVING_RMS_COMPONENTS
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
        "planar_velocity_rmse_mps": 0.05,
        "falls": 0,
        "timeouts": 0,
        "fall_free": True,
        **{
            key: target * stability_scale
            for key, _, _, target in (*rms_components, *grader.TAIL_COMPONENTS)
        },
        "torque": {
            "rms_applied_nm": 0.8,
            "max_per_joint_rms_applied_nm": 0.8,
            "peak_abs_computed_nm": 4.0,
            "computed_demand_over_rating_fraction": 0.10,
            "maximum_computed_over_rating_burst_s": 0.10,
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
        # Unused evaluator metrics are still recursively checked for NaN/Inf.
        "mean_reward_per_step": 1.0,
    }


def _report() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for _, command in grader.COMMAND_CONTRACT:
        for _ in range(grader.EXPECTED_COPIES_PER_COMMAND):
            rows.append(_row(command, len(rows)))
    # Copy order is deliberately irrelevant; indices and commands are audited.
    rows.reverse()
    return {
        "checkpoint": CHECKPOINT,
        "task": grader.TASK_ID,
        "command_frame": "navigation",
        "seed": grader.EXPECTED_SEED,
        "deterministic_policy": True,
        "startup_randomization_enabled": True,
        "policy_step_seconds": grader.EXPECTED_POLICY_STEP_SECONDS,
        "requested_steps": grader.EXPECTED_REQUESTED_STEPS,
        "warmup_steps": grader.EXPECTED_WARMUP_STEPS,
        "commands_evaluated_in_parallel": (
            len(grader.COMMAND_CONTRACT) * grader.EXPECTED_COPIES_PER_COMMAND
        ),
        "reset_stance_override": None,
        "results": rows,
    }


def _command_rows(
    report: dict[str, object], label: str
) -> list[dict[str, object]]:
    command = dict(grader.COMMAND_CONTRACT)[label]
    return [
        row
        for row in report["results"]
        if all(
            math.isclose(row["command"][key], expected, abs_tol=1.0e-12)
            for key, expected in zip(
                ("body_vx_mps", "body_vy_mps", "yaw_rate_radps"), command
            )
        )
    ]


def _set_per_joint_max_rms(row: dict[str, object], value: float) -> None:
    row["torque"]["max_per_joint_rms_applied_nm"] = value
    row["torque"]["per_joint"][0]["rms_applied_nm"] = value


def _set_component_values(
    report: dict[str, object], label: str, key: str, values: list[float]
) -> None:
    rows = sorted(_command_rows(report, label), key=lambda row: row["index"])
    assert len(rows) == len(values)
    for row, value in zip(rows, values):
        row[key] = value


class GradeStage2CRobustnessTest(unittest.TestCase):
    def test_safe_randomized_report_passes_and_uses_nominal_targets(self):
        grade = grader.grade_payload(_report())
        self.assertTrue(grade["accepted"])
        self.assertEqual(grade["copy_count"], 32)
        self.assertEqual(grade["total_falls"], 0)
        self.assertEqual(grade["total_timeouts"], 0)
        self.assertEqual(grade["iteration"], 90)
        self.assertEqual(
            grader.MOVING_RMS_TARGETS, nominal.ABSOLUTE_STABILITY_TARGETS
        )
        self.assertEqual(grader.STAND_RMS_TARGETS, nominal.STAND_ABSOLUTE_STABILITY_TARGETS)
        self.assertEqual(grader.TAIL_TARGETS, nominal.TAIL_STABILITY_TARGETS)

    def test_playback_contract_is_fail_closed(self):
        cases = (
            ("task", "wrong-task"),
            ("command_frame", "body"),
            ("seed", 60),
            ("deterministic_policy", False),
            ("startup_randomization_enabled", False),
            ("requested_steps", 499),
            ("warmup_steps", 24),
            ("commands_evaluated_in_parallel", 31),
            ("reset_stance_override", [0.0] * 18),
        )
        for key, value in cases:
            with self.subTest(key=key):
                report = _report()
                report[key] = value
                with self.assertRaises(grader.ReportError):
                    grader.grade_payload(report)

        missing_flag = _report()
        del missing_flag["startup_randomization_enabled"]
        with self.assertRaises(grader.ReportError):
            grader.grade_payload(missing_flag)

    def test_exact_copy_counts_commands_and_unique_indices_are_required(self):
        missing = _report()
        missing["results"].pop()
        with self.assertRaises(grader.ReportError):
            grader.grade_payload(missing)

        duplicate = _report()
        duplicate["results"][0]["index"] = duplicate["results"][1]["index"]
        with self.assertRaises(grader.ReportError):
            grader.grade_payload(duplicate)

        wrong_distribution = _report()
        row = _command_rows(wrong_distribution, "stand")[0]
        command = dict(grader.COMMAND_CONTRACT)["forward_0p16"]
        row["command"].update(
            body_vx_mps=command[0], body_vy_mps=command[1], yaw_rate_radps=command[2]
        )
        with self.assertRaises(grader.ReportError):
            grader.grade_payload(wrong_distribution)

    def test_missing_nonfinite_negative_and_bad_per_joint_rows_fail_closed(self):
        mutations = []

        missing = _report()
        del missing["results"][0]["tilt_p95_degrees"]
        mutations.append(missing)

        nonfinite_used = _report()
        nonfinite_used["results"][0]["base_height_std_m"] = float("nan")
        mutations.append(nonfinite_used)

        nonfinite_unused = _report()
        nonfinite_unused["results"][0]["mean_reward_per_step"] = float("inf")
        mutations.append(nonfinite_unused)

        overflowing_unused = _report()
        overflowing_unused["results"][0]["mean_reward_per_step"] = 10**10000
        mutations.append(overflowing_unused)

        missing_joint = _report()
        missing_joint["results"][0]["torque"]["per_joint"].pop()
        mutations.append(missing_joint)

        duplicate_joint = _report()
        joints = duplicate_joint["results"][0]["torque"]["per_joint"]
        joints[1]["name"] = joints[0]["name"]
        mutations.append(duplicate_joint)

        missing_joint_metric = _report()
        del missing_joint_metric["results"][0]["torque"]["per_joint"][0][
            "peak_abs_applied_nm"
        ]
        mutations.append(missing_joint_metric)

        negative_joint_metric = _report()
        negative_joint_metric["results"][0]["torque"]["per_joint"][0][
            "computed_demand_over_rating_fraction"
        ] = -0.01
        mutations.append(negative_joint_metric)

        for mutation in mutations:
            with self.subTest(mutation=mutations.index(mutation)):
                with self.assertRaises(grader.ReportError):
                    grader.grade_payload(mutation)

    def test_every_copy_tracking_falls_timeouts_and_rs05_gates_are_hard(self):
        report = _report()
        rows = sorted(report["results"], key=lambda row: row["index"])
        rows[0]["falls"] = 1
        rows[0]["fall_free"] = False
        rows[1]["timeouts"] = 1
        rows[2]["planar_velocity_rmse_mps"] = 0.101
        rows[3]["torque"]["peak_abs_computed_nm"] = 4.401
        rows[4]["torque"]["computed_demand_over_rating_fraction"] = 0.151
        rows[5]["torque"]["maximum_computed_over_rating_burst_s"] = 0.141
        _set_per_joint_max_rms(rows[6], 1.401)
        rows[7]["torque"]["per_joint"][0][
            "computed_demand_over_rating_fraction"
        ] = 0.351

        moving = _command_rows(report, "forward_0p16")[0]
        moving["mean_command_frame_linear_velocity_mps"][0] = 0.127

        grade = grader.grade_payload(report)
        self.assertFalse(grade["accepted"])
        self.assertGreaterEqual(grade["failed_operational_copy_count"], 9)
        reasons = "\n".join(
            reason
            for result in grade["results"]
            for copy_result in result["copies"]
            for reason in copy_result["operational_rejection_reasons"]
        )
        for expected in (
            "falls=1",
            "timeouts=1",
            "planar velocity RMSE",
            "peak absolute computed torque",
            "computed demand over rating fraction",
            "computed over-rating burst",
            "max per-joint RMS applied torque",
            "worst per-joint computed demand fraction",
            "achieved forward velocity",
        ):
            self.assertIn(expected, reasons)

    def test_stand_tracking_limits_are_hard_per_copy(self):
        report = _report()
        rows = _command_rows(report, "stand")
        rows[0]["mean_command_frame_linear_velocity_mps"][:2] = [0.0301, 0.0]
        rows[1]["mean_command_frame_angular_velocity_radps"][2] = -0.0801
        grade = grader.grade_payload(report)
        self.assertFalse(grade["operational_safe"])
        reasons = "\n".join(
            reason
            for result in grade["results"]
            for copy_result in result["copies"]
            for reason in copy_result["operational_rejection_reasons"]
        )
        self.assertIn("stand planar speed", reasons)
        self.assertIn("stand absolute yaw rate", reasons)

    def test_command_median_and_worst_copy_stability_gates_are_independent(self):
        median_failure = _report()
        target = grader.MOVING_RMS_TARGETS["base_height_std_m"]
        _set_component_values(
            median_failure,
            "forward_0p20",
            "base_height_std_m",
            [target * 0.8] * 4 + [target * 1.21] * 4,
        )
        grade = grader.grade_payload(median_failure)
        result = next(item for item in grade["results"] if item["label"] == "forward_0p20")
        self.assertFalse(result["stability_safe"])
        self.assertIn("median base-height std", "\n".join(result["stability_rejection_reasons"]))

        copy_failure = _report()
        tail_target = grader.TAIL_TARGETS["tilt_p95_degrees"]
        values = [tail_target * 0.8] * 7 + [
            tail_target * grader.MAXIMUM_COPY_TARGET_MULTIPLIER + 2.0e-6
        ]
        _set_component_values(
            copy_failure, "forward_0p30", "tilt_p95_degrees", values
        )
        grade = grader.grade_payload(copy_failure)
        result = next(item for item in grade["results"] if item["label"] == "forward_0p30")
        self.assertFalse(result["stability_safe"])
        self.assertIn("worst-copy tilt p95", "\n".join(result["stability_rejection_reasons"]))

    def test_stand_uses_tight_rms_targets_and_moving_yaw_tail_is_gated(self):
        report = _report()
        stand_target = grader.STAND_RMS_TARGETS["base_height_std_m"]
        _set_component_values(
            report, "stand", "base_height_std_m", [stand_target * 1.10] * 8
        )
        yaw_target = grader.TAIL_TARGETS["yaw_rate_rmse_radps"]
        _set_component_values(
            report,
            "forward_0p16",
            "yaw_rate_rmse_radps",
            [yaw_target * 1.10] * 8,
        )
        grade = grader.grade_payload(report)
        failed = {
            result["label"] for result in grade["results"] if not result["stability_safe"]
        }
        self.assertEqual(failed, {"stand", "forward_0p16"})

    def test_exact_boundaries_and_float_roundoff_pass(self):
        report = _report()
        rows = sorted(report["results"], key=lambda row: row["index"])
        for row in rows:
            row["planar_velocity_rmse_mps"] = 0.10 + 0.5e-6
            row["torque"]["peak_abs_computed_nm"] = 4.40 + 0.5e-6
            row["torque"]["computed_demand_over_rating_fraction"] = 0.15 + 0.5e-6
            row["torque"]["maximum_computed_over_rating_burst_s"] = 0.14 + 0.5e-6
            _set_per_joint_max_rms(row, 1.40 + 0.5e-6)
            row["torque"]["per_joint"][0][
                "computed_demand_over_rating_fraction"
            ] = 0.35 + 0.5e-6
        stand_rows = _command_rows(report, "stand")
        for row in stand_rows:
            row["mean_command_frame_linear_velocity_mps"][:2] = [0.03 + 0.5e-6, 0.0]
            row["mean_command_frame_angular_velocity_radps"][2] = 0.08 + 0.5e-6

        moving_rows = _command_rows(report, "forward_0p16")
        for row in moving_rows:
            row["mean_command_frame_linear_velocity_mps"][0] = 0.128 - 0.5e-6

        for label, _ in grader.COMMAND_CONTRACT:
            components = (
                grader.STAND_RMS_COMPONENTS
                if label == "stand"
                else grader.MOVING_RMS_COMPONENTS
            ) + grader.TAIL_COMPONENTS
            for key, _, _, target in components:
                _set_component_values(
                    report,
                    label,
                    key,
                    [target * 0.8] * 4
                    + [target * 1.2] * 3
                    + [
                        target * grader.MAXIMUM_COPY_TARGET_MULTIPLIER
                        + 0.5e-6
                    ],
                )

        grade = grader.grade_payload(report)
        self.assertTrue(grade["accepted"])

    def test_just_beyond_absolute_tolerance_fails(self):
        report = _report()
        row = _command_rows(report, "forward_0p30")[0]
        row["torque"]["maximum_computed_over_rating_burst_s"] = 0.14 + 1.1e-6
        grade = grader.grade_payload(report)
        self.assertFalse(grade["operational_safe"])

        report = _report()
        target = grader.MOVING_RMS_TARGETS["vertical_velocity_rms_mps"]
        _set_component_values(
            report,
            "forward_0p30",
            "vertical_velocity_rms_mps",
            [target * 0.8] * 4 + [target * 1.2 + 2.1e-6] * 4,
        )
        grade = grader.grade_payload(report)
        self.assertFalse(grade["stability_safe"])

    def test_single_report_and_exact_stage2c_checkpoint_are_required(self):
        with self.assertRaises(grader.ReportError):
            grader.grade_payload({"evaluations": [_report()]})

        report = _report()
        report["checkpoint"] = CHECKPOINT.replace("model_90.pt", "model_91.pt")
        with self.assertRaises(grader.ReportError):
            grader.grade_payload(report)

        report = _report()
        report["checkpoint"] = CHECKPOINT.replace(RUN_NAME, "some_other_run")
        with self.assertRaises(grader.ReportError):
            grader.grade_payload(report)

    def test_evaluator_emits_actual_randomization_metadata(self):
        source = (ISAACLAB_DIR / "evaluate_checkpoint.py").read_text(encoding="utf-8")
        self.assertIn("startup_randomization_enabled = env_cfg.events is not None", source)
        self.assertIn('reports[-1]["startup_randomization_enabled"]', source)
        self.assertIn("startup_randomization_enabled\n", source)

    def test_launcher_pins_one_randomized_eight_copy_playback(self):
        source = (
            ISAACLAB_DIR
            / "deploy"
            / "screen-phase2-recovery-stage2c-robust-best"
        ).read_text(encoding="utf-8")
        for expected in (
            "grade_batch_nominal.json",
            'grade.get("best_accepted_checkpoint")',
            "accepted_checkpoints_ranked",
            "verify_training_inactive",
            "verify_seed",
            "selected_checkpoint_sha256",
            "nominal_grade_sha256",
            "--copies 8",
            "--seconds 10",
            "--warmup-steps 25",
            "--seed 61",
            "grade_stage2c_robustness.py",
            "set -o noclobber",
            'ln -- "${eval_partial_host}" "${eval_json}"',
        ):
            self.assertIn(expected, source)
        self.assertNotIn("--disable-randomization", source)


if __name__ == "__main__":
    unittest.main()
