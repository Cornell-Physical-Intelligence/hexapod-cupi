"""CPU/static tests for the fail-closed Stage2D homotopy screen."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ISAACLAB_DIR = Path(__file__).parents[1]
MODULE_PATH = ISAACLAB_DIR / "grade_stage2d_homotopy.py"
SPEC = importlib.util.spec_from_file_location("hexapod_grade_stage2d", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
grader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = grader
SPEC.loader.exec_module(grader)


JOINT_NAMES = tuple(f"joint_{index:02d}" for index in range(18))
SHA256 = "a" * 64


def _run_name(stage: str) -> str:
    return f"2026-08-25_12-00-00_phase2_recovery_stage2d_{stage.lower()}_contract"


def _seed_checkpoint(stage: str) -> str:
    spec = grader.STAGE_SPECS[stage]
    return (
        "/workspace/hexapod/isaaclab/logs/rsl_rl/"
        f"{spec.experiment}/seed_immutable/model_seed.pt"
    )


def _candidate_checkpoint(stage: str, iteration: int, run_name: str | None = None) -> str:
    spec = grader.STAGE_SPECS[stage]
    return (
        "/workspace/hexapod/isaaclab/logs/rsl_rl/"
        f"{spec.experiment}/{run_name or _run_name(stage)}/model_{iteration}.pt"
    )


def _contract(stage: str = "C0"):
    return grader.make_contract(
        stage,
        _run_name(stage),
        _seed_checkpoint(stage),
        SHA256,
    )


def _row(
    label: str,
    command: tuple[float, float, float],
    index: int,
    *,
    stability_scale: float = 0.70,
    lateral_fraction: float = 0.60,
) -> dict[str, object]:
    vx, vy, yaw = command
    achieved_vx = vx
    achieved_vy = vy * lateral_fraction if label.startswith("oblique_") else 0.0
    achieved_yaw = yaw * 0.75 if label.startswith("yaw_anchor_") else 0.0
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
        "mean_command_frame_linear_velocity_mps": [achieved_vx, achieved_vy, 0.0],
        "mean_command_frame_angular_velocity_radps": [0.0, 0.0, achieved_yaw],
        "rmse_command_error": [
            abs(vx - achieved_vx),
            abs(vy - achieved_vy),
            abs(yaw - achieved_yaw),
        ],
        "planar_velocity_rmse_mps": 0.04,
        "yaw_rate_rmse_radps": 0.04,
        "falls": 0,
        "timeouts": 0,
        **{
            key: target * stability_scale
            for key, _, _, target in grader.STABILITY_COMPONENTS
        },
        "torque": {
            "rated_continuous_nm": 1.60,
            "rms_applied_nm": 0.70,
            "max_per_joint_rms_applied_nm": 1.10,
            "computed_demand_over_rating_fraction": 0.08,
            "maximum_computed_over_rating_burst_s": 0.08,
            "peak_abs_computed_nm": 3.70,
            "per_joint": [
                {
                    "name": name,
                    "rms_applied_nm": 0.70,
                    "peak_abs_applied_nm": 1.20,
                    "peak_abs_computed_nm": 3.70,
                    "applied_at_rating_fraction": 0.08,
                    "computed_demand_over_rating_fraction": 0.08,
                    "maximum_computed_over_rating_burst_s": 0.08,
                }
                for name in JOINT_NAMES
            ],
        },
    }


def _report(
    stage: str,
    checkpoint: str,
    *,
    stability_scale: float = 0.70,
    lateral_fraction: float = 0.60,
) -> dict[str, object]:
    spec = grader.STAGE_SPECS[stage]
    rows = [
        _row(
            label,
            command,
            index,
            stability_scale=stability_scale,
            lateral_fraction=lateral_fraction,
        )
        for index, (label, command) in enumerate(grader.command_contract(spec))
    ]
    rows.reverse()
    return {
        "checkpoint": checkpoint,
        "task": spec.task_id,
        "command_frame": "navigation",
        "seed": spec.evaluation_seed,
        "deterministic_policy": True,
        "policy_step_seconds": grader.EXPECTED_POLICY_STEP_SECONDS,
        "requested_steps": grader.EXPECTED_REQUESTED_STEPS,
        "warmup_steps": grader.EXPECTED_WARMUP_STEPS,
        "commands_evaluated_in_parallel": len(grader.command_contract(spec)),
        "reset_stance_override": None,
        "startup_randomization_enabled": False,
        "results": rows,
    }


def _payload(
    stage: str = "C0",
    *,
    seed_scale: float = 0.80,
    candidate_scale: float = 0.70,
) -> dict[str, object]:
    spec = grader.STAGE_SPECS[stage]
    return {
        "evaluations": [
            _report(stage, _seed_checkpoint(stage), stability_scale=seed_scale),
            *(
                _report(
                    stage,
                    _candidate_checkpoint(stage, iteration),
                    stability_scale=candidate_scale,
                )
                for iteration in spec.candidate_iterations
            ),
        ]
    }


def _candidate(payload: dict[str, object], iteration: int) -> dict[str, object]:
    suffix = f"/model_{iteration}.pt"
    return next(
        report
        for report in payload["evaluations"]
        if report["checkpoint"].endswith(suffix)
    )


def _result(report: dict[str, object], label: str, stage: str = "C0") -> dict[str, object]:
    expected = dict(grader.command_contract(grader.STAGE_SPECS[stage]))[label]
    return next(
        row
        for row in report["results"]
        if all(
            abs(row["command"][key] - value) < 1.0e-9
            for key, value in zip(
                ("body_vx_mps", "body_vy_mps", "yaw_rate_radps"), expected
            )
        )
    )


class GradeStage2DHomotopyTest(unittest.TestCase):
    def test_all_stage_contracts_are_exact_and_c5_is_true_pure_y(self):
        expected_fractions = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
        for index, minimum_fraction in enumerate(expected_fractions):
            stage = f"C{index}"
            with self.subTest(stage=stage):
                spec = grader.STAGE_SPECS[stage]
                self.assertEqual(spec.evaluation_seed, 61 + index)
                self.assertEqual(spec.minimum_signed_lateral_fraction, minimum_fraction)
                self.assertEqual(
                    spec.candidate_iterations,
                    (0, 5, 10, 15, 20, 25, 30, 35, 39)
                    if stage == "C5"
                    else (0, 5, 10, 15, 20, 24),
                )
                commands = grader.command_contract(spec)
                self.assertEqual(len(commands), 9)
                for label, (_, vy, _) in commands[3:]:
                    self.assertNotEqual(vy, 0.0, label)
                if stage == "C5":
                    self.assertTrue(all(vx == 0.0 for _, (vx, _, _) in commands[3:]))
                grade = grader.grade_payload(_payload(stage), _contract(stage))
                self.assertEqual(grade["accepted_checkpoint_count"], len(spec.candidate_iterations))

    def test_complete_batch_ranks_stability_before_tracking(self):
        payload = _payload("C0")
        calmer = _candidate(payload, 5)
        tracking = _candidate(payload, 10)
        for row in calmer["results"]:
            for key, _, _, target in grader.STABILITY_COMPONENTS:
                row[key] = target * 0.50
            if row["command"]["body_vy_mps"]:
                row["mean_command_frame_linear_velocity_mps"][1] = (
                    row["command"]["body_vy_mps"] * 0.30
                )
        for row in tracking["results"]:
            for key, _, _, target in grader.STABILITY_COMPONENTS:
                row[key] = target * 0.51
            if row["command"]["body_vy_mps"]:
                row["mean_command_frame_linear_velocity_mps"][1] = (
                    row["command"]["body_vy_mps"] * 1.00
                )
        grade = grader.grade_payload(payload, _contract("C0"))
        self.assertEqual(grade["best_accepted_checkpoint"], _candidate_checkpoint("C0", 5))

    def test_wrong_sign_and_stage_fraction_are_hard_for_every_stage(self):
        for stage in grader.STAGE_SPECS:
            with self.subTest(stage=stage):
                payload = _payload(stage)
                report = _candidate(payload, 0)
                negative = _result(report, "oblique_mid_negative", stage)
                negative["mean_command_frame_linear_velocity_mps"][1] = 0.01
                positive = _result(report, "oblique_mid_positive", stage)
                minimum = grader.STAGE_SPECS[stage].minimum_signed_lateral_fraction
                positive["mean_command_frame_linear_velocity_mps"][1] = (
                    positive["command"]["body_vy_mps"] * (minimum - 0.01)
                )
                grade = grader.grade_payload(payload, _contract(stage))
                evaluation = next(item for item in grade["evaluations"] if item["iteration"] == 0)
                reasons = "\n".join(
                    reason
                    for result in evaluation["results"]
                    for reason in result["operational_rejection_reasons"]
                )
                self.assertIn("lateral sign is wrong", reasons)
                self.assertIn("signed lateral command fraction", reasons)
                self.assertFalse(evaluation["accepted"])

    def test_forward_yaw_falls_and_timeouts_are_hard(self):
        payload = _payload("C2")
        report = _candidate(payload, 5)
        forward = _result(report, "forward_anchor", "C2")
        forward["mean_command_frame_linear_velocity_mps"][0] = 0.10
        forward["falls"] = 1
        forward["timeouts"] = 1
        yaw = _result(report, "yaw_anchor_negative", "C2")
        yaw["mean_command_frame_angular_velocity_radps"][2] = 0.10
        grade = grader.grade_payload(payload, _contract("C2"))
        evaluation = next(item for item in grade["evaluations"] if item["iteration"] == 5)
        reasons = "\n".join(
            reason
            for result in evaluation["results"]
            for reason in result["operational_rejection_reasons"]
        )
        for expected in ("falls=1", "timeouts=1", "forward anchor fraction", "yaw sign is wrong"):
            self.assertIn(expected, reasons)

    def test_each_command_local_stability_component_has_ten_percent_gate(self):
        for key, display, _, target in grader.STABILITY_COMPONENTS:
            with self.subTest(component=key):
                payload = _payload("C3", seed_scale=0.70, candidate_scale=0.65)
                report = _candidate(payload, 10)
                row = _result(report, "oblique_high_positive", "C3")
                row[key] = target * 0.70 * 1.101
                grade = grader.grade_payload(payload, _contract("C3"))
                evaluation = next(
                    item for item in grade["evaluations"] if item["iteration"] == 10
                )
                result = next(
                    item
                    for item in evaluation["results"]
                    if item["label"] == "oblique_high_positive"
                )
                self.assertFalse(result["relative_stability_safe"])
                reasons = "\n".join(result["stability_rejection_reasons"])
                self.assertIn(display, reasons)
                self.assertIn("command-local seed limit", reasons)

    def test_forward_and_yaw_tracking_must_retain_seed_per_command(self):
        payload = _payload("C2")
        report = _candidate(payload, 10)
        forward = _result(report, "forward_anchor", "C2")
        forward["mean_command_frame_linear_velocity_mps"][0] = 0.205
        yaw = _result(report, "yaw_anchor_positive", "C2")
        yaw["mean_command_frame_angular_velocity_radps"][2] = 0.11

        grade = grader.grade_payload(payload, _contract("C2"))
        evaluation = next(item for item in grade["evaluations"] if item["iteration"] == 10)
        reasons = "\n".join(
            reason
            for result in evaluation["results"]
            for reason in result["tracking_preservation_rejection_reasons"]
        )
        self.assertIn("forward command fraction", reasons)
        self.assertIn("signed yaw command fraction", reasons)
        self.assertFalse(evaluation["relative_tracking_safe"])

    def test_all_absolute_and_tail_platform_components_are_hard(self):
        payload = _payload("C1")
        report = _candidate(payload, 15)
        rows = list(report["results"])
        for component_index, (key, _, _, target) in enumerate(
            grader.STABILITY_COMPONENTS
        ):
            row = rows[component_index % len(rows)]
            row[key] = target + 2.0 * grader.GATE_ABS_TOLERANCE
        grade = grader.grade_payload(payload, _contract("C1"))
        evaluation = next(item for item in grade["evaluations"] if item["iteration"] == 15)
        reasons = "\n".join(
            reason
            for result in evaluation["results"]
            for reason in result["absolute_stability_rejection_reasons"]
        )
        for _, display, _, _ in grader.STABILITY_COMPONENTS:
            self.assertIn(display, reasons)

    def test_rs05_global_and_per_joint_limits_and_eighteen_rows_are_hard(self):
        payload = _payload("C4")
        report = _candidate(payload, 20)
        row = _result(report, "oblique_low_positive", "C4")
        row["torque"]["peak_abs_computed_nm"] = 4.41
        row["torque"]["computed_demand_over_rating_fraction"] = 0.151
        row["torque"]["maximum_computed_over_rating_burst_s"] = 0.141
        row["torque"]["max_per_joint_rms_applied_nm"] = 1.401
        joint = row["torque"]["per_joint"][0]
        joint["rms_applied_nm"] = 1.61
        joint["computed_demand_over_rating_fraction"] = 0.251
        joint["maximum_computed_over_rating_burst_s"] = 0.201
        joint["peak_abs_computed_nm"] = 5.51
        grade = grader.grade_payload(payload, _contract("C4"))
        evaluation = next(item for item in grade["evaluations"] if item["iteration"] == 20)
        reasons = "\n".join(
            reason
            for result in evaluation["results"]
            for reason in result["operational_rejection_reasons"]
        )
        for expected in (
            "peak absolute computed torque",
            "computed demand over rating fraction",
            "computed over-rating burst",
            "max per-joint RMS applied torque",
            "joint joint_00 RMS applied torque",
        ):
            self.assertIn(expected, reasons)

        malformed = _payload("C4")
        _result(_candidate(malformed, 20), "oblique_low_positive", "C4")["torque"]["per_joint"].pop()
        with self.assertRaisesRegex(grader.ReportError, "exactly 18"):
            grader.grade_payload(malformed, _contract("C4"))

    def test_batch_must_be_complete_unique_and_one_exact_run(self):
        payload = _payload("C0")
        payload["evaluations"].pop()
        with self.assertRaisesRegex(grader.ReportError, "exactly one immutable seed"):
            grader.grade_payload(payload, _contract("C0"))

        duplicate = _payload("C0")
        duplicate["evaluations"][2]["checkpoint"] = duplicate["evaluations"][1]["checkpoint"]
        with self.assertRaisesRegex(grader.ReportError, "duplicate checkpoint"):
            grader.grade_payload(duplicate, _contract("C0"))

        mixed = _payload("C0")
        mixed["evaluations"][1]["checkpoint"] = _candidate_checkpoint(
            "C0", 0, "2026-08-25_12-01-00_phase2_recovery_stage2d_c0_other"
        )
        with self.assertRaisesRegex(grader.ReportError, "exact absolute run directory"):
            grader.grade_payload(mixed, _contract("C0"))

        unexpected = _payload("C0")
        unexpected["evaluations"][1]["checkpoint"] = _candidate_checkpoint("C0", 23)
        with self.assertRaisesRegex(grader.ReportError, "unexpected C0 model_23"):
            grader.grade_payload(unexpected, _contract("C0"))

    def test_playback_commands_numbers_and_randomization_fail_closed(self):
        mutations = (
            ("task", "wrong-task", "task must be"),
            ("startup_randomization_enabled", True, "must be False"),
            ("policy_step_seconds", 0.04, "policy_step_seconds must be"),
            ("commands_evaluated_in_parallel", 8, "must be 9"),
        )
        for key, value, message in mutations:
            with self.subTest(key=key):
                payload = _payload("C5")
                payload["evaluations"][1][key] = value
                with self.assertRaisesRegex(grader.ReportError, message):
                    grader.grade_payload(payload, _contract("C5"))

        payload = _payload("C5")
        row = _result(_candidate(payload, 0), "oblique_mid_positive", "C5")
        row["tilt_rms_degrees"] = float("nan")
        with self.assertRaisesRegex(grader.ReportError, "must be finite"):
            grader.grade_payload(payload, _contract("C5"))

    def test_seed_and_run_provenance_validation_is_strict(self):
        with self.assertRaisesRegex(grader.ReportError, "64 hexadecimal"):
            grader.make_contract("C0", _run_name("C0"), _seed_checkpoint("C0"), "bad")
        with self.assertRaisesRegex(grader.ReportError, "staged child"):
            grader.make_contract(
                "C0",
                _run_name("C0"),
                "/workspace/other/model.pt",
                SHA256,
            )
        with self.assertRaisesRegex(grader.ReportError, "identify stage2d_c0"):
            grader.make_contract("C0", "wrong_run", _seed_checkpoint("C0"), SHA256)

    def test_launchers_encode_one_safe_dense_model_only_workflow(self):
        training = (
            ISAACLAB_DIR / "deploy/train-phase2-recovery-stage2d-homotopy-stage"
        ).read_text(encoding="utf-8")
        screen = (
            ISAACLAB_DIR / "deploy/screen-phase2-recovery-stage2d-homotopy-stage"
        ).read_text(encoding="utf-8")

        for index in range(6):
            self.assertIn(f"Stage2D-Homotopy-C{index}-Hexapod", training)
            self.assertIn(f"Stage2D-Homotopy-C{index}-Hexapod", screen)
            self.assertIn(f"stage2d_c{index}", training)
            self.assertIn(f"stage2d_c{index}", screen)
        self.assertEqual(training.count("max_iterations=25"), 5)
        self.assertEqual(training.count("max_iterations=40"), 1)
        self.assertIn("train_model_only_resume.py", training)
        self.assertIn("--num_envs 4096", training)
        self.assertIn("--std-min 0.08", training)
        self.assertIn("--std-max 0.18", training)
        self.assertIn("--resume", training)
        self.assertIn("--load_run", training)
        self.assertIn("load_run_regex='^'${load_run_regex}'$'", training)
        self.assertIn("checkpoint_regex='^'${checkpoint_regex}'$'", training)
        self.assertIn("${seed_directory//./[.]}", training)
        self.assertIn("${seed_filename//./[.]}", training)
        self.assertIn("expected_checkpoints=(0 5 10 15 20 24)", training)
        self.assertIn("expected_checkpoints=(0 5 10 15 20 25 30 35 39)", training)

        self.assertEqual(screen.count("evaluate_checkpoint.py"), 1)
        self.assertIn("checkpoint_numbers=(0 5 10 15 20 24)", screen)
        self.assertIn("checkpoint_numbers=(0 5 10 15 20 25 30 35 39)", screen)
        self.assertIn("oblique_mid=(0.00 0.09)", screen)
        self.assertIn("oblique_low=(0.00 0.08)", screen)
        self.assertIn("oblique_high=(0.00 0.10)", screen)
        self.assertIn("--disable-randomization", screen)
        self.assertIn("--steps 500", screen)
        self.assertIn("grade_stage2d_homotopy.py", screen)
        self.assertIn("--seed-sha256", screen)
        self.assertIn("refusing to overwrite", training)
        self.assertIn("refusing to overwrite", screen)
        self.assertIn("docker ps", training)
        self.assertIn("docker ps", screen)
        self.assertNotIn("docker rm", training)
        self.assertNotIn("docker rm", screen)
        self.assertNotIn("rm -f", training)
        self.assertNotIn("rm -f", screen)


if __name__ == "__main__":
    unittest.main()
