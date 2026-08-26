"""CPU/static tests for Stage2E fixed-command admission."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
MODULE_PATH = ROOT / "grade_stage2e_static.py"
SPEC = importlib.util.spec_from_file_location("hexapod_grade_stage2e_static", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
grader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = grader
SPEC.loader.exec_module(grader)


SHA256 = "b" * 64
JOINT_NAMES = tuple(f"joint_{index:02d}" for index in range(18))


def _run_name(stage: str) -> str:
    return f"2026-08-25_15-00-00_phase2_recovery_stage2e_{stage.lower()}_test"


def _seed(stage: str) -> str:
    spec = grader.STAGE_SPECS[stage]
    return (
        "/workspace/hexapod/isaaclab/logs/rsl_rl/"
        f"{spec.experiment}/seed_immutable/model_seed.pt"
    )


def _checkpoint(stage: str, iteration: int, run_name: str | None = None) -> str:
    spec = grader.STAGE_SPECS[stage]
    return (
        "/workspace/hexapod/isaaclab/logs/rsl_rl/"
        f"{spec.experiment}/{run_name or _run_name(stage)}/model_{iteration}.pt"
    )


def _contract(stage: str):
    return grader.make_contract(stage, _run_name(stage), _seed(stage), SHA256)


def _row(
    label: str,
    command: tuple[float, float, float],
    index: int,
    *,
    stability_scale: float,
) -> dict[str, object]:
    vx, vy, yaw = command
    height_target = 0.177 if (vx * vx + vy * vy) ** 0.5 <= 0.05 else 0.181
    x_fraction = 0.60 if vx < 0.0 else 1.0
    achieved_vx = vx * x_fraction
    achieved_vy = vy * 0.70
    achieved_yaw = yaw * 0.70
    stand_targets = dict(
        (key, target)
        for key, _, _, target in grader.STAND_RMS_STABILITY_COMPONENTS
    )
    return {
        "index": index,
        "command": {
            "frame": "navigation",
            "body_vx_mps": vx,
            "body_vy_mps": vy,
            "yaw_rate_radps": yaw,
        },
        "samples": grader.common.EXPECTED_SAMPLES,
        "measured_seconds": grader.common.EXPECTED_MEASURED_SECONDS,
        "mean_command_frame_linear_velocity_mps": [achieved_vx, achieved_vy, 0.0],
        "mean_command_frame_angular_velocity_radps": [0.0, 0.0, achieved_yaw],
        "planar_velocity_rmse_mps": 0.03,
        "yaw_rate_rmse_radps": 0.04,
        "mean_base_height_m": height_target,
        "falls": 0,
        "timeouts": 0,
        **{
            key: (
                stand_targets.get(key, target) if label == "stand" else target
            )
            * stability_scale
            for key, _, _, target in grader.common.STABILITY_COMPONENTS
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


def _report(stage: str, checkpoint: str, stability_scale: float) -> dict[str, object]:
    spec = grader.STAGE_SPECS[stage]
    rows = [
        _row(label, command, index, stability_scale=stability_scale)
        for index, (label, command) in enumerate(grader.command_contract(spec))
    ]
    rows.reverse()
    return {
        "checkpoint": checkpoint,
        "task": spec.task_id,
        "command_frame": "navigation",
        "seed": spec.evaluation_seed,
        "deterministic_policy": True,
        "policy_step_seconds": grader.common.EXPECTED_POLICY_STEP_SECONDS,
        "requested_steps": grader.common.EXPECTED_REQUESTED_STEPS,
        "warmup_steps": grader.common.EXPECTED_WARMUP_STEPS,
        "commands_evaluated_in_parallel": len(grader.command_contract(spec)),
        "reset_stance_override": None,
        "startup_randomization_enabled": False,
        "results": rows,
    }


def _payload(stage: str, seed_scale: float = 0.80, candidate_scale: float = 0.70):
    spec = grader.STAGE_SPECS[stage]
    return {
        "evaluations": [
            _report(stage, _seed(stage), seed_scale),
            *(
                _report(stage, _checkpoint(stage, iteration), candidate_scale)
                for iteration in spec.candidate_iterations
            ),
        ]
    }


def _candidate(payload: dict[str, object], iteration: int) -> dict[str, object]:
    return next(
        report
        for report in payload["evaluations"]
        if report["checkpoint"].endswith(f"model_{iteration}.pt")
    )


def _result(report: dict[str, object], label: str, stage: str) -> dict[str, object]:
    command = dict(grader.command_contract(grader.STAGE_SPECS[stage]))[label]
    return next(
        row
        for row in report["results"]
        if all(
            abs(row["command"][key] - value) < 1.0e-9
            for key, value in zip(
                ("body_vx_mps", "body_vy_mps", "yaw_rate_radps"), command
            )
        )
    )


class GradeStage2EStaticTest(unittest.TestCase):
    def test_all_stages_pass_complete_contract_with_reverse_and_pivot(self):
        for stage in grader.STAGE_SPECS:
            with self.subTest(stage=stage):
                spec = grader.STAGE_SPECS[stage]
                commands = dict(grader.command_contract(spec))
                self.assertIn("reverse_anchor", commands)
                self.assertLess(commands["reverse_anchor"][0], 0.0)
                self.assertEqual(commands["pivot_yaw_positive"][:2], (0.0, 0.0))
                self.assertEqual(
                    "yaw_bridge_positive" in commands,
                    stage != "E2",
                )
                grade = grader.grade_payload(_payload(stage), _contract(stage))
                self.assertTrue(grade["accepted"])
                self.assertEqual(
                    grade["accepted_checkpoint_count"], len(spec.candidate_iterations)
                )

    def test_wrong_reverse_and_pivot_signs_are_hard(self):
        payload = _payload("E2")
        report = _candidate(payload, 5)
        reverse = _result(report, "reverse_anchor", "E2")
        reverse["mean_command_frame_linear_velocity_mps"][0] = 0.03
        pivot = _result(report, "pivot_yaw_negative", "E2")
        pivot["mean_command_frame_angular_velocity_radps"][2] = 0.03
        grade = grader.grade_payload(payload, _contract("E2"))
        evaluation = next(item for item in grade["evaluations"] if item["iteration"] == 5)
        reasons = "\n".join(
            reason
            for result in evaluation["results"]
            for reason in result["operational_rejection_reasons"]
        )
        self.assertIn("signed x fraction sign is wrong", reasons)
        self.assertIn("signed yaw fraction sign is wrong", reasons)
        self.assertFalse(evaluation["accepted"])

    def test_anchor_tracking_and_each_stability_component_are_seed_relative(self):
        payload = _payload("E1")
        forward = _result(_candidate(payload, 10), "forward_anchor", "E1")
        forward["mean_command_frame_linear_velocity_mps"][0] = 0.205
        grade = grader.grade_payload(payload, _contract("E1"))
        evaluation = next(item for item in grade["evaluations"] if item["iteration"] == 10)
        reasons = "\n".join(
            reason
            for result in evaluation["results"]
            for reason in result["tracking_rejection_reasons"]
        )
        self.assertIn("retained x fraction", reasons)

        for key, display, _, target in grader.common.STABILITY_COMPONENTS:
            with self.subTest(component=key):
                payload = _payload("E1", seed_scale=0.70, candidate_scale=0.65)
                row = _result(_candidate(payload, 10), "reverse_anchor", "E1")
                row[key] = target * 0.70 * 1.101
                grade = grader.grade_payload(payload, _contract("E1"))
                evaluation = next(
                    item for item in grade["evaluations"] if item["iteration"] == 10
                )
                result = next(
                    item for item in evaluation["results"] if item["label"] == "reverse_anchor"
                )
                self.assertIn(display, "\n".join(result["stability_rejection_reasons"]))

    def test_stand_planar_rmse_is_strict_even_when_mean_speed_cancels(self):
        payload = _payload("E0")
        stand = _result(_candidate(payload, 0), "stand", "E0")
        stand["mean_command_frame_linear_velocity_mps"][:2] = [0.0, 0.0]
        stand["planar_velocity_rmse_mps"] = 0.0401
        grade = grader.grade_payload(payload, _contract("E0"))
        evaluation = next(item for item in grade["evaluations"] if item["iteration"] == 0)
        result = next(item for item in evaluation["results"] if item["label"] == "stand")
        self.assertIn("planar velocity RMSE", "\n".join(result["operational_rejection_reasons"]))
        self.assertFalse(result["accepted"])

    def test_stand_uses_strict_stage2c_rms_limits_and_keeps_tail_limits(self):
        strict_components = (
            ("base_height_std_m", 0.001),
            ("vertical_velocity_rms_mps", 0.020),
            ("roll_pitch_angular_velocity_rms_radps", 0.050),
            ("tilt_rms_degrees", 0.50),
        )
        for key, target in strict_components:
            with self.subTest(component=key):
                payload = _payload("E0")
                stand = _result(_candidate(payload, 0), "stand", "E0")
                # This remains below the looser moving target but must fail the
                # strict stand-only Stage2C deck gate.
                stand[key] = target * 1.01
                grade = grader.grade_payload(payload, _contract("E0"))
                evaluation = next(
                    item for item in grade["evaluations"] if item["iteration"] == 0
                )
                result = next(
                    item for item in evaluation["results"] if item["label"] == "stand"
                )
                self.assertFalse(result["accepted"])
                self.assertIn(
                    dict(
                        (component_key, display)
                        for component_key, display, _, _ in grader.STAND_RMS_STABILITY_COMPONENTS
                    )[key],
                    "\n".join(result["absolute_stability_rejection_reasons"]),
                )

        payload = _payload("E0")
        stand = _result(_candidate(payload, 0), "stand", "E0")
        stand["base_height_peak_to_peak_m"] = 0.01051
        grade = grader.grade_payload(payload, _contract("E0"))
        evaluation = next(item for item in grade["evaluations"] if item["iteration"] == 0)
        result = next(item for item in evaluation["results"] if item["label"] == "stand")
        self.assertIn(
            "base-height peak-to-peak",
            "\n".join(result["absolute_stability_rejection_reasons"]),
        )

    def test_height_gate_is_command_conditioned_and_pure_yaw_stays_low(self):
        payload = _payload("E2")
        candidate = _candidate(payload, 0)
        stand = _result(candidate, "stand", "E2")
        pivot = _result(candidate, "pivot_yaw_positive", "E2")
        moving = _result(candidate, "forward_anchor", "E2")
        self.assertEqual(stand["mean_base_height_m"], 0.177)
        self.assertEqual(pivot["mean_base_height_m"], 0.177)
        self.assertEqual(moving["mean_base_height_m"], 0.181)
        stand["mean_base_height_m"] = 0.18701
        pivot["mean_base_height_m"] = 0.18701
        moving["mean_base_height_m"] = 0.17099
        grade = grader.grade_payload(payload, _contract("E2"))
        evaluation = next(item for item in grade["evaluations"] if item["iteration"] == 0)
        for label, target in (
            ("stand", 0.177),
            ("pivot_yaw_positive", 0.177),
            ("forward_anchor", 0.181),
        ):
            with self.subTest(label=label):
                result = next(
                    item for item in evaluation["results"] if item["label"] == label
                )
                self.assertEqual(
                    result["metrics"]["command_conditioned_base_height_target_m"],
                    target,
                )
                self.assertIn(
                    "mean base-height error",
                    "\n".join(result["absolute_stability_rejection_reasons"]),
                )
                self.assertFalse(result["accepted"])

    def test_falls_rs05_per_joint_and_absolute_platform_gates_are_hard(self):
        payload = _payload("E0")
        row = _result(_candidate(payload, 15), "reverse_joystick_left_ccw", "E0")
        row["falls"] = 1
        row["timeouts"] = 1
        row["torque"]["peak_abs_computed_nm"] = 4.41
        row["torque"]["computed_demand_over_rating_fraction"] = 0.151
        row["torque"]["maximum_computed_over_rating_burst_s"] = 0.141
        row["torque"]["max_per_joint_rms_applied_nm"] = 1.401
        row["torque"]["per_joint"][0]["computed_demand_over_rating_fraction"] = 0.251
        row["maximum_tilt_degrees"] = 1.71
        grade = grader.grade_payload(payload, _contract("E0"))
        evaluation = next(item for item in grade["evaluations"] if item["iteration"] == 15)
        result = next(
            item for item in evaluation["results"] if item["label"] == "reverse_joystick_left_ccw"
        )
        reasons = "\n".join(result["rejection_reasons"])
        for expected in (
            "falls=1",
            "timeouts=1",
            "peak absolute computed torque",
            "computed demand over rating fraction",
            "computed over-rating burst",
            "max per-joint RMS applied torque",
            "worst per-joint",
            "maximum tilt",
        ):
            self.assertIn(expected, reasons)

    def test_batch_run_playback_and_joint_rows_fail_closed(self):
        incomplete = _payload("E2")
        incomplete["evaluations"].pop()
        with self.assertRaisesRegex(grader.ReportError, "exactly"):
            grader.grade_payload(incomplete, _contract("E2"))

        mixed = _payload("E2")
        mixed["evaluations"][1]["checkpoint"] = _checkpoint(
            "E2", 0, "2026-08-25_15-00-00_phase2_recovery_stage2e_e2_other"
        )
        with self.assertRaisesRegex(grader.ReportError, "exact run directory"):
            grader.grade_payload(mixed, _contract("E2"))

        randomized = _payload("E2")
        randomized["evaluations"][1]["startup_randomization_enabled"] = True
        with self.assertRaisesRegex(grader.ReportError, "must be False"):
            grader.grade_payload(randomized, _contract("E2"))

        joints = _payload("E2")
        _result(_candidate(joints, 0), "reverse_anchor", "E2")["torque"]["per_joint"].pop()
        with self.assertRaisesRegex(grader.common.ReportError, "exactly 18"):
            grader.grade_payload(joints, _contract("E2"))

    def test_stability_first_ranking_wins_over_tracking(self):
        payload = _payload("E0")
        calm = _candidate(payload, 5)
        accurate = _candidate(payload, 10)
        for row in calm["results"]:
            for key, _, _, target in grader.common.STABILITY_COMPONENTS:
                if row["command"]["body_vx_mps"] == 0.0 and row["command"]["body_vy_mps"] == 0.0 and row["command"]["yaw_rate_radps"] == 0.0:
                    target = dict(
                        (stand_key, stand_target)
                        for stand_key, _, _, stand_target in grader.STAND_RMS_STABILITY_COMPONENTS
                    ).get(key, target)
                row[key] = target * 0.50
        for row in accurate["results"]:
            for key, _, _, target in grader.common.STABILITY_COMPONENTS:
                if row["command"]["body_vx_mps"] == 0.0 and row["command"]["body_vy_mps"] == 0.0 and row["command"]["yaw_rate_radps"] == 0.0:
                    target = dict(
                        (stand_key, stand_target)
                        for stand_key, _, _, stand_target in grader.STAND_RMS_STABILITY_COMPONENTS
                    ).get(key, target)
                row[key] = target * 0.51
        reverse = _result(calm, "reverse_anchor", "E0")
        reverse["mean_command_frame_linear_velocity_mps"][0] = -0.0105
        grade = grader.grade_payload(payload, _contract("E0"))
        self.assertEqual(grade["best_accepted_checkpoint"], _checkpoint("E0", 5))

    def test_selected_checkpoint_sha_and_handoff_are_attached_from_host_files(self):
        stage = "E0"
        contract = _contract(stage)
        grade = grader.grade_payload(_payload(stage), contract)
        with tempfile.TemporaryDirectory() as temporary:
            experiment = Path(temporary) / contract.spec.experiment
            run_dir = experiment / contract.run_name
            run_dir.mkdir(parents=True)
            for iteration in contract.spec.candidate_iterations:
                (run_dir / f"model_{iteration}.pt").write_bytes(
                    f"checkpoint-{iteration}".encode("ascii")
                )
            hashes = grader.attach_checkpoint_provenance(grade, contract, run_dir)
            selected_host = run_dir.resolve() / Path(grade["selected_checkpoint"]).name
            self.assertTrue(grade["accepted"])
            self.assertEqual(grade["selected_checkpoint_host"], str(selected_host))
            self.assertEqual(grade["selected_checkpoint_sha256"], hashes[selected_host])
            self.assertEqual(
                grade["provenance_contract"]["selected_checkpoint_sha256"],
                hashes[selected_host],
            )
            self.assertEqual(grade["handoff"]["source_stage"], "E0")
            self.assertEqual(grade["handoff"]["next_stage"], "E1")
            self.assertEqual(
                grade["handoff"]["next_experiment"],
                grader.STAGE_SPECS["E1"].experiment,
            )

    def test_launchers_and_grade_schema_are_fail_closed(self):
        train = (
            ROOT / "deploy/train-phase2-recovery-stage2e-joystick-stage"
        ).read_text(encoding="utf-8")
        screen = (
            ROOT / "deploy/screen-phase2-recovery-stage2e-joystick-static"
        ).read_text(encoding="utf-8")
        grader_source = MODULE_PATH.read_text(encoding="utf-8")
        evaluator_source = (ROOT / "evaluate_checkpoint.py").read_text(encoding="utf-8")
        for index in range(3):
            self.assertIn(f"Stage2E-Joystick-E{index}", train)
            self.assertIn(f"Stage2E-Joystick-E{index}", screen)
            self.assertIn(f"stage2e_e{index}", train)
            self.assertIn(f"stage2e_e{index}", screen)
        for expected in (
            "train_model_only_resume.py",
            "--num_envs 4096",
            "--std-min 0.07",
            "--std-max 0.16",
            "load_run_regex='^'${load_run_regex}'$'",
            "checkpoint_regex='^'${checkpoint_regex}'$'",
        ):
            self.assertIn(expected, train)
        for expected in (
            'command "-${reverse_mid}" 0.0 0.0',
            "--command 0.0 0.0 0.24",
            "evaluate_checkpoint.py",
            "grade_stage2e_static.py",
            "checkpoint_hashes",
            "verify_run \"after grading\"",
            "--run-dir",
        ):
            self.assertIn(expected, screen)
        for expected in (
            '"accepted": bool(accepted)',
            '"selected_checkpoint"',
            '"selected_checkpoint_host"',
            '"selected_checkpoint_sha256"',
            '"provenance_contract"',
            '"handoff"',
        ):
            self.assertIn(expected, grader_source)
        for source in (train, screen):
            self.assertNotIn("docker rm", source)
            self.assertNotIn("rm -f", source)
            self.assertIn("refusing", source)

        checkpoint_loop = evaluator_source.index("for checkpoint in checkpoints:")
        reseed = evaluator_source.index("configure_seed(args.seed, True)", checkpoint_loop)
        reset = evaluator_source.index("observations, _ = env.reset()", checkpoint_loop)
        self.assertLess(reseed, reset)


if __name__ == "__main__":
    unittest.main()
