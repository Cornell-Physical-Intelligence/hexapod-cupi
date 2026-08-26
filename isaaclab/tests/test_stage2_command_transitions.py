"""CPU/static tests for final Stage2E joystick-transition admission."""

from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import sys
import unittest


ISAACLAB_DIR = Path(__file__).parents[1]
if str(ISAACLAB_DIR) not in sys.path:
    sys.path.insert(0, str(ISAACLAB_DIR))

import stage2_command_transition_contract as contract


GRADER_PATH = ISAACLAB_DIR / "grade_stage2_command_transitions.py"
SPEC = importlib.util.spec_from_file_location(
    "hexapod_grade_stage2_command_transitions", GRADER_PATH
)
assert SPEC is not None and SPEC.loader is not None
grader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = grader
SPEC.loader.exec_module(grader)


RUN_NAME = "2026-08-25_16-00-00_phase2_recovery_stage2e_e2_joystick_contract"
CHECKPOINT_SHA256 = "a" * 64
CHECKPOINT = (
    "/workspace/hexapod/isaaclab/logs/rsl_rl/"
    f"{grader.EXPECTED_EXPERIMENT}/{RUN_NAME}/model_40.pt"
)
HOST_RUN = Path(
    "/home/orionh/HEXAPOD/isaaclab/logs/rsl_rl"
) / grader.EXPECTED_EXPERIMENT / RUN_NAME
CHECKPOINT_HOST = str(HOST_RUN / "model_40.pt")
STATIC_GRADE_SHA256 = "c" * 64
JOINT_NAMES = tuple(grader.EXPECTED_JOINT_NAMES)


def _provenance() -> object:
    return grader.make_provenance(
        grader.EXPECTED_TASK_ID,
        grader.EXPECTED_EXPERIMENT,
        RUN_NAME,
        CHECKPOINT_SHA256,
    )


def _static_grade() -> dict[str, object]:
    evaluations = []
    for iteration in grader.EXPECTED_CANDIDATE_ITERATIONS:
        checkpoint = (
            "/workspace/hexapod/isaaclab/logs/rsl_rl/"
            f"{grader.EXPECTED_EXPERIMENT}/{RUN_NAME}/model_{iteration}.pt"
        )
        selected = iteration == 40
        evaluations.append(
            {
                "checkpoint": checkpoint,
                "iteration": iteration,
                "accepted": selected,
                "checkpoint_sha256": CHECKPOINT_SHA256 if selected else "d" * 64,
            }
        )
    return {
        "schema_version": grader.EXPECTED_STATIC_GRADE_SCHEMA_VERSION,
        "stage": "E2",
        "task": grader.EXPECTED_TASK_ID,
        "experiment": grader.EXPECTED_EXPERIMENT,
        "exact_run_name": RUN_NAME,
        "candidate_iterations": list(grader.EXPECTED_CANDIDATE_ITERATIONS),
        "candidate_count": len(grader.EXPECTED_CANDIDATE_ITERATIONS),
        "accepted": True,
        "accepted_checkpoint_count": 1,
        "accepted_checkpoints_ranked": [CHECKPOINT],
        "best_accepted_checkpoint": CHECKPOINT,
        "selected_checkpoint": CHECKPOINT,
        "selected_checkpoint_host": CHECKPOINT_HOST,
        "selected_checkpoint_sha256": CHECKPOINT_SHA256,
        "provenance_contract": {
            "task": grader.EXPECTED_TASK_ID,
            "experiment": grader.EXPECTED_EXPERIMENT,
            "exact_run_name": RUN_NAME,
            "seed_checkpoint": "/immutable/seed.pt",
            "seed_sha256": "e" * 64,
            "selected_checkpoint": CHECKPOINT,
            "selected_checkpoint_sha256": CHECKPOINT_SHA256,
        },
        "handoff": {
            "source_stage": "E2",
            "next_stage": "TRANSITION",
            "next_experiment": None,
            "selected_checkpoint": CHECKPOINT,
            "selected_checkpoint_host": CHECKPOINT_HOST,
            "selected_checkpoint_sha256": CHECKPOINT_SHA256,
        },
        "evaluations": evaluations,
    }


def _static_admission() -> object:
    return grader.validate_static_admission(
        _static_grade(),
        _provenance(),
        STATIC_GRADE_SHA256,
        run_dir=HOST_RUN,
    )


def _torque() -> dict[str, object]:
    return {
        "rated_continuous_nm": 1.60,
        "mean_abs_applied_nm": 0.5,
        "rms_applied_nm": 0.8,
        "peak_abs_applied_nm": 1.6,
        "peak_abs_computed_nm": 4.0,
        "computed_abs_p99_nm": 1.5,
        "computed_abs_p999_nm": 2.0,
        "max_per_joint_rms_applied_nm": 0.8,
        "maximum_computed_over_rating_burst_s": 0.10,
        "applied_at_rating_fraction": 0.08,
        "computed_demand_over_rating_fraction": 0.08,
        "mean_total_abs_mechanical_power_w": 5.0,
        "per_joint": [
            {
                "name": name,
                "rms_applied_nm": 0.8,
                "peak_abs_applied_nm": 1.6,
                "peak_abs_computed_nm": 4.0,
                "applied_at_rating_fraction": 0.08,
                "computed_demand_over_rating_fraction": 0.08,
                "maximum_computed_over_rating_burst_s": 0.10,
            }
            for name in JOINT_NAMES
        ],
    }


def _sync_torque_aggregates(torque: dict[str, object]) -> None:
    rows = torque["per_joint"]
    torque["rms_applied_nm"] = math.sqrt(
        sum(float(row["rms_applied_nm"]) ** 2 for row in rows) / len(rows)
    )
    torque["max_per_joint_rms_applied_nm"] = max(
        float(row["rms_applied_nm"]) for row in rows
    )
    torque["peak_abs_applied_nm"] = max(
        float(row["peak_abs_applied_nm"]) for row in rows
    )
    torque["peak_abs_computed_nm"] = max(
        float(row["peak_abs_computed_nm"]) for row in rows
    )
    torque["applied_at_rating_fraction"] = sum(
        float(row["applied_at_rating_fraction"]) for row in rows
    ) / len(rows)
    torque["computed_demand_over_rating_fraction"] = sum(
        float(row["computed_demand_over_rating_fraction"]) for row in rows
    ) / len(rows)
    torque["maximum_computed_over_rating_burst_s"] = max(
        float(row["maximum_computed_over_rating_burst_s"]) for row in rows
    )


def _metric_row(
    command: tuple[float, float, float],
    copy_index: int,
    *,
    samples: int,
    measured_seconds: float,
) -> dict[str, object]:
    vx, vy, yaw = command
    stand = command == (0.0, 0.0, 0.0)
    uses_stand_height = math.hypot(vx, vy) <= grader.EXPECTED_CONFIG_SNAPSHOT[
        "moving_command_threshold_mps"
    ]
    mean_height = grader.EXPECTED_CONFIG_SNAPSHOT[
        "stand_nominal_height_m" if uses_stand_height else "nominal_height_m"
    ]
    rms = grader.RMS_STAND_COMPONENTS if stand else grader.RMS_MOVING_COMPONENTS
    return {
        "index": copy_index,
        "command": {
            "frame": "navigation",
            "body_vx_mps": vx,
            "body_vy_mps": vy,
            "yaw_rate_radps": yaw,
        },
        "samples": samples,
        "measured_seconds": measured_seconds,
        "mean_command_frame_linear_velocity_mps": [vx, vy, 0.0],
        "mean_command_frame_angular_velocity_radps": [0.0, 0.0, yaw],
        "rmse_command_error": [0.02, 0.02, 0.02],
        "planar_velocity_rmse_mps": 0.04,
        "falls": 0,
        "timeouts": 0,
        "fall_free": True,
        "mean_base_height_m": mean_height,
        **{key: target * 0.70 for key, _, _, target in (*rms, *grader.TAIL_COMPONENTS)},
        "torque": _torque(),
        "mean_reward_per_step": 1.0,
    }


def _trace(
    copy_index: int, command: tuple[float, float, float]
) -> dict[str, object]:
    trace: dict[str, object] = {
        "copy_index": copy_index,
        "samples": contract.TRANSIENT_STEPS,
        "measured_seconds": contract.TRANSIENT_STEPS * contract.POLICY_STEP_SECONDS,
        "baseline_captured_before_first_action": True,
        "settling_dwell_steps": contract.SETTLING_DWELL_STEPS,
        "base_height_peak_to_peak_m": 0.010,
        "maximum_abs_vertical_velocity_mps": 0.15,
        "maximum_roll_pitch_angular_velocity_radps": 0.60,
        "maximum_tilt_degrees": 2.50,
    }
    _set_tracking_trace(
        trace,
        command,
        [[command[0], command[1], 0.0] for _ in range(contract.TRANSIENT_STEPS)],
        [command[2] for _ in range(contract.TRANSIENT_STEPS)],
    )
    return trace


def _set_tracking_trace(
    trace: dict[str, object],
    command: tuple[float, float, float],
    linear_samples: list[list[float]],
    yaw_samples: list[float],
) -> None:
    flags = [
        contract.tracking_within_settling_band(
            command,
            tuple(linear),
            (0.0, 0.0, yaw),
        )
        for linear, yaw in zip(linear_samples, yaw_samples)
    ]
    completion = contract.settling_completion_step(flags)
    trace.update(
        {
            "settled_sample_count": sum(flags),
            "settling_completed": completion is not None,
            "settling_completion_step": completion,
            "settling_time_s": (
                None
                if completion is None
                else completion * contract.POLICY_STEP_SECONDS
            ),
            "command_frame_linear_velocity_mps_samples": linear_samples,
            "command_frame_yaw_rate_radps_samples": yaw_samples,
            "maximum_x_tracking_error_mps": max(
                abs(sample[0] - command[0]) for sample in linear_samples
            ),
            "maximum_y_tracking_error_mps": max(
                abs(sample[1] - command[1]) for sample in linear_samples
            ),
            "maximum_planar_tracking_error_mps": max(
                math.hypot(sample[0] - command[0], sample[1] - command[1])
                for sample in linear_samples
            ),
            "maximum_yaw_tracking_error_radps": max(
                abs(sample - command[2]) for sample in yaw_samples
            ),
        }
    )


def _rollout_safety_row(copy_index: int) -> dict[str, object]:
    return {
        "index": copy_index,
        "samples": contract.TOTAL_STEPS,
        "measured_seconds": contract.TOTAL_DURATION_SECONDS,
        "falls": 0,
        "timeouts": 0,
        "fall_free": True,
        "torque": _torque(),
    }


def _segment(expected: dict[str, object]) -> dict[str, object]:
    command = tuple(expected["command"])
    results = [
        {
            "copy_index": copy_index,
            "transient_trace": _trace(copy_index, command),
            "transient_metrics": _metric_row(
                command,
                copy_index,
                samples=contract.TRANSIENT_STEPS,
                measured_seconds=(
                    contract.TRANSIENT_STEPS * contract.POLICY_STEP_SECONDS
                ),
            ),
            "steady_state_metrics": _metric_row(
                command,
                copy_index,
                samples=contract.STEADY_STEPS,
                measured_seconds=contract.STEADY_STEPS * contract.POLICY_STEP_SECONDS,
            ),
        }
        for copy_index in range(contract.COPIES)
    ]
    results.reverse()
    return {**copy.deepcopy(expected), "results": results}


def _report() -> dict[str, object]:
    rollout = [
        _rollout_safety_row(copy_index)
        for copy_index in range(contract.COPIES)
    ]
    rollout.reverse()
    return {
        "schema": contract.SCHEMA_ID,
        "checkpoint": CHECKPOINT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "task": grader.EXPECTED_TASK_ID,
        "task_config_class": grader.EXPECTED_TASK_CONFIG_CLASS,
        "agent_config_class": grader.EXPECTED_AGENT_CONFIG_CLASS,
        "command_frame": "navigation",
        "seed": contract.EVALUATION_SEED,
        "deterministic_policy": True,
        "startup_randomization_enabled": True,
        "copies": contract.COPIES,
        "policy_step_seconds": contract.POLICY_STEP_SECONDS,
        "rendered_steps": contract.TOTAL_STEPS,
        "rendered_duration_s": contract.TOTAL_DURATION_SECONDS,
        "episode_length_seconds": contract.TOTAL_DURATION_SECONDS + 5.0,
        "reset_stance_override": None,
        "config_snapshot": copy.deepcopy(grader.EXPECTED_CONFIG_SNAPSHOT),
        "continuous_episode_requested": True,
        "continuous_episode_achieved": True,
        "total_falls": 0,
        "total_timeouts": 0,
        "command_resampling_suppressed": True,
        "environment_reset_at_command_boundaries": False,
        "policy_state_reset_at_command_boundaries": False,
        "schedule_sha256": contract.SCHEDULE_SHA256,
        "schedule": contract.schedule_payload(),
        "segments": [_segment(item) for item in contract.schedule_payload()],
        "rollout_safety_metrics": rollout,
    }


def _segment_by_key(report: dict[str, object], key: str) -> dict[str, object]:
    return next(segment for segment in report["segments"] if segment["key"] == key)


def _copy(segment: dict[str, object], index: int = 0) -> dict[str, object]:
    return next(result for result in segment["results"] if result["copy_index"] == index)


class Stage2CommandTransitionContractTest(unittest.TestCase):
    def test_schedule_is_gap_free_and_covers_final_joystick_behaviors(self):
        schedule = contract.schedule_payload()
        self.assertEqual(len(schedule), 11)
        self.assertEqual(contract.TOTAL_STEPS, 1650)
        self.assertEqual(contract.TOTAL_DURATION_SECONDS, 33.0)
        self.assertEqual(schedule[0]["start_step"], 0)
        self.assertEqual(schedule[-1]["stop_step"], contract.TOTAL_STEPS)
        for previous, following in zip(schedule, schedule[1:]):
            self.assertEqual(previous["stop_step"], following["start_step"])
            self.assertEqual(following["previous_command"], previous["command"])
        commands = {item["key"]: item["command"] for item in schedule}
        self.assertLess(commands["reverse"][0], 0.0)
        self.assertGreater(commands["strafe_left"][1], 0.0)
        self.assertLess(commands["strafe_right"][1], 0.0)
        self.assertGreater(commands["yaw_left"][2], 0.0)
        self.assertLess(commands["yaw_right"][2], 0.0)
        triple = commands["combined_forward_right_yaw_left"]
        self.assertTrue(all(abs(value) > 0.0 for value in triple))
        reverse_triple = commands["combined_reverse_left_yaw_right"]
        self.assertLess(reverse_triple[0], 0.0)
        self.assertGreater(reverse_triple[1], 0.0)
        self.assertLess(reverse_triple[2], 0.0)
        self.assertEqual(contract.schedule_sha256(), contract.SCHEDULE_SHA256)

    def test_settling_helper_requires_signed_response_and_consecutive_dwell(self):
        self.assertFalse(
            contract.tracking_within_settling_band(
                (0.0, 0.09, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
            )
        )
        self.assertTrue(
            contract.tracking_within_settling_band(
                (0.0, -0.09, 0.0), (0.0, -0.05, 0.0), (0.0, 0.0, 0.0)
            )
        )
        flags = [False] * 5 + [True] * 9 + [False] + [True] * 10
        self.assertEqual(contract.settling_completion_step(flags), 25)
        self.assertIsNone(contract.settling_completion_step([True] * 9))
        with self.assertRaises(ValueError):
            contract.settling_completion_step([True], dwell_steps=0)

    def test_complete_safe_report_passes_all_forty_four_transitions(self):
        grade = grader.grade_payload(_report(), _provenance(), _static_admission())
        self.assertTrue(grade["accepted"])
        self.assertTrue(grade["continuous_episode_achieved"])
        self.assertTrue(grade["rollout_rs05_safe"])
        self.assertEqual(grade["segment_count"], 11)
        self.assertEqual(grade["copy_transition_count"], 44)
        self.assertEqual(grade["joint_names"], list(JOINT_NAMES))

    def test_only_exact_e2_task_experiment_run_checkpoint_and_hash_are_allowed(self):
        with self.assertRaises(grader.ReportError):
            grader.make_provenance(
                grader.EXPECTED_TASK_ID.replace("E2", "E1"),
                grader.EXPECTED_EXPERIMENT,
                RUN_NAME,
                CHECKPOINT_SHA256,
            )
        with self.assertRaises(grader.ReportError):
            grader.make_provenance(
                grader.EXPECTED_TASK_ID,
                grader.EXPECTED_EXPERIMENT.replace("e2", "e1"),
                RUN_NAME,
                CHECKPOINT_SHA256,
            )
        with self.assertRaises(grader.ReportError):
            grader.make_provenance(
                grader.EXPECTED_TASK_ID,
                grader.EXPECTED_EXPERIMENT,
                "wrong_run",
                CHECKPOINT_SHA256,
            )

        report = _report()
        report["checkpoint"] = CHECKPOINT.replace(RUN_NAME, f"{RUN_NAME}_other")
        with self.assertRaises(grader.ReportError):
            grader.grade_payload(report, _provenance(), _static_admission())

    def test_prerequisite_static_grade_mutations_fail_closed(self):
        def set_duplicate_iteration(payload: dict[str, object]) -> None:
            payload["evaluations"][1]["iteration"] = 0
            payload["evaluations"][1]["checkpoint"] = payload["evaluations"][0][
                "checkpoint"
            ]

        mutations = (
            ("accepted false", lambda item: item.__setitem__("accepted", False)),
            ("task mismatch", lambda item: item.__setitem__("task", "wrong-task")),
            (
                "run mismatch",
                lambda item: item.__setitem__("exact_run_name", f"{RUN_NAME}-other"),
            ),
            (
                "selected path mismatch",
                lambda item: item.__setitem__(
                    "selected_checkpoint", CHECKPOINT.replace("model_40", "model_35")
                ),
            ),
            (
                "selected host mismatch",
                lambda item: item.__setitem__(
                    "selected_checkpoint_host", CHECKPOINT_HOST + ".other"
                ),
            ),
            (
                "selected SHA mismatch",
                lambda item: item.__setitem__("selected_checkpoint_sha256", "b" * 64),
            ),
            (
                "ranking mismatch",
                lambda item: item.__setitem__(
                    "accepted_checkpoints_ranked",
                    [CHECKPOINT.replace("model_40", "model_35")],
                ),
            ),
            (
                "incomplete candidate set",
                lambda item: item["evaluations"].pop(),
            ),
            ("duplicate candidate iteration", set_duplicate_iteration),
            (
                "handoff mismatch",
                lambda item: item["handoff"].__setitem__("next_stage", "DONE"),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                static_grade = _static_grade()
                mutate(static_grade)
                with self.assertRaises(grader.ReportError):
                    grader.validate_static_admission(
                        static_grade,
                        _provenance(),
                        STATIC_GRADE_SHA256,
                        run_dir=HOST_RUN,
                    )

        with self.assertRaises(grader.ReportError):
            grader.validate_static_admission(
                _static_grade(),
                _provenance(),
                "not-a-sha",
                run_dir=HOST_RUN,
            )
        report = _report()
        report["checkpoint_sha256"] = "b" * 64
        with self.assertRaises(grader.ReportError):
            grader.grade_payload(report, _provenance(), _static_admission())

    def test_playback_and_exact_config_snapshot_are_fail_closed(self):
        mutations = (
            ("startup_randomization_enabled", False),
            ("copies", 3),
            ("rendered_steps", contract.TOTAL_STEPS - 1),
            ("continuous_episode_requested", False),
            ("environment_reset_at_command_boundaries", True),
            ("policy_state_reset_at_command_boundaries", True),
            ("reset_stance_override", [0.0] * 18),
        )
        for key, value in mutations:
            with self.subTest(key=key):
                report = _report()
                report[key] = value
                with self.assertRaises(grader.ReportError):
                    grader.grade_payload(report, _provenance(), _static_admission())

        for key in grader.EXPECTED_CONFIG_SNAPSHOT:
            with self.subTest(config=key):
                report = _report()
                value = report["config_snapshot"][key]
                report["config_snapshot"][key] = (
                    "wrong" if isinstance(value, str) else [9.0] * len(value)
                    if isinstance(value, list)
                    else float(value) + 0.01
                )
                with self.assertRaises(grader.ReportError):
                    grader.grade_payload(report, _provenance(), _static_admission())

    def test_schedule_hash_order_metadata_and_copy_indices_are_exact(self):
        report = _report()
        report["schedule_sha256"] = "0" * 64
        with self.assertRaises(grader.ReportError):
            grader.grade_payload(report, _provenance(), _static_admission())

        report = _report()
        report["segments"][1]["start_step"] += 1
        with self.assertRaises(grader.ReportError):
            grader.grade_payload(report, _provenance(), _static_admission())

        report = _report()
        report["segments"][0]["results"].pop()
        with self.assertRaises(grader.ReportError):
            grader.grade_payload(report, _provenance(), _static_admission())

        report = _report()
        rows = report["segments"][0]["results"]
        rows[0]["copy_index"] = rows[1]["copy_index"]
        with self.assertRaises(grader.ReportError):
            grader.grade_payload(report, _provenance(), _static_admission())

    def test_missing_nonfinite_and_malformed_joint_rows_fail_closed(self):
        reports: list[dict[str, object]] = []
        missing = _report()
        del _copy(_segment_by_key(missing, "forward"))["steady_state_metrics"][
            "tilt_p95_degrees"
        ]
        reports.append(missing)

        nonfinite = _report()
        _copy(_segment_by_key(nonfinite, "forward"))["transient_trace"][
            "maximum_tilt_degrees"
        ] = float("nan")
        reports.append(nonfinite)

        unused_nonfinite = _report()
        _copy(_segment_by_key(unused_nonfinite, "forward"))["transient_metrics"][
            "mean_reward_per_step"
        ] = float("inf")
        reports.append(unused_nonfinite)

        missing_joint = _report()
        _copy(_segment_by_key(missing_joint, "forward"))["steady_state_metrics"][
            "torque"
        ]["per_joint"].pop()
        reports.append(missing_joint)

        duplicate_joint = _report()
        joints = _copy(_segment_by_key(duplicate_joint, "forward"))[
            "steady_state_metrics"
        ]["torque"]["per_joint"]
        joints[1]["name"] = joints[0]["name"]
        reports.append(duplicate_joint)

        negative_joint = _report()
        _copy(_segment_by_key(negative_joint, "forward"))["transient_metrics"][
            "torque"
        ]["per_joint"][0]["computed_demand_over_rating_fraction"] = -0.1
        reports.append(negative_joint)

        for index, report in enumerate(reports):
            with self.subTest(index=index):
                with self.assertRaises(grader.ReportError):
                    grader.grade_payload(report, _provenance(), _static_admission())

    def test_global_torque_summaries_must_match_all_per_joint_rows(self):
        for key in (
            "rms_applied_nm",
            "max_per_joint_rms_applied_nm",
            "peak_abs_applied_nm",
            "peak_abs_computed_nm",
            "applied_at_rating_fraction",
            "computed_demand_over_rating_fraction",
            "maximum_computed_over_rating_burst_s",
        ):
            with self.subTest(key=key):
                report = _report()
                torque = _copy(_segment_by_key(report, "forward"))[
                    "steady_state_metrics"
                ]["torque"]
                torque[key] = float(torque[key]) + 0.01
                with self.assertRaises(grader.ReportError):
                    grader.grade_payload(
                        report, _provenance(), _static_admission()
                    )

    def test_fall_or_timeout_breaks_continuity_and_is_rejected(self):
        report = _report()
        segment_copy = _copy(_segment_by_key(report, "reverse"), 0)
        segment_copy["transient_metrics"]["falls"] = 1
        segment_copy["transient_metrics"]["fall_free"] = False
        rollout_copy = next(
            row for row in report["rollout_safety_metrics"] if row["index"] == 0
        )
        rollout_copy["falls"] = 1
        rollout_copy["fall_free"] = False
        report["total_falls"] = 1
        report["continuous_episode_achieved"] = False
        grade = grader.grade_payload(report, _provenance(), _static_admission())
        self.assertFalse(grade["accepted"])
        self.assertFalse(grade["continuous_episode_achieved"])
        self.assertEqual(grade["total_falls"], 1)

        report = _report()
        segment_copy = _copy(_segment_by_key(report, "stand_final"), 1)
        segment_copy["steady_state_metrics"]["timeouts"] = 1
        rollout_copy = next(
            row for row in report["rollout_safety_metrics"] if row["index"] == 1
        )
        rollout_copy["timeouts"] = 1
        report["total_timeouts"] = 1
        report["continuous_episode_achieved"] = False
        grade = grader.grade_payload(report, _provenance(), _static_admission())
        self.assertFalse(grade["accepted"])
        self.assertEqual(grade["total_timeouts"], 1)

    def test_signed_forward_reverse_lateral_yaw_and_combined_tracking_are_hard(self):
        report = _report()
        cases = (
            ("forward", 0, -0.10),
            ("reverse", 0, 0.10),
            ("strafe_left", 1, -0.05),
            ("strafe_right", 1, 0.05),
            ("yaw_left", 2, -0.10),
            ("yaw_right", 2, 0.10),
        )
        for key, axis, wrong_value in cases:
            row = _copy(_segment_by_key(report, key))["steady_state_metrics"]
            if axis < 2:
                row["mean_command_frame_linear_velocity_mps"][axis] = wrong_value
            else:
                row["mean_command_frame_angular_velocity_radps"][2] = wrong_value
        combined = _copy(
            _segment_by_key(report, "combined_forward_right_yaw_left")
        )["steady_state_metrics"]
        combined["mean_command_frame_linear_velocity_mps"][:2] = [-0.05, 0.05]
        combined["mean_command_frame_angular_velocity_radps"][2] = -0.05
        grade = grader.grade_payload(report, _provenance(), _static_admission())
        self.assertFalse(grade["accepted"])
        reasons = "\n".join(
            reason
            for segment in grade["segments"]
            for result in segment["copies"]
            for reason in result["operational_rejection_reasons"]
        )
        self.assertIn("forward sign is wrong", reasons)
        self.assertIn("lateral sign is wrong", reasons)
        self.assertIn("yaw sign is wrong", reasons)

    def test_steady_absolute_rms_and_all_tail_gates_are_hard(self):
        report = _report()
        rows = [
            _copy(_segment_by_key(report, "forward"), index)["steady_state_metrics"]
            for index in range(contract.COPIES)
        ]
        components = (*grader.RMS_MOVING_COMPONENTS, *grader.TAIL_COMPONENTS)
        for component_index, (key, _, _, target) in enumerate(components):
            rows[component_index % len(rows)][key] = target + 2.0e-6
        stand = _copy(_segment_by_key(report, "stand_final"))["steady_state_metrics"]
        stand["base_height_std_m"] = 0.001002
        grade = grader.grade_payload(report, _provenance(), _static_admission())
        self.assertFalse(grade["accepted"])
        reasons = "\n".join(
            reason
            for segment in grade["segments"]
            for result in segment["copies"]
            for reason in result["steady_stability_rejection_reasons"]
        )
        for _, display, _, _ in components:
            self.assertIn(display, reasons)

    def test_stand_rmse_and_command_conditioned_height_targets_are_hard(self):
        report = _report()
        stand = _copy(_segment_by_key(report, "stand_initial"))[
            "steady_state_metrics"
        ]
        stand["planar_velocity_rmse_mps"] = 0.040002
        moving = _copy(_segment_by_key(report, "forward"))["steady_state_metrics"]
        moving["mean_base_height_m"] = 0.191002
        pure_yaw = _copy(_segment_by_key(report, "yaw_left"))[
            "steady_state_metrics"
        ]
        # Pure yaw uses the environment's stand-height target because planar
        # command speed is zero, while retaining moving RMS gates.
        pure_yaw["mean_base_height_m"] = 0.187002
        grade = grader.grade_payload(report, _provenance(), _static_admission())
        self.assertFalse(grade["accepted"])
        operational = "\n".join(
            reason
            for segment in grade["segments"]
            for result in segment["copies"]
            for reason in result["operational_rejection_reasons"]
        )
        stability = "\n".join(
            reason
            for segment in grade["segments"]
            for result in segment["copies"]
            for reason in result["steady_stability_rejection_reasons"]
        )
        self.assertIn("stand planar velocity RMSE", operational)
        self.assertIn("mean base-height target error", stability)

    def test_transient_settling_and_deck_peaks_are_hard(self):
        report = _report()
        trace = _copy(_segment_by_key(report, "reverse"))["transient_trace"]
        _set_tracking_trace(
            trace,
            (-0.15, 0.0, 0.0),
            [[0.0, 0.0, 0.0] for _ in range(contract.TRANSIENT_STEPS)],
            [0.0 for _ in range(contract.TRANSIENT_STEPS)],
        )
        trace["base_height_peak_to_peak_m"] = 0.021002
        trace["maximum_abs_vertical_velocity_mps"] = 0.220002
        trace["maximum_roll_pitch_angular_velocity_radps"] = 0.900002
        trace["maximum_tilt_degrees"] = 3.400002
        grade = grader.grade_payload(report, _provenance(), _static_admission())
        self.assertFalse(grade["accepted"])
        reasons = "\n".join(
            reason
            for segment in grade["segments"]
            for copy_result in segment["copies"]
            for reason in copy_result["transient_rejection_reasons"]
        )
        self.assertIn("did not settle", reasons)
        self.assertIn("transient maximum tilt", reasons)

    def test_sign_flip_initial_error_is_allowed_but_excess_overshoot_is_not(self):
        report = _report()
        reverse = _copy(_segment_by_key(report, "reverse"))["transient_trace"]
        reverse_linear = [[-0.15, 0.0, 0.0] for _ in range(contract.TRANSIENT_STEPS)]
        reverse_linear[0] = [0.25, 0.0, 0.0]
        _set_tracking_trace(
            reverse,
            (-0.15, 0.0, 0.0),
            reverse_linear,
            [0.0 for _ in range(contract.TRANSIENT_STEPS)],
        )
        yaw = _copy(_segment_by_key(report, "yaw_right"))["transient_trace"]
        yaw_samples = [-0.24 for _ in range(contract.TRANSIENT_STEPS)]
        yaw_samples[0] = 0.24
        _set_tracking_trace(
            yaw,
            (0.0, 0.0, -0.24),
            [[0.0, 0.0, 0.0] for _ in range(contract.TRANSIENT_STEPS)],
            yaw_samples,
        )
        self.assertTrue(grader.grade_payload(report, _provenance(), _static_admission())["accepted"])

        reverse_linear[0] = [0.400002, 0.0, 0.0]
        _set_tracking_trace(
            reverse,
            (-0.15, 0.0, 0.0),
            reverse_linear,
            [0.0 for _ in range(contract.TRANSIENT_STEPS)],
        )
        grade = grader.grade_payload(report, _provenance(), _static_admission())
        self.assertFalse(grade["accepted"])
        reasons = "\n".join(
            reason
            for segment in grade["segments"]
            for copy_result in segment["copies"]
            for reason in copy_result["transient_rejection_reasons"]
        )
        self.assertIn("tracking-error overshoot", reasons)

    def test_transient_axis_gate_blocks_orthogonal_burst_hidden_by_planar_limit(self):
        report = _report()
        reverse = _copy(_segment_by_key(report, "reverse"))["transient_trace"]
        samples = [[-0.15, 0.0, 0.0] for _ in range(contract.TRANSIENT_STEPS)]
        # The x sign-flip accounts for a 0.40 m/s unavoidable delta.  A 0.15
        # m/s lateral burst still fits under the planar limit but must fail its
        # own inactive-y gate.
        samples[0] = [0.25, 0.150002, 0.0]
        _set_tracking_trace(
            reverse,
            (-0.15, 0.0, 0.0),
            samples,
            [0.0 for _ in range(contract.TRANSIENT_STEPS)],
        )
        grade = grader.grade_payload(report, _provenance(), _static_admission())
        self.assertFalse(grade["accepted"])
        reasons = "\n".join(
            reason
            for segment in grade["segments"]
            for result in segment["copies"]
            for reason in result["transient_rejection_reasons"]
        )
        self.assertIn("transient y tracking-error overshoot", reasons)

    def test_settling_summary_must_be_recomputable_from_raw_trace(self):
        for key, value in (
            ("settled_sample_count", contract.TRANSIENT_STEPS - 1),
            ("settling_completion_step", contract.SETTLING_DWELL_STEPS + 1),
            ("maximum_x_tracking_error_mps", 0.01),
        ):
            with self.subTest(key=key):
                report = _report()
                trace = _copy(_segment_by_key(report, "forward"))[
                    "transient_trace"
                ]
                trace[key] = value
                with self.assertRaises(grader.ReportError):
                    grader.grade_payload(
                        report, _provenance(), _static_admission()
                    )

    def test_lateral_and_yaw_sign_flip_symmetry_are_hard(self):
        report = _report()
        _copy(_segment_by_key(report, "strafe_left"))["steady_state_metrics"][
            "mean_command_frame_linear_velocity_mps"
        ][1] = 0.09 * 1.60
        _copy(_segment_by_key(report, "strafe_right"))["steady_state_metrics"][
            "mean_command_frame_linear_velocity_mps"
        ][1] = -0.09 * 0.45
        _copy(_segment_by_key(report, "yaw_left"))["steady_state_metrics"][
            "mean_command_frame_angular_velocity_radps"
        ][2] = 0.24 * 1.60
        _copy(_segment_by_key(report, "yaw_right"))["steady_state_metrics"][
            "mean_command_frame_angular_velocity_radps"
        ][2] = -0.24 * 0.40
        grade = grader.grade_payload(report, _provenance(), _static_admission())
        self.assertFalse(grade["accepted"])
        reasons = "\n".join(
            reason
            for segment in grade["segments"]
            for result in segment["copies"]
            for reason in result["operational_rejection_reasons"]
        )
        self.assertIn("lateral sign-flip symmetry ratio", reasons)
        self.assertIn("yaw sign-flip symmetry ratio", reasons)

    def test_rs05_is_hard_in_both_windows_and_across_window_boundaries(self):
        report = _report()
        result = _copy(_segment_by_key(report, "forward"))
        steady_torque = result["steady_state_metrics"]["torque"]
        steady_torque["per_joint"][0]["peak_abs_computed_nm"] = 4.41
        _sync_torque_aggregates(steady_torque)
        transient_torque = result["transient_metrics"]["torque"]
        transient_torque["per_joint"][0][
            "computed_demand_over_rating_fraction"
        ] = 0.351
        _sync_torque_aggregates(transient_torque)
        # Segment windows individually retain short 0.10 s bursts; the full
        # accumulator proves a continuous boundary-spanning 0.201 s run fails.
        rollout = next(
            row for row in report["rollout_safety_metrics"] if row["index"] == 0
        )
        rollout["torque"]["per_joint"][0][
            "maximum_computed_over_rating_burst_s"
        ] = 0.201
        _sync_torque_aggregates(rollout["torque"])
        grade = grader.grade_payload(report, _provenance(), _static_admission())
        self.assertFalse(grade["accepted"])
        self.assertFalse(grade["rollout_rs05_safe"])
        reasons = "\n".join(
            reason
            for result in grade["rollout_results"]
            for reason in result["rejection_reasons"]
        )
        self.assertIn("computed over-rating burst", reasons)

    def test_roundoff_at_hard_boundaries_passes_and_just_beyond_fails(self):
        report = _report()
        result = _copy(_segment_by_key(report, "forward"))
        steady = result["steady_state_metrics"]
        steady["tilt_p95_degrees"] = 1.40 + 0.5e-6
        transient = result["transient_trace"]
        command = (0.25, 0.0, 0.0)
        boundary_linear = [[0.0, 0.0, 0.0] for _ in range(40)] + [
            [command[0], command[1], 0.0] for _ in range(10)
        ]
        _set_tracking_trace(
            transient,
            command,
            boundary_linear,
            [0.0 for _ in range(contract.TRANSIENT_STEPS)],
        )
        transient["maximum_tilt_degrees"] = 3.40 + 0.5e-6
        transient_torque = result["transient_metrics"]["torque"]
        transient_torque["per_joint"][0][
            "computed_demand_over_rating_fraction"
        ] = 0.35 + 0.5e-6
        _sync_torque_aggregates(transient_torque)
        rollout = next(
            row for row in report["rollout_safety_metrics"] if row["index"] == 0
        )
        rollout["torque"]["per_joint"][0][
            "computed_demand_over_rating_fraction"
        ] = 0.25 + 0.5e-6
        _sync_torque_aggregates(rollout["torque"])
        self.assertTrue(grader.grade_payload(report, _provenance(), _static_admission())["accepted"])

        transient["maximum_tilt_degrees"] = 3.40 + 1.1e-6
        self.assertFalse(grader.grade_payload(report, _provenance(), _static_admission())["accepted"])

    def test_evaluator_and_launcher_preserve_reset_safe_exact_workflow(self):
        evaluator = (
            ISAACLAB_DIR / "evaluate_stage2_command_transitions.py"
        ).read_text(encoding="utf-8")
        launcher = (
            ISAACLAB_DIR / "deploy" / "screen-phase2-command-transitions"
        ).read_text(encoding="utf-8")
        for expected in (
            "_force_commands(raw_env, observations, commands)",
            "policy.reset(dones)",
            "rollout_accumulator.capture_pre_reset(reward)",
            '"continuous_episode_achieved"',
            '"reset_stance_override": None',
            'with json_path.open("x"',
            "PHASE2_RECOVERY_STAGE2E_E2_TASK_ID",
            '"command_frame_linear_velocity_mps_samples"',
            '"robot_usd_sha256"',
        ):
            self.assertIn(expected, evaluator)
        self.assertEqual(evaluator.count("observations, _ = env.reset()"), 1)
        for expected in (
            grader.EXPECTED_TASK_ID,
            grader.EXPECTED_EXPERIMENT,
            "verify_checkpoint",
            "verify_static_grade",
            "validate_static_admission",
            "--static-grade",
            "--static-grade-sha256",
            "verify_robot_usd",
            "--copies 4",
            "--seed 71",
            "grade_stage2_command_transitions.py",
            "refusing to overwrite/reuse artifact directory",
            "set -o noclobber",
        ):
            self.assertIn(expected, launcher)
        self.assertNotIn("--disable-randomization", launcher)
        self.assertNotIn("--checkpoint-sha256", launcher)


if __name__ == "__main__":
    unittest.main()
