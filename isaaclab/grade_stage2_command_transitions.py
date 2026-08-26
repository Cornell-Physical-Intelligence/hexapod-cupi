#!/usr/bin/env python3
"""Fail-closed grading for one terminal Stage2E joystick-transition report."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any

from stage2_command_transition_contract import (
    COPIES,
    EVALUATION_SEED,
    MAXIMUM_SETTLING_TIME_SECONDS,
    POLICY_STEP_SECONDS,
    SCHEMA_ID,
    SCHEDULE_SHA256,
    SEGMENT_STEPS,
    SETTLING_DWELL_STEPS,
    STEADY_STEPS,
    TOTAL_DURATION_SECONDS,
    TOTAL_STEPS,
    TRANSIENT_STEPS,
    schedule_payload,
    settling_completion_step,
    tracking_within_settling_band,
)


EXPECTED_JOINT_COUNT = 18
GATE_ABS_TOLERANCE = 1.0e-6
AGGREGATE_TORQUE_ABS_TOLERANCE = 1.0e-5
COMMAND_TOLERANCE = 1.0e-4
EXPECTED_TASK_ID = (
    "Isaac-Velocity-Omni-Recovery-Stage2E-Joystick-E2-"
    "Hexapod-RobStride-Direct-v0"
)
EXPECTED_EXPERIMENT = (
    "hexapod_robstride_phase2_recovery_stage2e_e2_joystick_direct"
)
EXPECTED_TASK_CONFIG_CLASS = (
    "hexapod_rl.phase2e_cfg.HexapodPhase2RecoveryStage2EE2EnvCfg"
)
EXPECTED_AGENT_CONFIG_CLASS = (
    "hexapod_rl.phase2e_cfg.HexapodPhase2RecoveryStage2EE2PPORunnerCfg"
)
EXPECTED_CANDIDATE_ITERATIONS = (0, 5, 10, 15, 20, 25, 30, 35, 40, 44)
EXPECTED_STATIC_GRADE_SCHEMA_VERSION = 1
EXPECTED_ROBOT_USD_PATH = (
    "/workspace/hexapod/robot/hexapod_mkii_mock_assy/usd/"
    "hexapod_mkii_robstride/hexapod_mkii_robstride.usda"
)
EXPECTED_ROBOT_USD_SHA256 = (
    "f0e55f8415e028fcd807cc6bce718916fd7dbb73414f7df7d8aaf4bf370ca5b0"
)
EXPECTED_JOINT_NAMES = sorted(
    (
        "revolute_1_1",
        "revolute_1_7",
        "revolute_2_5",
        "revolute_3",
        "revolute_4",
        "revolute_5",
        "revolute_1",
        "revolute_1_2",
        "revolute_1_3",
        "revolute_1_4",
        "revolute_1_5",
        "revolute_1_6",
        "revolute_2",
        "revolute_2_1",
        "revolute_2_2",
        "revolute_2_3",
        "revolute_2_4",
        "revolute_2_6",
    )
)
EXPECTED_CONFIG_SNAPSHOT: dict[str, Any] = {
    "action_scale": 0.20,
    "rated_torque_nm": 1.60,
    "nominal_height_m": 0.181,
    "stand_nominal_height_m": 0.177,
    "moving_command_threshold_mps": 0.05,
    "axis_command_active_threshold": 0.015,
    "reset_root_height_m": 0.185,
    "reset_femur_angle_rad": 0.60,
    "reset_tibia_angle_rad": 2.2335,
    "processed_joint_target_slew_limit_rad_per_20ms": 0.04,
    "velocity_command_sampling_mode": "stage2e_joystick_transitions",
    "velocity_command_resampling_time_range_s": [3.0, 3.0],
    "command_lin_vel_x_range_mps": [-0.20, 0.30],
    "command_lin_vel_y_range_mps": [-0.10, 0.10],
    "command_yaw_rate_range_rad_s": [-0.28, 0.28],
    "terminate_on_computed_torque_demand_nm": 5.5,
    "terminate_on_computed_torque_demand_duration_s": 0.10,
    "torque_demand_termination_grace_s": 0.50,
    "velocity_command_bucket_counts": [4, 8, 4, 4, 4, 8, 8],
    "velocity_command_bucket_stride": 13,
    "velocity_command_joystick_forward_range_mps": [0.03, 0.30],
    "velocity_command_joystick_reverse_abs_range_mps": [0.08, 0.20],
    "velocity_command_joystick_lateral_abs_range_mps": [0.03, 0.10],
    "velocity_command_joystick_yaw_abs_range_rad_s": [0.06, 0.28],
    "velocity_command_yaw_anchor_forward_range_mps": [0.0, 0.0],
    "startup_static_friction_range": [0.7, 1.2],
    "startup_dynamic_friction_range": [0.6, 1.0],
    "startup_restitution_range": [0.0, 0.02],
    "startup_material_num_buckets": 64,
    "startup_material_make_consistent": True,
    "startup_base_mass_additive_range_kg": [-0.20, 0.40],
    "robot_usd_path": EXPECTED_ROBOT_USD_PATH,
    "robot_usd_sha256": EXPECTED_ROBOT_USD_SHA256,
    "runtime_joint_names_sorted": EXPECTED_JOINT_NAMES,
    "actuator_saturation_effort_nm": 5.5,
    "actuator_effort_limit_nm": 1.6,
    "actuator_effort_limit_sim_nm": 5.5,
    "actuator_velocity_limit_rad_s": 480.0 * 2.0 * math.pi / 60.0,
    "actuator_velocity_limit_sim_rad_s": 528.0 * 2.0 * math.pi / 60.0,
    "actuator_stiffness_nm_rad": 30.0,
    "actuator_damping_nm_s_rad": 0.6,
    "actuator_armature_kg_m2": 7.0e-4,
    "actuator_friction_nm": 0.01,
    "actuator_dynamic_friction_nm": 0.01,
    "actuator_viscous_friction_nm_s_rad": 0.002,
}

RMS_MOVING_COMPONENTS: tuple[tuple[str, str, str, float], ...] = (
    ("base_height_std_m", "base-height std", "m", 0.0035),
    ("vertical_velocity_rms_mps", "vertical-velocity RMS", "m/s", 0.070),
    (
        "roll_pitch_angular_velocity_rms_radps",
        "roll/pitch-rate RMS",
        "rad/s",
        0.28,
    ),
    ("tilt_rms_degrees", "tilt RMS", "deg", 0.90),
)
RMS_STAND_COMPONENTS: tuple[tuple[str, str, str, float], ...] = (
    ("base_height_std_m", "base-height std", "m", 0.001),
    ("vertical_velocity_rms_mps", "vertical-velocity RMS", "m/s", 0.020),
    (
        "roll_pitch_angular_velocity_rms_radps",
        "roll/pitch-rate RMS",
        "rad/s",
        0.050,
    ),
    ("tilt_rms_degrees", "tilt RMS", "deg", 0.50),
)
TAIL_COMPONENTS: tuple[tuple[str, str, str, float], ...] = (
    ("base_height_peak_to_peak_m", "base-height peak-to-peak", "m", 0.0105),
    (
        "vertical_velocity_abs_p95_mps",
        "absolute vertical-velocity p95",
        "m/s",
        0.11,
    ),
    (
        "roll_pitch_angular_velocity_p95_radps",
        "roll/pitch-rate p95",
        "rad/s",
        0.45,
    ),
    ("tilt_p95_degrees", "tilt p95", "deg", 1.40),
    ("maximum_tilt_degrees", "maximum tilt", "deg", 1.70),
    ("yaw_rate_rmse_radps", "yaw-rate RMSE", "rad/s", 0.080),
)

THRESHOLDS: dict[str, float] = {
    "minimum_forward_fraction": 0.60,
    "maximum_forward_fraction": 1.60,
    "minimum_forward_anchor_fraction": 0.80,
    "maximum_forward_anchor_fraction": 1.30,
    "minimum_lateral_fraction": 0.45,
    "maximum_lateral_fraction": 1.75,
    "minimum_lateral_pair_symmetry_ratio": 0.65,
    "minimum_yaw_pair_symmetry_ratio": 0.65,
    "minimum_yaw_fraction": 0.40,
    "maximum_yaw_fraction": 1.60,
    "maximum_inactive_translation_mps": 0.06,
    "maximum_inactive_yaw_radps": 0.10,
    "maximum_planar_velocity_rmse_mps": 0.14,
    "maximum_forward_anchor_planar_velocity_rmse_mps": 0.12,
    "maximum_stand_planar_speed_mps": 0.03,
    "maximum_stand_planar_velocity_rmse_mps": 0.04,
    "maximum_stand_abs_yaw_rate_radps": 0.08,
    "maximum_mean_base_height_error_m": 0.010,
    "rated_continuous_nm": 1.60,
    "steady_peak_abs_computed_nm": 4.40,
    "steady_computed_demand_over_rating_fraction": 0.15,
    "steady_maximum_computed_over_rating_burst_s": 0.14,
    "steady_max_per_joint_rms_applied_nm": 1.40,
    "transient_peak_abs_computed_nm": 5.50,
    "transient_computed_demand_over_rating_fraction": 0.20,
    "transient_maximum_computed_over_rating_burst_s": 0.20,
    "transient_max_per_joint_rms_applied_nm": 1.60,
    "rollout_peak_abs_computed_nm": 5.50,
    "rollout_computed_demand_over_rating_fraction": 0.15,
    "rollout_maximum_computed_over_rating_burst_s": 0.20,
    "rollout_max_per_joint_rms_applied_nm": 1.40,
    "per_joint_rms_applied_nm": 1.60,
    "steady_per_joint_computed_demand_over_rating_fraction": 0.25,
    "transient_per_joint_computed_demand_over_rating_fraction": 0.35,
    "rollout_per_joint_computed_demand_over_rating_fraction": 0.25,
    "per_joint_maximum_computed_over_rating_burst_s": 0.20,
    "per_joint_peak_abs_computed_nm": 5.50,
    # Raw error immediately after a sign flip correctly includes the old
    # command-to-new-command delta. Gate only overshoot beyond that unavoidable
    # discontinuity; settling time owns recovery speed.
    "transient_planar_error_overshoot_margin_mps": 0.15,
    "transient_yaw_error_overshoot_margin_radps": 0.15,
    "transient_base_height_peak_to_peak_m": 0.021,
    "transient_maximum_abs_vertical_velocity_mps": 0.22,
    "transient_maximum_roll_pitch_angular_velocity_radps": 0.90,
    "transient_maximum_tilt_degrees": 3.40,
    "maximum_settling_time_s": MAXIMUM_SETTLING_TIME_SECONDS,
}

PER_JOINT_TORQUE_METRICS: tuple[str, ...] = (
    "rms_applied_nm",
    "peak_abs_applied_nm",
    "peak_abs_computed_nm",
    "applied_at_rating_fraction",
    "computed_demand_over_rating_fraction",
    "maximum_computed_over_rating_burst_s",
)


class ReportError(ValueError):
    """Raised when transition JSON or invocation provenance is malformed."""


@dataclass(frozen=True)
class Provenance:
    task: str
    experiment: str
    run_name: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class StaticAdmission:
    """Immutable selection proven by the prerequisite terminal E2 screen."""

    grade_sha256: str
    selected_checkpoint: str
    selected_checkpoint_host: str
    selected_checkpoint_sha256: str
    checkpoint_iteration: int


def make_provenance(
    task: str,
    experiment: str,
    run_name: str,
    checkpoint_sha256: str,
) -> Provenance:
    """Validate caller-owned exact task/run/checkpoint provenance."""

    if task != EXPECTED_TASK_ID:
        raise ReportError(
            "audited transition admission targets only the terminal Stage2E E2 task"
        )
    if experiment != EXPECTED_EXPERIMENT:
        raise ReportError(
            f"experiment must be exact Stage2E E2 experiment {EXPECTED_EXPERIMENT!r}"
        )
    if (
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name) is None
        or run_name in {".", ".."}
        or "stage2e_e2" not in run_name.lower()
    ):
        raise ReportError(f"run_name must safely identify stage2e_e2, got {run_name!r}")
    if not isinstance(checkpoint_sha256, str):
        raise ReportError("checkpoint_sha256 must be a string")
    normalized_sha = checkpoint_sha256.lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized_sha) is None:
        raise ReportError("checkpoint_sha256 must be 64 hexadecimal characters")
    return Provenance(task, experiment, run_name, normalized_sha)


def validate_static_admission(
    payload: Any,
    provenance: Provenance,
    static_grade_sha256: str,
    *,
    run_dir: str | Path | None = None,
) -> StaticAdmission:
    """Validate the prerequisite terminal-E2 static grade and its selection.

    The transition screen is deliberately not an independent checkpoint
    selector.  It may evaluate only the exact checkpoint already selected by
    an accepted terminal E2 fixed-command grade.
    """

    normalized_grade_sha = static_grade_sha256.lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized_grade_sha) is None:
        raise ReportError("static_grade_sha256 must be 64 hexadecimal characters")
    if not isinstance(payload, dict):
        raise ReportError("static grade must be a JSON object")
    _reject_nonfinite_numbers(payload, "static_grade")
    expected_scalars = {
        "schema_version": EXPECTED_STATIC_GRADE_SCHEMA_VERSION,
        "stage": "E2",
        "task": provenance.task,
        "experiment": provenance.experiment,
        "exact_run_name": provenance.run_name,
        "accepted": True,
        "candidate_iterations": list(EXPECTED_CANDIDATE_ITERATIONS),
        "candidate_count": len(EXPECTED_CANDIDATE_ITERATIONS),
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            raise ReportError(
                f"static_grade.{key} must be {expected!r}, got {payload.get(key)!r}"
            )

    selected = payload.get("selected_checkpoint")
    selected_host = payload.get("selected_checkpoint_host")
    selected_sha = payload.get("selected_checkpoint_sha256")
    if not all(isinstance(value, str) and value for value in (selected, selected_host)):
        raise ReportError("static grade must contain selected container and host paths")
    if not isinstance(selected_sha, str) or re.fullmatch(
        r"[0-9a-f]{64}", selected_sha.lower()
    ) is None:
        raise ReportError("static grade selected checkpoint SHA-256 is malformed")
    selected_sha = selected_sha.lower()
    if selected_sha != provenance.checkpoint_sha256:
        raise ReportError("static grade selected SHA differs from transition provenance")

    expected_container_parent = PurePosixPath(
        "/workspace/hexapod/isaaclab/logs/rsl_rl"
    ) / provenance.experiment / provenance.run_name
    selected_path = PurePosixPath(selected)
    match = re.fullmatch(r"model_(\d+)\.pt", selected_path.name)
    if selected_path.parent != expected_container_parent or match is None:
        raise ReportError("static grade selected checkpoint is outside the exact E2 run")
    iteration = int(match.group(1))
    if iteration not in EXPECTED_CANDIDATE_ITERATIONS:
        raise ReportError(
            f"static grade selected unexpected E2 checkpoint iteration {iteration}"
        )
    selected_host_path = Path(selected_host)
    if not selected_host_path.is_absolute() or selected_host_path.name != selected_path.name:
        raise ReportError("static grade selected host checkpoint path is invalid")
    if run_dir is not None:
        expected_host = Path(run_dir) / selected_path.name
        if selected_host_path != expected_host:
            raise ReportError(
                "static grade selected host checkpoint differs from exact run-dir"
            )
    elif (
        selected_host_path.parent.name != provenance.run_name
        or selected_host_path.parent.parent.name != provenance.experiment
    ):
        raise ReportError("static grade selected host path does not identify the exact run")

    ranked = payload.get("accepted_checkpoints_ranked")
    accepted_count = _nonnegative_int(
        payload.get("accepted_checkpoint_count"),
        "static_grade.accepted_checkpoint_count",
    )
    if (
        not isinstance(ranked, list)
        or not ranked
        or not all(isinstance(item, str) for item in ranked)
        or len(set(ranked)) != len(ranked)
        or accepted_count != len(ranked)
    ):
        raise ReportError("static grade accepted checkpoint ranking is inconsistent")
    if payload.get("best_accepted_checkpoint") != selected or ranked[0] != selected:
        raise ReportError("static grade selected checkpoint is not its best accepted checkpoint")

    provenance_contract = payload.get("provenance_contract")
    if not isinstance(provenance_contract, dict):
        raise ReportError("static_grade.provenance_contract must be an object")
    for key, expected in {
        "task": provenance.task,
        "experiment": provenance.experiment,
        "exact_run_name": provenance.run_name,
        "selected_checkpoint": selected,
        "selected_checkpoint_sha256": selected_sha,
    }.items():
        if provenance_contract.get(key) != expected:
            raise ReportError(
                f"static_grade.provenance_contract.{key} must be {expected!r}"
            )

    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != len(
        EXPECTED_CANDIDATE_ITERATIONS
    ):
        raise ReportError("static grade must contain the complete E2 candidate set")
    seen_iterations: set[int] = set()
    accepted_from_evaluations: set[str] = set()
    selected_matches = 0
    for index, candidate in enumerate(evaluations):
        path = f"static_grade.evaluations[{index}]"
        if not isinstance(candidate, dict):
            raise ReportError(f"{path} must be an object")
        checkpoint = candidate.get("checkpoint")
        if not isinstance(checkpoint, str):
            raise ReportError(f"{path}.checkpoint must be a string")
        checkpoint_path = PurePosixPath(checkpoint)
        checkpoint_match = re.fullmatch(r"model_(\d+)\.pt", checkpoint_path.name)
        if checkpoint_path.parent != expected_container_parent or checkpoint_match is None:
            raise ReportError(f"{path}.checkpoint is outside the exact E2 run")
        candidate_iteration = _nonnegative_int(candidate.get("iteration"), f"{path}.iteration")
        if candidate_iteration != int(checkpoint_match.group(1)):
            raise ReportError(f"{path}.iteration disagrees with checkpoint filename")
        if candidate_iteration in seen_iterations:
            raise ReportError("static grade has duplicate candidate iterations")
        seen_iterations.add(candidate_iteration)
        accepted = candidate.get("accepted")
        if not isinstance(accepted, bool):
            raise ReportError(f"{path}.accepted must be boolean")
        candidate_sha = candidate.get("checkpoint_sha256")
        if not isinstance(candidate_sha, str) or re.fullmatch(
            r"[0-9a-f]{64}", candidate_sha.lower()
        ) is None:
            raise ReportError(f"{path}.checkpoint_sha256 is malformed")
        if accepted:
            accepted_from_evaluations.add(checkpoint)
        if checkpoint == selected:
            selected_matches += 1
            if not accepted or candidate_sha.lower() != selected_sha:
                raise ReportError("selected static candidate is not accepted with exact SHA")
    if seen_iterations != set(EXPECTED_CANDIDATE_ITERATIONS):
        raise ReportError("static grade candidate iteration set is incomplete")
    if selected_matches != 1 or accepted_from_evaluations != set(ranked):
        raise ReportError("static grade accepted candidates disagree with ranking")

    handoff = payload.get("handoff")
    if not isinstance(handoff, dict):
        raise ReportError("static grade terminal handoff is missing")
    for key, expected in {
        "source_stage": "E2",
        "next_stage": "TRANSITION",
        "next_experiment": None,
        "selected_checkpoint": selected,
        "selected_checkpoint_host": selected_host,
        "selected_checkpoint_sha256": selected_sha,
    }.items():
        if handoff.get(key) != expected:
            raise ReportError(f"static_grade.handoff.{key} must be {expected!r}")

    return StaticAdmission(
        grade_sha256=normalized_grade_sha,
        selected_checkpoint=selected,
        selected_checkpoint_host=selected_host,
        selected_checkpoint_sha256=selected_sha,
        checkpoint_iteration=iteration,
    )


def _reject_nonfinite_numbers(value: Any, path: str = "report") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        try:
            converted = float(value)
        except OverflowError as exc:
            raise ReportError(f"{path} is outside finite float range") from exc
        if not math.isfinite(converted):
            raise ReportError(f"{path} must be finite, got {value!r}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite_numbers(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite_numbers(item, f"{path}.{key}")


def _finite_float(value: Any, path: str) -> float:
    if isinstance(value, bool):
        raise ReportError(f"{path} must be numeric, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReportError(f"{path} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ReportError(f"{path} must be finite, got {result!r}")
    return result


def _nonnegative_float(value: Any, path: str) -> float:
    result = _finite_float(value, path)
    if result < 0.0:
        raise ReportError(f"{path} must be nonnegative, got {result!r}")
    return result


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReportError(f"{path} must be a nonnegative integer, got {value!r}")
    return value


def _finite_vector(value: Any, length: int, path: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ReportError(f"{path} must be a {length}-element list")
    return tuple(
        _finite_float(item, f"{path}[{index}]")
        for index, item in enumerate(value)
    )


def _maximum_reason(
    reasons: list[str], name: str, actual: float, limit: float, unit: str = ""
) -> None:
    if actual > limit + GATE_ABS_TOLERANCE:
        suffix = f" {unit}" if unit else ""
        reasons.append(
            f"{name} {actual:.5f}{suffix} exceeds {limit:.5f}{suffix}"
        )


def _minimum_reason(
    reasons: list[str], name: str, actual: float, limit: float, unit: str = ""
) -> None:
    if actual < limit - GATE_ABS_TOLERANCE:
        suffix = f" {unit}" if unit else ""
        reasons.append(
            f"{name} {actual:.5f}{suffix} is below {limit:.5f}{suffix}"
        )


def _command_from_metric_row(
    row: dict[str, Any], path: str
) -> tuple[float, float, float]:
    command = row.get("command")
    if not isinstance(command, dict) or command.get("frame") != "navigation":
        raise ReportError(f"{path}.command must be a navigation-frame object")
    return (
        _finite_float(command.get("body_vx_mps"), f"{path}.command.body_vx_mps"),
        _finite_float(command.get("body_vy_mps"), f"{path}.command.body_vy_mps"),
        _finite_float(command.get("yaw_rate_radps"), f"{path}.command.yaw_rate_radps"),
    )


def _commands_match(
    actual: tuple[float, float, float], expected: tuple[float, float, float]
) -> bool:
    return all(
        abs(actual_value - expected_value) <= COMMAND_TOLERANCE
        for actual_value, expected_value in zip(actual, expected)
    )


def _per_joint_rows(torque: dict[str, Any], path: str) -> list[dict[str, Any]]:
    rows = torque.get("per_joint")
    if not isinstance(rows, list) or len(rows) != EXPECTED_JOINT_COUNT:
        actual = len(rows) if isinstance(rows, list) else None
        raise ReportError(
            f"{path}.per_joint must contain exactly {EXPECTED_JOINT_COUNT} rows, "
            f"got {actual!r}"
        )
    names: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        row_path = f"{path}.per_joint[{index}]"
        if not isinstance(row, dict):
            raise ReportError(f"{row_path} must be an object")
        name = row.get("name")
        if not isinstance(name, str) or not name or name != name.strip():
            raise ReportError(f"{row_path}.name must be a non-empty trimmed string")
        if name in names:
            raise ReportError(f"{path}.per_joint contains duplicate name {name!r}")
        names.add(name)
        validated.append(
            {
                "name": name,
                **{
                    key: _nonnegative_float(row.get(key), f"{row_path}.{key}")
                    for key in PER_JOINT_TORQUE_METRICS
                },
            }
        )
    if sorted(names) != EXPECTED_JOINT_NAMES:
        raise ReportError(
            f"{path}.per_joint names must be the exact 18 RobStride joints"
        )
    return validated


def _grade_torque(
    torque: Any, *, path: str, window: str
) -> tuple[list[str], list[str], dict[str, float]]:
    if not isinstance(torque, dict):
        raise ReportError(f"{path} must be an object")
    rated = _nonnegative_float(torque.get("rated_continuous_nm"), f"{path}.rated_continuous_nm")
    if not math.isclose(rated, THRESHOLDS["rated_continuous_nm"], abs_tol=GATE_ABS_TOLERANCE):
        raise ReportError(f"{path}.rated_continuous_nm must be 1.60")
    per_joint = _per_joint_rows(torque, path)
    metrics = {
        "rms_applied_nm": _nonnegative_float(
            torque.get("rms_applied_nm"), f"{path}.rms_applied_nm"
        ),
        "max_per_joint_rms_applied_nm": _nonnegative_float(
            torque.get("max_per_joint_rms_applied_nm"),
            f"{path}.max_per_joint_rms_applied_nm",
        ),
        "peak_abs_computed_nm": _nonnegative_float(
            torque.get("peak_abs_computed_nm"), f"{path}.peak_abs_computed_nm"
        ),
        "peak_abs_applied_nm": _nonnegative_float(
            torque.get("peak_abs_applied_nm"), f"{path}.peak_abs_applied_nm"
        ),
        "applied_at_rating_fraction": _nonnegative_float(
            torque.get("applied_at_rating_fraction"),
            f"{path}.applied_at_rating_fraction",
        ),
        "computed_demand_over_rating_fraction": _nonnegative_float(
            torque.get("computed_demand_over_rating_fraction"),
            f"{path}.computed_demand_over_rating_fraction",
        ),
        "maximum_computed_over_rating_burst_s": _nonnegative_float(
            torque.get("maximum_computed_over_rating_burst_s"),
            f"{path}.maximum_computed_over_rating_burst_s",
        ),
    }
    actual_max_rms = max(float(row["rms_applied_nm"]) for row in per_joint)
    expected_aggregates = {
        "max_per_joint_rms_applied_nm": actual_max_rms,
        "rms_applied_nm": math.sqrt(
            sum(float(row["rms_applied_nm"]) ** 2 for row in per_joint)
            / len(per_joint)
        ),
        "peak_abs_applied_nm": max(
            float(row["peak_abs_applied_nm"]) for row in per_joint
        ),
        "peak_abs_computed_nm": max(
            float(row["peak_abs_computed_nm"]) for row in per_joint
        ),
        "applied_at_rating_fraction": sum(
            float(row["applied_at_rating_fraction"]) for row in per_joint
        )
        / len(per_joint),
        "computed_demand_over_rating_fraction": sum(
            float(row["computed_demand_over_rating_fraction"])
            for row in per_joint
        )
        / len(per_joint),
        "maximum_computed_over_rating_burst_s": max(
            float(row["maximum_computed_over_rating_burst_s"])
            for row in per_joint
        ),
    }
    for key, expected in expected_aggregates.items():
        if not math.isclose(
            metrics[key],
            expected,
            rel_tol=0.0,
            abs_tol=AGGREGATE_TORQUE_ABS_TOLERANCE,
        ):
            raise ReportError(f"{path}.{key} disagrees with per-joint rows")

    reasons: list[str] = []
    if window not in {"steady", "transient", "rollout"}:
        raise ReportError(f"unsupported torque grading window {window!r}")
    prefix = window
    for key, display, threshold_suffix, unit in (
        ("peak_abs_computed_nm", "peak absolute computed torque", "peak_abs_computed_nm", "Nm"),
        (
            "computed_demand_over_rating_fraction",
            "computed demand over rating fraction",
            "computed_demand_over_rating_fraction",
            "",
        ),
        (
            "maximum_computed_over_rating_burst_s",
            "computed over-rating burst",
            "maximum_computed_over_rating_burst_s",
            "s",
        ),
        (
            "max_per_joint_rms_applied_nm",
            "max per-joint RMS applied torque",
            "max_per_joint_rms_applied_nm",
            "Nm",
        ),
    ):
        _maximum_reason(
            reasons,
            display,
            metrics[key],
            THRESHOLDS[f"{prefix}_{threshold_suffix}"],
            unit,
        )

    duty_limit = THRESHOLDS[
        f"{prefix}_per_joint_computed_demand_over_rating_fraction"
    ]
    for joint in per_joint:
        name = str(joint["name"])
        for key, display, limit, unit in (
            (
                "rms_applied_nm",
                "RMS applied torque",
                THRESHOLDS["per_joint_rms_applied_nm"],
                "Nm",
            ),
            (
                "computed_demand_over_rating_fraction",
                "computed demand over rating fraction",
                duty_limit,
                "",
            ),
            (
                "maximum_computed_over_rating_burst_s",
                "computed over-rating burst",
                THRESHOLDS["per_joint_maximum_computed_over_rating_burst_s"],
                "s",
            ),
            (
                "peak_abs_computed_nm",
                "peak absolute computed torque",
                THRESHOLDS["per_joint_peak_abs_computed_nm"],
                "Nm",
            ),
        ):
            _maximum_reason(
                reasons, f"joint {name} {display}", float(joint[key]), limit, unit
            )
    return reasons, [str(row["name"]) for row in per_joint], metrics


def _validate_metric_window(
    row: Any,
    *,
    path: str,
    copy_index: int,
    command: tuple[float, float, float],
    samples: int,
    measured_seconds: float,
) -> tuple[dict[str, Any], int, int]:
    if not isinstance(row, dict):
        raise ReportError(f"{path} must be an object")
    index = _nonnegative_int(row.get("index"), f"{path}.index")
    if index != copy_index:
        raise ReportError(f"{path}.index must be {copy_index}, got {index}")
    if not _commands_match(_command_from_metric_row(row, path), command):
        raise ReportError(f"{path}.command differs from schedule")
    actual_samples = _nonnegative_int(row.get("samples"), f"{path}.samples")
    if actual_samples != samples:
        raise ReportError(f"{path}.samples must be {samples}, got {actual_samples}")
    actual_seconds = _nonnegative_float(
        row.get("measured_seconds"), f"{path}.measured_seconds"
    )
    if not math.isclose(actual_seconds, measured_seconds, abs_tol=COMMAND_TOLERANCE):
        raise ReportError(
            f"{path}.measured_seconds must be {measured_seconds}, got {actual_seconds}"
        )
    falls = _nonnegative_int(row.get("falls"), f"{path}.falls")
    timeouts = _nonnegative_int(row.get("timeouts"), f"{path}.timeouts")
    fall_free = row.get("fall_free")
    if not isinstance(fall_free, bool) or fall_free != (falls == 0):
        raise ReportError(f"{path}.fall_free is inconsistent with falls")
    return row, falls, timeouts


def _validate_rollout_safety_row(
    row: Any, *, path: str, copy_index: int
) -> tuple[dict[str, Any], int, int]:
    """Validate the full-run reset/torque-only projection."""

    if not isinstance(row, dict):
        raise ReportError(f"{path} must be an object")
    expected_keys = {
        "index",
        "samples",
        "measured_seconds",
        "falls",
        "timeouts",
        "fall_free",
        "torque",
    }
    if set(row) != expected_keys:
        raise ReportError(
            f"{path} must contain only reset and torque safety fields"
        )
    if _nonnegative_int(row.get("index"), f"{path}.index") != copy_index:
        raise ReportError(f"{path}.index must be {copy_index}")
    samples = _nonnegative_int(row.get("samples"), f"{path}.samples")
    if samples != TOTAL_STEPS:
        raise ReportError(f"{path}.samples must be {TOTAL_STEPS}")
    seconds = _nonnegative_float(row.get("measured_seconds"), f"{path}.measured_seconds")
    if not math.isclose(seconds, TOTAL_DURATION_SECONDS, abs_tol=COMMAND_TOLERANCE):
        raise ReportError(f"{path}.measured_seconds must be {TOTAL_DURATION_SECONDS}")
    falls = _nonnegative_int(row.get("falls"), f"{path}.falls")
    timeouts = _nonnegative_int(row.get("timeouts"), f"{path}.timeouts")
    fall_free = row.get("fall_free")
    if not isinstance(fall_free, bool) or fall_free != (falls == 0):
        raise ReportError(f"{path}.fall_free is inconsistent with falls")
    return row, falls, timeouts


def _grade_steady(
    row: dict[str, Any],
    *,
    path: str,
    label: str,
    command: tuple[float, float, float],
) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
    operational_reasons: list[str] = []
    stability_reasons: list[str] = []
    achieved_linear = _finite_vector(
        row.get("mean_command_frame_linear_velocity_mps"),
        3,
        f"{path}.mean_command_frame_linear_velocity_mps",
    )
    achieved_angular = _finite_vector(
        row.get("mean_command_frame_angular_velocity_radps"),
        3,
        f"{path}.mean_command_frame_angular_velocity_radps",
    )
    planar_rmse = _nonnegative_float(
        row.get("planar_velocity_rmse_mps"), f"{path}.planar_velocity_rmse_mps"
    )
    vx, vy, yaw = command
    achieved_vx, achieved_vy = achieved_linear[:2]
    achieved_yaw = achieved_angular[2]
    stand = all(abs(value) <= COMMAND_TOLERANCE for value in command)

    if stand:
        _maximum_reason(
            operational_reasons,
            "stand planar speed",
            math.hypot(achieved_vx, achieved_vy),
            THRESHOLDS["maximum_stand_planar_speed_mps"],
            "m/s",
        )
        _maximum_reason(
            operational_reasons,
            "stand absolute yaw rate",
            abs(achieved_yaw),
            THRESHOLDS["maximum_stand_abs_yaw_rate_radps"],
            "rad/s",
        )
        _maximum_reason(
            operational_reasons,
            "stand planar velocity RMSE",
            planar_rmse,
            THRESHOLDS["maximum_stand_planar_velocity_rmse_mps"],
            "m/s",
        )
    else:
        for axis, commanded, achieved, minimum_key, maximum_key in (
            ("forward", vx, achieved_vx, "minimum_forward_fraction", "maximum_forward_fraction"),
            ("lateral", vy, achieved_vy, "minimum_lateral_fraction", "maximum_lateral_fraction"),
        ):
            if abs(commanded) > COMMAND_TOLERANCE:
                if label == "forward" and axis == "forward":
                    minimum_key = "minimum_forward_anchor_fraction"
                    maximum_key = "maximum_forward_anchor_fraction"
                signed_fraction = math.copysign(1.0, commanded) * achieved / abs(commanded)
                if achieved * commanded <= 0.0:
                    operational_reasons.append(
                        f"{axis} sign is wrong: command={commanded:+.4f}, achieved={achieved:+.4f}"
                    )
                _minimum_reason(
                    operational_reasons,
                    f"signed {axis} command fraction",
                    signed_fraction,
                    THRESHOLDS[minimum_key],
                )
                _maximum_reason(
                    operational_reasons,
                    f"signed {axis} command fraction",
                    signed_fraction,
                    THRESHOLDS[maximum_key],
                )
            else:
                _maximum_reason(
                    operational_reasons,
                    f"inactive {axis} leakage",
                    abs(achieved),
                    THRESHOLDS["maximum_inactive_translation_mps"],
                    "m/s",
                )
        if abs(yaw) > COMMAND_TOLERANCE:
            signed_yaw_fraction = math.copysign(1.0, yaw) * achieved_yaw / abs(yaw)
            if achieved_yaw * yaw <= 0.0:
                operational_reasons.append(
                    f"yaw sign is wrong: command={yaw:+.4f}, achieved={achieved_yaw:+.4f}"
                )
            _minimum_reason(
                operational_reasons,
                "signed yaw command fraction",
                signed_yaw_fraction,
                THRESHOLDS["minimum_yaw_fraction"],
            )
            _maximum_reason(
                operational_reasons,
                "signed yaw command fraction",
                signed_yaw_fraction,
                THRESHOLDS["maximum_yaw_fraction"],
            )
        else:
            _maximum_reason(
                operational_reasons,
                "inactive yaw leakage",
                abs(achieved_yaw),
                THRESHOLDS["maximum_inactive_yaw_radps"],
                "rad/s",
            )
        _maximum_reason(
            operational_reasons,
            "planar velocity RMSE",
            planar_rmse,
            THRESHOLDS[
                "maximum_forward_anchor_planar_velocity_rmse_mps"
                if label == "forward"
                else "maximum_planar_velocity_rmse_mps"
            ],
            "m/s",
        )

    mean_base_height = _nonnegative_float(
        row.get("mean_base_height_m"), f"{path}.mean_base_height_m"
    )
    uses_stand_height = math.hypot(vx, vy) <= float(
        EXPECTED_CONFIG_SNAPSHOT["moving_command_threshold_mps"]
    ) + COMMAND_TOLERANCE
    target_base_height = EXPECTED_CONFIG_SNAPSHOT[
        "stand_nominal_height_m" if uses_stand_height else "nominal_height_m"
    ]
    mean_base_height_error = abs(mean_base_height - float(target_base_height))
    _maximum_reason(
        stability_reasons,
        "mean base-height target error",
        mean_base_height_error,
        THRESHOLDS["maximum_mean_base_height_error_m"],
        "m",
    )

    components = (
        RMS_STAND_COMPONENTS if stand else RMS_MOVING_COMPONENTS
    ) + TAIL_COMPONENTS
    stability_values: dict[str, float] = {}
    for key, display, unit, target in components:
        value = _nonnegative_float(row.get(key), f"{path}.{key}")
        stability_values[key] = value
        _maximum_reason(stability_reasons, display, value, target, unit)

    torque_reasons, names, torque_metrics = _grade_torque(
        row.get("torque"), path=f"{path}.torque", window="steady"
    )
    return (
        operational_reasons,
        stability_reasons,
        torque_reasons,
        {
            "label": label,
            "achieved_command_frame_velocity": [achieved_vx, achieved_vy, achieved_yaw],
            "planar_velocity_rmse_mps": planar_rmse,
            "mean_base_height_m": mean_base_height,
            "mean_base_height_error_m": mean_base_height_error,
            **stability_values,
            **torque_metrics,
            "per_joint_names": names,
        },
    )


def _grade_transient_trace(
    trace: Any,
    *,
    path: str,
    copy_index: int,
    command: tuple[float, float, float],
    previous_command: tuple[float, float, float] | None,
) -> tuple[list[str], dict[str, Any]]:
    if not isinstance(trace, dict):
        raise ReportError(f"{path} must be an object")
    if _nonnegative_int(trace.get("copy_index"), f"{path}.copy_index") != copy_index:
        raise ReportError(f"{path}.copy_index must be {copy_index}")
    if _nonnegative_int(trace.get("samples"), f"{path}.samples") != TRANSIENT_STEPS:
        raise ReportError(f"{path}.samples must be {TRANSIENT_STEPS}")
    seconds = _nonnegative_float(trace.get("measured_seconds"), f"{path}.measured_seconds")
    if not math.isclose(
        seconds, TRANSIENT_STEPS * POLICY_STEP_SECONDS, abs_tol=COMMAND_TOLERANCE
    ):
        raise ReportError(f"{path}.measured_seconds is invalid")
    if trace.get("baseline_captured_before_first_action") is not True:
        raise ReportError(f"{path} did not capture the pre-action height baseline")
    dwell = _nonnegative_int(trace.get("settling_dwell_steps"), f"{path}.settling_dwell_steps")
    if dwell != SETTLING_DWELL_STEPS:
        raise ReportError(f"{path}.settling_dwell_steps must be {SETTLING_DWELL_STEPS}")
    settled_count = _nonnegative_int(
        trace.get("settled_sample_count"), f"{path}.settled_sample_count"
    )
    linear_trace_raw = trace.get("command_frame_linear_velocity_mps_samples")
    yaw_trace_raw = trace.get("command_frame_yaw_rate_radps_samples")
    if not isinstance(linear_trace_raw, list) or len(linear_trace_raw) != TRANSIENT_STEPS:
        raise ReportError(
            f"{path}.command_frame_linear_velocity_mps_samples must contain "
            f"exactly {TRANSIENT_STEPS} samples"
        )
    if not isinstance(yaw_trace_raw, list) or len(yaw_trace_raw) != TRANSIENT_STEPS:
        raise ReportError(
            f"{path}.command_frame_yaw_rate_radps_samples must contain exactly "
            f"{TRANSIENT_STEPS} samples"
        )
    linear_trace = [
        _finite_vector(sample, 3, f"{path}.command_frame_linear_velocity_mps_samples[{index}]")
        for index, sample in enumerate(linear_trace_raw)
    ]
    yaw_trace = [
        _finite_float(sample, f"{path}.command_frame_yaw_rate_radps_samples[{index}]")
        for index, sample in enumerate(yaw_trace_raw)
    ]
    settled_flags = [
        tracking_within_settling_band(command, linear, (0.0, 0.0, yaw))
        for linear, yaw in zip(linear_trace, yaw_trace)
    ]
    computed_settled_count = sum(settled_flags)
    computed_completion_step = settling_completion_step(
        settled_flags, dwell_steps=SETTLING_DWELL_STEPS
    )
    if settled_count != computed_settled_count:
        raise ReportError(f"{path}.settled_sample_count disagrees with tracking trace")

    reasons: list[str] = []
    completed = trace.get("settling_completed")
    if not isinstance(completed, bool):
        raise ReportError(f"{path}.settling_completed must be boolean")
    if completed != (computed_completion_step is not None):
        raise ReportError(f"{path}.settling_completed disagrees with tracking trace")
    completion_step = trace.get("settling_completion_step")
    settling_time = trace.get("settling_time_s")
    if completed:
        step = _nonnegative_int(completion_step, f"{path}.settling_completion_step")
        if step != computed_completion_step:
            raise ReportError(
                f"{path}.settling_completion_step disagrees with tracking trace"
            )
        time_value = _nonnegative_float(settling_time, f"{path}.settling_time_s")
        if not math.isclose(
            time_value, step * POLICY_STEP_SECONDS, abs_tol=GATE_ABS_TOLERANCE
        ):
            raise ReportError(f"{path}.settling_time_s disagrees with completion step")
        _maximum_reason(
            reasons,
            "settling time",
            time_value,
            THRESHOLDS["maximum_settling_time_s"],
            "s",
        )
    else:
        if completion_step is not None or settling_time is not None:
            raise ReportError(f"{path} has settling values despite incomplete status")
        reasons.append("tracking did not settle within the transient window")

    metrics: dict[str, float | int | bool | None] = {
        "settling_completed": completed,
        "settling_completion_step": completion_step,
        "settling_time_s": settling_time,
        "settled_sample_count": settled_count,
    }
    prior = command if previous_command is None else previous_command
    x_command_delta = abs(command[0] - prior[0])
    y_command_delta = abs(command[1] - prior[1])
    planar_command_delta = math.hypot(
        x_command_delta, y_command_delta
    )
    yaw_command_delta = abs(command[2] - prior[2])
    computed_x_error = max(abs(sample[0] - command[0]) for sample in linear_trace)
    computed_y_error = max(abs(sample[1] - command[1]) for sample in linear_trace)
    computed_planar_error = max(
        math.hypot(sample[0] - command[0], sample[1] - command[1])
        for sample in linear_trace
    )
    computed_yaw_error = max(abs(sample - command[2]) for sample in yaw_trace)
    summarized_errors = {
        "maximum_x_tracking_error_mps": computed_x_error,
        "maximum_y_tracking_error_mps": computed_y_error,
        "maximum_planar_tracking_error_mps": computed_planar_error,
        "maximum_yaw_tracking_error_radps": computed_yaw_error,
    }
    for key, computed in summarized_errors.items():
        claimed = _nonnegative_float(trace.get(key), f"{path}.{key}")
        if not math.isclose(
            claimed,
            computed,
            rel_tol=0.0,
            abs_tol=AGGREGATE_TORQUE_ABS_TOLERANCE,
        ):
            raise ReportError(f"{path}.{key} disagrees with tracking trace")
    x_error = computed_x_error
    y_error = computed_y_error
    planar_error = computed_planar_error
    yaw_error = computed_yaw_error
    x_error_limit = (
        x_command_delta + THRESHOLDS["transient_planar_error_overshoot_margin_mps"]
    )
    y_error_limit = (
        y_command_delta + THRESHOLDS["transient_planar_error_overshoot_margin_mps"]
    )
    planar_error_limit = (
        planar_command_delta
        + THRESHOLDS["transient_planar_error_overshoot_margin_mps"]
    )
    yaw_error_limit = (
        yaw_command_delta
        + THRESHOLDS["transient_yaw_error_overshoot_margin_radps"]
    )
    _maximum_reason(
        reasons,
        "transient x tracking-error overshoot",
        x_error,
        x_error_limit,
        "m/s",
    )
    _maximum_reason(
        reasons,
        "transient y tracking-error overshoot",
        y_error,
        y_error_limit,
        "m/s",
    )
    _maximum_reason(
        reasons,
        "transient planar tracking-error overshoot",
        planar_error,
        planar_error_limit,
        "m/s",
    )
    _maximum_reason(
        reasons,
        "transient yaw tracking-error overshoot",
        yaw_error,
        yaw_error_limit,
        "rad/s",
    )
    metrics.update(
        {
            "maximum_x_tracking_error_mps": x_error,
            "maximum_x_tracking_error_limit_mps": x_error_limit,
            "maximum_y_tracking_error_mps": y_error,
            "maximum_y_tracking_error_limit_mps": y_error_limit,
            "maximum_planar_tracking_error_mps": planar_error,
            "maximum_planar_tracking_error_limit_mps": planar_error_limit,
            "maximum_yaw_tracking_error_radps": yaw_error,
            "maximum_yaw_tracking_error_limit_radps": yaw_error_limit,
        }
    )
    for key, display, threshold_key, unit in (
        (
            "base_height_peak_to_peak_m",
            "transient base-height peak-to-peak",
            "transient_base_height_peak_to_peak_m",
            "m",
        ),
        (
            "maximum_abs_vertical_velocity_mps",
            "transient absolute vertical velocity",
            "transient_maximum_abs_vertical_velocity_mps",
            "m/s",
        ),
        (
            "maximum_roll_pitch_angular_velocity_radps",
            "transient roll/pitch rate",
            "transient_maximum_roll_pitch_angular_velocity_radps",
            "rad/s",
        ),
        (
            "maximum_tilt_degrees",
            "transient maximum tilt",
            "transient_maximum_tilt_degrees",
            "deg",
        ),
    ):
        value = _nonnegative_float(trace.get(key), f"{path}.{key}")
        metrics[key] = value
        _maximum_reason(reasons, display, value, THRESHOLDS[threshold_key], unit)
    return reasons, metrics


def _validate_report_contract(report: dict[str, Any], provenance: Provenance) -> tuple[str, int]:
    expected_scalars = {
        "schema": SCHEMA_ID,
        "task": provenance.task,
        "command_frame": "navigation",
        "seed": EVALUATION_SEED,
        "deterministic_policy": True,
        "startup_randomization_enabled": True,
        "copies": COPIES,
        "rendered_steps": TOTAL_STEPS,
        "reset_stance_override": None,
        "continuous_episode_requested": True,
        "command_resampling_suppressed": True,
        "environment_reset_at_command_boundaries": False,
        "policy_state_reset_at_command_boundaries": False,
        "schedule_sha256": SCHEDULE_SHA256,
    }
    for key, expected in expected_scalars.items():
        if key not in report or report.get(key) != expected:
            raise ReportError(f"report.{key} must be {expected!r}, got {report.get(key)!r}")
    if report.get("task_config_class") != EXPECTED_TASK_CONFIG_CLASS:
        raise ReportError(
            f"report.task_config_class must be {EXPECTED_TASK_CONFIG_CLASS!r}"
        )
    if report.get("agent_config_class") != EXPECTED_AGENT_CONFIG_CLASS:
        raise ReportError(
            f"report.agent_config_class must be {EXPECTED_AGENT_CONFIG_CLASS!r}"
        )
    config_snapshot = report.get("config_snapshot")
    if not isinstance(config_snapshot, dict):
        raise ReportError("report.config_snapshot must be an object")
    for key, expected in EXPECTED_CONFIG_SNAPSHOT.items():
        if key not in config_snapshot:
            raise ReportError(f"report.config_snapshot.{key} is missing")
        actual = config_snapshot[key]
        if isinstance(expected, float):
            if not math.isclose(
                _finite_float(actual, f"report.config_snapshot.{key}"),
                expected,
                abs_tol=GATE_ABS_TOLERANCE,
            ):
                raise ReportError(
                    f"report.config_snapshot.{key} must be {expected!r}"
                )
        elif isinstance(expected, list):
            if all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in expected
            ):
                vector = _finite_vector(
                    actual, len(expected), f"report.config_snapshot.{key}"
                )
                if any(
                    not math.isclose(item, target, abs_tol=GATE_ABS_TOLERANCE)
                    for item, target in zip(vector, expected)
                ):
                    raise ReportError(
                        f"report.config_snapshot.{key} must be {expected!r}"
                    )
            elif actual != expected:
                raise ReportError(
                    f"report.config_snapshot.{key} must be {expected!r}"
                )
        elif isinstance(expected, bool):
            if actual is not expected:
                raise ReportError(
                    f"report.config_snapshot.{key} must be {expected!r}"
                )
        elif isinstance(expected, int):
            if isinstance(actual, bool) or actual != expected:
                raise ReportError(
                    f"report.config_snapshot.{key} must be {expected!r}"
                )
        elif actual != expected:
            raise ReportError(
                f"report.config_snapshot.{key} must be {expected!r}"
            )
    policy_dt = _finite_float(report.get("policy_step_seconds"), "report.policy_step_seconds")
    if not math.isclose(policy_dt, POLICY_STEP_SECONDS, abs_tol=GATE_ABS_TOLERANCE):
        raise ReportError(f"report.policy_step_seconds must be {POLICY_STEP_SECONDS}")
    duration = _finite_float(report.get("rendered_duration_s"), "report.rendered_duration_s")
    if not math.isclose(duration, TOTAL_DURATION_SECONDS, abs_tol=GATE_ABS_TOLERANCE):
        raise ReportError(f"report.rendered_duration_s must be {TOTAL_DURATION_SECONDS}")
    episode_length = _finite_float(
        report.get("episode_length_seconds"), "report.episode_length_seconds"
    )
    if episode_length < TOTAL_DURATION_SECONDS + 5.0 - GATE_ABS_TOLERANCE:
        raise ReportError("report episode length does not protect the uninterrupted schedule")

    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ReportError("report.checkpoint must be non-empty")
    expected_parent = PurePosixPath(
        "/workspace/hexapod/isaaclab/logs/rsl_rl"
    ) / provenance.experiment / provenance.run_name
    checkpoint_path = PurePosixPath(checkpoint)
    match = re.fullmatch(r"model_(\d+)\.pt", checkpoint_path.name)
    if checkpoint_path.parent != expected_parent or match is None:
        raise ReportError("report checkpoint is not an exact model inside the exact run")
    if int(match.group(1)) not in EXPECTED_CANDIDATE_ITERATIONS:
        raise ReportError("report checkpoint iteration is outside the terminal E2 set")
    if report.get("checkpoint_sha256") != provenance.checkpoint_sha256:
        raise ReportError("report checkpoint SHA-256 differs from invocation provenance")
    if report.get("schedule") != schedule_payload():
        raise ReportError("report.schedule differs from the immutable schedule payload")
    return checkpoint, int(match.group(1))


def grade_payload(
    payload: Any,
    provenance: Provenance,
    static_admission: StaticAdmission,
) -> dict[str, Any]:
    """Validate and grade all 44 copy/segment transitions in one report."""

    if not isinstance(payload, dict):
        raise ReportError("top-level JSON value must be one report object")
    _reject_nonfinite_numbers(payload)
    checkpoint, iteration = _validate_report_contract(payload, provenance)
    if (
        checkpoint != static_admission.selected_checkpoint
        or iteration != static_admission.checkpoint_iteration
        or provenance.checkpoint_sha256
        != static_admission.selected_checkpoint_sha256
    ):
        raise ReportError(
            "transition report checkpoint does not match static E2 selection"
        )
    segments = payload.get("segments")
    expected_schedule = schedule_payload()
    if not isinstance(segments, list) or len(segments) != len(expected_schedule):
        raise ReportError(
            f"report.segments must contain exactly {len(expected_schedule)} segments"
        )

    rollout_rows = payload.get("rollout_safety_metrics")
    if not isinstance(rollout_rows, list) or len(rollout_rows) != COPIES:
        raise ReportError(
            f"report.rollout_safety_metrics must contain exactly {COPIES} rows"
        )
    rollout_results: list[dict[str, Any]] = []
    rollout_falls = 0
    rollout_timeouts = 0
    expected_joint_names: list[str] | None = None
    seen_rollout_indices: set[int] = set()
    for row_position, row in enumerate(rollout_rows):
        row_path = f"report.rollout_safety_metrics[{row_position}]"
        if not isinstance(row, dict):
            raise ReportError(f"{row_path} must be an object")
        copy_index = _nonnegative_int(row.get("index"), f"{row_path}.index")
        if copy_index in seen_rollout_indices or copy_index >= COPIES:
            raise ReportError(
                f"report.rollout_safety_metrics has invalid copy index {copy_index}"
            )
        seen_rollout_indices.add(copy_index)
        validated, falls, timeouts = _validate_rollout_safety_row(
            row,
            path=row_path,
            copy_index=copy_index,
        )
        torque_reasons, names, torque_metrics = _grade_torque(
            validated.get("torque"), path=f"{row_path}.torque", window="rollout"
        )
        if expected_joint_names is None:
            expected_joint_names = names
        elif names != expected_joint_names:
            raise ReportError(f"{row_path} joint names/order differ across rollout")
        reset_reasons: list[str] = []
        if falls:
            reset_reasons.append(f"falls={falls}, expected 0")
        if timeouts:
            reset_reasons.append(f"timeouts={timeouts}, expected 0")
        reasons = [*reset_reasons, *torque_reasons]
        rollout_falls += falls
        rollout_timeouts += timeouts
        rollout_results.append(
            {
                "copy_index": copy_index,
                "accepted": not reasons,
                "rejection_reasons": reasons,
                "falls": falls,
                "timeouts": timeouts,
                "torque_metrics": torque_metrics,
            }
        )
    if seen_rollout_indices != set(range(COPIES)):
        raise ReportError(f"rollout copy indices must be exactly 0..{COPIES - 1}")
    rollout_results.sort(key=lambda item: int(item["copy_index"]))
    reported_falls = _nonnegative_int(payload.get("total_falls"), "report.total_falls")
    reported_timeouts = _nonnegative_int(
        payload.get("total_timeouts"), "report.total_timeouts"
    )
    if reported_falls != rollout_falls or reported_timeouts != rollout_timeouts:
        raise ReportError("top-level fall/timeout totals disagree with full-rollout metrics")
    continuity = payload.get("continuous_episode_achieved")
    if not isinstance(continuity, bool) or continuity != (
        rollout_falls == 0 and rollout_timeouts == 0
    ):
        raise ReportError("report.continuous_episode_achieved is inconsistent")

    graded_segments: list[dict[str, Any]] = []
    total_falls = 0
    total_timeouts = 0
    for segment_index, (segment, expected) in enumerate(zip(segments, expected_schedule)):
        path = f"report.segments[{segment_index}]"
        if not isinstance(segment, dict):
            raise ReportError(f"{path} must be an object")
        for key, value in expected.items():
            if segment.get(key) != value:
                raise ReportError(f"{path}.{key} differs from immutable schedule")
        command = tuple(float(value) for value in expected["command"])
        results = segment.get("results")
        if not isinstance(results, list) or len(results) != COPIES:
            raise ReportError(f"{path}.results must contain exactly {COPIES} copies")
        if not all(isinstance(result, dict) for result in results):
            raise ReportError(f"{path}.results entries must be objects")
        seen_copy_indices: set[int] = set()
        graded_copies: list[dict[str, Any]] = []
        for result_position, result in enumerate(results):
            result_path = f"{path}.results[{result_position}]"
            copy_index = _nonnegative_int(
                result.get("copy_index"), f"{result_path}.copy_index"
            )
            if copy_index in seen_copy_indices:
                raise ReportError(f"{path}.results contains duplicate copy {copy_index}")
            if copy_index >= COPIES:
                raise ReportError(f"{result_path}.copy_index is outside 0..{COPIES - 1}")
            seen_copy_indices.add(copy_index)

            transient_row, transient_falls, transient_timeouts = _validate_metric_window(
                result.get("transient_metrics"),
                path=f"{result_path}.transient_metrics",
                copy_index=copy_index,
                command=command,
                samples=TRANSIENT_STEPS,
                measured_seconds=TRANSIENT_STEPS * POLICY_STEP_SECONDS,
            )
            steady_row, steady_falls, steady_timeouts = _validate_metric_window(
                result.get("steady_state_metrics"),
                path=f"{result_path}.steady_state_metrics",
                copy_index=copy_index,
                command=command,
                samples=STEADY_STEPS,
                measured_seconds=STEADY_STEPS * POLICY_STEP_SECONDS,
            )
            falls = transient_falls + steady_falls
            timeouts = transient_timeouts + steady_timeouts
            total_falls += falls
            total_timeouts += timeouts
            reset_reasons: list[str] = []
            if falls:
                reset_reasons.append(f"falls={falls}, expected 0")
            if timeouts:
                reset_reasons.append(f"timeouts={timeouts}, expected 0")

            transient_torque_reasons, transient_names, transient_torque = _grade_torque(
                transient_row.get("torque"),
                path=f"{result_path}.transient_metrics.torque",
                window="transient",
            )
            steady_operational, steady_stability, steady_torque_reasons, steady_metrics = _grade_steady(
                steady_row,
                path=f"{result_path}.steady_state_metrics",
                label=str(expected["key"]),
                command=command,
            )
            steady_names = list(steady_metrics.pop("per_joint_names"))
            if transient_names != steady_names:
                raise ReportError(f"{result_path} joint names differ between windows")
            if expected_joint_names is None:
                expected_joint_names = steady_names
            elif steady_names != expected_joint_names:
                raise ReportError(f"{result_path} joint names/order differ across report")
            transient_reasons, transient_metrics = _grade_transient_trace(
                result.get("transient_trace"),
                path=f"{result_path}.transient_trace",
                copy_index=copy_index,
                command=command,
                previous_command=(
                    None
                    if expected["previous_command"] is None
                    else tuple(float(value) for value in expected["previous_command"])
                ),
            )
            operational_reasons = [
                *reset_reasons,
                *steady_operational,
            ]
            rs05_reasons = [
                *transient_torque_reasons,
                *steady_torque_reasons,
            ]
            rejection_reasons = [
                *operational_reasons,
                *steady_stability,
                *transient_reasons,
                *rs05_reasons,
            ]
            graded_copies.append(
                {
                    "copy_index": copy_index,
                    "operational_safe": not operational_reasons,
                    "steady_stability_safe": not steady_stability,
                    "transient_safe": not transient_reasons,
                    "rs05_safe": not rs05_reasons,
                    "accepted": not rejection_reasons,
                    "operational_rejection_reasons": operational_reasons,
                    "steady_stability_rejection_reasons": steady_stability,
                    "transient_rejection_reasons": transient_reasons,
                    "rs05_rejection_reasons": rs05_reasons,
                    "rejection_reasons": rejection_reasons,
                    "falls": falls,
                    "timeouts": timeouts,
                    "transient_metrics": {
                        **transient_metrics,
                        **transient_torque,
                    },
                    "steady_state_metrics": steady_metrics,
                }
            )
        if seen_copy_indices != set(range(COPIES)):
            raise ReportError(f"{path}.results copy indices must be exactly 0..{COPIES - 1}")
        graded_copies.sort(key=lambda item: int(item["copy_index"]))
        graded_segments.append(
            {
                **expected,
                "accepted": all(copy["accepted"] for copy in graded_copies),
                "copies": graded_copies,
            }
        )

    segments_by_key = {segment["key"]: segment for segment in graded_segments}
    for (
        pair_name,
        positive_key,
        negative_key,
        velocity_axis,
        threshold_key,
    ) in (
        (
            "lateral",
            "strafe_left",
            "strafe_right",
            1,
            "minimum_lateral_pair_symmetry_ratio",
        ),
        (
            "yaw",
            "yaw_left",
            "yaw_right",
            2,
            "minimum_yaw_pair_symmetry_ratio",
        ),
    ):
        positive_segment = segments_by_key[positive_key]
        negative_segment = segments_by_key[negative_key]
        for copy_index in range(COPIES):
            positive_copy = positive_segment["copies"][copy_index]
            negative_copy = negative_segment["copies"][copy_index]
            positive_magnitude = abs(
                float(
                    positive_copy["steady_state_metrics"][
                        "achieved_command_frame_velocity"
                    ][velocity_axis]
                )
            )
            negative_magnitude = abs(
                float(
                    negative_copy["steady_state_metrics"][
                        "achieved_command_frame_velocity"
                    ][velocity_axis]
                )
            )
            maximum_magnitude = max(positive_magnitude, negative_magnitude)
            symmetry_ratio = (
                min(positive_magnitude, negative_magnitude) / maximum_magnitude
                if maximum_magnitude > 0.0
                else 0.0
            )
            for copy_result in (positive_copy, negative_copy):
                copy_result["steady_state_metrics"][
                    f"{pair_name}_pair_symmetry_ratio"
                ] = symmetry_ratio
            minimum_ratio = THRESHOLDS[threshold_key]
            if symmetry_ratio < minimum_ratio - GATE_ABS_TOLERANCE:
                reason = (
                    f"{pair_name} sign-flip symmetry ratio "
                    f"{symmetry_ratio:.5f} is below {minimum_ratio:.5f}"
                )
                for copy_result in (positive_copy, negative_copy):
                    copy_result["operational_safe"] = False
                    copy_result["accepted"] = False
                    copy_result["operational_rejection_reasons"].append(reason)
                    copy_result["rejection_reasons"].append(reason)
        positive_segment["accepted"] = all(
            copy["accepted"] for copy in positive_segment["copies"]
        )
        negative_segment["accepted"] = all(
            copy["accepted"] for copy in negative_segment["copies"]
        )

    if total_falls != rollout_falls or total_timeouts != rollout_timeouts:
        raise ReportError(
            "segment-window fall/timeout totals disagree with full-rollout metrics"
        )
    rollout_safe = all(result["accepted"] for result in rollout_results)
    accepted = (
        all(segment["accepted"] for segment in graded_segments)
        and rollout_safe
        and continuity
    )
    return {
        "schema_version": 1,
        "task": provenance.task,
        "experiment": provenance.experiment,
        "run_name": provenance.run_name,
        "checkpoint": checkpoint,
        "checkpoint_iteration": iteration,
        "checkpoint_sha256": provenance.checkpoint_sha256,
        "static_admission": {
            "accepted": True,
            "grade_sha256": static_admission.grade_sha256,
            "selected_checkpoint": static_admission.selected_checkpoint,
            "selected_checkpoint_host": static_admission.selected_checkpoint_host,
            "selected_checkpoint_sha256": (
                static_admission.selected_checkpoint_sha256
            ),
            "checkpoint_iteration": static_admission.checkpoint_iteration,
        },
        "schedule_sha256": SCHEDULE_SHA256,
        "playback_contract": {
            "seed": EVALUATION_SEED,
            "copies": COPIES,
            "policy_step_seconds": POLICY_STEP_SECONDS,
            "total_steps": TOTAL_STEPS,
            "total_duration_seconds": TOTAL_DURATION_SECONDS,
            "segment_steps": SEGMENT_STEPS,
            "transient_steps": TRANSIENT_STEPS,
            "steady_steps": STEADY_STEPS,
            "startup_randomization_enabled": True,
            "continuous_episode_requested": True,
            "environment_reset_at_command_boundaries": False,
            "policy_state_reset_at_command_boundaries": False,
        },
        "thresholds": THRESHOLDS,
        "moving_rms_targets": {
            key: target for key, _, _, target in RMS_MOVING_COMPONENTS
        },
        "stand_rms_targets": {
            key: target for key, _, _, target in RMS_STAND_COMPONENTS
        },
        "tail_targets": {key: target for key, _, _, target in TAIL_COMPONENTS},
        "joint_names": expected_joint_names,
        "segment_count": len(graded_segments),
        "copy_transition_count": len(graded_segments) * COPIES,
        "total_falls": rollout_falls,
        "total_timeouts": rollout_timeouts,
        "continuous_episode_achieved": continuity,
        "rollout_rs05_safe": rollout_safe,
        "failed_rollout_copy_count": sum(
            not result["accepted"] for result in rollout_results
        ),
        "failed_copy_transition_count": sum(
            not copy["accepted"]
            for segment in graded_segments
            for copy in segment["copies"]
        ),
        "accepted": accepted,
        "rollout_results": rollout_results,
        "segments": graded_segments,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grade one exact terminal-E2 continuous transition report."
    )
    parser.add_argument("input")
    parser.add_argument("--task", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--static-grade", required=True)
    parser.add_argument("--static-grade-sha256", required=True)
    parser.add_argument("--json", dest="json_path")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = _parse_args()
    try:
        static_grade_argument = Path(args.static_grade).expanduser()
        if static_grade_argument.is_symlink():
            raise ReportError("static grade must not be a symlink")
        static_grade_path = static_grade_argument.resolve(strict=True)
        if not static_grade_path.is_file():
            raise ReportError("static grade must be a canonical non-symlink file")
        expected_static_grade_path = Path(
            "/home/orionh/HEXAPOD/artifacts/phase2_recovery_stage2e_joystick/"
            f"e2/static/{args.run_name}/grade_static.json"
        )
        if static_grade_path != expected_static_grade_path:
            raise ReportError(
                "static grade must be the exact terminal E2 artifact for this run"
            )
        expected_static_grade_sha = args.static_grade_sha256.lower()
        if re.fullmatch(r"[0-9a-f]{64}", expected_static_grade_sha) is None:
            raise ReportError("static grade SHA-256 must be 64 hexadecimal characters")
        if _sha256(static_grade_path) != expected_static_grade_sha:
            raise ReportError("static grade SHA-256 mismatch before grading")
        static_payload = json.loads(static_grade_path.read_text(encoding="utf-8"))
        selected_sha = (
            static_payload.get("selected_checkpoint_sha256")
            if isinstance(static_payload, dict)
            else None
        )
        provenance = make_provenance(
            args.task, args.experiment, args.run_name, selected_sha
        )
        run_dir_argument = Path(args.run_dir).expanduser()
        if run_dir_argument.is_symlink():
            raise ReportError("run-dir must not be a symlink")
        run_dir = run_dir_argument.resolve(strict=True)
        if (
            not run_dir.is_dir()
            or run_dir.name != provenance.run_name
            or run_dir.parent.name != provenance.experiment
        ):
            raise ReportError("run-dir must be the exact canonical terminal E2 run")
        static_admission = validate_static_admission(
            static_payload,
            provenance,
            expected_static_grade_sha,
            run_dir=run_dir,
        )
        checkpoint_host = Path(static_admission.selected_checkpoint_host)
        if (
            checkpoint_host.resolve(strict=True) != checkpoint_host
            or not checkpoint_host.is_file()
            or checkpoint_host.is_symlink()
        ):
            raise ReportError("selected checkpoint must be canonical and non-symlink")
        if _sha256(checkpoint_host) != static_admission.selected_checkpoint_sha256:
            raise ReportError("selected checkpoint SHA-256 mismatch before grading")
        payload = json.loads(
            Path(args.input).expanduser().resolve().read_text(encoding="utf-8")
        )
        grade = grade_payload(payload, provenance, static_admission)
    except (OSError, json.JSONDecodeError, ReportError, TypeError) as exc:
        print(f"grading error: {exc}", file=sys.stderr)
        return 65

    status = "PASS" if grade["accepted"] else "FAIL"
    print(
        f"{status} checkpoint={grade['checkpoint']} "
        f"transitions={grade['copy_transition_count']} "
        f"failed={grade['failed_copy_transition_count']} "
        f"falls={grade['total_falls']} timeouts={grade['total_timeouts']}"
    )
    for segment in grade["segments"]:
        for copy in segment["copies"]:
            for reason in copy["rejection_reasons"]:
                print(
                    f"  {segment['key']}[copy={copy['copy_index']}]: {reason}"
                )
    if args.json_path:
        output_path = Path(args.json_path).expanduser().resolve()
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("x", encoding="utf-8") as output_file:
                json.dump(grade, output_file, indent=2, sort_keys=True, allow_nan=False)
                output_file.write("\n")
        except FileExistsError:
            print(f"refusing to overwrite grade: {output_path}", file=sys.stderr)
            return 73
        except OSError as exc:
            print(f"cannot write grade {output_path}: {exc}", file=sys.stderr)
            return 73
        print(f"grading_report={output_path}")
    try:
        if _sha256(static_grade_path) != static_admission.grade_sha256:
            raise ReportError("static grade changed during transition grading")
        if _sha256(checkpoint_host) != static_admission.selected_checkpoint_sha256:
            raise ReportError("selected checkpoint changed during transition grading")
    except (OSError, ReportError) as exc:
        print(f"grading error: {exc}", file=sys.stderr)
        return 65
    return 0 if grade["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
