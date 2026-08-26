#!/usr/bin/env python3
"""Fail-closed randomized robustness screen for one accepted Stage-2C policy.

The nominal batch grader selects the checkpoint.  This second pass evaluates
that one policy with eight independently randomized simulator copies of each
fixed command.  Admission requires every copy to remain operationally safe,
the median copy for each command to meet the nominal stability targets, and
even the worst copy to remain within 1.25 times those targets.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from statistics import median
import sys
from typing import Any


TASK_ID = (
    "Isaac-Velocity-Omni-Recovery-Stage2C-Stabilized-Forward-"
    "Hexapod-RobStride-Direct-v0"
)
EXPECTED_SEED = 61
EXPECTED_POLICY_STEP_SECONDS = 0.02
EXPECTED_REQUESTED_STEPS = 500
EXPECTED_WARMUP_STEPS = 25
EXPECTED_SAMPLES = EXPECTED_REQUESTED_STEPS - EXPECTED_WARMUP_STEPS
EXPECTED_MEASURED_SECONDS = EXPECTED_SAMPLES * EXPECTED_POLICY_STEP_SECONDS
EXPECTED_COPIES_PER_COMMAND = 8
EXPECTED_JOINT_COUNT = 18
EXPECTED_EXPERIMENT = (
    "hexapod_robstride_phase2_recovery_stage2c_stable_forward_direct"
)
EXPECTED_RUN_LABEL = (
    "phase2_recovery_stage2c_lower_stable_forward_from_stage2_model25_seed60"
)
EXPECTED_CANDIDATE_ITERATIONS = (*range(0, 120, 10), 119)

COMMAND_TOLERANCE = 1.0e-4
GATE_ABS_TOLERANCE = 1.0e-6
MAXIMUM_COPY_TARGET_MULTIPLIER = 1.25

COMMAND_CONTRACT: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("stand", (0.00, 0.00, 0.00)),
    ("forward_0p16", (0.16, 0.00, 0.00)),
    ("forward_0p20", (0.20, 0.00, 0.00)),
    ("forward_0p30", (0.30, 0.00, 0.00)),
)

# These values intentionally mirror the nominal Stage-2C admission gates.
THRESHOLDS: dict[str, float] = {
    "minimum_forward_command_fraction": 0.80,
    "maximum_forward_command_fraction": 1.30,
    "maximum_planar_velocity_rmse_mps": 0.10,
    "maximum_stand_planar_speed_mps": 0.03,
    "maximum_stand_abs_yaw_rate_radps": 0.08,
    "peak_abs_computed_nm": 4.40,
    "computed_demand_over_rating_fraction": 0.15,
    "maximum_computed_over_rating_burst_s": 0.14,
    "max_per_joint_rms_applied_nm": 1.40,
    # Randomized copies get a deliberately wider duty-cycle ceiling than the
    # nominal .25 gate, but it is still a hard per-motor bound.
    "maximum_per_joint_computed_demand_over_rating_fraction": 0.35,
}

MOVING_RMS_COMPONENTS: tuple[tuple[str, str, str, float], ...] = (
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
STAND_RMS_COMPONENTS: tuple[tuple[str, str, str, float], ...] = (
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

MOVING_RMS_TARGETS = {key: target for key, _, _, target in MOVING_RMS_COMPONENTS}
STAND_RMS_TARGETS = {key: target for key, _, _, target in STAND_RMS_COMPONENTS}
TAIL_TARGETS = {key: target for key, _, _, target in TAIL_COMPONENTS}

PER_JOINT_TORQUE_METRICS: tuple[str, ...] = (
    "rms_applied_nm",
    "peak_abs_applied_nm",
    "peak_abs_computed_nm",
    "applied_at_rating_fraction",
    "computed_demand_over_rating_fraction",
    "maximum_computed_over_rating_burst_s",
)


class ReportError(ValueError):
    """Raised when evaluator JSON does not match the robust playback contract."""


def _reject_nonfinite_numbers(value: Any, path: str = "report") -> None:
    """Reject NaN/Inf anywhere, including fields not used by a gate."""

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        try:
            numeric_value = float(value)
        except OverflowError as exc:
            raise ReportError(f"{path} is outside the finite float range") from exc
        if not math.isfinite(numeric_value):
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


def _command_from_row(row: dict[str, Any], path: str) -> tuple[float, float, float]:
    command = row.get("command")
    if not isinstance(command, dict):
        raise ReportError(f"{path}.command must be an object")
    if command.get("frame") != "navigation":
        raise ReportError(
            f"{path}.command.frame must be 'navigation', "
            f"got {command.get('frame')!r}"
        )
    return (
        _finite_float(command.get("body_vx_mps"), f"{path}.command.body_vx_mps"),
        _finite_float(command.get("body_vy_mps"), f"{path}.command.body_vy_mps"),
        _finite_float(command.get("yaw_rate_radps"), f"{path}.command.yaw_rate_radps"),
    )


def _commands_match(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> bool:
    return all(
        abs(left_value - right_value) <= COMMAND_TOLERANCE
        for left_value, right_value in zip(left, right)
    )


def _iteration_and_run_directory(checkpoint: str) -> tuple[int, str]:
    path = Path(checkpoint)
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    if match is None:
        raise ReportError(
            "checkpoint must end in model_<iteration>.pt, "
            f"got {checkpoint!r}"
        )
    iteration = int(match.group(1))
    if iteration not in EXPECTED_CANDIDATE_ITERATIONS:
        raise ReportError(
            f"checkpoint iteration {iteration} is outside the nominal Stage-2C batch"
        )
    if path.parent.parent.name != EXPECTED_EXPERIMENT:
        raise ReportError(
            f"checkpoint must be inside experiment {EXPECTED_EXPERIMENT!r}"
        )
    run_name = path.parent.name
    if not (
        run_name == EXPECTED_RUN_LABEL
        or run_name.endswith(f"_{EXPECTED_RUN_LABEL}")
    ):
        raise ReportError(
            f"checkpoint run must match label {EXPECTED_RUN_LABEL!r}, "
            f"got {run_name!r}"
        )
    return iteration, str(path.parent)


def _validate_playback_contract(report: dict[str, Any]) -> tuple[str, int, str]:
    if report.get("task") != TASK_ID:
        raise ReportError(f"report.task must be {TASK_ID!r}")
    if report.get("command_frame") != "navigation":
        raise ReportError("report.command_frame must be 'navigation'")
    if report.get("deterministic_policy") is not True:
        raise ReportError("report must be marked deterministic")
    if report.get("startup_randomization_enabled") is not True:
        raise ReportError(
            "report.startup_randomization_enabled must be true for the robust screen"
        )
    if report.get("seed") != EXPECTED_SEED:
        raise ReportError(
            f"report.seed must be {EXPECTED_SEED}, got {report.get('seed')!r}"
        )
    policy_step = _finite_float(
        report.get("policy_step_seconds"), "report.policy_step_seconds"
    )
    if not math.isclose(
        policy_step, EXPECTED_POLICY_STEP_SECONDS, abs_tol=GATE_ABS_TOLERANCE
    ):
        raise ReportError(
            "report.policy_step_seconds must be "
            f"{EXPECTED_POLICY_STEP_SECONDS}, got {policy_step}"
        )
    requested_steps = _nonnegative_int(
        report.get("requested_steps"), "report.requested_steps"
    )
    if requested_steps != EXPECTED_REQUESTED_STEPS:
        raise ReportError(
            f"report.requested_steps must be {EXPECTED_REQUESTED_STEPS}, "
            f"got {requested_steps}"
        )
    warmup_steps = _nonnegative_int(
        report.get("warmup_steps"), "report.warmup_steps"
    )
    if warmup_steps != EXPECTED_WARMUP_STEPS:
        raise ReportError(
            f"report.warmup_steps must be {EXPECTED_WARMUP_STEPS}, "
            f"got {warmup_steps}"
        )
    expected_parallel = len(COMMAND_CONTRACT) * EXPECTED_COPIES_PER_COMMAND
    parallel = _nonnegative_int(
        report.get("commands_evaluated_in_parallel"),
        "report.commands_evaluated_in_parallel",
    )
    if parallel != expected_parallel:
        raise ReportError(
            "report.commands_evaluated_in_parallel must be "
            f"{expected_parallel}, got {parallel}"
        )
    if "reset_stance_override" not in report:
        raise ReportError("report.reset_stance_override is missing")
    if report["reset_stance_override"] is not None:
        raise ReportError(
            "report.reset_stance_override must be null so the task stance is used"
        )
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ReportError("report.checkpoint must be a non-empty string")
    iteration, run_directory = _iteration_and_run_directory(checkpoint)
    return checkpoint, iteration, run_directory


def _group_rows(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = report.get("results")
    expected_count = len(COMMAND_CONTRACT) * EXPECTED_COPIES_PER_COMMAND
    if not isinstance(rows, list) or len(rows) != expected_count:
        actual = len(rows) if isinstance(rows, list) else None
        raise ReportError(
            f"report.results must contain exactly {expected_count} rows, got {actual!r}"
        )
    if not all(isinstance(row, dict) for row in rows):
        raise ReportError("every report.results row must be an object")

    seen_indices: set[int] = set()
    grouped: dict[str, list[dict[str, Any]]] = {
        label: [] for label, _ in COMMAND_CONTRACT
    }
    for position, row in enumerate(rows):
        path = f"report.results[{position}]"
        index = _nonnegative_int(row.get("index"), f"{path}.index")
        if index in seen_indices:
            raise ReportError(f"report.results contains duplicate index {index}")
        seen_indices.add(index)
        command = _command_from_row(row, path)
        matches = [
            label
            for label, expected in COMMAND_CONTRACT
            if _commands_match(command, expected)
        ]
        if len(matches) != 1:
            raise ReportError(
                f"{path}.command must match exactly one robust-screen command"
            )
        grouped[matches[0]].append(row)

    if seen_indices != set(range(expected_count)):
        raise ReportError(
            f"report.results indices must be exactly 0..{expected_count - 1}"
        )
    for label, command_rows in grouped.items():
        if len(command_rows) != EXPECTED_COPIES_PER_COMMAND:
            raise ReportError(
                f"command {label} must have exactly {EXPECTED_COPIES_PER_COMMAND} "
                f"copies, got {len(command_rows)}"
            )
        command_rows.sort(key=lambda row: int(row["index"]))
    return grouped


def _per_joint_torque_rows(
    torque: dict[str, Any], path: str
) -> list[dict[str, Any]]:
    rows = torque.get("per_joint")
    if not isinstance(rows, list) or len(rows) != EXPECTED_JOINT_COUNT:
        actual = len(rows) if isinstance(rows, list) else None
        raise ReportError(
            f"{path}.per_joint must contain exactly {EXPECTED_JOINT_COUNT} "
            f"rows, got {actual!r}"
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
    return validated


def _grade_copy(
    row: dict[str, Any], *, label: str, command: tuple[float, float, float]
) -> dict[str, Any]:
    index = int(row["index"])
    path = f"{label}[index={index}]"
    reasons: list[str] = []
    samples = _nonnegative_int(row.get("samples"), f"{path}.samples")
    if samples != EXPECTED_SAMPLES:
        raise ReportError(
            f"{path}.samples must be {EXPECTED_SAMPLES}, got {samples}"
        )
    measured_seconds = _nonnegative_float(
        row.get("measured_seconds"), f"{path}.measured_seconds"
    )
    if not math.isclose(
        measured_seconds, EXPECTED_MEASURED_SECONDS, abs_tol=COMMAND_TOLERANCE
    ):
        raise ReportError(
            f"{path}.measured_seconds must be {EXPECTED_MEASURED_SECONDS}, "
            f"got {measured_seconds}"
        )
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
    falls = _nonnegative_int(row.get("falls"), f"{path}.falls")
    timeouts = _nonnegative_int(row.get("timeouts"), f"{path}.timeouts")
    fall_free = row.get("fall_free")
    if not isinstance(fall_free, bool) or fall_free != (falls == 0):
        raise ReportError(f"{path}.fall_free must be a boolean consistent with falls")
    if falls:
        reasons.append(f"falls={falls}, expected 0")
    if timeouts:
        reasons.append(f"timeouts={timeouts}, expected 0")

    rms_components = STAND_RMS_COMPONENTS if label == "stand" else MOVING_RMS_COMPONENTS
    stability = {
        key: _nonnegative_float(row.get(key), f"{path}.{key}")
        for key, _, _, _ in (*rms_components, *TAIL_COMPONENTS)
    }

    torque = row.get("torque")
    if not isinstance(torque, dict):
        raise ReportError(f"{path}.torque must be an object")
    per_joint = _per_joint_torque_rows(torque, f"{path}.torque")
    torque_metrics = {
        "rms_applied_nm": _nonnegative_float(
            torque.get("rms_applied_nm"), f"{path}.torque.rms_applied_nm"
        ),
        "max_per_joint_rms_applied_nm": _nonnegative_float(
            torque.get("max_per_joint_rms_applied_nm"),
            f"{path}.torque.max_per_joint_rms_applied_nm",
        ),
        "peak_abs_computed_nm": _nonnegative_float(
            torque.get("peak_abs_computed_nm"),
            f"{path}.torque.peak_abs_computed_nm",
        ),
        "computed_demand_over_rating_fraction": _nonnegative_float(
            torque.get("computed_demand_over_rating_fraction"),
            f"{path}.torque.computed_demand_over_rating_fraction",
        ),
        "maximum_computed_over_rating_burst_s": _nonnegative_float(
            torque.get("maximum_computed_over_rating_burst_s"),
            f"{path}.torque.maximum_computed_over_rating_burst_s",
        ),
    }
    actual_max_joint_rms = max(float(item["rms_applied_nm"]) for item in per_joint)
    if not math.isclose(
        torque_metrics["max_per_joint_rms_applied_nm"],
        actual_max_joint_rms,
        abs_tol=GATE_ABS_TOLERANCE,
    ):
        raise ReportError(
            f"{path}.torque.max_per_joint_rms_applied_nm disagrees with per_joint rows"
        )
    worst_joint = max(
        per_joint,
        key=lambda item: float(item["computed_demand_over_rating_fraction"]),
    )
    worst_joint_duty = float(worst_joint["computed_demand_over_rating_fraction"])

    achieved_vx, achieved_vy = achieved_linear[:2]
    achieved_yaw = achieved_angular[2]
    planar_speed = math.hypot(achieved_vx, achieved_vy)
    if label == "stand":
        _maximum_reason(
            reasons,
            "stand planar speed",
            planar_speed,
            THRESHOLDS["maximum_stand_planar_speed_mps"],
            "m/s",
        )
        _maximum_reason(
            reasons,
            "stand absolute yaw rate",
            abs(achieved_yaw),
            THRESHOLDS["maximum_stand_abs_yaw_rate_radps"],
            "rad/s",
        )
    else:
        _minimum_reason(
            reasons,
            "achieved forward velocity",
            achieved_vx,
            command[0] * THRESHOLDS["minimum_forward_command_fraction"],
            "m/s",
        )
        _maximum_reason(
            reasons,
            "achieved forward velocity",
            achieved_vx,
            command[0] * THRESHOLDS["maximum_forward_command_fraction"],
            "m/s",
        )
    _maximum_reason(
        reasons,
        "planar velocity RMSE",
        planar_rmse,
        THRESHOLDS["maximum_planar_velocity_rmse_mps"],
        "m/s",
    )
    for key, display_name, unit in (
        ("peak_abs_computed_nm", "peak absolute computed torque", "Nm"),
        (
            "computed_demand_over_rating_fraction",
            "computed demand over rating fraction",
            "",
        ),
        (
            "maximum_computed_over_rating_burst_s",
            "computed over-rating burst",
            "s",
        ),
        (
            "max_per_joint_rms_applied_nm",
            "max per-joint RMS applied torque",
            "Nm",
        ),
    ):
        _maximum_reason(reasons, display_name, torque_metrics[key], THRESHOLDS[key], unit)
    _maximum_reason(
        reasons,
        f"worst per-joint computed demand fraction ({worst_joint['name']})",
        worst_joint_duty,
        THRESHOLDS["maximum_per_joint_computed_demand_over_rating_fraction"],
    )

    return {
        "index": index,
        "operational_safe": not reasons,
        "operational_rejection_reasons": reasons,
        "metrics": {
            "achieved_command_frame_velocity": [achieved_vx, achieved_vy, achieved_yaw],
            "achieved_planar_speed_mps": planar_speed,
            "planar_velocity_rmse_mps": planar_rmse,
            "samples": samples,
            "measured_seconds": measured_seconds,
            "falls": falls,
            "timeouts": timeouts,
            **stability,
            **torque_metrics,
            "per_joint_torque_row_count": len(per_joint),
            "worst_per_joint_computed_demand_over_rating_name": worst_joint["name"],
            "worst_per_joint_computed_demand_over_rating_fraction": worst_joint_duty,
        },
    }


def _grade_command_stability(
    label: str, copy_results: list[dict[str, Any]]
) -> dict[str, Any]:
    components = (
        STAND_RMS_COMPONENTS if label == "stand" else MOVING_RMS_COMPONENTS
    ) + TAIL_COMPONENTS
    reasons: list[str] = []
    summaries: dict[str, dict[str, Any]] = {}
    for key, display_name, unit, target in components:
        values = [float(result["metrics"][key]) for result in copy_results]
        median_value = float(median(values))
        maximum_value = max(values)
        median_met = median_value <= target + GATE_ABS_TOLERANCE
        copy_limit = target * MAXIMUM_COPY_TARGET_MULTIPLIER
        every_copy_met = maximum_value <= copy_limit + GATE_ABS_TOLERANCE
        if not median_met:
            reasons.append(
                f"median {display_name} {median_value:.5f} {unit} exceeds "
                f"nominal target {target:.5f} {unit}"
            )
        if not every_copy_met:
            reasons.append(
                f"worst-copy {display_name} {maximum_value:.5f} {unit} exceeds "
                f"robust limit {copy_limit:.5f} {unit}"
            )
        summaries[key] = {
            "target": target,
            "median": median_value,
            "median_target_met": median_met,
            "maximum": maximum_value,
            "maximum_copy_limit": copy_limit,
            "every_copy_limit_met": every_copy_met,
        }
    return {
        "stability_safe": not reasons,
        "stability_rejection_reasons": reasons,
        "stability_components": summaries,
    }


def grade_payload(payload: Any) -> dict[str, Any]:
    """Validate and grade one randomized 32-environment Stage-2C report."""

    if not isinstance(payload, dict):
        raise ReportError("top-level JSON value must be one report object")
    if "evaluations" in payload:
        raise ReportError("robustness input must be one report, not an evaluations batch")
    _reject_nonfinite_numbers(payload)
    checkpoint, iteration, run_directory = _validate_playback_contract(payload)
    grouped_rows = _group_rows(payload)

    command_results: list[dict[str, Any]] = []
    for label, command in COMMAND_CONTRACT:
        copies = [
            _grade_copy(row, label=label, command=command)
            for row in grouped_rows[label]
        ]
        stability_grade = _grade_command_stability(label, copies)
        command_results.append(
            {
                "label": label,
                "command": list(command),
                "copy_count": len(copies),
                "operational_safe": all(copy["operational_safe"] for copy in copies),
                **stability_grade,
                "copies": copies,
            }
        )

    operational_safe = all(result["operational_safe"] for result in command_results)
    stability_safe = all(result["stability_safe"] for result in command_results)
    accepted = operational_safe and stability_safe
    all_copies = [copy for result in command_results for copy in result["copies"]]
    return {
        "schema_version": 1,
        "task": TASK_ID,
        "checkpoint": checkpoint,
        "iteration": iteration,
        "candidate_run_directory": run_directory,
        "playback_contract": {
            "seed": EXPECTED_SEED,
            "policy_step_seconds": EXPECTED_POLICY_STEP_SECONDS,
            "requested_steps": EXPECTED_REQUESTED_STEPS,
            "warmup_steps": EXPECTED_WARMUP_STEPS,
            "deterministic_policy": True,
            "startup_randomization_enabled": True,
            "reset_stance_override": None,
            "copies_per_command": EXPECTED_COPIES_PER_COMMAND,
        },
        "contract": [
            {"label": label, "command": list(command)}
            for label, command in COMMAND_CONTRACT
        ],
        "thresholds": THRESHOLDS,
        "moving_rms_targets": MOVING_RMS_TARGETS,
        "stand_rms_targets": STAND_RMS_TARGETS,
        "tail_targets": TAIL_TARGETS,
        "maximum_copy_target_multiplier": MAXIMUM_COPY_TARGET_MULTIPLIER,
        "copy_count": len(all_copies),
        "total_falls": sum(copy["metrics"]["falls"] for copy in all_copies),
        "total_timeouts": sum(copy["metrics"]["timeouts"] for copy in all_copies),
        "failed_operational_copy_count": sum(
            not copy["operational_safe"] for copy in all_copies
        ),
        "failed_stability_command_count": sum(
            not result["stability_safe"] for result in command_results
        ),
        "operational_safe": operational_safe,
        "stability_safe": stability_safe,
        "accepted": accepted,
        "results": command_results,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grade one randomized eight-copy-per-command Stage-2C replay."
    )
    parser.add_argument("input", help="Evaluator JSON from evaluate_checkpoint.py")
    parser.add_argument("--json", dest="json_path", help="Write grading JSON here")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        input_path = Path(args.input).expanduser().resolve()
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        grade = grade_payload(payload)
    except (OSError, json.JSONDecodeError, ReportError) as exc:
        print(f"grading error: {exc}", file=sys.stderr)
        return 65

    status = "PASS" if grade["accepted"] else "FAIL"
    print(
        f"{status} checkpoint={grade['checkpoint']} copies={grade['copy_count']} "
        f"operational_safe={grade['operational_safe']} "
        f"stability_safe={grade['stability_safe']} "
        f"falls={grade['total_falls']} timeouts={grade['total_timeouts']}"
    )
    for result in grade["results"]:
        for reason in result["stability_rejection_reasons"]:
            print(f"  {result['label']}: {reason}")
        for copy in result["copies"]:
            for reason in copy["operational_rejection_reasons"]:
                print(f"  {result['label']}[index={copy['index']}]: {reason}")

    if args.json_path:
        output_path = Path(args.json_path).expanduser().resolve()
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("x", encoding="utf-8") as output_file:
                json.dump(grade, output_file, indent=2, sort_keys=True)
                output_file.write("\n")
        except FileExistsError:
            print(
                f"refusing to overwrite existing grading report: {output_path}",
                file=sys.stderr,
            )
            return 73
        except OSError as exc:
            print(f"cannot write grading report {output_path}: {exc}", file=sys.stderr)
            return 73
        print(f"grading_report={output_path}")
    return 0 if grade["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
