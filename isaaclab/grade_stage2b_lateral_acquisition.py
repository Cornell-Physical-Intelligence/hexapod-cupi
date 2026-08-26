#!/usr/bin/env python3
"""Grade the deterministic Stage-2B true-lateral acquisition batch.

The immutable Stage-2 model-25 seed is evaluated first through the Stage-2B
task.  Every candidate is then compared with that seed under the exact same
eight commands.  This makes the stability gate command-local: a calm command
cannot conceal a platform-wiggle regression on another command.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any


TASK_ID = (
    "Isaac-Velocity-Omni-Recovery-Stage2B-Lateral-"
    "Hexapod-RobStride-Direct-v0"
)
EXPECTED_SEED = 58
EXPECTED_SEED_CHECKPOINT_SUFFIX = (
    "/seed_from_stage2_model25/model_25_stage2_seed.pt"
)
EXPECTED_SEED_SHA256 = (
    "2cb28a0f4f4388e709111a09568c72d0770390d2ba72e08ee17eaaf0f27b6893"
)
EXPECTED_CANDIDATE_ITERATIONS = (*range(0, 100, 10), 99)
COMMAND_TOLERANCE = 1.0e-4
GATE_ABS_TOLERANCE = 1.0e-6

COMMAND_CONTRACT: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("forward_0p20", (0.20, 0.00, 0.00)),
    ("forward_0p30", (0.30, 0.00, 0.00)),
    ("pure_y_positive", (0.00, 0.10, 0.00)),
    ("pure_y_negative", (0.00, -0.10, 0.00)),
    ("bridge_y_positive", (0.22, 0.08, 0.00)),
    ("bridge_y_negative", (0.22, -0.08, 0.00)),
    ("yaw_positive", (0.23, 0.00, 0.25)),
    ("yaw_negative", (0.23, 0.00, -0.25)),
)

LATERAL_SYMMETRY_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("pure_y", "pure_y_positive", "pure_y_negative"),
    ("bridge_y", "bridge_y_positive", "bridge_y_negative"),
)

THRESHOLDS: dict[str, float] = {
    "minimum_forward_command_fraction": 0.80,
    "minimum_pure_lateral_abs_velocity_mps": 0.040,
    "minimum_bridge_lateral_abs_velocity_mps": 0.032,
    "minimum_yaw_abs_rate_radps": 0.10,
    "minimum_lateral_pair_symmetry_ratio": 0.65,
    "maximum_pure_lateral_abs_forward_velocity_mps": 0.08,
    "maximum_uncommanded_abs_yaw_rate_radps": 0.15,
    # Preserve the established Stage-2 tracking gates as well as the explicit
    # Stage-2B acquisition floors above.
    "maximum_planar_velocity_rmse_mps": 0.12,
    "maximum_lateral_rmse_mps": 0.10,
    "maximum_yaw_rate_rmse_radps": 0.18,
    "maximum_tilt_degrees": 20.0,
    "max_per_joint_rms_applied_nm": 1.60,
    "computed_demand_over_rating_fraction": 0.20,
    "maximum_computed_over_rating_burst_s": 0.20,
    "peak_abs_computed_nm": 5.50,
    "maximum_stability_regression_fraction": 0.10,
    # Ranking normalizers only; the hard stability gate is seed-relative.
    "preferred_base_height_std_m": 0.012,
    "preferred_vertical_velocity_rms_mps": 0.10,
    "preferred_roll_pitch_angular_velocity_rms_radps": 0.35,
    "preferred_tilt_rms_degrees": 5.0,
}

STABILITY_COMPONENTS: tuple[tuple[str, str, str], ...] = (
    ("base_height_std_m", "base-height std", "m"),
    ("vertical_velocity_rms_mps", "vertical-velocity RMS", "m/s"),
    (
        "roll_pitch_angular_velocity_rms_radps",
        "roll/pitch-rate RMS",
        "rad/s",
    ),
    ("tilt_rms_degrees", "tilt RMS", "deg"),
)


class ReportError(ValueError):
    """Raised when evaluator JSON does not match the Stage-2B screen contract."""


def _finite_float(value: Any, path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReportError(f"{path} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ReportError(f"{path} must be finite, got {result!r}")
    return result


def _finite_vector(value: Any, length: int, path: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ReportError(f"{path} must be a {length}-element list")
    return tuple(
        _finite_float(item, f"{path}[{index}]")
        for index, item in enumerate(value)
    )


def _reports(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ReportError("top-level JSON value must be an object")
    reports = payload.get("evaluations", [payload])
    expected_count = 1 + len(EXPECTED_CANDIDATE_ITERATIONS)
    if not isinstance(reports, list) or len(reports) != expected_count:
        raise ReportError(
            "evaluations must contain exactly the immutable seed plus "
            f"{len(EXPECTED_CANDIDATE_ITERATIONS)} candidates "
            f"({expected_count} reports total)"
        )
    if not all(isinstance(report, dict) for report in reports):
        raise ReportError("every evaluation must be an object")
    return reports


def _command_from_row(
    row: dict[str, Any], path: str
) -> tuple[float, float, float]:
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


def _validate_playback_contract(
    report: dict[str, Any], report_index: int, reference: dict[str, Any] | None
) -> None:
    prefix = f"evaluations[{report_index}]"
    if report.get("task") != TASK_ID:
        raise ReportError(
            f"{prefix}.task must be {TASK_ID!r}, got {report.get('task')!r}"
        )
    if report.get("command_frame") != "navigation":
        raise ReportError(
            f"{prefix}.command_frame must be 'navigation', "
            f"got {report.get('command_frame')!r}"
        )
    if report.get("deterministic_policy") is not True:
        raise ReportError(f"{prefix} is not marked deterministic")
    if report.get("seed") != EXPECTED_SEED:
        raise ReportError(
            f"{prefix}.seed must be {EXPECTED_SEED}, got {report.get('seed')!r}"
        )
    if report.get("commands_evaluated_in_parallel") != len(COMMAND_CONTRACT):
        raise ReportError(
            f"{prefix}.commands_evaluated_in_parallel must be "
            f"{len(COMMAND_CONTRACT)}"
        )
    if reference is None:
        return
    for key in (
        "task",
        "command_frame",
        "seed",
        "policy_step_seconds",
        "requested_steps",
        "warmup_steps",
        "commands_evaluated_in_parallel",
        "reset_stance_override",
    ):
        if report.get(key) != reference.get(key):
            raise ReportError(
                f"{prefix}.{key}={report.get(key)!r} differs from seed "
                f"value {reference.get(key)!r}"
            )


def _ordered_contract_rows(
    report: dict[str, Any], report_index: int
) -> list[dict[str, Any]]:
    rows = report.get("results")
    if not isinstance(rows, list):
        raise ReportError(f"evaluations[{report_index}].results must be a list")
    if len(rows) != len(COMMAND_CONTRACT):
        raise ReportError(
            f"evaluations[{report_index}] has {len(rows)} result rows; "
            f"expected exactly {len(COMMAND_CONTRACT)}"
        )

    unmatched = list(enumerate(rows))
    ordered: list[dict[str, Any]] = []
    for label, expected in COMMAND_CONTRACT:
        matches = [
            (position, row_index, row)
            for position, (row_index, row) in enumerate(unmatched)
            if isinstance(row, dict)
            and _commands_match(
                _command_from_row(
                    row, f"evaluations[{report_index}].results[{row_index}]"
                ),
                expected,
            )
        ]
        if len(matches) != 1:
            raise ReportError(
                f"evaluations[{report_index}] must contain command "
                f"{label}={expected} exactly once; found {len(matches)}"
            )
        position, _, row = matches[0]
        ordered.append(row)
        unmatched.pop(position)
    return ordered


def _maximum_reason(
    reasons: list[str], name: str, actual: float, limit: float, unit: str = ""
) -> None:
    if actual > limit + GATE_ABS_TOLERANCE:
        suffix = f" {unit}" if unit else ""
        reasons.append(
            f"{name} {actual:.4f}{suffix} exceeds {limit:.4f}{suffix}"
        )


def _minimum_reason(
    reasons: list[str], name: str, actual: float, limit: float, unit: str = ""
) -> None:
    if actual < limit - GATE_ABS_TOLERANCE:
        suffix = f" {unit}" if unit else ""
        reasons.append(
            f"{name} {actual:.4f}{suffix} is below {limit:.4f}{suffix}"
        )


def _grade_axis_row(
    row: dict[str, Any],
    *,
    label: str,
    command: tuple[float, float, float],
) -> dict[str, Any]:
    path = label
    reasons: list[str] = []
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
    rmse = _finite_vector(
        row.get("rmse_command_error"), 3, f"{path}.rmse_command_error"
    )
    planar_rmse = _finite_float(
        row.get("planar_velocity_rmse_mps"), f"{path}.planar_velocity_rmse_mps"
    )
    yaw_rmse = _finite_float(
        row.get("yaw_rate_rmse_radps"), f"{path}.yaw_rate_rmse_radps"
    )
    try:
        falls = int(row.get("falls", -1))
        timeouts = int(row.get("timeouts", -1))
    except (TypeError, ValueError) as exc:
        raise ReportError(f"{path}.falls and timeouts must be integers") from exc
    maximum_tilt = _finite_float(
        row.get("maximum_tilt_degrees"), f"{path}.maximum_tilt_degrees"
    )
    stability = {
        key: _finite_float(row.get(key), f"{path}.{key}")
        for key, _, _ in STABILITY_COMPONENTS
    }
    torque = row.get("torque")
    if not isinstance(torque, dict):
        raise ReportError(f"{path}.torque must be an object")
    torque_metrics = {
        "rms_applied_nm": _finite_float(
            torque.get("rms_applied_nm"), f"{path}.torque.rms_applied_nm"
        ),
        "max_per_joint_rms_applied_nm": _finite_float(
            torque.get("max_per_joint_rms_applied_nm"),
            f"{path}.torque.max_per_joint_rms_applied_nm",
        ),
        "computed_demand_over_rating_fraction": _finite_float(
            torque.get("computed_demand_over_rating_fraction"),
            f"{path}.torque.computed_demand_over_rating_fraction",
        ),
        "maximum_computed_over_rating_burst_s": _finite_float(
            torque.get("maximum_computed_over_rating_burst_s"),
            f"{path}.torque.maximum_computed_over_rating_burst_s",
        ),
        "peak_abs_computed_nm": _finite_float(
            torque.get("peak_abs_computed_nm"),
            f"{path}.torque.peak_abs_computed_nm",
        ),
    }

    vx, vy, yaw = command
    achieved_vx, achieved_vy = achieved_linear[:2]
    achieved_yaw = achieved_angular[2]

    if falls != 0:
        reasons.append(f"falls={falls}, expected 0")
    if timeouts != 0:
        reasons.append(f"timeouts={timeouts}, expected 0")

    if vx > COMMAND_TOLERANCE:
        minimum_forward = THRESHOLDS["minimum_forward_command_fraction"] * vx
        _minimum_reason(
            reasons,
            "forward velocity preservation",
            achieved_vx,
            minimum_forward,
            "m/s",
        )
    elif label.startswith("pure_y_"):
        _maximum_reason(
            reasons,
            "pure-y absolute forward leakage",
            abs(achieved_vx),
            THRESHOLDS["maximum_pure_lateral_abs_forward_velocity_mps"],
            "m/s",
        )

    if abs(vy) > COMMAND_TOLERANCE:
        if achieved_vy * vy <= 0.0:
            reasons.append(
                f"lateral sign is wrong: command={vy:+.4f}, "
                f"achieved={achieved_vy:+.4f}"
            )
        lateral_floor_key = (
            "minimum_pure_lateral_abs_velocity_mps"
            if label.startswith("pure_y_")
            else "minimum_bridge_lateral_abs_velocity_mps"
        )
        _minimum_reason(
            reasons,
            "absolute lateral velocity acquisition",
            abs(achieved_vy),
            THRESHOLDS[lateral_floor_key],
            "m/s",
        )
        _maximum_reason(
            reasons,
            "lateral RMSE",
            rmse[1],
            THRESHOLDS["maximum_lateral_rmse_mps"],
            "m/s",
        )

    if abs(yaw) > COMMAND_TOLERANCE:
        if achieved_yaw * yaw <= 0.0:
            reasons.append(
                f"yaw sign is wrong: command={yaw:+.4f}, "
                f"achieved={achieved_yaw:+.4f}"
            )
        _minimum_reason(
            reasons,
            "absolute yaw-rate preservation",
            abs(achieved_yaw),
            THRESHOLDS["minimum_yaw_abs_rate_radps"],
            "rad/s",
        )
        _maximum_reason(
            reasons,
            "yaw-rate RMSE",
            yaw_rmse,
            THRESHOLDS["maximum_yaw_rate_rmse_radps"],
            "rad/s",
        )
    else:
        _maximum_reason(
            reasons,
            "uncommanded absolute yaw rate",
            abs(achieved_yaw),
            THRESHOLDS["maximum_uncommanded_abs_yaw_rate_radps"],
            "rad/s",
        )

    _maximum_reason(
        reasons,
        "planar velocity RMSE",
        planar_rmse,
        THRESHOLDS["maximum_planar_velocity_rmse_mps"],
        "m/s",
    )
    _maximum_reason(
        reasons,
        "maximum tilt",
        maximum_tilt,
        THRESHOLDS["maximum_tilt_degrees"],
        "deg",
    )
    for metric_key, reason_name, unit in (
        (
            "max_per_joint_rms_applied_nm",
            "max per-joint RMS applied torque",
            "Nm",
        ),
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
        ("peak_abs_computed_nm", "peak absolute computed torque", "Nm"),
    ):
        _maximum_reason(
            reasons,
            reason_name,
            torque_metrics[metric_key],
            THRESHOLDS[metric_key],
            unit,
        )

    tracking_components = [rmse[0] / THRESHOLDS["maximum_planar_velocity_rmse_mps"]]
    if abs(vy) > COMMAND_TOLERANCE:
        tracking_components.append(
            rmse[1] / THRESHOLDS["maximum_lateral_rmse_mps"]
        )
    if abs(yaw) > COMMAND_TOLERANCE:
        tracking_components.append(
            yaw_rmse / THRESHOLDS["maximum_yaw_rate_rmse_radps"]
        )
    stability_score = (
        stability["base_height_std_m"]
        / THRESHOLDS["preferred_base_height_std_m"]
        + stability["vertical_velocity_rms_mps"]
        / THRESHOLDS["preferred_vertical_velocity_rms_mps"]
        + stability["roll_pitch_angular_velocity_rms_radps"]
        / THRESHOLDS["preferred_roll_pitch_angular_velocity_rms_radps"]
        + stability["tilt_rms_degrees"]
        / THRESHOLDS["preferred_tilt_rms_degrees"]
    ) / 4.0
    normalized_load_score = (
        torque_metrics["max_per_joint_rms_applied_nm"]
        / THRESHOLDS["max_per_joint_rms_applied_nm"]
        + torque_metrics["computed_demand_over_rating_fraction"]
        / THRESHOLDS["computed_demand_over_rating_fraction"]
        + torque_metrics["maximum_computed_over_rating_burst_s"]
        / THRESHOLDS["maximum_computed_over_rating_burst_s"]
        + torque_metrics["peak_abs_computed_nm"]
        / THRESHOLDS["peak_abs_computed_nm"]
    ) / 4.0

    return {
        "label": label,
        "command": list(command),
        "axis_safe": not reasons,
        "axis_rejection_reasons": reasons,
        "metrics": {
            "achieved_command_frame_velocity": [
                achieved_vx,
                achieved_vy,
                achieved_yaw,
            ],
            "rmse_command_error": list(rmse),
            "planar_velocity_rmse_mps": planar_rmse,
            "yaw_rate_rmse_radps": yaw_rmse,
            "falls": falls,
            "timeouts": timeouts,
            "maximum_tilt_degrees": maximum_tilt,
            **stability,
            **torque_metrics,
            "normalized_tracking_score": sum(tracking_components)
            / len(tracking_components),
            "normalized_stability_score": stability_score,
            "normalized_load_score": normalized_load_score,
        },
    }


def _apply_lateral_symmetry_gates(results: list[dict[str, Any]]) -> None:
    by_label = {result["label"]: result for result in results}
    floor = THRESHOLDS["minimum_lateral_pair_symmetry_ratio"]
    for pair_label, positive_label, negative_label in LATERAL_SYMMETRY_PAIRS:
        positive = by_label[positive_label]
        negative = by_label[negative_label]
        positive_abs = abs(
            float(positive["metrics"]["achieved_command_frame_velocity"][1])
        )
        negative_abs = abs(
            float(negative["metrics"]["achieved_command_frame_velocity"][1])
        )
        maximum = max(positive_abs, negative_abs)
        ratio = min(positive_abs, negative_abs) / maximum if maximum > 0.0 else 0.0
        for result in (positive, negative):
            result["metrics"]["lateral_pair_symmetry_ratio"] = ratio
        if ratio < floor - GATE_ABS_TOLERANCE:
            reason = (
                f"{pair_label} lateral symmetry ratio {ratio:.4f} is below "
                f"{floor:.4f} (|positive|={positive_abs:.4f}, "
                f"|negative|={negative_abs:.4f})"
            )
            for result in (positive, negative):
                result["axis_rejection_reasons"].append(reason)
                result["axis_safe"] = False


def _apply_seed_stability_gate(
    candidate: dict[str, Any], seed: dict[str, Any]
) -> None:
    reasons: list[str] = []
    regressions: dict[str, float] = {}
    allowed_factor = 1.0 + THRESHOLDS["maximum_stability_regression_fraction"]
    for key, label, unit in STABILITY_COMPONENTS:
        actual = float(candidate["metrics"][key])
        baseline = float(seed["metrics"][key])
        limit = baseline * allowed_factor
        regression = actual / baseline - 1.0 if baseline > 0.0 else math.inf
        regressions[key] = regression
        if actual > limit + GATE_ABS_TOLERANCE:
            reasons.append(
                f"{label} {actual:.5f} {unit} exceeds seed-relative limit "
                f"{limit:.5f} {unit} (seed={baseline:.5f}, "
                f"regression={100.0 * regression:.1f}%)"
            )

    actual_score = float(candidate["metrics"]["normalized_stability_score"])
    seed_score = float(seed["metrics"]["normalized_stability_score"])
    score_limit = seed_score * allowed_factor
    score_regression = actual_score / seed_score - 1.0 if seed_score > 0.0 else math.inf
    regressions["normalized_stability_score"] = score_regression
    if actual_score > score_limit + GATE_ABS_TOLERANCE:
        reasons.append(
            f"normalized stability score {actual_score:.5f} exceeds seed-relative "
            f"limit {score_limit:.5f} (seed={seed_score:.5f}, "
            f"regression={100.0 * score_regression:.1f}%)"
        )

    candidate["stability_regression_fraction"] = regressions
    candidate["stability_within_seed_limit"] = not reasons
    candidate["stability_rejection_reasons"] = reasons
    candidate["accepted"] = candidate["axis_safe"] and not reasons
    candidate["rejection_reasons"] = [
        *candidate["axis_rejection_reasons"],
        *reasons,
    ]


def _iteration_from_checkpoint(checkpoint: str) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", Path(checkpoint).name)
    if match is None:
        raise ReportError(
            f"candidate checkpoint must end in model_<iteration>.pt: {checkpoint!r}"
        )
    return int(match.group(1))


def _aggregate_evaluation(
    report: dict[str, Any],
    report_index: int,
    seed_results: list[dict[str, Any]],
) -> dict[str, Any]:
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ReportError(
            f"evaluations[{report_index}].checkpoint must be a non-empty string"
        )
    rows = _ordered_contract_rows(report, report_index)
    results = [
        _grade_axis_row(row, label=label, command=command)
        for row, (label, command) in zip(rows, COMMAND_CONTRACT)
    ]
    _apply_lateral_symmetry_gates(results)
    for result, seed_result in zip(results, seed_results):
        _apply_seed_stability_gate(result, seed_result)

    axis_safe = all(result["axis_safe"] for result in results)
    stability_safe = all(result["stability_within_seed_limit"] for result in results)
    mean_stability = sum(
        result["metrics"]["normalized_stability_score"] for result in results
    ) / len(results)
    seed_mean_stability = sum(
        result["metrics"]["normalized_stability_score"]
        for result in seed_results
    ) / len(seed_results)
    return {
        "checkpoint": checkpoint,
        "iteration": _iteration_from_checkpoint(checkpoint),
        "axis_safe": axis_safe,
        "stability_within_seed_limit": stability_safe,
        "accepted": axis_safe and stability_safe,
        "failed_axis_command_count": sum(not result["axis_safe"] for result in results),
        "failed_stability_command_count": sum(
            not result["stability_within_seed_limit"] for result in results
        ),
        "total_falls": sum(result["metrics"]["falls"] for result in results),
        "total_timeouts": sum(result["metrics"]["timeouts"] for result in results),
        "mean_normalized_tracking_score": sum(
            result["metrics"]["normalized_tracking_score"] for result in results
        )
        / len(results),
        "mean_normalized_stability_score": mean_stability,
        "mean_normalized_load_score": sum(
            result["metrics"]["normalized_load_score"] for result in results
        )
        / len(results),
        "mean_stability_regression_fraction": (
            mean_stability / seed_mean_stability - 1.0
            if seed_mean_stability > 0.0
            else math.inf
        ),
        "maximum_stability_regression_fraction": max(
            max(result["stability_regression_fraction"].values())
            for result in results
        ),
        "maximum_max_per_joint_rms_applied_nm": max(
            result["metrics"]["max_per_joint_rms_applied_nm"]
            for result in results
        ),
        "maximum_computed_demand_over_rating_fraction": max(
            result["metrics"]["computed_demand_over_rating_fraction"]
            for result in results
        ),
        "maximum_computed_over_rating_burst_s": max(
            result["metrics"]["maximum_computed_over_rating_burst_s"]
            for result in results
        ),
        "maximum_peak_abs_computed_nm": max(
            result["metrics"]["peak_abs_computed_nm"] for result in results
        ),
        "results": results,
    }


def grade_payload(payload: Any) -> dict[str, Any]:
    """Validate and grade the seed plus the complete Stage-2B batch."""

    reports = _reports(payload)
    seed_report = reports[0]
    _validate_playback_contract(seed_report, 0, None)
    seed_checkpoint = seed_report.get("checkpoint")
    if not isinstance(seed_checkpoint, str) or not seed_checkpoint.endswith(
        EXPECTED_SEED_CHECKPOINT_SUFFIX
    ):
        raise ReportError(
            "evaluations[0] must be the immutable staged Stage-2 model25 seed "
            f"ending in {EXPECTED_SEED_CHECKPOINT_SUFFIX!r}, "
            f"got {seed_checkpoint!r}"
        )
    seed_rows = _ordered_contract_rows(seed_report, 0)
    seed_results = [
        _grade_axis_row(row, label=label, command=command)
        for row, (label, command) in zip(seed_rows, COMMAND_CONTRACT)
    ]
    _apply_lateral_symmetry_gates(seed_results)
    seed_reference = {
        "checkpoint": seed_checkpoint,
        "expected_sha256": EXPECTED_SEED_SHA256,
        "mean_normalized_stability_score": sum(
            result["metrics"]["normalized_stability_score"]
            for result in seed_results
        )
        / len(seed_results),
        "results": seed_results,
    }

    seen_checkpoints = {seed_checkpoint}
    candidates: list[dict[str, Any]] = []
    seen_iterations: set[int] = set()
    for report_index, report in enumerate(reports[1:], start=1):
        _validate_playback_contract(report, report_index, seed_report)
        checkpoint = report.get("checkpoint")
        if not isinstance(checkpoint, str):
            raise ReportError(
                f"evaluations[{report_index}].checkpoint must be a string"
            )
        if checkpoint in seen_checkpoints:
            raise ReportError(f"duplicate checkpoint in report: {checkpoint!r}")
        seen_checkpoints.add(checkpoint)
        iteration = _iteration_from_checkpoint(checkpoint)
        if iteration not in EXPECTED_CANDIDATE_ITERATIONS:
            raise ReportError(
                f"unexpected Stage-2B candidate model_{iteration}.pt; expected "
                f"iterations {EXPECTED_CANDIDATE_ITERATIONS}"
            )
        if iteration in seen_iterations:
            raise ReportError(
                f"duplicate Stage-2B candidate iteration model_{iteration}.pt"
            )
        seen_iterations.add(iteration)
        candidates.append(
            _aggregate_evaluation(report, report_index, seed_results)
        )

    if seen_iterations != set(EXPECTED_CANDIDATE_ITERATIONS):
        missing = sorted(set(EXPECTED_CANDIDATE_ITERATIONS) - seen_iterations)
        raise ReportError(f"Stage-2B candidate batch is incomplete; missing {missing}")

    def rank_key(candidate: dict[str, Any]) -> tuple[float, ...]:
        # Platform steadiness is the primary objective, then tracking, actuator
        # load, and finally the earliest checkpoint as a deterministic tie-break.
        return (
            candidate["mean_normalized_stability_score"],
            candidate["mean_normalized_tracking_score"],
            candidate["mean_normalized_load_score"],
            candidate["maximum_computed_demand_over_rating_fraction"],
            candidate["maximum_max_per_joint_rms_applied_nm"],
            candidate["maximum_peak_abs_computed_nm"],
            candidate["iteration"],
        )

    axis_safe = sorted(
        (candidate for candidate in candidates if candidate["axis_safe"]),
        key=rank_key,
    )
    accepted = [
        candidate for candidate in axis_safe if candidate["stability_within_seed_limit"]
    ]
    return {
        "schema_version": 1,
        "task": TASK_ID,
        "contract": [
            {"label": label, "command": list(command)}
            for label, command in COMMAND_CONTRACT
        ],
        "candidate_iterations": list(EXPECTED_CANDIDATE_ITERATIONS),
        "thresholds": THRESHOLDS,
        "seed_reference": seed_reference,
        "candidate_count": len(candidates),
        "axis_safe_candidate_count": len(axis_safe),
        "accepted_checkpoint_count": len(accepted),
        "best_axis_safe_checkpoint": axis_safe[0]["checkpoint"] if axis_safe else None,
        "best_accepted_checkpoint": accepted[0]["checkpoint"] if accepted else None,
        "axis_safe_checkpoints_ranked": [
            candidate["checkpoint"] for candidate in axis_safe
        ],
        "accepted_checkpoints_ranked": [
            candidate["checkpoint"] for candidate in accepted
        ],
        "evaluations": candidates,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply Stage-2B lateral acquisition, RS05, and command-local "
            "seed-relative stability gates."
        )
    )
    parser.add_argument("input", help="Evaluator JSON produced by evaluate_checkpoint.py")
    parser.add_argument("--json", dest="json_path", help="Write grading JSON here")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    grade = grade_payload(payload)

    seed = grade["seed_reference"]
    print(
        f"SEED checkpoint={seed['checkpoint']} "
        f"stability_score={seed['mean_normalized_stability_score']:.3f}"
    )
    for evaluation in grade["evaluations"]:
        status = "PASS" if evaluation["accepted"] else "FAIL"
        print(
            f"{status} checkpoint={evaluation['checkpoint']} "
            f"axis_safe={evaluation['axis_safe']} "
            f"stability_safe={evaluation['stability_within_seed_limit']} "
            f"axis_failures={evaluation['failed_axis_command_count']} "
            f"stability_failures={evaluation['failed_stability_command_count']} "
            f"tracking={evaluation['mean_normalized_tracking_score']:.3f} "
            f"stability={evaluation['mean_normalized_stability_score']:.3f} "
            f"load={evaluation['mean_normalized_load_score']:.3f} "
            f"falls={evaluation['total_falls']}"
        )
        for result in evaluation["results"]:
            for reason in result["rejection_reasons"]:
                print(f"  {result['label']}: {reason}")
    print(f"best_accepted_checkpoint={grade['best_accepted_checkpoint']}")

    if args.json_path:
        output_path = Path(args.json_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(grade, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"grading_report={output_path}")
    return 0 if grade["accepted_checkpoint_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
