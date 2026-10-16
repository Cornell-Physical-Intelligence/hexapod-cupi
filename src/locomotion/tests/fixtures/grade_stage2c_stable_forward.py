#!/usr/bin/env python3
"""Grade the deterministic Stage-2C lower stable-forward checkpoint batch.

The immutable Stage-2 model-25 checkpoint is replayed through the Stage-2C
task before every candidate.  Stability admission is command-local: each
moving command must improve its matching seed composite by at least 15% while
no component regresses more than 5%; standing permits at most 10% component
regression.  Every candidate must also satisfy command-specific absolute RMS
limits, motion-tail limits, and per-joint RS05 demand limits.  A checkpoint is
accepted only when the operational, absolute, and seed-relative gates all pass.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


TASK_ID = (
    "Isaac-Velocity-Omni-Recovery-Stage2C-Stabilized-Forward-"
    "Hexapod-RobStride-Direct-v0"
)
EXPECTED_SEED = 60
EXPECTED_POLICY_STEP_SECONDS = 0.02
EXPECTED_REQUESTED_STEPS = 500
EXPECTED_WARMUP_STEPS = 25
EXPECTED_SAMPLES = EXPECTED_REQUESTED_STEPS - EXPECTED_WARMUP_STEPS
EXPECTED_MEASURED_SECONDS = EXPECTED_SAMPLES * EXPECTED_POLICY_STEP_SECONDS
EXPECTED_EXPERIMENT = (
    "hexapod_robstride_phase2_recovery_stage2c_stable_forward_direct"
)
EXPECTED_RUN_LABEL = (
    "phase2_recovery_stage2c_lower_stable_forward_from_stage2_model25_seed60"
)
EXPECTED_SEED_CHECKPOINT_SUFFIX = (
    f"/{EXPECTED_EXPERIMENT}/"
    "seed_from_stage2_model25/model_25_stage2_seed.pt"
)
EXPECTED_SEED_SHA256 = (
    "2cb28a0f4f4388e709111a09568c72d0770390d2ba72e08ee17eaaf0f27b6893"
)
EXPECTED_CANDIDATE_ITERATIONS = (*range(0, 120, 10), 119)
EXPECTED_JOINT_COUNT = 18

COMMAND_TOLERANCE = 1.0e-4
GATE_ABS_TOLERANCE = 1.0e-6

COMMAND_CONTRACT: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("stand", (0.00, 0.00, 0.00)),
    ("forward_0p16", (0.16, 0.00, 0.00)),
    ("forward_0p20", (0.20, 0.00, 0.00)),
    ("forward_0p30", (0.30, 0.00, 0.00)),
)

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
    "maximum_per_joint_computed_demand_over_rating_fraction": 0.25,
    "minimum_moving_stability_improvement_fraction": 0.15,
    "maximum_moving_component_regression_fraction": 0.05,
    "maximum_stand_component_regression_fraction": 0.10,
}

# These are both the moving-command composite normalizers and hard absolute
# targets.  The same normalizers remain in use for seed-relative comparisons
# so the ranking contract stays stable while admission becomes fail-closed.
STABILITY_COMPONENTS: tuple[tuple[str, str, str, float], ...] = (
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
ABSOLUTE_STABILITY_TARGETS = {
    key: target for key, _, _, target in STABILITY_COMPONENTS
}

STAND_STABILITY_COMPONENTS: tuple[tuple[str, str, str, float], ...] = (
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
STAND_ABSOLUTE_STABILITY_TARGETS = {
    key: target for key, _, _, target in STAND_STABILITY_COMPONENTS
}

TAIL_STABILITY_COMPONENTS: tuple[tuple[str, str, str, float], ...] = (
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
TAIL_STABILITY_TARGETS = {
    key: target for key, _, _, target in TAIL_STABILITY_COMPONENTS
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
    """Raised when evaluator JSON does not match the Stage-2C contract."""


def _finite_float(value: Any, path: str) -> float:
    if isinstance(value, bool):
        raise ReportError(f"{path} must be numeric, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReportError(f"{path} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise ReportError(f"{path} must be finite, got {result!r}")
    return result


def _nonnegative_float(value: Any, path: str) -> float:
    result = _finite_float(value, path)
    if result < 0.0:
        raise ReportError(f"{path} must be nonnegative, got {result!r}")
    return result


def _finite_vector(value: Any, length: int, path: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ReportError(f"{path} must be a {length}-element list")
    return tuple(
        _finite_float(item, f"{path}[{index}]")
        for index, item in enumerate(value)
    )


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReportError(f"{path} must be a nonnegative integer, got {value!r}")
    return value


def _per_joint_torque_rows(
    torque: dict[str, Any], path: str
) -> list[dict[str, Any]]:
    """Validate and return the complete, uniquely named 18-motor report."""

    rows = torque.get("per_joint")
    if not isinstance(rows, list) or len(rows) != EXPECTED_JOINT_COUNT:
        actual_count = len(rows) if isinstance(rows, list) else None
        raise ReportError(
            f"{path}.per_joint must contain exactly {EXPECTED_JOINT_COUNT} "
            f"rows, got {actual_count!r}"
        )

    seen_names: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        row_path = f"{path}.per_joint[{index}]"
        if not isinstance(row, dict):
            raise ReportError(f"{row_path} must be an object")
        name = row.get("name")
        if not isinstance(name, str) or not name or name != name.strip():
            raise ReportError(
                f"{row_path}.name must be a non-empty, trimmed string, got {name!r}"
            )
        if name in seen_names:
            raise ReportError(f"{path}.per_joint contains duplicate name {name!r}")
        seen_names.add(name)
        metrics = {
            key: _nonnegative_float(row.get(key), f"{row_path}.{key}")
            for key in PER_JOINT_TORQUE_METRICS
        }
        validated.append({"name": name, **metrics})
    return validated


def _reports(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ReportError("top-level JSON value must be an object")
    reports = payload.get("evaluations")
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
    parallel = _nonnegative_int(
        report.get("commands_evaluated_in_parallel"),
        f"{prefix}.commands_evaluated_in_parallel",
    )
    if parallel != len(COMMAND_CONTRACT):
        raise ReportError(
            f"{prefix}.commands_evaluated_in_parallel must be "
            f"{len(COMMAND_CONTRACT)}, got {parallel}"
        )
    policy_step = _finite_float(
        report.get("policy_step_seconds"), f"{prefix}.policy_step_seconds"
    )
    if not math.isclose(
        policy_step, EXPECTED_POLICY_STEP_SECONDS, abs_tol=GATE_ABS_TOLERANCE
    ):
        raise ReportError(
            f"{prefix}.policy_step_seconds must be "
            f"{EXPECTED_POLICY_STEP_SECONDS}, got {policy_step}"
        )
    requested_steps = _nonnegative_int(
        report.get("requested_steps"), f"{prefix}.requested_steps"
    )
    if requested_steps != EXPECTED_REQUESTED_STEPS:
        raise ReportError(
            f"{prefix}.requested_steps must be {EXPECTED_REQUESTED_STEPS}, "
            f"got {requested_steps}"
        )
    warmup_steps = _nonnegative_int(
        report.get("warmup_steps"), f"{prefix}.warmup_steps"
    )
    if warmup_steps != EXPECTED_WARMUP_STEPS:
        raise ReportError(
            f"{prefix}.warmup_steps must be {EXPECTED_WARMUP_STEPS}, "
            f"got {warmup_steps}"
        )
    if "reset_stance_override" not in report:
        raise ReportError(f"{prefix}.reset_stance_override is missing")
    if report["reset_stance_override"] is not None:
        raise ReportError(
            f"{prefix}.reset_stance_override must be null so the Stage-2C "
            f"task stance is used, got {report['reset_stance_override']!r}"
        )

    if reference is None:
        return
    for key in (
        "task",
        "command_frame",
        "seed",
        "deterministic_policy",
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
    if not all(isinstance(row, dict) for row in rows):
        raise ReportError(f"evaluations[{report_index}] result rows must be objects")

    unmatched = list(enumerate(rows))
    ordered: list[dict[str, Any]] = []
    for expected_index, (label, expected) in enumerate(COMMAND_CONTRACT):
        matches = [
            (position, row_index, row)
            for position, (row_index, row) in enumerate(unmatched)
            if _commands_match(
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
        position, row_index, row = matches[0]
        actual_index = _nonnegative_int(
            row.get("index"),
            f"evaluations[{report_index}].results[{row_index}].index",
        )
        if actual_index != expected_index:
            raise ReportError(
                f"evaluations[{report_index}] command {label} must have "
                f"index {expected_index}, got {actual_index}"
            )
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


def _grade_operational_row(
    row: dict[str, Any], *, label: str, command: tuple[float, float, float]
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
    planar_rmse = _nonnegative_float(
        row.get("planar_velocity_rmse_mps"), f"{path}.planar_velocity_rmse_mps"
    )
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
    falls = _nonnegative_int(row.get("falls"), f"{path}.falls")
    timeouts = _nonnegative_int(row.get("timeouts"), f"{path}.timeouts")

    stability = {
        key: _nonnegative_float(row.get(key), f"{path}.{key}")
        for key, _, _, _ in STABILITY_COMPONENTS
    }
    tail_stability = {
        key: _nonnegative_float(row.get(key), f"{path}.{key}")
        for key, _, _, _ in TAIL_STABILITY_COMPONENTS
    }
    normalized_stability = {
        key: stability[key] / target
        for key, _, _, target in STABILITY_COMPONENTS
    }
    stability_composite = sum(normalized_stability.values()) / len(
        normalized_stability
    )
    absolute_components = (
        STAND_STABILITY_COMPONENTS if label == "stand" else STABILITY_COMPONENTS
    )
    absolute_component_targets_met = {
        key: stability[key] <= target + GATE_ABS_TOLERANCE
        for key, _, _, target in absolute_components
    }
    absolute_rms_target_reasons = [
        (
            f"{display_name} {stability[key]:.5f} {unit} exceeds absolute "
            f"target {target:.5f} {unit}"
        )
        for key, display_name, unit, target in absolute_components
        if not absolute_component_targets_met[key]
    ]
    tail_targets_met = {
        key: tail_stability[key] <= target + GATE_ABS_TOLERANCE
        for key, _, _, target in TAIL_STABILITY_COMPONENTS
    }
    tail_target_reasons = [
        (
            f"{display_name} {tail_stability[key]:.5f} {unit} exceeds absolute "
            f"target {target:.5f} {unit}"
        )
        for key, display_name, unit, target in TAIL_STABILITY_COMPONENTS
        if not tail_targets_met[key]
    ]
    absolute_target_reasons = [
        *absolute_rms_target_reasons,
        *tail_target_reasons,
    ]

    torque = row.get("torque")
    if not isinstance(torque, dict):
        raise ReportError(f"{path}.torque must be an object")
    per_joint_torque = _per_joint_torque_rows(torque, f"{path}.torque")
    torque_metrics = {
        "rms_applied_nm": _nonnegative_float(
            torque.get("rms_applied_nm"), f"{path}.torque.rms_applied_nm"
        ),
        "max_per_joint_rms_applied_nm": _nonnegative_float(
            torque.get("max_per_joint_rms_applied_nm"),
            f"{path}.torque.max_per_joint_rms_applied_nm",
        ),
        "computed_demand_over_rating_fraction": _nonnegative_float(
            torque.get("computed_demand_over_rating_fraction"),
            f"{path}.torque.computed_demand_over_rating_fraction",
        ),
        "maximum_computed_over_rating_burst_s": _nonnegative_float(
            torque.get("maximum_computed_over_rating_burst_s"),
            f"{path}.torque.maximum_computed_over_rating_burst_s",
        ),
        "peak_abs_computed_nm": _nonnegative_float(
            torque.get("peak_abs_computed_nm"),
            f"{path}.torque.peak_abs_computed_nm",
        ),
    }
    worst_per_joint = max(
        per_joint_torque,
        key=lambda item: item["computed_demand_over_rating_fraction"],
    )
    worst_per_joint_demand_fraction = float(
        worst_per_joint["computed_demand_over_rating_fraction"]
    )

    achieved_vx, achieved_vy = achieved_linear[:2]
    achieved_yaw = achieved_angular[2]
    achieved_planar_speed = math.hypot(achieved_vx, achieved_vy)
    commanded_vx = command[0]

    if falls:
        reasons.append(f"falls={falls}, expected 0")
    if timeouts:
        reasons.append(f"timeouts={timeouts}, expected 0")

    if label == "stand":
        _maximum_reason(
            reasons,
            "stand planar speed",
            achieved_planar_speed,
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
            commanded_vx * THRESHOLDS["minimum_forward_command_fraction"],
            "m/s",
        )
        _maximum_reason(
            reasons,
            "achieved forward velocity",
            achieved_vx,
            commanded_vx * THRESHOLDS["maximum_forward_command_fraction"],
            "m/s",
        )

    _maximum_reason(
        reasons,
        "planar velocity RMSE",
        planar_rmse,
        THRESHOLDS["maximum_planar_velocity_rmse_mps"],
        "m/s",
    )
    for metric_key, reason_name, unit in (
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
        _maximum_reason(
            reasons,
            reason_name,
            torque_metrics[metric_key],
            THRESHOLDS[metric_key],
            unit,
        )
    _maximum_reason(
        reasons,
        (
            "worst per-joint computed demand over rating fraction "
            f"({worst_per_joint['name']})"
        ),
        worst_per_joint_demand_fraction,
        THRESHOLDS[
            "maximum_per_joint_computed_demand_over_rating_fraction"
        ],
    )

    normalized_load_score = (
        torque_metrics["peak_abs_computed_nm"] / THRESHOLDS["peak_abs_computed_nm"]
        + torque_metrics["computed_demand_over_rating_fraction"]
        / THRESHOLDS["computed_demand_over_rating_fraction"]
        + torque_metrics["maximum_computed_over_rating_burst_s"]
        / THRESHOLDS["maximum_computed_over_rating_burst_s"]
        + torque_metrics["max_per_joint_rms_applied_nm"]
        / THRESHOLDS["max_per_joint_rms_applied_nm"]
    ) / 4.0

    return {
        "label": label,
        "command": list(command),
        "operational_safe": not reasons,
        "operational_rejection_reasons": reasons,
        "absolute_stability_safe": not absolute_target_reasons,
        "absolute_stability_rejection_reasons": absolute_target_reasons,
        "metrics": {
            "achieved_command_frame_velocity": [
                achieved_vx,
                achieved_vy,
                achieved_yaw,
            ],
            "achieved_planar_speed_mps": achieved_planar_speed,
            "planar_velocity_rmse_mps": planar_rmse,
            "normalized_tracking_score": (
                planar_rmse / THRESHOLDS["maximum_planar_velocity_rmse_mps"]
            ),
            "samples": samples,
            "measured_seconds": measured_seconds,
            "falls": falls,
            "timeouts": timeouts,
            **stability,
            **tail_stability,
            "normalized_stability_components": normalized_stability,
            "normalized_stability_composite": stability_composite,
            "absolute_stability_threshold_profile": (
                "stand" if label == "stand" else "moving"
            ),
            "absolute_stability_component_targets_met": (
                absolute_component_targets_met
            ),
            "absolute_stability_rms_targets_met": all(
                absolute_component_targets_met.values()
            ),
            "tail_stability_targets_met_by_component": tail_targets_met,
            "tail_stability_targets_met": all(tail_targets_met.values()),
            "absolute_stability_targets_met": not absolute_target_reasons,
            "moving_normalized_stability_composite_at_or_below_one": (
                stability_composite <= 1.0 + GATE_ABS_TOLERANCE
            ),
            "absolute_stability_target_reasons": absolute_target_reasons,
            **torque_metrics,
            "per_joint_torque_row_count": len(per_joint_torque),
            "worst_per_joint_computed_demand_over_rating_name": (
                worst_per_joint["name"]
            ),
            "worst_per_joint_computed_demand_over_rating_fraction": (
                worst_per_joint_demand_fraction
            ),
            "normalized_load_score": normalized_load_score,
        },
    }


def _fractional_change(actual: float, baseline: float) -> float | None:
    if baseline <= GATE_ABS_TOLERANCE:
        return 0.0 if actual <= GATE_ABS_TOLERANCE else None
    return actual / baseline - 1.0


def _apply_seed_stability_gate(
    candidate: dict[str, Any], seed: dict[str, Any]
) -> None:
    label = candidate["label"]
    moving = label != "stand"
    max_regression = THRESHOLDS[
        (
            "maximum_moving_component_regression_fraction"
            if moving
            else "maximum_stand_component_regression_fraction"
        )
    ]
    reasons: list[str] = []
    component_changes: dict[str, float | None] = {}

    for key, display_name, unit, _ in STABILITY_COMPONENTS:
        actual = float(candidate["metrics"][key])
        baseline = float(seed["metrics"][key])
        limit = baseline * (1.0 + max_regression)
        change = _fractional_change(actual, baseline)
        component_changes[key] = change
        if actual > limit + GATE_ABS_TOLERANCE:
            change_text = (
                "undefined from zero seed"
                if change is None
                else f"{100.0 * change:.1f}%"
            )
            reasons.append(
                f"{display_name} {actual:.5f} {unit} exceeds seed-relative "
                f"limit {limit:.5f} {unit} (seed={baseline:.5f}, "
                f"change={change_text})"
            )

    actual_composite = float(
        candidate["metrics"]["normalized_stability_composite"]
    )
    seed_composite = float(seed["metrics"]["normalized_stability_composite"])
    composite_improvement = (
        1.0 - actual_composite / seed_composite
        if seed_composite > GATE_ABS_TOLERANCE
        else (0.0 if actual_composite <= GATE_ABS_TOLERANCE else None)
    )
    required_improvement: float | None = None
    if moving:
        required_improvement = THRESHOLDS[
            "minimum_moving_stability_improvement_fraction"
        ]
        composite_limit = seed_composite * (1.0 - required_improvement)
        if actual_composite > composite_limit + GATE_ABS_TOLERANCE:
            improvement_text = (
                "undefined from zero seed"
                if composite_improvement is None
                else f"{100.0 * composite_improvement:.1f}%"
            )
            reasons.append(
                f"normalized stability composite {actual_composite:.5f} "
                f"exceeds required-improvement limit {composite_limit:.5f} "
                f"(seed={seed_composite:.5f}, improvement={improvement_text}, "
                f"required={100.0 * required_improvement:.1f}%)"
            )

    candidate["relative_stability"] = {
        "mode": "moving_improvement" if moving else "stand_no_regression",
        "seed_normalized_stability_composite": seed_composite,
        "candidate_normalized_stability_composite": actual_composite,
        "composite_improvement_fraction": composite_improvement,
        "required_composite_improvement_fraction": required_improvement,
        "maximum_component_regression_fraction": max_regression,
        "component_change_fractions": component_changes,
    }
    candidate["relative_stability_safe"] = not reasons
    candidate["stability_rejection_reasons"] = reasons
    candidate["accepted"] = (
        candidate["operational_safe"]
        and candidate["absolute_stability_safe"]
        and not reasons
    )
    candidate["rejection_reasons"] = [
        *candidate["operational_rejection_reasons"],
        *candidate["absolute_stability_rejection_reasons"],
        *reasons,
    ]


def _iteration_from_checkpoint(checkpoint: str) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", Path(checkpoint).name)
    if match is None:
        raise ReportError(
            f"candidate checkpoint must end in model_<iteration>.pt: {checkpoint!r}"
        )
    return int(match.group(1))


def _candidate_run_directory(checkpoint: str) -> str:
    path = Path(checkpoint)
    if path.parent.parent.name != EXPECTED_EXPERIMENT:
        raise ReportError(
            "candidate checkpoint must be inside the Stage-2C experiment "
            f"{EXPECTED_EXPERIMENT!r}: {checkpoint!r}"
        )
    run_name = path.parent.name
    if not (
        run_name == EXPECTED_RUN_LABEL
        or run_name.endswith(f"_{EXPECTED_RUN_LABEL}")
    ):
        raise ReportError(
            "candidate checkpoint run directory must match the Stage-2C run "
            f"label {EXPECTED_RUN_LABEL!r}: {checkpoint!r}"
        )
    return str(path.parent)


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
        _grade_operational_row(row, label=label, command=command)
        for row, (label, command) in zip(rows, COMMAND_CONTRACT)
    ]
    for result, seed_result in zip(results, seed_results):
        _apply_seed_stability_gate(result, seed_result)

    moving_results = [result for result in results if result["label"] != "stand"]
    defined_moving_improvements = [
        result["relative_stability"]["composite_improvement_fraction"]
        for result in moving_results
        if result["relative_stability"]["composite_improvement_fraction"]
        is not None
    ]
    operational_safe = all(result["operational_safe"] for result in results)
    absolute_stability_safe = all(
        result["absolute_stability_safe"] for result in results
    )
    stability_safe = all(result["relative_stability_safe"] for result in results)
    return {
        "checkpoint": checkpoint,
        "iteration": _iteration_from_checkpoint(checkpoint),
        "operational_safe": operational_safe,
        "absolute_stability_safe": absolute_stability_safe,
        "relative_stability_safe": stability_safe,
        "accepted": operational_safe and absolute_stability_safe and stability_safe,
        "failed_operational_command_count": sum(
            not result["operational_safe"] for result in results
        ),
        "failed_absolute_stability_command_count": sum(
            not result["absolute_stability_safe"] for result in results
        ),
        "failed_relative_stability_command_count": sum(
            not result["relative_stability_safe"] for result in results
        ),
        "absolute_stability_target_command_count": sum(
            result["metrics"]["absolute_stability_targets_met"]
            for result in results
        ),
        "absolute_stability_targets_met_all_commands": all(
            result["metrics"]["absolute_stability_targets_met"]
            for result in results
        ),
        "total_falls": sum(result["metrics"]["falls"] for result in results),
        "total_timeouts": sum(
            result["metrics"]["timeouts"] for result in results
        ),
        "mean_normalized_tracking_score": sum(
            result["metrics"]["normalized_tracking_score"] for result in results
        )
        / len(results),
        "mean_moving_normalized_stability_composite": sum(
            result["metrics"]["normalized_stability_composite"]
            for result in moving_results
        )
        / len(moving_results),
        "maximum_moving_normalized_stability_composite": max(
            result["metrics"]["normalized_stability_composite"]
            for result in moving_results
        ),
        "mean_moving_stability_improvement_fraction": (
            sum(defined_moving_improvements) / len(defined_moving_improvements)
            if defined_moving_improvements
            else None
        ),
        "stand_normalized_stability_composite": results[0]["metrics"][
            "normalized_stability_composite"
        ],
        "mean_normalized_load_score": sum(
            result["metrics"]["normalized_load_score"] for result in results
        )
        / len(results),
        "maximum_max_per_joint_rms_applied_nm": max(
            result["metrics"]["max_per_joint_rms_applied_nm"]
            for result in results
        ),
        "maximum_computed_demand_over_rating_fraction": max(
            result["metrics"]["computed_demand_over_rating_fraction"]
            for result in results
        ),
        "maximum_per_joint_computed_demand_over_rating_fraction": max(
            result["metrics"][
                "worst_per_joint_computed_demand_over_rating_fraction"
            ]
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
    """Validate and grade the seed plus the complete Stage-2C batch."""

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
        _grade_operational_row(row, label=label, command=command)
        for row, (label, command) in zip(seed_rows, COMMAND_CONTRACT)
    ]
    seed_moving = [result for result in seed_results if result["label"] != "stand"]
    seed_reference = {
        "checkpoint": seed_checkpoint,
        "expected_sha256": EXPECTED_SEED_SHA256,
        "operational_safe": all(
            result["operational_safe"] for result in seed_results
        ),
        "absolute_stability_safe": all(
            result["absolute_stability_safe"] for result in seed_results
        ),
        "mean_moving_normalized_stability_composite": sum(
            result["metrics"]["normalized_stability_composite"]
            for result in seed_moving
        )
        / len(seed_moving),
        "results": seed_results,
    }

    seen_checkpoints = {seed_checkpoint}
    seen_iterations: set[int] = set()
    candidate_run_directory: str | None = None
    candidates: list[dict[str, Any]] = []
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
                f"unexpected Stage-2C candidate model_{iteration}.pt; expected "
                f"iterations {EXPECTED_CANDIDATE_ITERATIONS}"
            )
        if iteration in seen_iterations:
            raise ReportError(
                f"duplicate Stage-2C candidate iteration model_{iteration}.pt"
            )
        seen_iterations.add(iteration)
        run_directory = _candidate_run_directory(checkpoint)
        if candidate_run_directory is None:
            candidate_run_directory = run_directory
        elif run_directory != candidate_run_directory:
            raise ReportError(
                "all Stage-2C candidates must come from one exact run directory; "
                f"got {run_directory!r} and {candidate_run_directory!r}"
            )
        candidates.append(
            _aggregate_evaluation(report, report_index, seed_results)
        )

    if seen_iterations != set(EXPECTED_CANDIDATE_ITERATIONS):
        missing = sorted(set(EXPECTED_CANDIDATE_ITERATIONS) - seen_iterations)
        raise ReportError(f"Stage-2C candidate batch is incomplete; missing {missing}")

    def rank_key(candidate: dict[str, Any]) -> tuple[float, ...]:
        # The user's primary objective is moving-platform steadiness.  Tracking
        # and then the earliest iteration are deterministic secondary keys.
        return (
            candidate["mean_moving_normalized_stability_composite"],
            candidate["mean_normalized_tracking_score"],
            candidate["iteration"],
        )

    operationally_safe = sorted(
        (candidate for candidate in candidates if candidate["operational_safe"]),
        key=rank_key,
    )
    accepted = [candidate for candidate in operationally_safe if candidate["accepted"]]
    return {
        "schema_version": 1,
        "task": TASK_ID,
        "contract": [
            {"label": label, "command": list(command)}
            for label, command in COMMAND_CONTRACT
        ],
        "playback_contract": {
            "seed": EXPECTED_SEED,
            "policy_step_seconds": EXPECTED_POLICY_STEP_SECONDS,
            "requested_steps": EXPECTED_REQUESTED_STEPS,
            "warmup_steps": EXPECTED_WARMUP_STEPS,
            "deterministic_policy": True,
            "randomization": "disabled by audited launcher",
            "reset_stance_override": None,
        },
        "candidate_experiment": EXPECTED_EXPERIMENT,
        "candidate_run_directory": candidate_run_directory,
        "candidate_iterations": list(EXPECTED_CANDIDATE_ITERATIONS),
        "thresholds": THRESHOLDS,
        "absolute_stability_targets": ABSOLUTE_STABILITY_TARGETS,
        "stand_absolute_stability_targets": STAND_ABSOLUTE_STABILITY_TARGETS,
        "tail_stability_targets": TAIL_STABILITY_TARGETS,
        "absolute_targets_are_admission_gates": True,
        "seed_reference": seed_reference,
        "candidate_count": len(candidates),
        "operationally_safe_candidate_count": len(operationally_safe),
        "absolute_stability_safe_candidate_count": sum(
            candidate["absolute_stability_safe"] for candidate in candidates
        ),
        "accepted_checkpoint_count": len(accepted),
        "best_operationally_safe_checkpoint": (
            operationally_safe[0]["checkpoint"] if operationally_safe else None
        ),
        "best_accepted_checkpoint": (
            accepted[0]["checkpoint"] if accepted else None
        ),
        "operationally_safe_checkpoints_ranked": [
            candidate["checkpoint"] for candidate in operationally_safe
        ],
        "accepted_checkpoints_ranked": [
            candidate["checkpoint"] for candidate in accepted
        ],
        "evaluations": candidates,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply Stage-2C tracking, RS05, and command-local stable-forward "
            "gates to a deterministic checkpoint batch."
        )
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

    seed = grade["seed_reference"]
    print(
        f"SEED checkpoint={seed['checkpoint']} "
        "moving_stability="
        f"{seed['mean_moving_normalized_stability_composite']:.3f}"
    )
    for evaluation in grade["evaluations"]:
        status = "PASS" if evaluation["accepted"] else "FAIL"
        print(
            f"{status} checkpoint={evaluation['checkpoint']} "
            f"operational_safe={evaluation['operational_safe']} "
            f"absolute_stability_safe={evaluation['absolute_stability_safe']} "
            f"relative_stability_safe={evaluation['relative_stability_safe']} "
            "operational_failures="
            f"{evaluation['failed_operational_command_count']} "
            "absolute_stability_failures="
            f"{evaluation['failed_absolute_stability_command_count']} "
            "relative_stability_failures="
            f"{evaluation['failed_relative_stability_command_count']} "
            "moving_stability="
            f"{evaluation['mean_moving_normalized_stability_composite']:.3f} "
            f"tracking={evaluation['mean_normalized_tracking_score']:.3f} "
            "absolute_targets="
            f"{evaluation['absolute_stability_target_command_count']}/4 "
            f"falls={evaluation['total_falls']}"
        )
        for result in evaluation["results"]:
            for reason in result["rejection_reasons"]:
                print(f"  {result['label']}: {reason}")
    print(f"best_accepted_checkpoint={grade['best_accepted_checkpoint']}")

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
    return 0 if grade["accepted_checkpoint_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
