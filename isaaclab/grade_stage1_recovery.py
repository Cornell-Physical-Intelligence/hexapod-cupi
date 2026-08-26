#!/usr/bin/env python3
"""Grade deterministic Stage-1 recovery checkpoint evaluations.

This module is deliberately pure Python: it consumes the JSON written by
``evaluate_checkpoint.py`` and can run on the Spark host after the Isaac Lab
container exits.  A checkpoint passes only when every command in the fixed
Stage-1 contract passes every hard tracking, stability, and RS05 load gate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


TASK_ID = "Isaac-Velocity-Omni-Recovery-Stage1-Hexapod-RobStride-Direct-v0"
COMMAND_TOLERANCE = 1.0e-4
GATE_ABS_TOLERANCE = 1.0e-6

COMMAND_CONTRACT: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("forward_0p20", (0.20, 0.00, 0.00)),
    ("forward_0p30", (0.30, 0.00, 0.00)),
    ("forward_yaw_left", (0.25, 0.00, 0.15)),
    ("forward_yaw_right", (0.25, 0.00, -0.15)),
    ("forward_lateral_left", (0.25, 0.06, 0.00)),
    ("forward_lateral_right", (0.25, -0.06, 0.00)),
    ("gentle_three_axis_left", (0.25, 0.05, 0.12)),
    ("gentle_three_axis_right", (0.25, -0.05, -0.12)),
)

THRESHOLDS: dict[str, float] = {
    # Preserve the accepted Phase-1 gait instead of trading all forward motion
    # for a numerically favorable lateral/yaw error.
    "minimum_forward_command_fraction": 0.80,
    "maximum_planar_velocity_rmse_mps": 0.12,
    # A sign check alone lets floating-point noise masquerade as a learned axis.
    "minimum_introduced_axis_command_fraction": 0.40,
    "maximum_lateral_rmse_mps": 0.08,
    "maximum_yaw_rate_rmse_radps": 0.12,
    "maximum_uncommanded_abs_yaw_rate_radps": 0.15,
    # Established deterministic stability and RobStride 05 gates.
    "maximum_tilt_degrees": 20.0,
    "max_per_joint_rms_applied_nm": 1.60,
    "computed_demand_over_rating_fraction": 0.20,
    "maximum_computed_over_rating_burst_s": 0.20,
    "peak_abs_computed_nm": 5.50,
    # Preference thresholds are reported but do not independently reject.
    "preferred_maximum_tilt_degrees": 15.0,
    "preferred_base_height_std_m": 0.012,
    "preferred_vertical_velocity_rms_mps": 0.10,
    "preferred_roll_pitch_angular_velocity_rms_radps": 0.35,
    "preferred_tilt_rms_degrees": 5.0,
    "preferred_rms_applied_nm": 1.10,
    "preferred_computed_demand_over_rating_fraction": 0.15,
}


class ReportError(ValueError):
    """Raised when an evaluator report does not match the screening contract."""


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
    return tuple(_finite_float(item, f"{path}[{index}]") for index, item in enumerate(value))


def _reports(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ReportError("top-level JSON value must be an object")
    reports = payload.get("evaluations", [payload])
    if not isinstance(reports, list) or not reports:
        raise ReportError("evaluations must be a non-empty list")
    if not all(isinstance(report, dict) for report in reports):
        raise ReportError("every evaluation must be an object")
    return reports


def _command_from_row(row: dict[str, Any], path: str) -> tuple[float, float, float]:
    command = row.get("command")
    if not isinstance(command, dict):
        raise ReportError(f"{path}.command must be an object")
    if command.get("frame") != "navigation":
        raise ReportError(
            f"{path}.command.frame must be 'navigation', got {command.get('frame')!r}"
        )
    return (
        _finite_float(command.get("body_vx_mps"), f"{path}.command.body_vx_mps"),
        _finite_float(command.get("body_vy_mps"), f"{path}.command.body_vy_mps"),
        _finite_float(command.get("yaw_rate_radps"), f"{path}.command.yaw_rate_radps"),
    )


def _commands_match(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> bool:
    return all(abs(lhs - rhs) <= COMMAND_TOLERANCE for lhs, rhs in zip(left, right))


def _ordered_contract_rows(report: dict[str, Any], report_index: int) -> list[dict[str, Any]]:
    if report.get("task") != TASK_ID:
        raise ReportError(
            f"evaluations[{report_index}].task must be {TASK_ID!r}, got {report.get('task')!r}"
        )
    if report.get("command_frame") != "navigation":
        raise ReportError(
            f"evaluations[{report_index}].command_frame must be 'navigation', "
            f"got {report.get('command_frame')!r}"
        )
    if report.get("deterministic_policy") is not True:
        raise ReportError(f"evaluations[{report_index}] is not marked deterministic")
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
                f"evaluations[{report_index}] must contain command {label}={expected} exactly once; "
                f"found {len(matches)}"
            )
        position, _, row = matches[0]
        ordered.append(row)
        unmatched.pop(position)
    return ordered


def _grade_row(
    row: dict[str, Any],
    *,
    label: str,
    command: tuple[float, float, float],
) -> dict[str, Any]:
    reasons: list[str] = []
    advisories: list[str] = []

    achieved_linear = _finite_vector(
        row.get("mean_command_frame_linear_velocity_mps"),
        3,
        f"{label}.mean_command_frame_linear_velocity_mps",
    )
    achieved_angular = _finite_vector(
        row.get("mean_command_frame_angular_velocity_radps"),
        3,
        f"{label}.mean_command_frame_angular_velocity_radps",
    )
    rmse = _finite_vector(row.get("rmse_command_error"), 3, f"{label}.rmse_command_error")
    planar_rmse = _finite_float(
        row.get("planar_velocity_rmse_mps"), f"{label}.planar_velocity_rmse_mps"
    )
    yaw_rmse = _finite_float(
        row.get("yaw_rate_rmse_radps"), f"{label}.yaw_rate_rmse_radps"
    )
    tilt = _finite_float(row.get("maximum_tilt_degrees"), f"{label}.maximum_tilt_degrees")
    base_height_std = _finite_float(
        row.get("base_height_std_m"), f"{label}.base_height_std_m"
    )
    vertical_velocity_rms = _finite_float(
        row.get("vertical_velocity_rms_mps"),
        f"{label}.vertical_velocity_rms_mps",
    )
    roll_pitch_rate_rms = _finite_float(
        row.get("roll_pitch_angular_velocity_rms_radps"),
        f"{label}.roll_pitch_angular_velocity_rms_radps",
    )
    tilt_rms = _finite_float(
        row.get("tilt_rms_degrees"), f"{label}.tilt_rms_degrees"
    )
    falls = int(row.get("falls", -1))
    timeouts = int(row.get("timeouts", -1))
    torque = row.get("torque")
    if not isinstance(torque, dict):
        raise ReportError(f"{label}.torque must be an object")

    def maximum(name: str, actual: float, limit: float, unit: str = "") -> None:
        if actual > limit + GATE_ABS_TOLERANCE:
            suffix = f" {unit}" if unit else ""
            reasons.append(f"{name} {actual:.4f}{suffix} exceeds {limit:.4f}{suffix}")

    vx, vy, yaw = command
    if falls != 0:
        reasons.append(f"falls={falls}, expected 0")
    if timeouts != 0:
        reasons.append(f"timeouts={timeouts}, expected 0")

    minimum_forward = THRESHOLDS["minimum_forward_command_fraction"] * vx
    if achieved_linear[0] < minimum_forward - GATE_ABS_TOLERANCE:
        reasons.append(
            f"forward velocity {achieved_linear[0]:+.4f} m/s is below preservation floor "
            f"{minimum_forward:+.4f} m/s"
        )
    maximum(
        "planar velocity RMSE",
        planar_rmse,
        THRESHOLDS["maximum_planar_velocity_rmse_mps"],
        "m/s",
    )

    introduced_axes = (
        ("lateral", vy, achieved_linear[1]),
        ("yaw", yaw, achieved_angular[2]),
    )
    for axis_name, commanded, achieved in introduced_axes:
        if abs(commanded) <= COMMAND_TOLERANCE:
            continue
        if achieved * commanded <= 0.0:
            reasons.append(
                f"{axis_name} sign is wrong: command={commanded:+.4f}, achieved={achieved:+.4f}"
            )
        minimum_magnitude = (
            THRESHOLDS["minimum_introduced_axis_command_fraction"] * abs(commanded)
        )
        if abs(achieved) < minimum_magnitude - GATE_ABS_TOLERANCE:
            reasons.append(
                f"{axis_name} magnitude {abs(achieved):.4f} is below acquisition floor "
                f"{minimum_magnitude:.4f}"
            )

    if abs(vy) > COMMAND_TOLERANCE:
        maximum(
            "lateral RMSE",
            rmse[1],
            THRESHOLDS["maximum_lateral_rmse_mps"],
            "m/s",
        )
    if abs(yaw) > COMMAND_TOLERANCE:
        maximum(
            "yaw-rate RMSE",
            yaw_rmse,
            THRESHOLDS["maximum_yaw_rate_rmse_radps"],
            "rad/s",
        )
    else:
        maximum(
            "uncommanded absolute yaw rate",
            abs(achieved_angular[2]),
            THRESHOLDS["maximum_uncommanded_abs_yaw_rate_radps"],
            "rad/s",
        )

    maximum("maximum tilt", tilt, THRESHOLDS["maximum_tilt_degrees"], "deg")
    max_joint_rms = _finite_float(
        torque.get("max_per_joint_rms_applied_nm"),
        f"{label}.torque.max_per_joint_rms_applied_nm",
    )
    overall_rms = _finite_float(
        torque.get("rms_applied_nm"), f"{label}.torque.rms_applied_nm"
    )
    demand_fraction = _finite_float(
        torque.get("computed_demand_over_rating_fraction"),
        f"{label}.torque.computed_demand_over_rating_fraction",
    )
    over_rating_burst = _finite_float(
        torque.get("maximum_computed_over_rating_burst_s"),
        f"{label}.torque.maximum_computed_over_rating_burst_s",
    )
    raw_peak = _finite_float(
        torque.get("peak_abs_computed_nm"), f"{label}.torque.peak_abs_computed_nm"
    )
    maximum(
        "max per-joint RMS applied torque",
        max_joint_rms,
        THRESHOLDS["max_per_joint_rms_applied_nm"],
        "Nm",
    )
    maximum(
        "computed demand over rating fraction",
        demand_fraction,
        THRESHOLDS["computed_demand_over_rating_fraction"],
    )
    maximum(
        "computed over-rating burst",
        over_rating_burst,
        THRESHOLDS["maximum_computed_over_rating_burst_s"],
        "s",
    )
    maximum(
        "peak absolute computed torque",
        raw_peak,
        THRESHOLDS["peak_abs_computed_nm"],
        "Nm",
    )

    if tilt > THRESHOLDS["preferred_maximum_tilt_degrees"]:
        advisories.append(f"maximum tilt {tilt:.4f} deg exceeds preferred 15 deg")
    if base_height_std > THRESHOLDS["preferred_base_height_std_m"]:
        advisories.append(
            f"base-height std {base_height_std:.4f} m exceeds preferred "
            f"{THRESHOLDS['preferred_base_height_std_m']:.4f} m"
        )
    if vertical_velocity_rms > THRESHOLDS["preferred_vertical_velocity_rms_mps"]:
        advisories.append(
            f"vertical-velocity RMS {vertical_velocity_rms:.4f} m/s exceeds preferred "
            f"{THRESHOLDS['preferred_vertical_velocity_rms_mps']:.4f} m/s"
        )
    if roll_pitch_rate_rms > THRESHOLDS[
        "preferred_roll_pitch_angular_velocity_rms_radps"
    ]:
        advisories.append(
            f"roll/pitch-rate RMS {roll_pitch_rate_rms:.4f} rad/s exceeds preferred "
            f"{THRESHOLDS['preferred_roll_pitch_angular_velocity_rms_radps']:.4f} rad/s"
        )
    if tilt_rms > THRESHOLDS["preferred_tilt_rms_degrees"]:
        advisories.append(
            f"tilt RMS {tilt_rms:.4f} deg exceeds preferred "
            f"{THRESHOLDS['preferred_tilt_rms_degrees']:.4f} deg"
        )
    if overall_rms > THRESHOLDS["preferred_rms_applied_nm"]:
        advisories.append(f"overall applied RMS torque {overall_rms:.4f} Nm exceeds preferred 1.1 Nm")
    if demand_fraction > THRESHOLDS["preferred_computed_demand_over_rating_fraction"]:
        advisories.append(
            f"computed demand over rating fraction {demand_fraction:.4f} exceeds preferred 0.15"
        )

    normalized_tracking_components = [
        rmse[0] / THRESHOLDS["maximum_planar_velocity_rmse_mps"]
    ]
    if abs(vy) > COMMAND_TOLERANCE:
        normalized_tracking_components.append(
            rmse[1] / THRESHOLDS["maximum_lateral_rmse_mps"]
        )
    if abs(yaw) > COMMAND_TOLERANCE:
        normalized_tracking_components.append(
            yaw_rmse / THRESHOLDS["maximum_yaw_rate_rmse_radps"]
        )
    normalized_stability_score = (
        base_height_std / THRESHOLDS["preferred_base_height_std_m"]
        + vertical_velocity_rms
        / THRESHOLDS["preferred_vertical_velocity_rms_mps"]
        + roll_pitch_rate_rms
        / THRESHOLDS["preferred_roll_pitch_angular_velocity_rms_radps"]
        + tilt_rms / THRESHOLDS["preferred_tilt_rms_degrees"]
    ) / 4.0

    return {
        "label": label,
        "command": list(command),
        "accepted": not reasons,
        "rejection_reasons": reasons,
        "advisories": advisories,
        "metrics": {
            "achieved_command_frame_velocity": [
                achieved_linear[0],
                achieved_linear[1],
                achieved_angular[2],
            ],
            "rmse_command_error": list(rmse),
            "planar_velocity_rmse_mps": planar_rmse,
            "yaw_rate_rmse_radps": yaw_rmse,
            "falls": falls,
            "timeouts": timeouts,
            "maximum_tilt_degrees": tilt,
            "base_height_std_m": base_height_std,
            "vertical_velocity_rms_mps": vertical_velocity_rms,
            "roll_pitch_angular_velocity_rms_radps": roll_pitch_rate_rms,
            "tilt_rms_degrees": tilt_rms,
            "rms_applied_nm": overall_rms,
            "max_per_joint_rms_applied_nm": max_joint_rms,
            "computed_demand_over_rating_fraction": demand_fraction,
            "maximum_computed_over_rating_burst_s": over_rating_burst,
            "peak_abs_computed_nm": raw_peak,
            "normalized_tracking_score": sum(normalized_tracking_components)
            / len(normalized_tracking_components),
            "normalized_stability_score": normalized_stability_score,
        },
    }


def grade_payload(payload: Any) -> dict[str, Any]:
    """Validate and grade a single- or multi-checkpoint evaluator payload."""

    evaluations: list[dict[str, Any]] = []
    seen_checkpoints: set[str] = set()
    for report_index, report in enumerate(_reports(payload)):
        checkpoint = report.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ReportError(f"evaluations[{report_index}].checkpoint must be a non-empty string")
        if checkpoint in seen_checkpoints:
            raise ReportError(f"duplicate checkpoint in report: {checkpoint}")
        seen_checkpoints.add(checkpoint)

        ordered_rows = _ordered_contract_rows(report, report_index)
        command_results = [
            _grade_row(row, label=label, command=command)
            for row, (label, command) in zip(ordered_rows, COMMAND_CONTRACT)
        ]
        failed_commands = sum(not result["accepted"] for result in command_results)
        total_falls = sum(result["metrics"]["falls"] for result in command_results)
        total_timeouts = sum(result["metrics"]["timeouts"] for result in command_results)
        evaluations.append(
            {
                "checkpoint": checkpoint,
                "accepted": failed_commands == 0,
                "failed_command_count": failed_commands,
                "total_falls": total_falls,
                "total_timeouts": total_timeouts,
                "mean_normalized_tracking_score": sum(
                    result["metrics"]["normalized_tracking_score"]
                    for result in command_results
                )
                / len(command_results),
                "mean_normalized_stability_score": sum(
                    result["metrics"]["normalized_stability_score"]
                    for result in command_results
                )
                / len(command_results),
                "maximum_tilt_degrees": max(
                    result["metrics"]["maximum_tilt_degrees"]
                    for result in command_results
                ),
                "maximum_computed_demand_over_rating_fraction": max(
                    result["metrics"]["computed_demand_over_rating_fraction"]
                    for result in command_results
                ),
                "maximum_peak_abs_computed_nm": max(
                    result["metrics"]["peak_abs_computed_nm"]
                    for result in command_results
                ),
                "results": command_results,
            }
        )

    accepted = [evaluation for evaluation in evaluations if evaluation["accepted"]]
    accepted.sort(
        key=lambda evaluation: (
            evaluation["mean_normalized_stability_score"],
            evaluation["mean_normalized_tracking_score"],
            evaluation["maximum_computed_demand_over_rating_fraction"],
            evaluation["maximum_peak_abs_computed_nm"],
            evaluation["checkpoint"],
        )
    )
    return {
        "schema_version": 1,
        "task": TASK_ID,
        "contract": [
            {"label": label, "command": list(command)}
            for label, command in COMMAND_CONTRACT
        ],
        "thresholds": THRESHOLDS,
        "all_checkpoints_accepted": len(accepted) == len(evaluations),
        "accepted_checkpoint_count": len(accepted),
        "best_accepted_checkpoint": accepted[0]["checkpoint"] if accepted else None,
        "accepted_checkpoints_ranked": [
            evaluation["checkpoint"] for evaluation in accepted
        ],
        "evaluations": evaluations,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the deterministic Stage-1 recovery acceptance gates."
    )
    parser.add_argument("input", help="Evaluator JSON produced by evaluate_checkpoint.py")
    parser.add_argument("--json", dest="json_path", help="Write the grading report to this path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    grade = grade_payload(payload)

    for evaluation in grade["evaluations"]:
        status = "PASS" if evaluation["accepted"] else "FAIL"
        print(
            f"{status} checkpoint={evaluation['checkpoint']} "
            f"failed_commands={evaluation['failed_command_count']} "
            f"falls={evaluation['total_falls']} timeouts={evaluation['total_timeouts']} "
            f"tracking_score={evaluation['mean_normalized_tracking_score']:.3f} "
            f"stability_score={evaluation['mean_normalized_stability_score']:.3f} "
            f"max_demand_over={100.0 * evaluation['maximum_computed_demand_over_rating_fraction']:.2f}% "
            f"max_raw_peak={evaluation['maximum_peak_abs_computed_nm']:.3f}Nm"
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
