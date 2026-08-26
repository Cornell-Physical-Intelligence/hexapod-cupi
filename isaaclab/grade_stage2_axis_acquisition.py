#!/usr/bin/env python3
"""Grade deterministic Stage-2 signed-axis acquisition checkpoints.

The first evaluation must be the immutable Stage-1 model-25 seed played through
the Stage-2 task.  Every candidate is compared against that seed under each
matching command, so the halfway stance and simulator configuration are
identical for both tracking and stability-regression decisions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any


TASK_ID = "Isaac-Velocity-Omni-Recovery-Stage2-Hexapod-RobStride-Direct-v0"
EXPECTED_SEED = 57
EXPECTED_SEED_CHECKPOINT_SUFFIX = (
    "/seed_from_stage1_model25/model_25_stage1_seed.pt"
)
EXPECTED_SEED_SHA256 = (
    "f9f474ae10c4091b0efd2392bd9f0c31f63d4739ebb7428df3bb0827587d5b27"
)
COMMAND_TOLERANCE = 1.0e-4
GATE_ABS_TOLERANCE = 1.0e-6

COMMAND_CONTRACT: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("forward_0p20", (0.20, 0.00, 0.00)),
    ("forward_0p30", (0.30, 0.00, 0.00)),
    ("forward_lateral_left", (0.25, 0.10, 0.00)),
    ("forward_lateral_right", (0.25, -0.10, 0.00)),
    ("forward_yaw_left", (0.25, 0.00, 0.25)),
    ("forward_yaw_right", (0.25, 0.00, -0.25)),
)

THRESHOLDS: dict[str, float] = {
    "minimum_forward_command_fraction": 0.80,
    "maximum_planar_velocity_rmse_mps": 0.12,
    "minimum_introduced_axis_command_fraction": 0.40,
    "maximum_lateral_rmse_mps": 0.10,
    "maximum_yaw_rate_rmse_radps": 0.18,
    "maximum_uncommanded_abs_yaw_rate_radps": 0.15,
    "maximum_tilt_degrees": 20.0,
    "max_per_joint_rms_applied_nm": 1.60,
    "computed_demand_over_rating_fraction": 0.20,
    "maximum_computed_over_rating_burst_s": 0.20,
    "peak_abs_computed_nm": 5.50,
    # Each matched command and stability component must remain within this
    # multiplicative margin of the seed played through the same Stage-2 task.
    "maximum_stability_regression_fraction": 0.10,
    # Normalizers used only to make the stability ranking dimensionless.
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
    """Raised when evaluator JSON does not match the Stage-2 screen contract."""


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
    if not isinstance(reports, list) or len(reports) < 2:
        raise ReportError("evaluations must contain the seed and at least one candidate")
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


def _grade_axis_row(
    row: dict[str, Any],
    *,
    label: str,
    command: tuple[float, float, float],
) -> dict[str, Any]:
    reasons: list[str] = []
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
    rmse = _finite_vector(
        row.get("rmse_command_error"), 3, f"{label}.rmse_command_error"
    )
    planar_rmse = _finite_float(
        row.get("planar_velocity_rmse_mps"), f"{label}.planar_velocity_rmse_mps"
    )
    yaw_rmse = _finite_float(
        row.get("yaw_rate_rmse_radps"), f"{label}.yaw_rate_rmse_radps"
    )
    falls = int(row.get("falls", -1))
    timeouts = int(row.get("timeouts", -1))
    maximum_tilt = _finite_float(
        row.get("maximum_tilt_degrees"), f"{label}.maximum_tilt_degrees"
    )
    stability = {
        key: _finite_float(row.get(key), f"{label}.{key}")
        for key, _, _ in STABILITY_COMPONENTS
    }
    torque = row.get("torque")
    if not isinstance(torque, dict):
        raise ReportError(f"{label}.torque must be an object")
    torque_metrics = {
        "rms_applied_nm": _finite_float(
            torque.get("rms_applied_nm"), f"{label}.torque.rms_applied_nm"
        ),
        "max_per_joint_rms_applied_nm": _finite_float(
            torque.get("max_per_joint_rms_applied_nm"),
            f"{label}.torque.max_per_joint_rms_applied_nm",
        ),
        "computed_demand_over_rating_fraction": _finite_float(
            torque.get("computed_demand_over_rating_fraction"),
            f"{label}.torque.computed_demand_over_rating_fraction",
        ),
        "maximum_computed_over_rating_burst_s": _finite_float(
            torque.get("maximum_computed_over_rating_burst_s"),
            f"{label}.torque.maximum_computed_over_rating_burst_s",
        ),
        "peak_abs_computed_nm": _finite_float(
            torque.get("peak_abs_computed_nm"),
            f"{label}.torque.peak_abs_computed_nm",
        ),
    }

    vx, vy, yaw = command
    if falls != 0:
        reasons.append(f"falls={falls}, expected 0")
    if timeouts != 0:
        reasons.append(f"timeouts={timeouts}, expected 0")

    minimum_forward = THRESHOLDS["minimum_forward_command_fraction"] * vx
    if achieved_linear[0] < minimum_forward - GATE_ABS_TOLERANCE:
        reasons.append(
            f"forward velocity {achieved_linear[0]:+.4f} m/s is below preservation "
            f"floor {minimum_forward:+.4f} m/s"
        )
    _maximum_reason(
        reasons,
        "planar velocity RMSE",
        planar_rmse,
        THRESHOLDS["maximum_planar_velocity_rmse_mps"],
        "m/s",
    )

    for axis_name, commanded, achieved in (
        ("lateral", vy, achieved_linear[1]),
        ("yaw", yaw, achieved_angular[2]),
    ):
        if abs(commanded) <= COMMAND_TOLERANCE:
            continue
        if achieved * commanded <= 0.0:
            reasons.append(
                f"{axis_name} sign is wrong: command={commanded:+.4f}, "
                f"achieved={achieved:+.4f}"
            )
        minimum_axis = (
            THRESHOLDS["minimum_introduced_axis_command_fraction"]
            * abs(commanded)
        )
        if abs(achieved) < minimum_axis - GATE_ABS_TOLERANCE:
            reasons.append(
                f"{axis_name} magnitude {abs(achieved):.4f} is below acquisition "
                f"floor {minimum_axis:.4f}"
            )

    if abs(vy) > COMMAND_TOLERANCE:
        _maximum_reason(
            reasons,
            "lateral RMSE",
            rmse[1],
            THRESHOLDS["maximum_lateral_rmse_mps"],
            "m/s",
        )
    if abs(yaw) > COMMAND_TOLERANCE:
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
            abs(achieved_angular[2]),
            THRESHOLDS["maximum_uncommanded_abs_yaw_rate_radps"],
            "rad/s",
        )

    _maximum_reason(
        reasons,
        "maximum tilt",
        maximum_tilt,
        THRESHOLDS["maximum_tilt_degrees"],
        "deg",
    )
    for metric_key, reason_name, threshold_key, unit in (
        (
            "max_per_joint_rms_applied_nm",
            "max per-joint RMS applied torque",
            "max_per_joint_rms_applied_nm",
            "Nm",
        ),
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
            "peak_abs_computed_nm",
            "peak absolute computed torque",
            "peak_abs_computed_nm",
            "Nm",
        ),
    ):
        _maximum_reason(
            reasons,
            reason_name,
            torque_metrics[metric_key],
            THRESHOLDS[threshold_key],
            unit,
        )

    tracking_components = [
        rmse[0] / THRESHOLDS["maximum_planar_velocity_rmse_mps"]
    ]
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

    return {
        "label": label,
        "command": list(command),
        "axis_safe": not reasons,
        "axis_rejection_reasons": reasons,
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
            "maximum_tilt_degrees": maximum_tilt,
            **stability,
            **torque_metrics,
            "normalized_tracking_score": sum(tracking_components)
            / len(tracking_components),
            "normalized_stability_score": stability_score,
        },
    }


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
    return int(match.group(1)) if match is not None else 2**31 - 1


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
        "mean_stability_regression_fraction": (
            mean_stability / seed_mean_stability - 1.0
            if seed_mean_stability > 0.0
            else math.inf
        ),
        "maximum_stability_regression_fraction": max(
            max(result["stability_regression_fraction"].values())
            for result in results
        ),
        "maximum_computed_demand_over_rating_fraction": max(
            result["metrics"]["computed_demand_over_rating_fraction"]
            for result in results
        ),
        "maximum_peak_abs_computed_nm": max(
            result["metrics"]["peak_abs_computed_nm"] for result in results
        ),
        "results": results,
    }


def grade_payload(payload: Any) -> dict[str, Any]:
    """Validate and grade seed plus Stage-2 checkpoint evaluator reports."""

    reports = _reports(payload)
    seed_report = reports[0]
    _validate_playback_contract(seed_report, 0, None)
    seed_checkpoint = seed_report.get("checkpoint")
    if not isinstance(seed_checkpoint, str) or not seed_checkpoint.endswith(
        EXPECTED_SEED_CHECKPOINT_SUFFIX
    ):
        raise ReportError(
            "evaluations[0] must be the immutable staged Stage-1 seed ending in "
            f"{EXPECTED_SEED_CHECKPOINT_SUFFIX!r}, got {seed_checkpoint!r}"
        )
    seed_rows = _ordered_contract_rows(seed_report, 0)
    seed_results = [
        _grade_axis_row(row, label=label, command=command)
        for row, (label, command) in zip(seed_rows, COMMAND_CONTRACT)
    ]
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
    for report_index, report in enumerate(reports[1:], start=1):
        _validate_playback_contract(report, report_index, seed_report)
        checkpoint = report.get("checkpoint")
        if checkpoint in seen_checkpoints:
            raise ReportError(f"duplicate checkpoint in report: {checkpoint!r}")
        seen_checkpoints.add(checkpoint)
        candidates.append(
            _aggregate_evaluation(report, report_index, seed_results)
        )

    def rank_key(candidate: dict[str, Any]) -> tuple[float, ...]:
        return (
            candidate["mean_normalized_stability_score"],
            candidate["mean_normalized_tracking_score"],
            candidate["maximum_computed_demand_over_rating_fraction"],
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
        description="Apply Stage-2 axis acquisition and seed-relative stability gates."
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
