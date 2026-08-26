#!/usr/bin/env python3
"""Compare Stage-2C probe checkpoints evaluated with extra speed commands.

The canonical Stage-2C grader intentionally accepts only its four-command,
full-checkpoint-batch admission artifact.  Probe batches often add commands at
0.23, 0.24, and 0.25 m/s to expose behavior around a reward threshold.  This
utility leaves the canonical grader fail-closed, selects its four command rows
from a larger evaluator report, and reuses the canonical per-row grading
logic.  Extra rows are summarized as diagnostics; they never silently replace
an admission row.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import grade_stage2c_stable_forward as admission

DIAGNOSTIC_COMMANDS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("forward_0p23_diagnostic", (0.23, 0.0, 0.0)),
    ("forward_0p24_diagnostic", (0.24, 0.0, 0.0)),
    ("forward_0p25_diagnostic", (0.25, 0.0, 0.0)),
)

SWEEP_SPEEDS = (0.20, 0.23, 0.24, 0.25, 0.30)


def _sample_contract(
    diagnostic_expected_samples: int | None,
) -> dict[str, Any]:
    """Build an explicit formal or short-duration evaluator contract."""

    formal_samples = admission.EXPECTED_SAMPLES
    formal_seconds = admission.EXPECTED_MEASURED_SECONDS
    if diagnostic_expected_samples is None:
        return {
            "mode": "formal_admission",
            "expected_samples": formal_samples,
            "expected_measured_seconds": formal_seconds,
            "warmup_steps": admission.EXPECTED_WARMUP_STEPS,
            "requested_steps": admission.EXPECTED_REQUESTED_STEPS,
            "policy_step_seconds": admission.EXPECTED_POLICY_STEP_SECONDS,
            "formal_expected_samples": formal_samples,
            "formal_expected_measured_seconds": formal_seconds,
            "formal_admission_eligible": True,
        }
    if (
        isinstance(diagnostic_expected_samples, bool)
        or not isinstance(diagnostic_expected_samples, int)
        or diagnostic_expected_samples <= 0
        or diagnostic_expected_samples >= formal_samples
    ):
        raise admission.ReportError(
            "diagnostic_expected_samples must be a positive integer below the "
            f"formal {formal_samples}-sample contract, got "
            f"{diagnostic_expected_samples!r}"
        )
    measured_seconds = (
        diagnostic_expected_samples * admission.EXPECTED_POLICY_STEP_SECONDS
    )
    return {
        "mode": "diagnostic_short_duration",
        "expected_samples": diagnostic_expected_samples,
        "expected_measured_seconds": measured_seconds,
        "warmup_steps": admission.EXPECTED_WARMUP_STEPS,
        "requested_steps": (
            diagnostic_expected_samples + admission.EXPECTED_WARMUP_STEPS
        ),
        "policy_step_seconds": admission.EXPECTED_POLICY_STEP_SECONDS,
        "formal_expected_samples": formal_samples,
        "formal_expected_measured_seconds": formal_seconds,
        "formal_admission_eligible": False,
    }


def _grade_row_for_sample_contract(
    row: dict[str, Any],
    *,
    label: str,
    command: tuple[float, float, float],
    sample_contract: dict[str, Any],
) -> dict[str, Any]:
    """Grade one row after fail-closed validation of its sample contract.

    The canonical grader remains untouched and fail-closed at 475 samples.
    Diagnostic rows are copied and adapted only for reuse of its metric/gate
    calculations, then their true duration is restored in the returned data.
    """

    samples = admission._nonnegative_int(row.get("samples"), f"{label}.samples")
    expected_samples = int(sample_contract["expected_samples"])
    if samples != expected_samples:
        raise admission.ReportError(
            f"{label}.samples must be {expected_samples} for "
            f"{sample_contract['mode']}, got {samples}"
        )
    measured_seconds = admission._nonnegative_float(
        row.get("measured_seconds"), f"{label}.measured_seconds"
    )
    expected_seconds = float(sample_contract["expected_measured_seconds"])
    if not math.isclose(
        measured_seconds,
        expected_seconds,
        abs_tol=admission.COMMAND_TOLERANCE,
    ):
        raise admission.ReportError(
            f"{label}.measured_seconds must be {expected_seconds} for "
            f"{sample_contract['mode']}, got {measured_seconds}"
        )
    if sample_contract["formal_admission_eligible"]:
        return admission._grade_operational_row(
            row, label=label, command=command
        )

    canonical_row = dict(row)
    canonical_row["samples"] = admission.EXPECTED_SAMPLES
    canonical_row["measured_seconds"] = admission.EXPECTED_MEASURED_SECONDS
    result = admission._grade_operational_row(
        canonical_row, label=label, command=command
    )
    result["metrics"]["samples"] = samples
    result["metrics"]["measured_seconds"] = measured_seconds
    return result


def _reports(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise admission.ReportError("top-level JSON value must be an object")
    reports = payload.get("evaluations")
    if not isinstance(reports, list) or not reports:
        raise admission.ReportError(
            "evaluations must be a non-empty list of evaluator reports"
        )
    if not all(isinstance(report, dict) for report in reports):
        raise admission.ReportError("every evaluation must be an object")
    return reports


def _result_rows(report: dict[str, Any], report_index: int) -> list[dict[str, Any]]:
    rows = report.get("results")
    if not isinstance(rows, list):
        raise admission.ReportError(
            f"evaluations[{report_index}].results must be a list"
        )
    if not all(isinstance(row, dict) for row in rows):
        raise admission.ReportError(
            f"evaluations[{report_index}] result rows must be objects"
        )
    return rows


def _match_command_row(
    rows: list[dict[str, Any]],
    *,
    report_index: int,
    label: str,
    command: tuple[float, float, float],
    required: bool,
) -> tuple[int, dict[str, Any]] | None:
    matches: list[tuple[int, dict[str, Any]]] = []
    for row_index, row in enumerate(rows):
        actual = admission._command_from_row(
            row, f"evaluations[{report_index}].results[{row_index}]"
        )
        if admission._commands_match(actual, command):
            matches.append((row_index, row))
    if len(matches) > 1:
        raise admission.ReportError(
            f"evaluations[{report_index}] contains command {label}={command} "
            f"more than once ({len(matches)} matches)"
        )
    if not matches:
        if required:
            raise admission.ReportError(
                f"evaluations[{report_index}] is missing command {label}={command}"
            )
        return None
    return matches[0]


def _select_commands(
    rows: list[dict[str, Any]],
    *,
    report_index: int,
    contract: Iterable[tuple[str, tuple[float, float, float]]],
    required: bool,
) -> list[tuple[str, tuple[float, float, float], int, dict[str, Any]]]:
    selected = []
    for label, command in contract:
        match = _match_command_row(
            rows,
            report_index=report_index,
            label=label,
            command=command,
            required=required,
        )
        if match is not None:
            row_index, row = match
            selected.append((label, command, row_index, row))
    return selected


def _checkpoint_label(checkpoint: str) -> str:
    path = Path(checkpoint)
    run_name = path.parent.name or "unknown_run"
    run_name = re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_", "", run_name)
    return f"{run_name}/{path.name}"


def _continuous_gate_ratios(result: dict[str, Any]) -> list[tuple[str, float]]:
    """Return finite-gate utilization ratios; values above one are failures."""

    label = str(result["label"])
    metrics = result["metrics"]
    ratios: list[tuple[str, float]] = [
        (
            "planar_velocity_rmse",
            metrics["planar_velocity_rmse_mps"]
            / admission.THRESHOLDS["maximum_planar_velocity_rmse_mps"],
        ),
        (
            "torque_peak",
            metrics["peak_abs_computed_nm"]
            / admission.THRESHOLDS["peak_abs_computed_nm"],
        ),
        (
            "torque_duty",
            metrics["computed_demand_over_rating_fraction"]
            / admission.THRESHOLDS["computed_demand_over_rating_fraction"],
        ),
        (
            "torque_burst",
            metrics["maximum_computed_over_rating_burst_s"]
            / admission.THRESHOLDS["maximum_computed_over_rating_burst_s"],
        ),
        (
            "max_per_joint_rms_applied",
            metrics["max_per_joint_rms_applied_nm"]
            / admission.THRESHOLDS["max_per_joint_rms_applied_nm"],
        ),
        (
            "worst_per_joint_torque_duty",
            metrics["worst_per_joint_computed_demand_over_rating_fraction"]
            / admission.THRESHOLDS[
                "maximum_per_joint_computed_demand_over_rating_fraction"
            ],
        ),
    ]

    rms_components = (
        admission.STAND_STABILITY_COMPONENTS
        if label == "stand"
        else admission.STABILITY_COMPONENTS
    )
    ratios.extend(
        (f"stability_{key}", metrics[key] / target)
        for key, _, _, target in rms_components
    )
    ratios.extend(
        (f"tail_{key}", metrics[key] / target)
        for key, _, _, target in admission.TAIL_STABILITY_COMPONENTS
    )

    achieved_vx, achieved_vy, achieved_yaw = metrics["achieved_command_frame_velocity"]
    if label == "stand":
        ratios.extend(
            (
                (
                    "stand_planar_speed",
                    math.hypot(achieved_vx, achieved_vy)
                    / admission.THRESHOLDS["maximum_stand_planar_speed_mps"],
                ),
                (
                    "stand_abs_yaw_rate",
                    abs(achieved_yaw)
                    / admission.THRESHOLDS["maximum_stand_abs_yaw_rate_radps"],
                ),
            )
        )
    else:
        commanded_vx = float(result["command"][0])
        minimum = (
            commanded_vx * admission.THRESHOLDS["minimum_forward_command_fraction"]
        )
        maximum = (
            commanded_vx * admission.THRESHOLDS["maximum_forward_command_fraction"]
        )
        # Keep exported JSON finite even for a policy that stalls or walks
        # backward.  Positive velocities retain the intuitive limit/actual
        # ratio; non-positive velocities use one plus normalized shortfall.
        lower_ratio = (
            minimum / achieved_vx
            if achieved_vx > 0.0
            else 1.0 + (minimum - achieved_vx) / minimum
        )
        ratios.extend(
            (
                ("minimum_forward_velocity", lower_ratio),
                ("maximum_forward_velocity", achieved_vx / maximum),
            )
        )
    return ratios


def _worst_gate(results: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        (ratio, result["label"], metric)
        for result in results
        for metric, ratio in _continuous_gate_ratios(result)
    ]
    ratio, label, metric = max(candidates, key=lambda item: item[0])
    return {
        "command": label,
        "metric": metric,
        "ratio_to_limit": ratio,
        "normalized_violation": max(0.0, ratio - 1.0),
    }


def _sweep_point(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    return {
        "command_vx_mps": float(result["command"][0]),
        "achieved_vx_mps": metrics["achieved_command_frame_velocity"][0],
        "planar_velocity_rmse_mps": metrics["planar_velocity_rmse_mps"],
        "normalized_stability_composite": metrics["normalized_stability_composite"],
        "falls": metrics["falls"],
        "timeouts": metrics["timeouts"],
        "torque_duty_fraction": metrics["computed_demand_over_rating_fraction"],
        "torque_burst_s": metrics["maximum_computed_over_rating_burst_s"],
        "torque_peak_nm": metrics["peak_abs_computed_nm"],
        "worst_per_joint_torque_duty_fraction": metrics[
            "worst_per_joint_computed_demand_over_rating_fraction"
        ],
        "worst_per_joint_name": metrics[
            "worst_per_joint_computed_demand_over_rating_name"
        ],
    }


_JUMP_NORMALIZERS: dict[str, float] = {
    "achieved_vx_mps": 0.30,
    "planar_velocity_rmse_mps": admission.THRESHOLDS[
        "maximum_planar_velocity_rmse_mps"
    ],
    "normalized_stability_composite": 1.0,
    "torque_duty_fraction": admission.THRESHOLDS[
        "computed_demand_over_rating_fraction"
    ],
    "torque_burst_s": admission.THRESHOLDS["maximum_computed_over_rating_burst_s"],
    "torque_peak_nm": admission.THRESHOLDS["peak_abs_computed_nm"],
    "worst_per_joint_torque_duty_fraction": admission.THRESHOLDS[
        "maximum_per_joint_computed_demand_over_rating_fraction"
    ],
}


def _adjacent_jump(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    deltas = {key: float(right[key]) - float(left[key]) for key in _JUMP_NORMALIZERS}
    normalized_deltas = {
        key: delta / _JUMP_NORMALIZERS[key] for key, delta in deltas.items()
    }
    jump_metric = max(normalized_deltas, key=lambda key: abs(normalized_deltas[key]))
    command_delta = right["command_vx_mps"] - left["command_vx_mps"]
    return {
        "from_command_vx_mps": left["command_vx_mps"],
        "to_command_vx_mps": right["command_vx_mps"],
        "command_delta_mps": command_delta,
        "achieved_vx_gain_per_command_delta": (
            deltas["achieved_vx_mps"] / command_delta
            if abs(command_delta) > admission.COMMAND_TOLERANCE
            else None
        ),
        "metric_deltas": deltas,
        "normalized_metric_deltas": normalized_deltas,
        "largest_normalized_jump_metric": jump_metric,
        "largest_normalized_jump": abs(normalized_deltas[jump_metric]),
        "falls_delta": int(right["falls"]) - int(left["falls"]),
        "timeouts_delta": int(right["timeouts"]) - int(left["timeouts"]),
    }


def _threshold_diagnostics(
    admission_results: list[dict[str, Any]],
    diagnostic_results: list[dict[str, Any]],
) -> dict[str, Any]:
    points_by_speed = {
        round(float(result["command"][0]), 4): _sweep_point(result)
        for result in (*admission_results, *diagnostic_results)
        if float(result["command"][0]) > 0.0
    }
    points = [
        points_by_speed[round(speed, 4)]
        for speed in SWEEP_SPEEDS
        if round(speed, 4) in points_by_speed
    ]
    jumps = [
        _adjacent_jump(points[index], points[index + 1])
        for index in range(len(points) - 1)
    ]
    boundary_jump = next(
        (
            jump
            for jump in jumps
            if math.isclose(
                jump["from_command_vx_mps"],
                0.23,
                abs_tol=admission.COMMAND_TOLERANCE,
            )
            and math.isclose(
                jump["to_command_vx_mps"],
                0.24,
                abs_tol=admission.COMMAND_TOLERANCE,
            )
        ),
        None,
    )
    largest_jump = max(
        jumps, key=lambda jump: jump["largest_normalized_jump"], default=None
    )
    expected_speeds = {round(speed, 4) for speed in SWEEP_SPEEDS}
    return {
        "complete": expected_speeds.issubset(points_by_speed),
        "missing_command_vx_mps": sorted(expected_speeds - points_by_speed.keys()),
        "points": points,
        "adjacent_jumps": jumps,
        "threshold_boundary_0p23_to_0p24": boundary_jump,
        "largest_adjacent_normalized_jump": largest_jump,
    }


def _analyze_report(
    report: dict[str, Any],
    report_index: int,
    *,
    sample_contract: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise admission.ReportError(
            f"evaluations[{report_index}].checkpoint must be a non-empty string"
        )
    rows = _result_rows(report, report_index)
    admission_rows = _select_commands(
        rows,
        report_index=report_index,
        contract=admission.COMMAND_CONTRACT,
        required=True,
    )
    diagnostics = _select_commands(
        rows,
        report_index=report_index,
        contract=DIAGNOSTIC_COMMANDS,
        required=False,
    )
    admission_results = [
        _grade_row_for_sample_contract(
            row,
            label=label,
            command=command,
            sample_contract=sample_contract,
        )
        for label, command, _, row in admission_rows
    ]
    diagnostic_results = [
        _grade_row_for_sample_contract(
            row,
            label=label,
            command=command,
            sample_contract=sample_contract,
        )
        for label, command, _, row in diagnostics
    ]
    moving_results = [
        result for result in admission_results if result["label"] != "stand"
    ]
    operational_safe = all(result["operational_safe"] for result in admission_results)
    absolute_stability_safe = all(
        result["absolute_stability_safe"] for result in admission_results
    )
    worst_gate = _worst_gate(admission_results)
    return {
        "checkpoint": checkpoint,
        "label": _checkpoint_label(checkpoint),
        "input_report_index": report_index,
        "input_result_row_count": len(rows),
        "formal_admission_eligible": sample_contract[
            "formal_admission_eligible"
        ],
        "sample_contract_mode": sample_contract["mode"],
        "selected_admission_row_indices": [item[2] for item in admission_rows],
        "selected_diagnostic_row_indices": [item[2] for item in diagnostics],
        "four_command_contract_safe": operational_safe and absolute_stability_safe,
        "operational_safe": operational_safe,
        "absolute_stability_safe": absolute_stability_safe,
        "failed_operational_command_count": sum(
            not result["operational_safe"] for result in admission_results
        ),
        "failed_absolute_stability_command_count": sum(
            not result["absolute_stability_safe"] for result in admission_results
        ),
        "total_falls": sum(result["metrics"]["falls"] for result in admission_results),
        "total_timeouts": sum(
            result["metrics"]["timeouts"] for result in admission_results
        ),
        "mean_moving_planar_velocity_rmse_mps": sum(
            result["metrics"]["planar_velocity_rmse_mps"] for result in moving_results
        )
        / len(moving_results),
        "mean_moving_normalized_tracking_score": sum(
            result["metrics"]["normalized_tracking_score"] for result in moving_results
        )
        / len(moving_results),
        "mean_moving_normalized_stability_composite": sum(
            result["metrics"]["normalized_stability_composite"]
            for result in moving_results
        )
        / len(moving_results),
        "maximum_moving_normalized_stability_composite": max(
            result["metrics"]["normalized_stability_composite"]
            for result in moving_results
        ),
        "stand_normalized_stability_composite": admission_results[0]["metrics"][
            "normalized_stability_composite"
        ],
        "maximum_torque_duty_fraction": max(
            result["metrics"]["computed_demand_over_rating_fraction"]
            for result in admission_results
        ),
        "maximum_torque_burst_s": max(
            result["metrics"]["maximum_computed_over_rating_burst_s"]
            for result in admission_results
        ),
        "maximum_torque_peak_nm": max(
            result["metrics"]["peak_abs_computed_nm"] for result in admission_results
        ),
        "maximum_worst_per_joint_torque_duty_fraction": max(
            result["metrics"]["worst_per_joint_computed_demand_over_rating_fraction"]
            for result in admission_results
        ),
        "worst_normalized_gate": worst_gate,
        "admission_results": admission_results,
        "diagnostic_results": diagnostic_results,
        "threshold_diagnostics": _threshold_diagnostics(
            admission_results, diagnostic_results
        ),
    }


def _baseline_index(evaluations: list[dict[str, Any]], selector: str | None) -> int:
    if selector is None:
        return 0
    exact = [
        index
        for index, evaluation in enumerate(evaluations)
        if evaluation["checkpoint"] == selector or evaluation["label"] == selector
    ]
    if len(exact) == 1:
        return exact[0]
    partial = [
        index
        for index, evaluation in enumerate(evaluations)
        if selector in evaluation["checkpoint"] or selector in evaluation["label"]
    ]
    if len(partial) != 1:
        raise admission.ReportError(
            f"baseline selector {selector!r} matched {len(partial)} evaluations; "
            "provide a unique checkpoint or label substring"
        )
    return partial[0]


def analyze_payload(
    payload: Any,
    *,
    baseline: str | None = None,
    diagnostic_expected_samples: int | None = None,
) -> dict[str, Any]:
    """Analyze arbitrary Stage-2C checkpoints with admission plus probe rows."""

    sample_contract = _sample_contract(diagnostic_expected_samples)
    evaluations = [
        _analyze_report(
            report,
            report_index,
            sample_contract=sample_contract,
        )
        for report_index, report in enumerate(_reports(payload))
    ]
    baseline_index = _baseline_index(evaluations, baseline)
    baseline_evaluation = evaluations[baseline_index]
    baseline_stability = baseline_evaluation[
        "mean_moving_normalized_stability_composite"
    ]
    baseline_tracking = baseline_evaluation["mean_moving_planar_velocity_rmse_mps"]
    for evaluation in evaluations:
        stability = evaluation["mean_moving_normalized_stability_composite"]
        tracking = evaluation["mean_moving_planar_velocity_rmse_mps"]
        evaluation["relative_to_baseline"] = {
            "moving_stability_improvement_fraction": (
                1.0 - stability / baseline_stability
                if baseline_stability > admission.GATE_ABS_TOLERANCE
                else (0.0 if stability <= admission.GATE_ABS_TOLERANCE else None)
            ),
            "moving_tracking_rmse_improvement_fraction": (
                1.0 - tracking / baseline_tracking
                if baseline_tracking > admission.GATE_ABS_TOLERANCE
                else (0.0 if tracking <= admission.GATE_ABS_TOLERANCE else None)
            ),
        }

    def rank_key(evaluation: dict[str, Any]) -> tuple[float, ...]:
        return (
            evaluation["total_falls"],
            evaluation["total_timeouts"],
            not evaluation["four_command_contract_safe"],
            evaluation["failed_operational_command_count"]
            + evaluation["failed_absolute_stability_command_count"],
            evaluation["worst_normalized_gate"]["ratio_to_limit"],
            evaluation["mean_moving_normalized_stability_composite"],
            evaluation["mean_moving_planar_velocity_rmse_mps"],
            evaluation["input_report_index"],
        )

    ranked = sorted(evaluations, key=rank_key)
    formal_admission_eligible = bool(
        sample_contract["formal_admission_eligible"]
    )
    return {
        "schema_version": 1,
        "task": admission.TASK_ID,
        "analysis_kind": (
            "stage2c_admission_plus_threshold_probe"
            if formal_admission_eligible
            else "stage2c_diagnostic_short_duration_probe_screen"
        ),
        "formal_admission_eligible": formal_admission_eligible,
        "sample_contract": sample_contract,
        "admission_contract": [
            {"label": label, "command": list(command)}
            for label, command in admission.COMMAND_CONTRACT
        ],
        "diagnostic_contract": [
            {"label": label, "command": list(command)}
            for label, command in DIAGNOSTIC_COMMANDS
        ],
        "baseline_checkpoint": baseline_evaluation["checkpoint"],
        "candidate_count": len(evaluations),
        "four_command_contract_safe_count": sum(
            evaluation["four_command_contract_safe"] for evaluation in evaluations
        ),
        "ranked_checkpoints": [evaluation["checkpoint"] for evaluation in ranked],
        "evaluations": evaluations,
    }


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if not math.isfinite(value):
        return "inf"
    return f"{value:.{digits}f}"


def _print_summary(analysis: dict[str, Any]) -> None:
    by_checkpoint = {
        evaluation["checkpoint"]: evaluation for evaluation in analysis["evaluations"]
    }
    contract = analysis["sample_contract"]
    print(
        f"mode={contract['mode']} samples={contract['expected_samples']} "
        f"measured_seconds={contract['expected_measured_seconds']:.3f} "
        "formal_admission_eligible="
        f"{str(analysis['formal_admission_eligible']).lower()}"
    )
    print(f"baseline={analysis['baseline_checkpoint']}")
    print(
        "rank status falls track_rmse deck_mean deck_worst duty burst peak "
        "joint_duty worst_gate candidate"
    )
    for rank, checkpoint in enumerate(analysis["ranked_checkpoints"], start=1):
        evaluation = by_checkpoint[checkpoint]
        worst = evaluation["worst_normalized_gate"]
        if analysis["formal_admission_eligible"]:
            status = "PASS" if evaluation["four_command_contract_safe"] else "FAIL"
        else:
            status = (
                "DIAG-PASS"
                if evaluation["four_command_contract_safe"]
                else "DIAG-FAIL"
            )
        print(
            f"{rank:>4} {status:>9} {evaluation['total_falls']:>5} "
            f"{evaluation['mean_moving_planar_velocity_rmse_mps']:.4f} "
            f"{evaluation['mean_moving_normalized_stability_composite']:.3f} "
            f"{evaluation['maximum_moving_normalized_stability_composite']:.3f} "
            f"{evaluation['maximum_torque_duty_fraction']:.3f} "
            f"{evaluation['maximum_torque_burst_s']:.3f} "
            f"{evaluation['maximum_torque_peak_nm']:.3f} "
            f"{evaluation['maximum_worst_per_joint_torque_duty_fraction']:.3f} "
            f"{worst['ratio_to_limit']:.2f}x:{worst['command']}/"
            f"{worst['metric']} {evaluation['label']}"
        )
        threshold = evaluation["threshold_diagnostics"][
            "threshold_boundary_0p23_to_0p24"
        ]
        if threshold is None:
            print("     threshold .23->.24: unavailable")
        else:
            deltas = threshold["metric_deltas"]
            print(
                "     threshold .23->.24: "
                f"dvx={deltas['achieved_vx_mps']:+.4f} "
                f"dtrack={deltas['planar_velocity_rmse_mps']:+.4f} "
                "ddeck="
                f"{deltas['normalized_stability_composite']:+.3f} "
                f"dduty={deltas['torque_duty_fraction']:+.3f} "
                f"dburst={deltas['torque_burst_s']:+.3f} "
                f"dpeak={deltas['torque_peak_nm']:+.3f} "
                "largest_normalized_jump="
                f"{threshold['largest_normalized_jump']:.3f}:"
                f"{threshold['largest_normalized_jump_metric']}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Stage-2C evaluator reports containing the four admission "
            "commands plus optional 0.23/0.24/0.25 m/s probes."
        )
    )
    parser.add_argument("input", help="Evaluator JSON from evaluate_checkpoint.py")
    parser.add_argument(
        "--baseline",
        help=(
            "Unique checkpoint/label substring used for relative deltas "
            "(default: first evaluation)"
        ),
    )
    parser.add_argument(
        "--diagnostic-expected-samples",
        type=int,
        help=(
            "Explicitly analyze a shorter post-warmup screen (for example "
            "275 samples from a 6 s run). This mode is never formal-admission "
            "eligible; the default remains the canonical 475 samples."
        ),
    )
    parser.add_argument("--json", dest="json_path", help="Write analysis JSON here")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        input_path = Path(args.input).expanduser().resolve()
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        analysis = analyze_payload(
            payload,
            baseline=args.baseline,
            diagnostic_expected_samples=args.diagnostic_expected_samples,
        )
    except (OSError, json.JSONDecodeError, admission.ReportError) as exc:
        print(f"analysis error: {exc}", file=sys.stderr)
        return 65

    _print_summary(analysis)
    if args.json_path:
        output_path = Path(args.json_path).expanduser().resolve()
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("x", encoding="utf-8") as output_file:
                json.dump(analysis, output_file, indent=2, sort_keys=True)
                output_file.write("\n")
        except FileExistsError:
            print(
                f"refusing to overwrite existing analysis: {output_path}",
                file=sys.stderr,
            )
            return 73
        except OSError as exc:
            print(f"cannot write analysis {output_path}: {exc}", file=sys.stderr)
            return 73
        print(f"analysis_json={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
