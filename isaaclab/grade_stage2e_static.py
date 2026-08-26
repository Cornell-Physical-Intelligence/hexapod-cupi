#!/usr/bin/env python3
"""Fail-closed fixed-command grading for one Stage2E E0--E2 run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import grade_stage2d_homotopy as common


@dataclass(frozen=True)
class StageSpec:
    name: str
    task_id: str
    experiment: str
    evaluation_seed: int
    reverse_midpoint_mps: float
    yaw_bridge_forward_mps: float
    joystick_forward_midpoint_mps: float
    joystick_lateral_midpoint_mps: float
    joystick_yaw_midpoint_radps: float
    minimum_reverse_fraction: float
    minimum_pivot_fraction: float
    candidate_iterations: tuple[int, ...]


STAGE_SPECS = {
    "E0": StageSpec(
        "E0",
        "Isaac-Velocity-Omni-Recovery-Stage2E-Joystick-E0-Hexapod-RobStride-Direct-v0",
        "hexapod_robstride_phase2_recovery_stage2e_e0_joystick_bridge_direct",
        72,
        0.035,
        0.17,
        0.15,
        0.07,
        0.17,
        0.15,
        0.10,
        (0, 5, 10, 15, 20, 25, 29),
    ),
    "E1": StageSpec(
        "E1",
        "Isaac-Velocity-Omni-Recovery-Stage2E-Joystick-E1-Hexapod-RobStride-Direct-v0",
        "hexapod_robstride_phase2_recovery_stage2e_e1_reverse_bridge_direct",
        73,
        0.07,
        0.075,
        0.155,
        0.07,
        0.17,
        0.25,
        0.20,
        (0, 5, 10, 15, 20, 25, 30, 34),
    ),
    "E2": StageSpec(
        "E2",
        "Isaac-Velocity-Omni-Recovery-Stage2E-Joystick-E2-Hexapod-RobStride-Direct-v0",
        "hexapod_robstride_phase2_recovery_stage2e_e2_joystick_direct",
        74,
        0.14,
        0.0,
        0.165,
        0.065,
        0.17,
        0.35,
        0.40,
        (0, 5, 10, 15, 20, 25, 30, 35, 40, 44),
    ),
}


@dataclass(frozen=True)
class GradeContract:
    spec: StageSpec
    run_name: str
    seed_checkpoint: str
    seed_sha256: str


THRESHOLDS = {
    "minimum_forward_fraction": 0.80,
    "minimum_combined_forward_fraction": 0.60,
    "minimum_lateral_fraction": 0.40,
    "minimum_combined_lateral_fraction": 0.35,
    "minimum_yaw_fraction": 0.40,
    "minimum_combined_yaw_fraction": 0.35,
    "maximum_signed_axis_fraction": 1.75,
    "minimum_anchor_retention_fraction": 0.90,
    "maximum_stand_planar_speed_mps": 0.03,
    "maximum_stand_planar_rmse_mps": 0.04,
    "maximum_stand_abs_yaw_rate_radps": 0.08,
    "maximum_uncommanded_axis_speed": 0.06,
    "maximum_anchor_planar_rmse_mps": 0.14,
    "maximum_combined_planar_rmse_mps": 0.18,
    "maximum_yaw_rmse_radps": 0.20,
    "maximum_stability_regression_fraction": 0.10,
    "maximum_mean_base_height_error_m": 0.010,
}


# Standing keeps the strict Stage2C RMS deck contract.  The moving limits in
# the Stage2D seed remain appropriate for every non-stand command, while the
# tail limits are hard in both modes.
STAND_RMS_STABILITY_COMPONENTS = (
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


def _command_height_target(command: tuple[float, float, float]) -> float:
    """Return the inherited low/deep stance target for this command."""

    vx, vy, _ = command
    return 0.177 if math.hypot(vx, vy) <= 0.05 + common.COMMAND_TOLERANCE else 0.181


class ReportError(ValueError):
    """Raised when a static screen does not exactly match its contract."""


def make_contract(
    stage: str, run_name: str, seed_checkpoint: str, seed_sha256: str
) -> GradeContract:
    canonical = stage.upper()
    if canonical not in STAGE_SPECS:
        raise ReportError(f"stage must be one of {tuple(STAGE_SPECS)}")
    spec = STAGE_SPECS[canonical]
    if (
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name) is None
        or run_name in {".", ".."}
        or f"stage2e_e{canonical[1:]}" not in run_name.lower()
    ):
        raise ReportError(f"run_name must safely identify stage2e_{canonical.lower()}")
    seed_path = Path(seed_checkpoint)
    if (
        not seed_path.is_absolute()
        or seed_path.suffix != ".pt"
        or len(seed_path.parents) < 2
        or seed_path.parent.parent.name != spec.experiment
    ):
        raise ReportError(
            f"seed_checkpoint must be an absolute staged child of {spec.experiment}"
        )
    normalized_sha = seed_sha256.lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized_sha) is None:
        raise ReportError("seed_sha256 must be exactly 64 hexadecimal characters")
    return GradeContract(spec, run_name, seed_checkpoint, normalized_sha)


def command_contract(
    spec: StageSpec,
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    """Return anchors, reverse-only, pivot, and both joystick quadrants."""

    commands: list[tuple[str, tuple[float, float, float]]] = [
        ("stand", (0.0, 0.0, 0.0)),
        ("forward_anchor", (0.25, 0.0, 0.0)),
        ("reverse_anchor", (-spec.reverse_midpoint_mps, 0.0, 0.0)),
        ("lateral_anchor_positive", (0.0, 0.09, 0.0)),
        ("lateral_anchor_negative", (0.0, -0.09, 0.0)),
        ("accepted_forward_yaw_positive", (0.23, 0.0, 0.24)),
        ("accepted_forward_yaw_negative", (0.23, 0.0, -0.24)),
    ]
    if spec.yaw_bridge_forward_mps > common.COMMAND_TOLERANCE:
        commands.extend(
            (
                ("yaw_bridge_positive", (spec.yaw_bridge_forward_mps, 0.0, 0.24)),
                ("yaw_bridge_negative", (spec.yaw_bridge_forward_mps, 0.0, -0.24)),
            )
        )
    commands.extend(
        (
            ("pivot_yaw_positive", (0.0, 0.0, 0.24)),
            ("pivot_yaw_negative", (0.0, 0.0, -0.24)),
        )
    )
    for x_name, vx in (
        ("forward_joystick", spec.joystick_forward_midpoint_mps),
        ("reverse_joystick", -spec.reverse_midpoint_mps),
    ):
        for y_name, y_sign in (("left", 1.0), ("right", -1.0)):
            for yaw_name, yaw_sign in (("ccw", 1.0), ("cw", -1.0)):
                commands.append(
                    (
                        f"{x_name}_{y_name}_{yaw_name}",
                        (
                            vx,
                            y_sign * spec.joystick_lateral_midpoint_mps,
                            yaw_sign * spec.joystick_yaw_midpoint_radps,
                        ),
                    )
                )
    return tuple(commands)


def _signed_fraction(command: float, achieved: float) -> float:
    return math.copysign(1.0, command) * achieved / abs(command)


def _check_fraction(
    reasons: list[str], name: str, command: float, achieved: float, minimum: float
) -> float:
    fraction = _signed_fraction(command, achieved)
    if command * achieved <= 0.0:
        reasons.append(
            f"{name} sign is wrong: command={command:+.5f}, achieved={achieved:+.5f}"
        )
    common._minimum_reason(reasons, name, fraction, minimum)
    common._maximum_reason(
        reasons,
        name,
        fraction,
        THRESHOLDS["maximum_signed_axis_fraction"],
    )
    return fraction


def _validate_playback(
    report: dict[str, Any], index: int, contract: GradeContract
) -> None:
    expected = {
        "task": contract.spec.task_id,
        "command_frame": "navigation",
        "seed": contract.spec.evaluation_seed,
        "deterministic_policy": True,
        "policy_step_seconds": common.EXPECTED_POLICY_STEP_SECONDS,
        "requested_steps": common.EXPECTED_REQUESTED_STEPS,
        "warmup_steps": common.EXPECTED_WARMUP_STEPS,
        "commands_evaluated_in_parallel": len(command_contract(contract.spec)),
        "reset_stance_override": None,
        "startup_randomization_enabled": False,
    }
    for key, value in expected.items():
        actual = report.get(key)
        if actual != value:
            raise ReportError(
                f"evaluations[{index}].{key} must be {value!r}, got {actual!r}"
            )


def _row_command(row: dict[str, Any], path: str) -> tuple[float, float, float]:
    command = row.get("command")
    if not isinstance(command, dict) or command.get("frame") != "navigation":
        raise ReportError(f"{path}.command must be a navigation-frame object")
    return tuple(
        common._finite_float(command.get(key), f"{path}.command.{key}")
        for key in ("body_vx_mps", "body_vy_mps", "yaw_rate_radps")
    )


def _ordered_rows(
    report: dict[str, Any], report_index: int, spec: StageSpec
) -> list[dict[str, Any]]:
    rows = report.get("results")
    commands = command_contract(spec)
    if not isinstance(rows, list) or len(rows) != len(commands):
        raise ReportError(
            f"evaluations[{report_index}].results must contain exactly {len(commands)} rows"
        )
    unmatched = list(enumerate(rows))
    ordered = []
    for expected_index, (label, command) in enumerate(commands):
        matches = [
            (position, row_index, row)
            for position, (row_index, row) in enumerate(unmatched)
            if isinstance(row, dict)
            and common._commands_match(
                _row_command(row, f"evaluations[{report_index}].results[{row_index}]"),
                command,
            )
        ]
        if len(matches) != 1:
            raise ReportError(
                f"evaluations[{report_index}] must contain {label}={command} exactly once"
            )
        position, row_index, row = matches[0]
        if common._nonnegative_int(
            row.get("index"),
            f"evaluations[{report_index}].results[{row_index}].index",
        ) != expected_index:
            raise ReportError(f"{label} row index must be {expected_index}")
        ordered.append(row)
        unmatched.pop(position)
    return ordered


def _grade_row(
    row: dict[str, Any], label: str, command: tuple[float, float, float], spec: StageSpec
) -> dict[str, Any]:
    reasons: list[str] = []
    achieved_linear = common._finite_vector(
        row.get("mean_command_frame_linear_velocity_mps"),
        3,
        f"{label}.mean_command_frame_linear_velocity_mps",
    )
    achieved_angular = common._finite_vector(
        row.get("mean_command_frame_angular_velocity_radps"),
        3,
        f"{label}.mean_command_frame_angular_velocity_radps",
    )
    planar_rmse = common._nonnegative_float(
        row.get("planar_velocity_rmse_mps"), f"{label}.planar_velocity_rmse_mps"
    )
    yaw_rmse = common._nonnegative_float(
        row.get("yaw_rate_rmse_radps"), f"{label}.yaw_rate_rmse_radps"
    )
    if common._nonnegative_int(row.get("samples"), f"{label}.samples") != common.EXPECTED_SAMPLES:
        raise ReportError(f"{label}.samples must be {common.EXPECTED_SAMPLES}")
    measured = common._nonnegative_float(
        row.get("measured_seconds"), f"{label}.measured_seconds"
    )
    if not math.isclose(measured, common.EXPECTED_MEASURED_SECONDS, abs_tol=1.0e-4):
        raise ReportError(f"{label}.measured_seconds has the wrong duration")
    falls = common._nonnegative_int(row.get("falls"), f"{label}.falls")
    timeouts = common._nonnegative_int(row.get("timeouts"), f"{label}.timeouts")
    if falls:
        reasons.append(f"falls={falls}, expected 0")
    if timeouts:
        reasons.append(f"timeouts={timeouts}, expected 0")

    stability = {
        key: common._nonnegative_float(row.get(key), f"{label}.{key}")
        for key, _, _, _ in common.STABILITY_COMPONENTS
    }
    mean_base_height = common._nonnegative_float(
        row.get("mean_base_height_m"), f"{label}.mean_base_height_m"
    )
    height_target = _command_height_target(command)
    height_error = abs(mean_base_height - height_target)
    absolute_stability_reasons: list[str] = []
    absolute_components = (
        STAND_RMS_STABILITY_COMPONENTS + common.TAIL_STABILITY_COMPONENTS
        if label == "stand"
        else common.STABILITY_COMPONENTS
    )
    for key, display, unit, target in absolute_components:
        common._maximum_reason(
            absolute_stability_reasons, display, stability[key], target, unit
        )
    common._maximum_reason(
        absolute_stability_reasons,
        "mean base-height error",
        height_error,
        THRESHOLDS["maximum_mean_base_height_error_m"],
        "m",
    )
    normalized_stability = sum(
        stability[key] / target for key, _, _, target in common.STABILITY_COMPONENTS
    ) / len(common.STABILITY_COMPONENTS)

    torque = row.get("torque")
    if not isinstance(torque, dict):
        raise ReportError(f"{label}.torque must be an object")
    rated = common._nonnegative_float(
        torque.get("rated_continuous_nm"), f"{label}.torque.rated_continuous_nm"
    )
    if not math.isclose(rated, 1.60, abs_tol=common.GATE_ABS_TOLERANCE):
        raise ReportError(f"{label}.torque.rated_continuous_nm must be 1.6")
    per_joint = common._per_joint_torque_rows(torque, f"{label}.torque")
    torque_metrics = {
        key: common._nonnegative_float(torque.get(key), f"{label}.torque.{key}")
        for key in (
            "rms_applied_nm",
            "max_per_joint_rms_applied_nm",
            "computed_demand_over_rating_fraction",
            "maximum_computed_over_rating_burst_s",
            "peak_abs_computed_nm",
        )
    }
    for key, display, unit in (
        ("peak_abs_computed_nm", "peak absolute computed torque", "Nm"),
        ("computed_demand_over_rating_fraction", "computed demand over rating fraction", ""),
        ("maximum_computed_over_rating_burst_s", "computed over-rating burst", "s"),
        ("max_per_joint_rms_applied_nm", "max per-joint RMS applied torque", "Nm"),
    ):
        common._maximum_reason(reasons, display, torque_metrics[key], common.THRESHOLDS[key], unit)
    worst_joint = max(
        per_joint, key=lambda item: item["computed_demand_over_rating_fraction"]
    )
    common._maximum_reason(
        reasons,
        f"worst per-joint computed demand over rating fraction ({worst_joint['name']})",
        float(worst_joint["computed_demand_over_rating_fraction"]),
        common.THRESHOLDS[
            "maximum_per_joint_computed_demand_over_rating_fraction"
        ],
    )

    vx, vy, yaw = command
    achieved_vx, achieved_vy = achieved_linear[:2]
    achieved_yaw = achieved_angular[2]
    fractions: dict[str, float] = {}
    if label == "stand":
        common._maximum_reason(
            reasons,
            "stand planar speed",
            math.hypot(achieved_vx, achieved_vy),
            THRESHOLDS["maximum_stand_planar_speed_mps"],
            "m/s",
        )
        common._maximum_reason(
            reasons,
            "stand absolute yaw rate",
            abs(achieved_yaw),
            THRESHOLDS["maximum_stand_abs_yaw_rate_radps"],
            "rad/s",
        )
    else:
        if abs(vx) > common.COMMAND_TOLERANCE:
            minimum_x = (
                spec.minimum_reverse_fraction
                if vx < 0.0
                else (
                    THRESHOLDS["minimum_combined_forward_fraction"]
                    if "joystick" in label
                    else THRESHOLDS["minimum_forward_fraction"]
                )
            )
            fractions["x"] = _check_fraction(
                reasons, "signed x fraction", vx, achieved_vx, minimum_x
            )
        else:
            common._maximum_reason(
                reasons,
                "absolute x leakage",
                abs(achieved_vx),
                THRESHOLDS["maximum_uncommanded_axis_speed"],
                "m/s",
            )
        if abs(vy) > common.COMMAND_TOLERANCE:
            minimum_y = (
                THRESHOLDS["minimum_combined_lateral_fraction"]
                if "joystick" in label
                else THRESHOLDS["minimum_lateral_fraction"]
            )
            fractions["y"] = _check_fraction(
                reasons, "signed y fraction", vy, achieved_vy, minimum_y
            )
        else:
            common._maximum_reason(
                reasons,
                "absolute y leakage",
                abs(achieved_vy),
                THRESHOLDS["maximum_uncommanded_axis_speed"],
                "m/s",
            )
        if abs(yaw) > common.COMMAND_TOLERANCE:
            minimum_yaw = (
                spec.minimum_pivot_fraction
                if label.startswith("pivot_yaw")
                else (
                    THRESHOLDS["minimum_combined_yaw_fraction"]
                    if "joystick" in label
                    else THRESHOLDS["minimum_yaw_fraction"]
                )
            )
            fractions["yaw"] = _check_fraction(
                reasons, "signed yaw fraction", yaw, achieved_yaw, minimum_yaw
            )
        else:
            common._maximum_reason(
                reasons,
                "absolute yaw leakage",
                abs(achieved_yaw),
                THRESHOLDS["maximum_stand_abs_yaw_rate_radps"],
                "rad/s",
            )

    common._maximum_reason(
        reasons,
        "planar velocity RMSE",
        planar_rmse,
        (
            THRESHOLDS["maximum_stand_planar_rmse_mps"]
            if label == "stand"
            else (
                THRESHOLDS["maximum_combined_planar_rmse_mps"]
                if "joystick" in label
                else THRESHOLDS["maximum_anchor_planar_rmse_mps"]
            )
        ),
        "m/s",
    )
    common._maximum_reason(
        reasons,
        "yaw-rate RMSE",
        yaw_rmse,
        THRESHOLDS["maximum_yaw_rmse_radps"],
        "rad/s",
    )
    tracking_score = (
        sum(abs(1.0 - value) for value in fractions.values()) / len(fractions)
        if fractions
        else math.hypot(achieved_vx, achieved_vy) + abs(achieved_yaw)
    )
    load_score = (
        torque_metrics["peak_abs_computed_nm"] / common.THRESHOLDS["peak_abs_computed_nm"]
        + torque_metrics["computed_demand_over_rating_fraction"]
        / common.THRESHOLDS["computed_demand_over_rating_fraction"]
        + torque_metrics["maximum_computed_over_rating_burst_s"]
        / common.THRESHOLDS["maximum_computed_over_rating_burst_s"]
        + torque_metrics["max_per_joint_rms_applied_nm"]
        / common.THRESHOLDS["max_per_joint_rms_applied_nm"]
    ) / 4.0
    return {
        "label": label,
        "command": list(command),
        "operational_safe": not reasons,
        "operational_rejection_reasons": reasons,
        "absolute_stability_safe": not absolute_stability_reasons,
        "absolute_stability_rejection_reasons": absolute_stability_reasons,
        "metrics": {
            **stability,
            "mean_base_height_m": mean_base_height,
            "command_conditioned_base_height_target_m": height_target,
            "mean_base_height_error_m": height_error,
            "normalized_stability_composite": normalized_stability,
            "axis_fractions": fractions,
            "normalized_tracking_score": tracking_score,
            "normalized_load_score": load_score,
            "falls": falls,
            "timeouts": timeouts,
            "per_joint_names": [item["name"] for item in per_joint],
        },
    }


def _apply_seed_gates(candidate: dict[str, Any], seed: dict[str, Any]) -> None:
    if candidate["metrics"]["per_joint_names"] != seed["metrics"]["per_joint_names"]:
        raise ReportError(f"{candidate['label']} per-joint names/order differ from seed")
    stability_reasons = []
    factor = 1.0 + THRESHOLDS["maximum_stability_regression_fraction"]
    changes = []
    for key, display, unit, _ in common.STABILITY_COMPONENTS:
        actual = float(candidate["metrics"][key])
        baseline = float(seed["metrics"][key])
        common._maximum_reason(
            stability_reasons,
            f"command-local {display}",
            actual,
            baseline * factor,
            unit,
        )
        change = common._fractional_change(actual, baseline)
        if change is not None:
            changes.append(change)
    tracking_reasons = []
    if candidate["label"] in {
        "forward_anchor",
        "lateral_anchor_positive",
        "lateral_anchor_negative",
        "accepted_forward_yaw_positive",
        "accepted_forward_yaw_negative",
    }:
        for axis, seed_fraction in seed["metrics"]["axis_fractions"].items():
            actual_fraction = candidate["metrics"]["axis_fractions"].get(axis)
            if actual_fraction is None:
                raise ReportError(f"{candidate['label']} lost tracked seed axis {axis}")
            minimum = float(seed_fraction) * THRESHOLDS[
                "minimum_anchor_retention_fraction"
            ]
            common._minimum_reason(
                tracking_reasons,
                f"command-local retained {axis} fraction",
                float(actual_fraction),
                minimum,
            )
    candidate["relative_stability_safe"] = not stability_reasons
    candidate["tracking_preserved"] = not tracking_reasons
    candidate["stability_rejection_reasons"] = stability_reasons
    candidate["tracking_rejection_reasons"] = tracking_reasons
    candidate["maximum_stability_regression_fraction"] = max(changes, default=1.0)
    candidate["rejection_reasons"] = [
        *candidate["operational_rejection_reasons"],
        *candidate["absolute_stability_rejection_reasons"],
        *stability_reasons,
        *tracking_reasons,
    ]
    candidate["accepted"] = not candidate["rejection_reasons"]


def _iteration(checkpoint: str) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", Path(checkpoint).name)
    if match is None:
        raise ReportError(f"candidate checkpoint is not model_<iteration>.pt: {checkpoint}")
    return int(match.group(1))


def grade_payload(payload: Any, contract: GradeContract) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("evaluations"), list):
        raise ReportError("top-level evaluations list is required")
    reports = payload["evaluations"]
    expected_count = 1 + len(contract.spec.candidate_iterations)
    if len(reports) != expected_count or not all(isinstance(item, dict) for item in reports):
        raise ReportError(f"evaluations must contain exactly {expected_count} reports")
    for index, report in enumerate(reports):
        _validate_playback(report, index, contract)

    seed_report = reports[0]
    if seed_report.get("checkpoint") != contract.seed_checkpoint:
        raise ReportError("first evaluation must be the exact immutable seed")
    seed_rows = _ordered_rows(seed_report, 0, contract.spec)
    seed_results = [
        _grade_row(row, label, command, contract.spec)
        for row, (label, command) in zip(seed_rows, command_contract(contract.spec))
    ]

    expected_directory = Path(contract.seed_checkpoint).parent.parent / contract.run_name
    expected_iterations = set(contract.spec.candidate_iterations)
    seen_iterations = set()
    candidates = []
    for report_index, report in enumerate(reports[1:], start=1):
        checkpoint = report.get("checkpoint")
        if not isinstance(checkpoint, str):
            raise ReportError(f"evaluations[{report_index}].checkpoint must be a string")
        path = Path(checkpoint)
        if not path.is_absolute() or path.parent != expected_directory:
            raise ReportError(f"candidate must use exact run directory {expected_directory}")
        iteration = _iteration(checkpoint)
        if iteration not in expected_iterations or iteration in seen_iterations:
            raise ReportError(f"unexpected or duplicate candidate iteration {iteration}")
        seen_iterations.add(iteration)
        rows = _ordered_rows(report, report_index, contract.spec)
        results = [
            _grade_row(row, label, command, contract.spec)
            for row, (label, command) in zip(rows, command_contract(contract.spec))
        ]
        for result, seed_result in zip(results, seed_results):
            _apply_seed_gates(result, seed_result)
        candidates.append(
            {
                "checkpoint": checkpoint,
                "iteration": iteration,
                "accepted": all(result["accepted"] for result in results),
                "mean_normalized_stability_composite": sum(
                    result["metrics"]["normalized_stability_composite"]
                    for result in results
                )
                / len(results),
                "maximum_stability_regression_fraction": max(
                    result["maximum_stability_regression_fraction"]
                    for result in results
                ),
                "mean_normalized_tracking_score": sum(
                    result["metrics"]["normalized_tracking_score"]
                    for result in results
                )
                / len(results),
                "mean_normalized_load_score": sum(
                    result["metrics"]["normalized_load_score"] for result in results
                )
                / len(results),
                "results": results,
            }
        )
    if seen_iterations != expected_iterations:
        raise ReportError(
            f"candidate set is incomplete; missing {sorted(expected_iterations - seen_iterations)}"
        )
    accepted = sorted(
        (candidate for candidate in candidates if candidate["accepted"]),
        key=lambda candidate: (
            candidate["mean_normalized_stability_composite"],
            candidate["maximum_stability_regression_fraction"],
            candidate["mean_normalized_tracking_score"],
            candidate["mean_normalized_load_score"],
            candidate["iteration"],
        ),
    )
    return {
        "schema_version": 1,
        "stage": contract.spec.name,
        "task": contract.spec.task_id,
        "experiment": contract.spec.experiment,
        "exact_run_name": contract.run_name,
        "command_contract": [
            {"label": label, "command": list(command)}
            for label, command in command_contract(contract.spec)
        ],
        "candidate_iterations": list(contract.spec.candidate_iterations),
        "thresholds": THRESHOLDS,
        "candidate_count": len(candidates),
        "accepted": bool(accepted),
        "accepted_checkpoint_count": len(accepted),
        "best_accepted_checkpoint": accepted[0]["checkpoint"] if accepted else None,
        "selected_checkpoint": accepted[0]["checkpoint"] if accepted else None,
        "selected_checkpoint_host": None,
        "selected_checkpoint_sha256": None,
        "accepted_checkpoints_ranked": [item["checkpoint"] for item in accepted],
        "provenance_contract": {
            "task": contract.spec.task_id,
            "experiment": contract.spec.experiment,
            "exact_run_name": contract.run_name,
            "seed_checkpoint": contract.seed_checkpoint,
            "seed_sha256": contract.seed_sha256,
        },
        "evaluations": candidates,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attach_checkpoint_provenance(
    grade: dict[str, Any], contract: GradeContract, run_directory: str | Path
) -> dict[Path, str]:
    """Attach exact host/container checkpoint identity to a completed grade."""

    provided_run_dir = Path(run_directory).expanduser()
    if provided_run_dir.is_symlink():
        raise ReportError("run-dir must not be a symlink")
    run_dir = provided_run_dir.resolve(strict=True)
    if (
        not run_dir.is_dir()
        or run_dir.name != contract.run_name
        or run_dir.parent.name != contract.spec.experiment
    ):
        raise ReportError(
            "run-dir must be the exact non-symlink host candidate directory "
            f"inside the {contract.spec.experiment} experiment"
        )
    candidate_hashes: dict[Path, str] = {}
    for candidate in grade["evaluations"]:
        host_checkpoint = run_dir / Path(candidate["checkpoint"]).name
        if not host_checkpoint.is_file() or host_checkpoint.is_symlink():
            raise ReportError(
                f"candidate checkpoint is missing or a symlink: {host_checkpoint}"
            )
        checkpoint_sha = _sha256(host_checkpoint)
        candidate["checkpoint_sha256"] = checkpoint_sha
        candidate_hashes[host_checkpoint] = checkpoint_sha
    if grade["selected_checkpoint"] is not None:
        selected_name = Path(grade["selected_checkpoint"]).name
        selected_host = run_dir / selected_name
        selected_sha = candidate_hashes[selected_host]
        grade["selected_checkpoint_host"] = str(selected_host)
        grade["selected_checkpoint_sha256"] = selected_sha
        grade["provenance_contract"]["selected_checkpoint"] = grade[
            "selected_checkpoint"
        ]
        grade["provenance_contract"]["selected_checkpoint_sha256"] = selected_sha
        next_stage = {"E0": "E1", "E1": "E2", "E2": "TRANSITION"}[
            contract.spec.name
        ]
        next_experiment = (
            STAGE_SPECS[next_stage].experiment
            if next_stage in STAGE_SPECS
            else None
        )
        grade["handoff"] = {
            "source_stage": contract.spec.name,
            "next_stage": next_stage,
            "next_experiment": next_experiment,
            "selected_checkpoint": grade["selected_checkpoint"],
            "selected_checkpoint_host": str(selected_host),
            "selected_checkpoint_sha256": selected_sha,
        }
    return candidate_hashes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--stage", required=True, choices=tuple(STAGE_SPECS))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--seed-checkpoint", required=True)
    parser.add_argument("--seed-sha256", required=True)
    parser.add_argument("--json", dest="json_path", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    contract = make_contract(
        args.stage, args.run_name, args.seed_checkpoint, args.seed_sha256
    )
    seed_file = Path(args.seed_file).expanduser().resolve(strict=True)
    if not seed_file.is_file() or seed_file.name != Path(contract.seed_checkpoint).name:
        raise ReportError("seed-file must match the immutable seed checkpoint basename")
    if _sha256(seed_file) != contract.seed_sha256:
        raise ReportError("immutable seed SHA mismatch before grading")
    input_path = Path(args.input).expanduser().resolve(strict=True)
    grade = grade_payload(json.loads(input_path.read_text(encoding="utf-8")), contract)
    candidate_hashes = attach_checkpoint_provenance(grade, contract, args.run_dir)
    for candidate in grade["evaluations"]:
        print(
            f"{'PASS' if candidate['accepted'] else 'FAIL'} "
            f"checkpoint={candidate['checkpoint']} "
            f"stability={candidate['mean_normalized_stability_composite']:.4f}"
        )
        for result in candidate["results"]:
            for reason in result["rejection_reasons"]:
                print(f"  {result['label']}: {reason}")
    print(f"best_accepted_checkpoint={grade['best_accepted_checkpoint']}")
    output = Path(args.json_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as stream:
            json.dump(grade, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise ReportError(f"refusing to overwrite grading report: {output}") from exc
    if _sha256(seed_file) != contract.seed_sha256:
        raise ReportError("immutable seed changed during grading")
    for candidate_path, expected_sha in candidate_hashes.items():
        if _sha256(candidate_path) != expected_sha:
            raise ReportError(f"candidate changed during grading: {candidate_path}")
    return 0 if grade["accepted_checkpoint_count"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReportError, common.ReportError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"GRADE_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(65)
