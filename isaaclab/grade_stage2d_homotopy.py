#!/usr/bin/env python3
"""Fail-closed grading for one Stage2D C0--C5 homotopy screen.

The first evaluation is an immutable staged seed.  Every expected candidate
checkpoint is replayed in the same Kit process under the same stage-specific
commands.  Admission requires signed lateral acquisition in both directions,
preserved forward/yaw anchors, absolute platform and RS05 safety, and no
command-local stability component more than ten percent worse than the seed.
"""

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


EXPECTED_POLICY_STEP_SECONDS = 0.02
EXPECTED_REQUESTED_STEPS = 500
EXPECTED_WARMUP_STEPS = 25
EXPECTED_SAMPLES = EXPECTED_REQUESTED_STEPS - EXPECTED_WARMUP_STEPS
EXPECTED_MEASURED_SECONDS = EXPECTED_SAMPLES * EXPECTED_POLICY_STEP_SECONDS
EXPECTED_JOINT_COUNT = 18
COMMAND_TOLERANCE = 1.0e-4
GATE_ABS_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class StageSpec:
    """Immutable task and screening contract for one homotopy stage."""

    name: str
    index: int
    task_id: str
    experiment: str
    evaluation_seed: int
    oblique_forward_range: tuple[float, float]
    oblique_lateral_abs_range: tuple[float, float]
    minimum_signed_lateral_fraction: float
    candidate_iterations: tuple[int, ...]


def _stage_spec(
    index: int,
    forward_range: tuple[float, float],
    lateral_range: tuple[float, float],
    minimum_fraction: float,
) -> StageSpec:
    final_iteration = 39 if index == 5 else 24
    regular = tuple(range(0, 40 if index == 5 else 25, 5))
    return StageSpec(
        name=f"C{index}",
        index=index,
        task_id=(
            "Isaac-Velocity-Omni-Recovery-Stage2D-Homotopy-"
            f"C{index}-Hexapod-RobStride-Direct-v0"
        ),
        experiment=(
            "hexapod_robstride_phase2_recovery_stage2d_"
            f"c{index}{'_pure_y' if index == 5 else ''}_direct"
        ),
        evaluation_seed=61 + index,
        oblique_forward_range=forward_range,
        oblique_lateral_abs_range=lateral_range,
        minimum_signed_lateral_fraction=minimum_fraction,
        candidate_iterations=(*regular, final_iteration),
    )


# Acquisition floors rise monotonically as lateral command magnitude and its
# share of the requested heading grow: the gentlest bridge only has to show a
# real 20% response, then each stage adds five points.  C5's 45% of the 0.09
# m/s midpoint is 0.0405 m/s, the previously proven Stage2B pure-y floor.
STAGE_SPECS: dict[str, StageSpec] = {
    "C0": _stage_spec(0, (0.22, 0.26), (0.02, 0.04), 0.20),
    "C1": _stage_spec(1, (0.20, 0.24), (0.04, 0.06), 0.25),
    "C2": _stage_spec(2, (0.14, 0.18), (0.06, 0.08), 0.30),
    "C3": _stage_spec(3, (0.08, 0.12), (0.08, 0.10), 0.35),
    "C4": _stage_spec(4, (0.03, 0.06), (0.08, 0.10), 0.40),
    "C5": _stage_spec(5, (0.0, 0.0), (0.08, 0.10), 0.45),
}


@dataclass(frozen=True)
class GradeContract:
    """Invocation-owned provenance that must match every evaluator report."""

    spec: StageSpec
    run_name: str
    seed_checkpoint: str
    seed_sha256: str


THRESHOLDS: dict[str, float] = {
    "minimum_forward_anchor_fraction": 0.80,
    "maximum_forward_anchor_fraction": 1.30,
    "minimum_oblique_forward_fraction": 0.60,
    "maximum_oblique_forward_fraction": 1.60,
    "minimum_yaw_command_fraction": 0.40,
    "maximum_yaw_command_fraction": 1.60,
    "maximum_signed_lateral_fraction": 1.75,
    "minimum_lateral_pair_symmetry_ratio": 0.65,
    "maximum_anchor_abs_lateral_velocity_mps": 0.06,
    "maximum_pure_y_abs_forward_velocity_mps": 0.06,
    "maximum_uncommanded_abs_yaw_rate_radps": 0.10,
    "maximum_anchor_planar_velocity_rmse_mps": 0.12,
    "maximum_oblique_planar_velocity_rmse_mps": 0.14,
    "maximum_yaw_rate_rmse_radps": 0.18,
    "rated_continuous_nm": 1.60,
    "peak_abs_computed_nm": 4.40,
    "computed_demand_over_rating_fraction": 0.15,
    "maximum_computed_over_rating_burst_s": 0.14,
    "max_per_joint_rms_applied_nm": 1.40,
    "maximum_per_joint_rms_applied_nm": 1.60,
    "maximum_per_joint_computed_demand_over_rating_fraction": 0.25,
    "maximum_per_joint_computed_over_rating_burst_s": 0.20,
    "maximum_per_joint_peak_abs_computed_nm": 5.50,
    "maximum_stability_regression_fraction": 0.10,
    "minimum_anchor_tracking_retention_fraction": 0.90,
}

# These are inherited from the accepted Stage2C platform contract.  They are
# both absolute hard gates and the normalizers for stability-first ranking.
RMS_STABILITY_COMPONENTS: tuple[tuple[str, str, str, float], ...] = (
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
STABILITY_COMPONENTS = RMS_STABILITY_COMPONENTS + TAIL_STABILITY_COMPONENTS

PER_JOINT_TORQUE_METRICS: tuple[str, ...] = (
    "rms_applied_nm",
    "peak_abs_applied_nm",
    "peak_abs_computed_nm",
    "applied_at_rating_fraction",
    "computed_demand_over_rating_fraction",
    "maximum_computed_over_rating_burst_s",
)


class ReportError(ValueError):
    """Raised when input does not exactly match the Stage2D contract."""


def make_contract(
    stage: str,
    run_name: str,
    seed_checkpoint: str,
    seed_sha256: str,
) -> GradeContract:
    """Validate caller-owned provenance and construct an immutable contract."""

    canonical_stage = stage.upper()
    if canonical_stage not in STAGE_SPECS:
        raise ReportError(f"stage must be one of {tuple(STAGE_SPECS)}, got {stage!r}")
    if (
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name) is None
        or run_name in {".", ".."}
    ):
        raise ReportError(f"run_name must be one safe exact directory name, got {run_name!r}")
    spec = STAGE_SPECS[canonical_stage]
    if f"stage2d_c{spec.index}" not in run_name.lower():
        raise ReportError(
            f"run_name must identify stage2d_c{spec.index}, got {run_name!r}"
        )
    seed_path = Path(seed_checkpoint)
    if not seed_path.is_absolute() or seed_path.name in {"", ".", ".."}:
        raise ReportError("seed_checkpoint must be an absolute checkpoint path")
    if seed_path.suffix != ".pt":
        raise ReportError("seed_checkpoint must end in .pt")
    if len(seed_path.parents) < 2 or seed_path.parent.parent.name != spec.experiment:
        raise ReportError(
            "seed_checkpoint must be an immutable staged child of experiment "
            f"{spec.experiment!r}, got {seed_checkpoint!r}"
        )
    normalized_sha = seed_sha256.lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized_sha) is None:
        raise ReportError("seed_sha256 must be exactly 64 hexadecimal characters")
    return GradeContract(spec, run_name, seed_checkpoint, normalized_sha)


def command_contract(
    spec: StageSpec,
) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    """Return anchors plus midpoint/lower/upper paired oblique commands."""

    forward_low, forward_high = spec.oblique_forward_range
    lateral_low, lateral_high = spec.oblique_lateral_abs_range
    forward_mid = (forward_low + forward_high) / 2.0
    lateral_mid = (lateral_low + lateral_high) / 2.0
    return (
        ("forward_anchor", (0.25, 0.0, 0.0)),
        ("yaw_anchor_positive", (0.23, 0.0, 0.24)),
        ("yaw_anchor_negative", (0.23, 0.0, -0.24)),
        ("oblique_mid_positive", (forward_mid, lateral_mid, 0.0)),
        ("oblique_mid_negative", (forward_mid, -lateral_mid, 0.0)),
        ("oblique_low_positive", (forward_low, lateral_low, 0.0)),
        ("oblique_low_negative", (forward_low, -lateral_low, 0.0)),
        ("oblique_high_positive", (forward_high, lateral_high, 0.0)),
        ("oblique_high_negative", (forward_high, -lateral_high, 0.0)),
    )


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


def _commands_match(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> bool:
    return all(
        abs(left_value - right_value) <= COMMAND_TOLERANCE
        for left_value, right_value in zip(left, right)
    )


def _reports(payload: Any, contract: GradeContract) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ReportError("top-level JSON value must be an object")
    reports = payload.get("evaluations")
    expected_count = 1 + len(contract.spec.candidate_iterations)
    if not isinstance(reports, list) or len(reports) != expected_count:
        raise ReportError(
            "evaluations must contain exactly one immutable seed plus "
            f"{len(contract.spec.candidate_iterations)} candidates "
            f"({expected_count} reports total)"
        )
    if not all(isinstance(report, dict) for report in reports):
        raise ReportError("every evaluation must be an object")
    return reports


def _validate_playback_contract(
    report: dict[str, Any],
    report_index: int,
    contract: GradeContract,
    reference: dict[str, Any] | None,
) -> None:
    prefix = f"evaluations[{report_index}]"
    command_count = len(command_contract(contract.spec))
    expected = {
        "task": contract.spec.task_id,
        "command_frame": "navigation",
        "seed": contract.spec.evaluation_seed,
        "deterministic_policy": True,
        "commands_evaluated_in_parallel": command_count,
        "requested_steps": EXPECTED_REQUESTED_STEPS,
        "warmup_steps": EXPECTED_WARMUP_STEPS,
        "reset_stance_override": None,
        "startup_randomization_enabled": False,
    }
    for key, value in expected.items():
        if key not in report:
            raise ReportError(f"{prefix}.{key} is missing")
        if report.get(key) != value:
            raise ReportError(
                f"{prefix}.{key} must be {value!r}, got {report.get(key)!r}"
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
    if reference is not None:
        for key in (*expected, "policy_step_seconds"):
            if report.get(key) != reference.get(key):
                raise ReportError(
                    f"{prefix}.{key} differs from immutable seed playback"
                )


def _command_from_row(
    row: dict[str, Any], path: str
) -> tuple[float, float, float]:
    command = row.get("command")
    if not isinstance(command, dict):
        raise ReportError(f"{path}.command must be an object")
    if command.get("frame") != "navigation":
        raise ReportError(f"{path}.command.frame must be 'navigation'")
    return (
        _finite_float(command.get("body_vx_mps"), f"{path}.command.body_vx_mps"),
        _finite_float(command.get("body_vy_mps"), f"{path}.command.body_vy_mps"),
        _finite_float(command.get("yaw_rate_radps"), f"{path}.command.yaw_rate_radps"),
    )


def _ordered_contract_rows(
    report: dict[str, Any], report_index: int, contract: GradeContract
) -> list[dict[str, Any]]:
    rows = report.get("results")
    commands = command_contract(contract.spec)
    if not isinstance(rows, list) or len(rows) != len(commands):
        actual = len(rows) if isinstance(rows, list) else None
        raise ReportError(
            f"evaluations[{report_index}].results must contain exactly "
            f"{len(commands)} rows, got {actual!r}"
        )
    if not all(isinstance(row, dict) for row in rows):
        raise ReportError(f"evaluations[{report_index}] result rows must be objects")

    unmatched = list(enumerate(rows))
    ordered: list[dict[str, Any]] = []
    for expected_index, (label, expected_command) in enumerate(commands):
        matches = [
            (position, row_index, row)
            for position, (row_index, row) in enumerate(unmatched)
            if _commands_match(
                _command_from_row(
                    row, f"evaluations[{report_index}].results[{row_index}]"
                ),
                expected_command,
            )
        ]
        if len(matches) != 1:
            raise ReportError(
                f"evaluations[{report_index}] must contain {label}="
                f"{expected_command} exactly once; found {len(matches)}"
            )
        position, row_index, row = matches[0]
        actual_index = _nonnegative_int(
            row.get("index"),
            f"evaluations[{report_index}].results[{row_index}].index",
        )
        if actual_index != expected_index:
            raise ReportError(
                f"evaluations[{report_index}] {label} index must be "
                f"{expected_index}, got {actual_index}"
            )
        ordered.append(row)
        unmatched.pop(position)
    return ordered


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
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        row_path = f"{path}.per_joint[{index}]"
        if not isinstance(row, dict):
            raise ReportError(f"{row_path} must be an object")
        name = row.get("name")
        if not isinstance(name, str) or not name or name != name.strip():
            raise ReportError(f"{row_path}.name must be a non-empty trimmed string")
        if name in seen:
            raise ReportError(f"{path}.per_joint contains duplicate name {name!r}")
        seen.add(name)
        metrics = {
            key: _nonnegative_float(row.get(key), f"{row_path}.{key}")
            for key in PER_JOINT_TORQUE_METRICS
        }
        validated.append({"name": name, **metrics})
    return validated


def _grade_row(
    row: dict[str, Any],
    *,
    label: str,
    command: tuple[float, float, float],
    spec: StageSpec,
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
    planar_rmse = _nonnegative_float(
        row.get("planar_velocity_rmse_mps"), f"{path}.planar_velocity_rmse_mps"
    )
    yaw_rmse = _nonnegative_float(
        row.get("yaw_rate_rmse_radps"), f"{path}.yaw_rate_rmse_radps"
    )
    samples = _nonnegative_int(row.get("samples"), f"{path}.samples")
    if samples != EXPECTED_SAMPLES:
        raise ReportError(f"{path}.samples must be {EXPECTED_SAMPLES}, got {samples}")
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
    if falls:
        reasons.append(f"falls={falls}, expected 0")
    if timeouts:
        reasons.append(f"timeouts={timeouts}, expected 0")

    stability = {
        key: _nonnegative_float(row.get(key), f"{path}.{key}")
        for key, _, _, _ in STABILITY_COMPONENTS
    }
    normalized_stability = {
        key: stability[key] / target
        for key, _, _, target in STABILITY_COMPONENTS
    }
    stability_composite = sum(normalized_stability.values()) / len(
        normalized_stability
    )
    absolute_stability_reasons: list[str] = []
    for key, display_name, unit, target in STABILITY_COMPONENTS:
        _maximum_reason(
            absolute_stability_reasons,
            display_name,
            stability[key],
            target,
            unit,
        )

    torque = row.get("torque")
    if not isinstance(torque, dict):
        raise ReportError(f"{path}.torque must be an object")
    rated_torque = _nonnegative_float(
        torque.get("rated_continuous_nm"), f"{path}.torque.rated_continuous_nm"
    )
    if not math.isclose(
        rated_torque,
        THRESHOLDS["rated_continuous_nm"],
        abs_tol=GATE_ABS_TOLERANCE,
    ):
        raise ReportError(
            f"{path}.torque.rated_continuous_nm must be "
            f"{THRESHOLDS['rated_continuous_nm']}, got {rated_torque}"
        )
    per_joint = _per_joint_torque_rows(torque, f"{path}.torque")
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
    for key, display, unit in (
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
        _maximum_reason(reasons, display, torque_metrics[key], THRESHOLDS[key], unit)

    per_joint_reasons: list[str] = []
    for joint in per_joint:
        name = str(joint["name"])
        for key, display, threshold_key, unit in (
            (
                "rms_applied_nm",
                "RMS applied torque",
                "maximum_per_joint_rms_applied_nm",
                "Nm",
            ),
            (
                "computed_demand_over_rating_fraction",
                "computed demand over rating fraction",
                "maximum_per_joint_computed_demand_over_rating_fraction",
                "",
            ),
            (
                "maximum_computed_over_rating_burst_s",
                "computed over-rating burst",
                "maximum_per_joint_computed_over_rating_burst_s",
                "s",
            ),
            (
                "peak_abs_computed_nm",
                "peak absolute computed torque",
                "maximum_per_joint_peak_abs_computed_nm",
                "Nm",
            ),
        ):
            _maximum_reason(
                per_joint_reasons,
                f"joint {name} {display}",
                float(joint[key]),
                THRESHOLDS[threshold_key],
                unit,
            )
    reasons.extend(per_joint_reasons)

    vx, vy, yaw = command
    achieved_vx, achieved_vy = achieved_linear[:2]
    achieved_yaw = achieved_angular[2]
    forward_fraction: float | None = None
    yaw_fraction: float | None = None
    signed_lateral_fraction: float | None = None

    if label == "forward_anchor":
        forward_fraction = achieved_vx / vx
        _minimum_reason(
            reasons,
            "forward anchor fraction",
            forward_fraction,
            THRESHOLDS["minimum_forward_anchor_fraction"],
        )
        _maximum_reason(
            reasons,
            "forward anchor fraction",
            forward_fraction,
            THRESHOLDS["maximum_forward_anchor_fraction"],
        )
        _maximum_reason(
            reasons,
            "forward anchor absolute lateral velocity",
            abs(achieved_vy),
            THRESHOLDS["maximum_anchor_abs_lateral_velocity_mps"],
            "m/s",
        )
    elif label.startswith("yaw_anchor_"):
        forward_fraction = achieved_vx / vx
        _minimum_reason(
            reasons,
            "yaw-anchor forward fraction",
            forward_fraction,
            THRESHOLDS["minimum_forward_anchor_fraction"],
        )
        _maximum_reason(
            reasons,
            "yaw-anchor forward fraction",
            forward_fraction,
            THRESHOLDS["maximum_forward_anchor_fraction"],
        )
        yaw_fraction = math.copysign(1.0, yaw) * achieved_yaw / abs(yaw)
        if achieved_yaw * yaw <= 0.0:
            reasons.append(
                f"yaw sign is wrong: command={yaw:+.4f}, achieved={achieved_yaw:+.4f}"
            )
        _minimum_reason(
            reasons,
            "signed yaw fraction",
            yaw_fraction,
            THRESHOLDS["minimum_yaw_command_fraction"],
        )
        _maximum_reason(
            reasons,
            "signed yaw fraction",
            yaw_fraction,
            THRESHOLDS["maximum_yaw_command_fraction"],
        )
        _maximum_reason(
            reasons,
            "yaw-anchor yaw-rate RMSE",
            yaw_rmse,
            THRESHOLDS["maximum_yaw_rate_rmse_radps"],
            "rad/s",
        )
        _maximum_reason(
            reasons,
            "yaw-anchor absolute lateral velocity",
            abs(achieved_vy),
            THRESHOLDS["maximum_anchor_abs_lateral_velocity_mps"],
            "m/s",
        )
    elif label.startswith("oblique_"):
        signed_lateral_fraction = (
            math.copysign(1.0, vy) * achieved_vy / abs(vy)
        )
        if achieved_vy * vy <= 0.0:
            reasons.append(
                f"lateral sign is wrong: command={vy:+.4f}, "
                f"achieved={achieved_vy:+.4f}"
            )
        _minimum_reason(
            reasons,
            f"{spec.name} signed lateral command fraction",
            signed_lateral_fraction,
            spec.minimum_signed_lateral_fraction,
        )
        _maximum_reason(
            reasons,
            "signed lateral command fraction",
            signed_lateral_fraction,
            THRESHOLDS["maximum_signed_lateral_fraction"],
        )
        if vx > COMMAND_TOLERANCE:
            forward_fraction = achieved_vx / vx
            _minimum_reason(
                reasons,
                "oblique forward fraction",
                forward_fraction,
                THRESHOLDS["minimum_oblique_forward_fraction"],
            )
            _maximum_reason(
                reasons,
                "oblique forward fraction",
                forward_fraction,
                THRESHOLDS["maximum_oblique_forward_fraction"],
            )
        else:
            _maximum_reason(
                reasons,
                "pure-y absolute forward leakage",
                abs(achieved_vx),
                THRESHOLDS["maximum_pure_y_abs_forward_velocity_mps"],
                "m/s",
            )

    if abs(yaw) <= COMMAND_TOLERANCE:
        _maximum_reason(
            reasons,
            "uncommanded absolute yaw rate",
            abs(achieved_yaw),
            THRESHOLDS["maximum_uncommanded_abs_yaw_rate_radps"],
            "rad/s",
        )
    rmse_limit = THRESHOLDS[
        (
            "maximum_oblique_planar_velocity_rmse_mps"
            if label.startswith("oblique_")
            else "maximum_anchor_planar_velocity_rmse_mps"
        )
    ]
    _maximum_reason(
        reasons, "planar velocity RMSE", planar_rmse, rmse_limit, "m/s"
    )

    if label.startswith("oblique_"):
        tracking_components = [abs(achieved_vy - vy) / abs(vy)]
        if vx > COMMAND_TOLERANCE:
            tracking_components.append(abs(achieved_vx - vx) / vx)
        else:
            tracking_components.append(
                abs(achieved_vx)
                / THRESHOLDS["maximum_pure_y_abs_forward_velocity_mps"]
            )
    elif label.startswith("yaw_anchor_"):
        tracking_components = [
            abs(achieved_vx - vx) / vx,
            abs(achieved_yaw - yaw) / abs(yaw),
        ]
    else:
        tracking_components = [abs(achieved_vx - vx) / vx]

    normalized_load_score = (
        torque_metrics["peak_abs_computed_nm"]
        / THRESHOLDS["peak_abs_computed_nm"]
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
        "absolute_stability_safe": not absolute_stability_reasons,
        "absolute_stability_rejection_reasons": absolute_stability_reasons,
        "metrics": {
            "achieved_command_frame_velocity": [
                achieved_vx,
                achieved_vy,
                achieved_yaw,
            ],
            "rmse_command_error": list(rmse),
            "planar_velocity_rmse_mps": planar_rmse,
            "yaw_rate_rmse_radps": yaw_rmse,
            "forward_command_fraction": forward_fraction,
            "signed_yaw_command_fraction": yaw_fraction,
            "signed_lateral_command_fraction": signed_lateral_fraction,
            "normalized_tracking_score": sum(tracking_components)
            / len(tracking_components),
            "samples": samples,
            "measured_seconds": measured_seconds,
            "falls": falls,
            "timeouts": timeouts,
            **stability,
            "normalized_stability_components": normalized_stability,
            "normalized_stability_composite": stability_composite,
            "absolute_stability_targets_met": not absolute_stability_reasons,
            **torque_metrics,
            "per_joint_torque_row_count": len(per_joint),
            "per_joint_torque_names": [str(item["name"]) for item in per_joint],
            "normalized_load_score": normalized_load_score,
        },
    }


def _apply_lateral_symmetry_gates(results: list[dict[str, Any]]) -> None:
    by_label = {result["label"]: result for result in results}
    for prefix in ("oblique_mid", "oblique_low", "oblique_high"):
        positive = by_label[f"{prefix}_positive"]
        negative = by_label[f"{prefix}_negative"]
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
        floor = THRESHOLDS["minimum_lateral_pair_symmetry_ratio"]
        if ratio < floor - GATE_ABS_TOLERANCE:
            reason = (
                f"{prefix} lateral symmetry ratio {ratio:.4f} is below "
                f"{floor:.4f}"
            )
            for result in (positive, negative):
                result["operational_rejection_reasons"].append(reason)
                result["operational_safe"] = False


def _fractional_change(actual: float, baseline: float) -> float | None:
    if baseline <= GATE_ABS_TOLERANCE:
        return 0.0 if actual <= GATE_ABS_TOLERANCE else None
    return actual / baseline - 1.0


def _apply_seed_stability_gate(
    candidate: dict[str, Any], seed: dict[str, Any]
) -> None:
    if (
        candidate["metrics"]["per_joint_torque_names"]
        != seed["metrics"]["per_joint_torque_names"]
    ):
        raise ReportError(
            f"{candidate['label']} per-joint motor names/order differ from seed"
        )
    reasons: list[str] = []
    tracking_reasons: list[str] = []
    retention = THRESHOLDS["minimum_anchor_tracking_retention_fraction"]
    for key, display_name in (
        ("forward_command_fraction", "forward command fraction"),
        ("signed_yaw_command_fraction", "signed yaw command fraction"),
    ):
        actual_value = candidate["metrics"][key]
        seed_value = seed["metrics"][key]
        if actual_value is None and seed_value is None:
            continue
        if actual_value is None or seed_value is None:
            raise ReportError(
                f"{candidate['label']} {display_name} presence differs from seed"
            )
        minimum = float(seed_value) * retention
        if float(actual_value) < minimum - GATE_ABS_TOLERANCE:
            tracking_reasons.append(
                f"{display_name} {float(actual_value):.5f} retains less than "
                f"{100.0 * retention:.1f}% of command-local seed "
                f"{float(seed_value):.5f}"
            )

    changes: dict[str, float | None] = {}
    factor = 1.0 + THRESHOLDS["maximum_stability_regression_fraction"]
    for key, display_name, unit, _ in STABILITY_COMPONENTS:
        actual = float(candidate["metrics"][key])
        baseline = float(seed["metrics"][key])
        limit = baseline * factor
        change = _fractional_change(actual, baseline)
        changes[key] = change
        if actual > limit + GATE_ABS_TOLERANCE:
            change_text = "undefined from zero seed" if change is None else f"{100.0 * change:.1f}%"
            reasons.append(
                f"{display_name} {actual:.5f} {unit} exceeds command-local "
                f"seed limit {limit:.5f} {unit} (seed={baseline:.5f}, "
                f"change={change_text})"
            )

    actual_composite = float(
        candidate["metrics"]["normalized_stability_composite"]
    )
    seed_composite = float(seed["metrics"]["normalized_stability_composite"])
    composite_limit = seed_composite * factor
    composite_change = _fractional_change(actual_composite, seed_composite)
    changes["normalized_stability_composite"] = composite_change
    if actual_composite > composite_limit + GATE_ABS_TOLERANCE:
        reasons.append(
            f"normalized stability composite {actual_composite:.5f} exceeds "
            f"command-local seed limit {composite_limit:.5f}"
        )

    candidate["stability_change_fractions"] = changes
    candidate["relative_tracking_safe"] = not tracking_reasons
    candidate["tracking_preservation_rejection_reasons"] = tracking_reasons
    candidate["relative_stability_safe"] = not reasons
    candidate["stability_rejection_reasons"] = reasons
    candidate["accepted"] = (
        candidate["operational_safe"]
        and candidate["absolute_stability_safe"]
        and not tracking_reasons
        and not reasons
    )
    candidate["rejection_reasons"] = [
        *candidate["operational_rejection_reasons"],
        *candidate["absolute_stability_rejection_reasons"],
        *tracking_reasons,
        *reasons,
    ]


def _iteration_from_checkpoint(checkpoint: str) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", Path(checkpoint).name)
    if match is None:
        raise ReportError(
            f"candidate checkpoint must end in model_<iteration>.pt: {checkpoint!r}"
        )
    return int(match.group(1))


def _candidate_run_directory(
    checkpoint: str, contract: GradeContract
) -> str:
    path = Path(checkpoint)
    expected_directory = Path(contract.seed_checkpoint).parent.parent / contract.run_name
    if not path.is_absolute() or path.parent != expected_directory:
        raise ReportError(
            "candidate checkpoint must use exact absolute run directory "
            f"{str(expected_directory)!r}, got {checkpoint!r}"
        )
    return str(path.parent)


def _aggregate_evaluation(
    report: dict[str, Any],
    report_index: int,
    contract: GradeContract,
    seed_results: list[dict[str, Any]],
) -> dict[str, Any]:
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ReportError(
            f"evaluations[{report_index}].checkpoint must be a non-empty string"
        )
    rows = _ordered_contract_rows(report, report_index, contract)
    results = [
        _grade_row(row, label=label, command=command, spec=contract.spec)
        for row, (label, command) in zip(rows, command_contract(contract.spec))
    ]
    _apply_lateral_symmetry_gates(results)
    for result, seed_result in zip(results, seed_results):
        _apply_seed_stability_gate(result, seed_result)

    operational_safe = all(result["operational_safe"] for result in results)
    absolute_stability_safe = all(
        result["absolute_stability_safe"] for result in results
    )
    relative_stability_safe = all(
        result["relative_stability_safe"] for result in results
    )
    relative_tracking_safe = all(
        result["relative_tracking_safe"] for result in results
    )
    stability_changes = [
        change
        for result in results
        for change in result["stability_change_fractions"].values()
        if change is not None
    ]
    return {
        "checkpoint": checkpoint,
        "iteration": _iteration_from_checkpoint(checkpoint),
        "operational_safe": operational_safe,
        "absolute_stability_safe": absolute_stability_safe,
        "relative_stability_safe": relative_stability_safe,
        "relative_tracking_safe": relative_tracking_safe,
        "accepted": (
            operational_safe
            and absolute_stability_safe
            and relative_stability_safe
            and relative_tracking_safe
        ),
        "failed_operational_command_count": sum(
            not result["operational_safe"] for result in results
        ),
        "failed_absolute_stability_command_count": sum(
            not result["absolute_stability_safe"] for result in results
        ),
        "failed_relative_stability_command_count": sum(
            not result["relative_stability_safe"] for result in results
        ),
        "failed_relative_tracking_command_count": sum(
            not result["relative_tracking_safe"] for result in results
        ),
        "total_falls": sum(result["metrics"]["falls"] for result in results),
        "total_timeouts": sum(
            result["metrics"]["timeouts"] for result in results
        ),
        "mean_normalized_tracking_score": sum(
            result["metrics"]["normalized_tracking_score"] for result in results
        )
        / len(results),
        "mean_normalized_stability_composite": sum(
            result["metrics"]["normalized_stability_composite"]
            for result in results
        )
        / len(results),
        # An undefined ratio can only arise from a zero seed component with a
        # nonzero candidate, which the command-local gate already rejects.
        "maximum_stability_regression_fraction": max(
            stability_changes, default=1.0
        ),
        "mean_normalized_load_score": sum(
            result["metrics"]["normalized_load_score"] for result in results
        )
        / len(results),
        "results": results,
    }


def grade_payload(payload: Any, contract: GradeContract) -> dict[str, Any]:
    """Validate and grade one immutable seed plus one complete stage run."""

    reports = _reports(payload, contract)
    seed_report = reports[0]
    _validate_playback_contract(seed_report, 0, contract, None)
    seed_checkpoint = seed_report.get("checkpoint")
    if seed_checkpoint != contract.seed_checkpoint:
        raise ReportError(
            "evaluations[0].checkpoint must exactly equal immutable staged seed "
            f"{contract.seed_checkpoint!r}, got {seed_checkpoint!r}"
        )
    seed_rows = _ordered_contract_rows(seed_report, 0, contract)
    seed_results = [
        _grade_row(row, label=label, command=command, spec=contract.spec)
        for row, (label, command) in zip(
            seed_rows, command_contract(contract.spec)
        )
    ]
    _apply_lateral_symmetry_gates(seed_results)
    seed_reference = {
        "checkpoint": contract.seed_checkpoint,
        "expected_sha256": contract.seed_sha256,
        "operational_safe": all(
            result["operational_safe"] for result in seed_results
        ),
        "absolute_stability_safe": all(
            result["absolute_stability_safe"] for result in seed_results
        ),
        "mean_normalized_stability_composite": sum(
            result["metrics"]["normalized_stability_composite"]
            for result in seed_results
        )
        / len(seed_results),
        "results": seed_results,
    }

    seen_checkpoints = {contract.seed_checkpoint}
    seen_iterations: set[int] = set()
    run_directory: str | None = None
    candidates: list[dict[str, Any]] = []
    expected_iterations = set(contract.spec.candidate_iterations)
    for report_index, report in enumerate(reports[1:], start=1):
        _validate_playback_contract(report, report_index, contract, seed_report)
        checkpoint = report.get("checkpoint")
        if not isinstance(checkpoint, str):
            raise ReportError(
                f"evaluations[{report_index}].checkpoint must be a string"
            )
        if checkpoint in seen_checkpoints:
            raise ReportError(f"duplicate checkpoint in report: {checkpoint!r}")
        seen_checkpoints.add(checkpoint)
        iteration = _iteration_from_checkpoint(checkpoint)
        if iteration not in expected_iterations:
            raise ReportError(
                f"unexpected {contract.spec.name} model_{iteration}.pt; expected "
                f"{contract.spec.candidate_iterations}"
            )
        if iteration in seen_iterations:
            raise ReportError(f"duplicate candidate iteration model_{iteration}.pt")
        seen_iterations.add(iteration)
        candidate_directory = _candidate_run_directory(checkpoint, contract)
        if run_directory is None:
            run_directory = candidate_directory
        elif candidate_directory != run_directory:
            raise ReportError("all candidates must come from one exact run directory")
        candidates.append(
            _aggregate_evaluation(
                report, report_index, contract, seed_results
            )
        )
    if seen_iterations != expected_iterations:
        missing = sorted(expected_iterations - seen_iterations)
        raise ReportError(f"candidate batch is incomplete; missing {missing}")

    def rank_key(candidate: dict[str, Any]) -> tuple[float, ...]:
        # User priority: platform steadiness first.  Tracking, actuator load,
        # and earliest iteration are only deterministic secondary keys.
        return (
            candidate["mean_normalized_stability_composite"],
            candidate["maximum_stability_regression_fraction"],
            candidate["mean_normalized_tracking_score"],
            candidate["mean_normalized_load_score"],
            candidate["iteration"],
        )

    accepted = sorted(
        (candidate for candidate in candidates if candidate["accepted"]),
        key=rank_key,
    )
    return {
        "schema_version": 1,
        "stage": contract.spec.name,
        "task": contract.spec.task_id,
        "experiment": contract.spec.experiment,
        "exact_run_name": contract.run_name,
        "contract": [
            {"label": label, "command": list(command)}
            for label, command in command_contract(contract.spec)
        ],
        "candidate_iterations": list(contract.spec.candidate_iterations),
        "minimum_signed_lateral_fraction": (
            contract.spec.minimum_signed_lateral_fraction
        ),
        "thresholds": THRESHOLDS,
        "seed_reference": seed_reference,
        "candidate_count": len(candidates),
        "accepted_checkpoint_count": len(accepted),
        "best_accepted_checkpoint": (
            accepted[0]["checkpoint"] if accepted else None
        ),
        "accepted_checkpoints_ranked": [
            candidate["checkpoint"] for candidate in accepted
        ],
        "evaluations": candidates,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Evaluator batch JSON")
    parser.add_argument("--stage", required=True, choices=tuple(STAGE_SPECS))
    parser.add_argument("--run-name", required=True, help="Exact candidate directory name")
    parser.add_argument(
        "--seed-file", required=True, help="Host path to immutable staged seed"
    )
    parser.add_argument(
        "--seed-checkpoint",
        required=True,
        help="Exact checkpoint path recorded by evaluator inside its container",
    )
    parser.add_argument("--seed-sha256", required=True)
    parser.add_argument("--json", dest="json_path", help="Write grading JSON here")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    contract = make_contract(
        args.stage, args.run_name, args.seed_checkpoint, args.seed_sha256
    )
    seed_file = Path(args.seed_file).expanduser().resolve(strict=True)
    if not seed_file.is_file():
        raise ReportError(f"seed_file is not a regular file: {seed_file}")
    if seed_file.name != Path(contract.seed_checkpoint).name:
        raise ReportError("seed_file and seed_checkpoint basenames differ")
    actual_sha = _sha256(seed_file)
    if actual_sha != contract.seed_sha256:
        raise ReportError(
            f"immutable seed SHA mismatch: expected {contract.seed_sha256}, "
            f"got {actual_sha}"
        )

    input_path = Path(args.input).expanduser().resolve(strict=True)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    grade = grade_payload(payload, contract)
    for evaluation in grade["evaluations"]:
        status = "PASS" if evaluation["accepted"] else "FAIL"
        print(
            f"{status} checkpoint={evaluation['checkpoint']} "
            f"stability={evaluation['mean_normalized_stability_composite']:.4f} "
            f"tracking={evaluation['mean_normalized_tracking_score']:.4f} "
            f"falls={evaluation['total_falls']}"
        )
        for result in evaluation["results"]:
            for reason in result["rejection_reasons"]:
                print(f"  {result['label']}: {reason}")
    print(f"best_accepted_checkpoint={grade['best_accepted_checkpoint']}")

    if args.json_path:
        output_path = Path(args.json_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output_path.open("x", encoding="utf-8") as stream:
                json.dump(grade, stream, indent=2, sort_keys=True)
                stream.write("\n")
        except FileExistsError as exc:
            raise ReportError(
                f"refusing to overwrite grading report: {output_path}"
            ) from exc
        print(f"grading_report={output_path}")

    after_sha = _sha256(seed_file)
    if after_sha != contract.seed_sha256:
        raise ReportError("immutable seed changed during grading")
    return 0 if grade["accepted_checkpoint_count"] > 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReportError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"GRADE_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(65)
