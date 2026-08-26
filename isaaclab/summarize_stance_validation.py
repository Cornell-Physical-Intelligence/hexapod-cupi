#!/usr/bin/env python3
"""Summarize bounded zero-action stance-validation logs.

The Isaac validator intentionally emits simple ``key=value`` lines so this
grader can run on the Spark host without importing Isaac Sim.  A failed or
timed-out candidate is retained in the comparison instead of preventing the
remaining stance results from being reported.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence


APPLIED_TORQUE_LIMIT_NM = 1.61
COMPUTED_TORQUE_PEAK_LIMIT_NM = 5.50
SATURATION_FRACTION_LIMIT = 0.005
MINIMUM_BASE_HEIGHT_M = 0.055
TIMEOUT_EXIT_CODE = 124

FLOAT_METRICS = (
    "stance_root_height_m",
    "stance_femur_angle_rad",
    "stance_tibia_angle_rad",
    "min_base_height_m",
    "mean_base_height_m",
    "post_settle_mean_base_height_m",
    "post_settle_mean_height_std_m",
    "post_settle_max_height_std_m",
    "post_settle_mean_vertical_velocity_rms_mps",
    "post_settle_mean_roll_pitch_rate_rms_radps",
    "post_settle_mean_tilt_rms_deg",
    "post_settle_non_foot_contact_fraction",
    "max_abs_torque_nm",
    "max_abs_computed_torque_nm",
    "post_settle_mean_abs_computed_torque_nm",
    "post_settle_torque_saturation_fraction",
)
INTEGER_METRICS = (
    "joint_count",
    "body_count",
    "foot_count",
    "unexpected_terminations",
    "unexpected_truncations",
)
REQUIRED_METRICS = frozenset(FLOAT_METRICS + INTEGER_METRICS)
KEY_VALUE_PATTERN = re.compile(r"^([a-z][a-z0-9_]*)=([^\s]+)$")


@dataclass(frozen=True)
class CandidateInput:
    """One launched validator process and its requested reset stance."""

    name: str
    root_height_m: float
    femur_angle_rad: float
    tibia_angle_rad: float
    log_path: Path
    exit_code: int


def _finite_float(value: str, key: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{key} must be finite, got {parsed!r}")
    return parsed


def parse_validator_log(path: Path) -> tuple[dict[str, float | int], bool, list[str]]:
    """Return selected metrics, pass-marker presence, and parsing diagnostics."""

    if not path.is_file():
        return {}, False, [f"log file is missing: {path}"]

    raw_values: dict[str, str] = {}
    validation_pass = False
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line == "VALIDATION_PASS":
            validation_pass = True
            continue
        match = KEY_VALUE_PATTERN.fullmatch(line)
        if match:
            raw_values[match.group(1)] = match.group(2)

    metrics: dict[str, float | int] = {}
    diagnostics: list[str] = []
    for key in FLOAT_METRICS:
        if key not in raw_values:
            continue
        try:
            metrics[key] = _finite_float(raw_values[key], key)
        except ValueError as exc:
            diagnostics.append(str(exc))
    for key in INTEGER_METRICS:
        if key not in raw_values:
            continue
        try:
            metrics[key] = int(raw_values[key])
        except ValueError:
            diagnostics.append(f"{key} must be an integer, got {raw_values[key]!r}")

    missing = sorted(REQUIRED_METRICS.difference(metrics))
    if missing:
        diagnostics.append("missing required metrics: " + ", ".join(missing))
    return metrics, validation_pass, diagnostics


def summarize_candidate(candidate: CandidateInput) -> dict[str, Any]:
    """Grade one candidate while preserving metrics from a failed validator."""

    metrics, marker_present, diagnostics = parse_validator_log(candidate.log_path)
    unsafe_reasons: list[str] = []

    if metrics.get("max_abs_torque_nm", 0.0) > APPLIED_TORQUE_LIMIT_NM + 1.0e-9:
        unsafe_reasons.append("applied_torque_exceeded_rs05_continuous_limit")
    if (
        metrics.get("max_abs_computed_torque_nm", 0.0)
        > COMPUTED_TORQUE_PEAK_LIMIT_NM + 1.0e-9
    ):
        unsafe_reasons.append("computed_torque_demand_exceeded_rs05_peak_limit")
    if (
        metrics.get("post_settle_torque_saturation_fraction", 0.0)
        > SATURATION_FRACTION_LIMIT + 1.0e-12
    ):
        unsafe_reasons.append("sustained_computed_torque_demand_exceeded_rs05_rating")
    if metrics.get("post_settle_non_foot_contact_fraction", 0.0) > 0.0:
        unsafe_reasons.append("coxa_femur_or_tibia_shaft_contacted_ground")
    if metrics.get("min_base_height_m", math.inf) < MINIMUM_BASE_HEIGHT_M:
        unsafe_reasons.append("base_height_below_fall_threshold")
    if metrics.get("unexpected_terminations", 0) != 0:
        unsafe_reasons.append("unexpected_termination")
    if metrics.get("unexpected_truncations", 0) != 0:
        unsafe_reasons.append("unexpected_truncation")

    reported_stance = (
        ("stance_root_height_m", candidate.root_height_m),
        ("stance_femur_angle_rad", candidate.femur_angle_rad),
        ("stance_tibia_angle_rad", candidate.tibia_angle_rad),
    )
    for metric_name, requested_value in reported_stance:
        if metric_name in metrics and not math.isclose(
            float(metrics[metric_name]), requested_value, rel_tol=0.0, abs_tol=5.0e-6
        ):
            diagnostics.append(
                f"{metric_name}={metrics[metric_name]!r} does not match requested "
                f"{requested_value!r}"
            )

    if candidate.exit_code == TIMEOUT_EXIT_CODE:
        process_status = "timeout"
    elif unsafe_reasons:
        process_status = "unsafe"
    elif candidate.exit_code != 0 or diagnostics or not marker_present:
        process_status = "error"
    else:
        process_status = "pass"

    error_reasons = list(diagnostics)
    if candidate.exit_code != 0 and candidate.exit_code != TIMEOUT_EXIT_CODE:
        error_reasons.append(f"validator exited with status {candidate.exit_code}")
    if candidate.exit_code == TIMEOUT_EXIT_CODE:
        error_reasons.append("validator exceeded its wall-clock timeout")
    if not marker_present:
        error_reasons.append("VALIDATION_PASS marker is absent")

    return {
        "name": candidate.name,
        "requested_stance": {
            "root_height_m": candidate.root_height_m,
            "femur_angle_rad": candidate.femur_angle_rad,
            "tibia_angle_rad": candidate.tibia_angle_rad,
        },
        "status": process_status,
        "safe": process_status == "pass",
        "validator_exit_code": candidate.exit_code,
        "validation_pass_marker": marker_present,
        "unsafe_reasons": unsafe_reasons,
        "error_reasons": error_reasons,
        "metrics": metrics,
        "log_file": str(candidate.log_path),
    }


def build_summary(
    candidates: Sequence[CandidateInput],
    *,
    steps: int,
    num_envs: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Build the concise machine-readable comparison payload."""

    if not candidates:
        raise ValueError("at least one stance candidate is required")
    results = [summarize_candidate(candidate) for candidate in candidates]
    safe_candidates = [result["name"] for result in results if result["safe"]]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_contract": {
            "zero_action": True,
            "steps": steps,
            "policy_rate_hz": 50,
            "simulated_seconds": steps / 50.0,
            "num_envs": num_envs,
            "per_candidate_wall_timeout_seconds": timeout_seconds,
            "applied_torque_limit_nm": APPLIED_TORQUE_LIMIT_NM,
            "computed_torque_peak_limit_nm": COMPUTED_TORQUE_PEAK_LIMIT_NM,
            "post_settle_saturation_fraction_limit": SATURATION_FRACTION_LIMIT,
        },
        "candidate_count": len(results),
        "safe_candidate_count": len(safe_candidates),
        "all_candidates_safe": len(safe_candidates) == len(results),
        "safe_candidates": safe_candidates,
        "candidates": results,
    }


def _candidate_from_cli(values: Sequence[str]) -> CandidateInput:
    name, root_height, femur_angle, tibia_angle, log_path, exit_code = values
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
        raise ValueError(f"invalid candidate name: {name!r}")
    return CandidateInput(
        name=name,
        root_height_m=_finite_float(root_height, f"{name}.root_height_m"),
        femur_angle_rad=_finite_float(femur_angle, f"{name}.femur_angle_rad"),
        tibia_angle_rad=_finite_float(tibia_angle, f"{name}.tibia_angle_rad"),
        log_path=Path(log_path),
        exit_code=int(exit_code),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, type=Path, help="Comparison JSON output path.")
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--num-envs", required=True, type=int)
    parser.add_argument("--timeout-seconds", required=True, type=int)
    parser.add_argument(
        "--candidate",
        nargs=6,
        action="append",
        required=True,
        metavar=("NAME", "ROOT_Z", "FEMUR", "TIBIA", "LOG", "EXIT_CODE"),
        help="Repeat once per candidate in launch order.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    candidates = [_candidate_from_cli(values) for values in args.candidate]
    if args.steps < 1 or args.num_envs < 1 or args.timeout_seconds < 1:
        raise ValueError("steps, num-envs, and timeout-seconds must be positive")
    summary = build_summary(
        candidates,
        steps=args.steps,
        num_envs=args.num_envs,
        timeout_seconds=args.timeout_seconds,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.json.with_suffix(args.json.suffix + ".tmp")
    temporary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(args.json)

    print("candidate status height_std_mm vz_rms_mps rp_rate_rms_radps nonfoot torque_sat")
    for result in summary["candidates"]:
        metrics = result["metrics"]
        height_std_mm = 1000.0 * float(metrics.get("post_settle_mean_height_std_m", math.nan))
        print(
            f"{result['name']:11s} {result['status']:7s} "
            f"{height_std_mm:13.3f} "
            f"{float(metrics.get('post_settle_mean_vertical_velocity_rms_mps', math.nan)):10.5f} "
            f"{float(metrics.get('post_settle_mean_roll_pitch_rate_rms_radps', math.nan)):18.5f} "
            f"{float(metrics.get('post_settle_non_foot_contact_fraction', math.nan)):7.5f} "
            f"{float(metrics.get('post_settle_torque_saturation_fraction', math.nan)):10.6f}"
        )
    print(f"comparison_json={args.json}")
    return 0 if summary["all_candidates_safe"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
