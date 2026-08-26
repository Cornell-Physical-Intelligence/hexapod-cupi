#!/usr/bin/env python3
"""Fail-closed merge of two deterministic Stage-2C probe evaluator shards.

The sharded launcher repeats one immutable parent checkpoint in two independent
Isaac processes.  This utility accepts the shards only when both processes
produce exactly the same parent report and every report satisfies the complete
nominal Stage-2C playback contract.  Candidate membership and order come from
the caller, never from filenames discovered in the reports.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import grade_stage2c_stable_forward as admission


EXPECTED_ACTION_PROCESSING: dict[str, float | str] = {
    "processed_joint_target_slew_limit_rad_per_20ms": 0.04,
    "source": "cli_override",
}

_REPORT_METADATA: tuple[tuple[str, object], ...] = (
    ("task", admission.TASK_ID),
    ("command_frame", "navigation"),
    ("deterministic_policy", True),
    ("seed", admission.EXPECTED_SEED),
    ("policy_step_seconds", admission.EXPECTED_POLICY_STEP_SECONDS),
    ("requested_steps", admission.EXPECTED_REQUESTED_STEPS),
    ("warmup_steps", admission.EXPECTED_WARMUP_STEPS),
    ("commands_evaluated_in_parallel", len(admission.COMMAND_CONTRACT)),
    ("reset_stance_override", None),
    ("startup_randomization_enabled", False),
)


class MergeError(ValueError):
    """Raised when a shard violates the deterministic merge contract."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MergeError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise MergeError(f"non-finite JSON number {value!r} is not permitted")


def load_payload(path: Path) -> Any:
    """Read strict JSON, rejecting duplicate keys and non-finite constants."""

    if path.is_symlink() or not path.is_file():
        raise MergeError(f"shard is missing, not regular, or a symlink: {path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_json,
        )
    except OSError as exc:
        raise MergeError(f"cannot read shard {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MergeError(f"invalid JSON in shard {path}: {exc}") from exc


def _same_typed_value(actual: Any, expected: object) -> bool:
    """Compare metadata without Python's bool/int equality aliasing."""

    if expected is None:
        return actual is None
    return type(actual) is type(expected) and actual == expected


def _validate_action_processing(value: Any, path: str) -> None:
    if not isinstance(value, dict):
        raise MergeError(f"{path} must be an object")
    if set(value) != set(EXPECTED_ACTION_PROCESSING):
        raise MergeError(
            f"{path} must contain exactly {sorted(EXPECTED_ACTION_PROCESSING)}, "
            f"got {sorted(value)}"
        )
    resolved = value["processed_joint_target_slew_limit_rad_per_20ms"]
    if (
        type(resolved) is not float
        or not math.isfinite(resolved)
        or resolved
        != EXPECTED_ACTION_PROCESSING[
            "processed_joint_target_slew_limit_rad_per_20ms"
        ]
    ):
        raise MergeError(
            f"{path}.processed_joint_target_slew_limit_rad_per_20ms must be "
            "the exact float 0.04"
        )
    if value["source"] != EXPECTED_ACTION_PROCESSING["source"]:
        raise MergeError(f"{path}.source must be 'cli_override'")


def _validate_report(report: Any, path: str, expected_checkpoint: str) -> None:
    if not isinstance(report, dict):
        raise MergeError(f"{path} must be an object")
    checkpoint = report.get("checkpoint")
    if type(checkpoint) is not str or checkpoint != expected_checkpoint:
        raise MergeError(
            f"{path}.checkpoint must be {expected_checkpoint!r}, got {checkpoint!r}"
        )
    for key, expected in _REPORT_METADATA:
        actual = report.get(key)
        if not _same_typed_value(actual, expected):
            raise MergeError(
                f"{path}.{key} must be {expected!r} with type "
                f"{type(expected).__name__}, got {actual!r}"
            )
    _validate_action_processing(report.get("action_processing"), f"{path}.action_processing")

    # Reuse the canonical command/index validator.  This rejects missing,
    # duplicate, reordered-index, or extra command rows while leaving numerical
    # metric grading to analyze_stage2c_probe_sweep.py.
    try:
        rows = admission._ordered_contract_rows(report, 0)
    except admission.ReportError as exc:
        raise MergeError(f"{path}.results violates command contract: {exc}") from exc
    for row_index, row in enumerate(rows):
        samples = row.get("samples")
        measured_seconds = row.get("measured_seconds")
        if type(samples) is not int or samples != admission.EXPECTED_SAMPLES:
            raise MergeError(
                f"{path}.results[{row_index}].samples must be "
                f"{admission.EXPECTED_SAMPLES}, got {samples!r}"
            )
        if (
            type(measured_seconds) is not float
            or not math.isfinite(measured_seconds)
            or measured_seconds != admission.EXPECTED_MEASURED_SECONDS
        ):
            raise MergeError(
                f"{path}.results[{row_index}].measured_seconds must be the exact "
                f"float {admission.EXPECTED_MEASURED_SECONDS}, got "
                f"{measured_seconds!r}"
            )


def _validate_payload(
    payload: Any,
    *,
    shard_name: str,
    expected_checkpoints: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise MergeError(f"{shard_name} top-level JSON must be an object")
    if set(payload) != {"action_processing", "evaluations"}:
        raise MergeError(
            f"{shard_name} top-level keys must be exactly "
            "['action_processing', 'evaluations']"
        )
    _validate_action_processing(
        payload["action_processing"], f"{shard_name}.action_processing"
    )
    reports = payload["evaluations"]
    if not isinstance(reports, list) or len(reports) != len(expected_checkpoints):
        actual_count = len(reports) if isinstance(reports, list) else "not a list"
        raise MergeError(
            f"{shard_name}.evaluations must contain exactly "
            f"{len(expected_checkpoints)} reports, got {actual_count}"
        )
    for report_index, (report, checkpoint) in enumerate(
        zip(reports, expected_checkpoints)
    ):
        _validate_report(
            report,
            f"{shard_name}.evaluations[{report_index}]",
            checkpoint,
        )
    return reports


def merge_payloads(
    shard_a_payload: Any,
    shard_b_payload: Any,
    *,
    parent_checkpoint: str,
    candidate_checkpoints: list[str],
) -> dict[str, Any]:
    """Validate two shards and return parent-once reports in caller order."""

    if type(parent_checkpoint) is not str or not parent_checkpoint:
        raise MergeError("parent checkpoint must be a non-empty string")
    if len(candidate_checkpoints) < 2:
        raise MergeError("at least two candidate checkpoints are required")
    if any(type(path) is not str or not path for path in candidate_checkpoints):
        raise MergeError("every candidate checkpoint must be a non-empty string")
    if len(set(candidate_checkpoints)) != len(candidate_checkpoints):
        raise MergeError("candidate checkpoint list contains duplicates")
    if parent_checkpoint in candidate_checkpoints:
        raise MergeError("immutable parent must not also appear as a candidate")

    expected_a = [parent_checkpoint, *candidate_checkpoints[0::2]]
    expected_b = [parent_checkpoint, *candidate_checkpoints[1::2]]
    reports_a = _validate_payload(
        shard_a_payload, shard_name="shard_a", expected_checkpoints=expected_a
    )
    reports_b = _validate_payload(
        shard_b_payload, shard_name="shard_b", expected_checkpoints=expected_b
    )
    if reports_a[0] != reports_b[0]:
        raise MergeError(
            "immutable parent reports differ across shards; refusing causal merge"
        )

    report_by_checkpoint: dict[str, dict[str, Any]] = {}
    for report in (*reports_a[1:], *reports_b[1:]):
        checkpoint = report["checkpoint"]
        if checkpoint in report_by_checkpoint:
            raise MergeError(f"duplicate candidate report {checkpoint!r}")
        report_by_checkpoint[checkpoint] = report
    if set(report_by_checkpoint) != set(candidate_checkpoints):
        missing = sorted(set(candidate_checkpoints) - set(report_by_checkpoint))
        unexpected = sorted(set(report_by_checkpoint) - set(candidate_checkpoints))
        raise MergeError(
            f"candidate report membership differs: missing={missing}, "
            f"unexpected={unexpected}"
        )

    return {
        "action_processing": dict(EXPECTED_ACTION_PROCESSING),
        "evaluations": [
            reports_a[0],
            *(report_by_checkpoint[path] for path in candidate_checkpoints),
        ],
    }


def write_json_exclusive_atomic(path: Path, payload: Any) -> None:
    """Publish complete JSON atomically without ever replacing an existing path."""

    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="x",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            json.dump(payload, output_file, indent=2, sort_keys=True, allow_nan=False)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        # Hard-link publication is atomic and fails with EEXIST rather than
        # replacing a concurrently created output.  The temporary file lives in
        # the same directory, so the operation never crosses filesystems.
        os.link(temporary_path, path)
        temporary_path.unlink()
        temporary_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-a", required=True, type=Path)
    parser.add_argument("--shard-b", required=True, type=Path)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument(
        "--candidate-checkpoint",
        required=True,
        action="append",
        dest="candidate_checkpoints",
    )
    parser.add_argument("--json", required=True, dest="json_path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        payload = merge_payloads(
            load_payload(args.shard_a),
            load_payload(args.shard_b),
            parent_checkpoint=args.parent_checkpoint,
            candidate_checkpoints=args.candidate_checkpoints,
        )
        write_json_exclusive_atomic(args.json_path, payload)
    except FileExistsError:
        print(f"refusing to overwrite merged report: {args.json_path}", file=sys.stderr)
        return 73
    except (MergeError, OSError, ValueError) as exc:
        print(f"merge error: {exc}", file=sys.stderr)
        return 65
    print(f"merged_json={args.json_path.expanduser().absolute()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
