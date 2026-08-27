"""One documented entry point for the loose evaluation scripts.

The nineteen scripts at `isaaclab/` top level run inside the Isaac container and
are referenced by the hardened launchers by path.  Nothing here moves, wraps, or
reinterprets them: this module owns a mapping from `(action, stage)` to an
existing script path, and `main` replaces the current process with that script
under the same interpreter, forwarding the remaining argv untouched.  No
argument is added, removed, reordered, or validated on the script's behalf, so
the exit code, stdout, and stderr are exactly the script's own.

Usage::

    python -m hexapod_eval <action> --stage <stage> [args ...]
    python -m hexapod_eval --list

`--stage` must be the token immediately after `<action>`; every token after the
stage value is forwarded verbatim.  That rule is what lets a target script have
its own `--stage` flag (Stage2D's `C0`-`C5`, Stage2E's `E0`-`E2`) without
ambiguity::

    python -m hexapod_eval grade --stage stage2d-homotopy batch.json \
        --stage C0 --run-name ... --json out.json
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Mapping, Sequence


# The repository root, resolved from this file: hexapod_eval/hexapod_eval ->
# hexapod_eval -> packages -> repo root.  `HEXAPOD_EVAL_SCRIPT_ROOT` overrides
# it for an installed copy of this package that sits outside the tree.
_DEFAULT_SCRIPT_ROOT = Path(__file__).resolve().parents[3] / "isaaclab"

# (action, stage) -> path relative to the script root.  Every entry names a
# script that already exists and already has its own CLI; this table adds no
# behavior of its own.
_MAPPING: dict[tuple[str, str], str] = {
    ("grade", "stage1-recovery"): "grade_stage1_recovery.py",
    ("grade", "stage2-axis-acquisition"): "grade_stage2_axis_acquisition.py",
    ("grade", "stage2b-lateral-acquisition"): (
        "grade_stage2b_lateral_acquisition.py"
    ),
    ("grade", "stage2c-stable-forward"): "grade_stage2c_stable_forward.py",
    ("grade", "stage2c-robustness"): "grade_stage2c_robustness.py",
    ("grade", "stage2d-homotopy"): "grade_stage2d_homotopy.py",
    ("grade", "stage2e-static"): "grade_stage2e_static.py",
    ("grade", "stage2-command-transitions"): (
        "grade_stage2_command_transitions.py"
    ),
    ("analyze", "stage2c-probe-sweep"): "analyze_stage2c_probe_sweep.py",
    ("merge", "stage2c-probe-shards"): "merge_stage2c_probe_shards.py",
    ("summarize", "stance-validation"): "summarize_stance_validation.py",
    ("evaluate", "checkpoint"): "evaluate_checkpoint.py",
    ("evaluate", "stage2-command-transitions"): (
        "evaluate_stage2_command_transitions.py"
    ),
}

MAPPING: Mapping[tuple[str, str], str] = MappingProxyType(dict(_MAPPING))

USAGE = (
    "usage: python -m hexapod_eval <action> --stage <stage> [args ...]\n"
    "       python -m hexapod_eval --list"
)


class DispatchError(ValueError):
    """Raised when argv does not name a mapped (action, stage) pair."""


def script_root() -> Path:
    """Return the directory holding the evaluation scripts."""

    override = os.environ.get("HEXAPOD_EVAL_SCRIPT_ROOT")
    if override:
        return Path(override).expanduser()
    return _DEFAULT_SCRIPT_ROOT


def script_path(action: str, stage: str) -> Path:
    """Return the absolute path of the script mapped to this pair."""

    try:
        relative = MAPPING[(action, stage)]
    except KeyError:
        raise DispatchError(
            f"no evaluation script is mapped to action={action!r} "
            f"stage={stage!r}\n\n{format_table()}"
        ) from None
    return script_root() / relative


def actions() -> tuple[str, ...]:
    """Return the mapped actions, in table order."""

    seen: list[str] = []
    for action, _ in MAPPING:
        if action not in seen:
            seen.append(action)
    return tuple(seen)


def format_table() -> str:
    """Render the full mapping table."""

    rows = [("ACTION", "STAGE", "SCRIPT")]
    rows.extend(
        (action, stage, f"isaaclab/{relative}")
        for (action, stage), relative in MAPPING.items()
    )
    widths = [max(len(row[column]) for row in rows) for column in range(3)]
    lines = [
        f"{row[0]:<{widths[0]}}  {row[1]:<{widths[1]}}  {row[2]}" for row in rows
    ]
    return "\n".join(lines)


def parse(argv: Sequence[str]) -> tuple[str, str, list[str]]:
    """Split argv into `(action, stage, forwarded)`.

    `--stage` must be the token immediately after the action, either as
    `--stage <value>` or `--stage=<value>`.  Everything after it is forwarded
    untouched, including a second `--stage` belonging to the target script.
    """

    tokens = list(argv)
    if not tokens:
        raise DispatchError(f"missing action\n\n{USAGE}\n\n{format_table()}")
    action = tokens[0]
    if action.startswith("-"):
        raise DispatchError(
            f"first argument must be an action, got {action!r}\n\n{USAGE}\n\n"
            f"{format_table()}"
        )
    rest = tokens[1:]
    if not rest:
        raise DispatchError(
            f"action {action!r} requires --stage\n\n{USAGE}\n\n{format_table()}"
        )
    if rest[0] == "--stage":
        if len(rest) < 2:
            raise DispatchError(
                f"--stage requires a value\n\n{USAGE}\n\n{format_table()}"
            )
        return action, rest[1], rest[2:]
    if rest[0].startswith("--stage="):
        return action, rest[0][len("--stage=") :], rest[1:]
    raise DispatchError(
        f"expected --stage immediately after action {action!r}, got "
        f"{rest[0]!r}\n\n{USAGE}\n\n{format_table()}"
    )


def resolve(argv: Sequence[str]) -> tuple[Path, list[str]]:
    """Return the script to run and the argv to forward to it."""

    action, stage, forwarded = parse(argv)
    return script_path(action, stage), forwarded


def main(argv: Sequence[str] | None = None) -> int:
    """Exec the mapped script, or print the table for `--list`/`--help`."""

    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens or tokens[0] in {"--list", "-l"}:
        print(USAGE)
        print()
        print(format_table())
        return 0 if tokens else 2
    if tokens[0] in {"--help", "-h"}:
        print(USAGE)
        print()
        print(format_table())
        return 0

    try:
        script, forwarded = resolve(tokens)
    except DispatchError as exc:
        print(f"hexapod_eval: {exc}", file=sys.stderr)
        return 2

    if not script.is_file():
        print(
            f"hexapod_eval: mapped script is missing: {script}", file=sys.stderr
        )
        return 2

    # Replace this process so the script owns stdio and the exit code exactly.
    # `sys.executable` keeps the Isaac container's interpreter when the
    # dispatcher is launched with it.
    os.execv(sys.executable, [sys.executable, str(script), *forwarded])
