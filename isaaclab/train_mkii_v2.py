#!/usr/bin/env python3
"""Opt-in entry point for future scratch training of the corrected CAD v2.

This is an in-container trainer adapter, not a GPU scheduler: invoke it only
inside the project's workload-gated launcher after the corrected asset and
standing acceptance gates pass. --dry-run prints an invocation without gym,
Isaac Sim, torch, registration, or any training execution.

The upstream Isaac Lab trainer keeps ownership of AppLauncher, simulator
imports and PPO. This wrapper only selects/registers the new task and supplies
source-package paths that the Spark's non-installed bind mount needs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_SCRIPT = Path(
    os.environ.get(
        "ISAACLAB_RSL_RL_TRAIN_SCRIPT",
        "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py",
    )
)
MKII_V2_TASK_ID = "Isaac-Velocity-Flat-Hexapod-MKII-V2-Direct-v0"


def prepare_training_args(argv: Sequence[str]) -> list[str]:
    """Bind the task and refuse checkpoint adoption without a v2 lineage gate.

    This first entry point is intentionally scratch-only. A future resume
    path needs explicit checkpoint/asset/action-manifest compatibility checks;
    merely accepting a checkpoint filename here would silently adopt the
    archived mock or incorrect-inertia CAD lineage.
    """
    values = list(argv)
    task_values = []
    index = 0
    checkpoint_keys = {"resume", "checkpoint", "load-run", "load-checkpoint"}
    protected_keys = checkpoint_keys | {"task"}
    while index < len(values):
        value = values[index]
        if value == "--task":
            if index + 1 >= len(values) or values[index + 1].startswith("-"):
                raise ValueError("--task needs the CAD v2 task ID")
            task_values.append(values[index + 1])
            index += 2
            continue
        if value.startswith("--task="):
            task_values.append(value.split("=", 1)[1])
            index += 1
            continue
        # Cover stock CLI and ordinary Hydra overrides (+/++ included).
        key = value.split("=", 1)[0].lstrip("-+")
        key = key.replace("_", "-")
        # The upstream trainer may use argparse's default allow_abbrev=True.
        # Our wrapper parser setting does not propagate to that parser, so
        # --tas, --resu and --load_c must never reach it. Reject all proper
        # prefixes, including prefixes ambiguous between protected options;
        # complete unrelated flags such as --rendering_mode remain unchanged.
        if value.startswith("--") and key and any(
            option.startswith(key) and option != key for option in protected_keys
        ):
            raise ValueError(
                "Abbreviated protected options are not accepted by the CAD v2 "
                "scratch-only entry point; use the complete --task option"
            )
        leaf = key.rsplit(".", 1)[-1]
        if leaf in checkpoint_keys:
            raise ValueError(
                "CAD v2 entry point is scratch-only; checkpoint/resume options "
                "require a future compatibility-checked resume entry point"
            )
        if "=" in value and key in {"task", "env-task", "env.task"}:
            raise ValueError("Use --task for task selection; task overrides are not accepted")
        index += 1
    if len(task_values) > 1:
        raise ValueError("Pass --task at most once")
    if task_values and task_values[0] != MKII_V2_TASK_ID:
        raise ValueError(f"This entry point requires --task {MKII_V2_TASK_ID}")
    return values if task_values else ["--task", MKII_V2_TASK_ID, *values]


def source_paths() -> list[str]:
    """Package roots needed with only PYTHONPATH=/workspace/hexapod/isaaclab."""
    return [
        str(REPO_ROOT / "isaaclab"),
        str(REPO_ROOT / "packages/hexapod_env"),
        str(REPO_ROOT / "packages/hexapod_core"),
    ]


def launch_plan(train_script: Path, train_args: Sequence[str]) -> dict:
    """Describe the future invocation; this is not an asset or GPU admission."""
    return {
        "execution": "not_started",
        "mode": "scratch_only",
        "task_id": MKII_V2_TASK_ID,
        "train_script": str(train_script),
        "train_script_exists": train_script.is_file(),
        "argv": [str(train_script), *train_args],
        "source_paths": source_paths(),
        "usd_path_container": (
            "/workspace/hexapod/robot/hexapod_mkii_assy/usd/"
            "hexapod_mkii_serial_v2/hexapod_mkii_serial_v2.usda"
        ),
        "usd_override_env_var": "HEXAPOD_MKII_V2_USD_PATH",
        "required_before_execution": [
            "corrected URDF/USD integrity and reset geometry gates pass",
            "new CAD v2 standing/PhysX acceptance passes and joint order is recorded",
            "project workload gate admits the job without competing with unrelated GPU work",
        ],
    }


def run_training_script(train_script: Path, train_args: Sequence[str]) -> None:
    """Register strings only, then delegate AppLauncher/import ordering upstream."""
    train_args = prepare_training_args(train_args)
    train_script = train_script.expanduser().resolve()
    if not train_script.is_file():
        raise FileNotFoundError(train_script)
    if train_script == Path(__file__).resolve():
        raise ValueError("The upstream train script cannot be this wrapper")
    original_argv = sys.argv
    original_sys_path = list(sys.path)
    try:
        paths = [*source_paths(), str(train_script.parent)]
        sys.path[:0] = [path for path in paths if path not in sys.path]
        # This module imports neither Isaac Sim nor the environment config.
        # gym registration stores entry-point strings for the upstream trainer
        # to resolve at its normal simulator/config import boundary.
        from hexapod_env.tasks.mkii_v2.register import MKII_V2_FLAT_TASK_ID, register_mkii_v2

        if MKII_V2_FLAT_TASK_ID != MKII_V2_TASK_ID:
            raise RuntimeError("CAD v2 wrapper and registration task IDs disagree")
        register_mkii_v2()
        sys.argv = [str(train_script), *train_args]
        runpy.run_path(str(train_script), run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.path[:] = original_sys_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--dry-run", action="store_true", help="Print the launch plan without imports or execution")
    args, downstream = parser.parse_known_args(list(argv) if argv is not None else sys.argv[1:])
    train_args = prepare_training_args(downstream)
    train_script = args.train_script.expanduser().resolve()
    if args.dry_run:
        print(json.dumps(launch_plan(train_script, train_args), indent=2))
        return 0
    run_training_script(train_script, train_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
