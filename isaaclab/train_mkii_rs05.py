#!/usr/bin/env python3
"""Scratch-only entry point for PPO training of the RS05 direct task.

This is an in-container trainer adapter, not a GPU scheduler. Invoke it inside
the project's workload-gated launcher, after the standing capture of this task
passes the unchanged scorer. The upstream Isaac Lab trainer keeps ownership of
AppLauncher, the simulator imports and PPO; this wrapper selects and registers
the task and supplies the source paths the Spark bind mount needs.

Measure transitions per second at ``--num_envs 1024`` and record that number
before running at the 4096 default. The larger scale changes memory, contact
counts and step time, so the smaller measurement decides whether 4096 fits the
host and how long an update takes.

The entry point refuses every checkpoint and resume option. A future resume
path needs explicit checkpoint, asset and action-manifest compatibility checks;
accepting a checkpoint filename here would adopt the historical mock, CAD or
paper-walk lineage in silence. ``--dry-run`` prints the plan with no gym, Isaac
Sim, torch, registration or training.
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
MKII_RS05_TASK_ID = "Isaac-Velocity-Flat-Hexapod-MKII-RS05-Direct-v0"
MKII_RS05_USD_PATH_CONTAINER = (
    "/workspace/hexapod/artifacts/mkii_updated_2026-09-10/usd_002/rs05_mass_corrected/robot.usda"
)
DEFAULT_NUM_ENVS = 4096


def prepare_training_args(argv: Sequence[str]) -> list[str]:
    """Bind the task, add the scale defaults and refuse checkpoint adoption."""

    values = list(argv)
    task_values = []
    index = 0
    checkpoint_keys = {"resume", "checkpoint", "load-run", "load-checkpoint"}
    protected_keys = checkpoint_keys | {"task"}
    has_num_envs = False
    has_headless = False
    while index < len(values):
        value = values[index]
        if value == "--task":
            if index + 1 >= len(values) or values[index + 1].startswith("-"):
                raise ValueError("--task needs the RS05 task ID")
            task_values.append(values[index + 1])
            index += 2
            continue
        if value.startswith("--task="):
            task_values.append(value.split("=", 1)[1])
            index += 1
            continue
        if value == "--headless":
            has_headless = True
        # Cover stock CLI and ordinary Hydra overrides (+/++ included).
        key = value.split("=", 1)[0].lstrip("-+")
        key = key.replace("_", "-")
        # The upstream trainer may use argparse's default allow_abbrev=True.
        # Reject every proper prefix of a protected option, including one that
        # is ambiguous between them; complete unrelated flags stay unchanged.
        if value.startswith("--") and key and any(
            option.startswith(key) and option != key for option in protected_keys
        ):
            raise ValueError(
                "Abbreviated protected options are not accepted by the RS05 "
                "scratch-only entry point; use the complete --task option"
            )
        leaf = key.rsplit(".", 1)[-1]
        if leaf in checkpoint_keys:
            raise ValueError(
                "The RS05 entry point is scratch-only; checkpoint and resume options "
                "require a future compatibility-checked resume entry point"
            )
        if leaf == "num-envs":
            has_num_envs = True
        if "=" in value and key in {"task", "env-task", "env.task"}:
            raise ValueError("Use --task for task selection; task overrides are not accepted")
        index += 1
    if len(task_values) > 1:
        raise ValueError("Pass --task at most once")
    if task_values and task_values[0] != MKII_RS05_TASK_ID:
        raise ValueError(f"This entry point requires --task {MKII_RS05_TASK_ID}")
    defaults = []
    if not has_num_envs:
        defaults += ["--num_envs", str(DEFAULT_NUM_ENVS)]
    if not has_headless:
        defaults.append("--headless")
    selection = [] if task_values else ["--task", MKII_RS05_TASK_ID]
    return [*selection, *defaults, *values]


def source_paths() -> list[str]:
    """Package roots needed with only PYTHONPATH=/workspace/hexapod/isaaclab."""

    return [
        str(REPO_ROOT / "isaaclab"),
        str(REPO_ROOT / "packages/hexapod_env"),
        str(REPO_ROOT / "packages/hexapod_core"),
    ]


def launch_plan(train_script: Path, train_args: Sequence[str]) -> dict:
    """Describe the future invocation. This is no admission and no GPU claim."""

    return {
        "execution": "not_started",
        "mode": "scratch_only",
        "task_id": MKII_RS05_TASK_ID,
        "train_script": str(train_script),
        "train_script_exists": train_script.is_file(),
        "argv": [str(train_script), *train_args],
        "source_paths": source_paths(),
        "default_num_envs": DEFAULT_NUM_ENVS,
        "usd_path_container": MKII_RS05_USD_PATH_CONTAINER,
        "usd_override_env_var": "HEXAPOD_MKII_RS05_USD_PATH",
        "required_before_execution": [
            "the standing capture of this task passes the unchanged scorer at the intended replica count",
            "transitions per second are measured and recorded at --num_envs 1024 before the 4096 default",
            "the project workload gate admits the job and the Spark reservation, GPU lock and process ownership are verified",
        ],
        "scope": "A launch description; it qualifies no behavior and starts no job.",
    }


def run_training_script(train_script: Path, train_args: Sequence[str]) -> None:
    """Register strings only, then delegate the simulator startup upstream."""

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
        # This module imports neither Isaac Sim nor the environment config. The
        # registration stores entry-point strings for the upstream trainer to
        # resolve at its own simulator and config import boundary.
        from hexapod_env.tasks.mkii_rs05.register import MKII_RS05_FLAT_TASK_ID, register_mkii_rs05

        if MKII_RS05_FLAT_TASK_ID != MKII_RS05_TASK_ID:
            raise RuntimeError("The RS05 wrapper and registration task IDs disagree")
        register_mkii_rs05()
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
