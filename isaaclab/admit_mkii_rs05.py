#!/usr/bin/env python3
"""Record a standing capture of the RS05 direct task for the frozen scorer.

The runner holds every replica at its declared neutral stance for 1000 control
steps and writes the 8000 substep rows in the layout
``experiments/paper_walk/env.py:score_diagnostic`` reads. It grades nothing:
after a run, the unchanged scorer reads the directory and decides.

Run it inside the project's workload-gated launcher on the Spark host. It needs
the GPU simulator, so it starts nothing on a machine without Isaac Sim.
``--dry-run`` prints the plan with no gym, Isaac Sim or torch import.

The capture episode outlasts the scored window, so no reset happens inside it.
A capture at one replica and a capture at N replicas are separate results with
separate identities.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
MKII_RS05_TASK_ID = "Isaac-Velocity-Flat-Hexapod-MKII-RS05-Direct-v0"
MKII_RS05_USD_SHA256 = "3be98420adc7923923757d6557d8492d5930f84128c1b1366b9c6db1bb03207c"
#: The frozen scorer reads exactly 1000 control steps and 8000 substeps.
SCORER_CONTROLS = 1000
SCORER_SETTLE_CONTROLS = 200
CONTROL_DT_S = 0.02
#: Extra control steps of episode length, so the scored window holds no reset.
EPISODE_MARGIN_CONTROLS = 50


def source_paths() -> list[str]:
    """Package roots needed with only PYTHONPATH=/workspace/hexapod/isaaclab."""

    return [
        str(REPO_ROOT / "isaaclab"),
        str(REPO_ROOT / "packages/hexapod_env"),
        str(REPO_ROOT / "packages/hexapod_core"),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--num-envs", type=int, required=True, help="Replicas in this capture")
    parser.add_argument("--controls", type=int, default=SCORER_CONTROLS)
    parser.add_argument("--settle-controls", type=int, default=SCORER_SETTLE_CONTROLS)
    parser.add_argument("--geometry", type=Path, required=True, help="Prepared geometry.json")
    parser.add_argument("--geometry-extrema", type=Path, required=True, help="Prepared geometry_extrema.npz")
    parser.add_argument("--output", type=Path, required=True, help="New capture directory")
    parser.add_argument("--usd", type=Path, default=None, help="USD override; the asset default otherwise")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without any import or run")
    return parser


def prepare_arguments(argv):
    """Parse and check the arguments. This starts nothing."""

    args = build_parser().parse_args(list(argv))
    if args.num_envs < 1:
        raise ValueError("A capture needs at least one replica")
    if args.controls != SCORER_CONTROLS:
        raise ValueError(f"The frozen scorer reads exactly {SCORER_CONTROLS} control steps")
    if not 0 <= args.settle_controls < args.controls:
        raise ValueError("The settle window must lie inside the capture")
    for name, path in (("geometry", args.geometry), ("geometry extrema", args.geometry_extrema)):
        if not path.is_file():
            raise ValueError(f"The {name} input does not exist: {path}")
    if args.usd is not None and not args.usd.is_file():
        raise ValueError(f"The USD override does not exist: {args.usd}")
    if args.output.exists():
        raise ValueError(f"The capture directory already exists: {args.output}")
    return args


def capture_episode_seconds(args) -> float:
    """Episode length that keeps the scored window free of a reset."""

    return (args.controls + EPISODE_MARGIN_CONTROLS) * CONTROL_DT_S


def launch_plan(args) -> dict:
    """Describe the capture. This is no admission and no running job."""

    return {
        "execution": "not_started",
        "task_id": MKII_RS05_TASK_ID,
        "num_envs": args.num_envs,
        "controls": args.controls,
        "substeps": args.controls * 8,
        "settle_controls": args.settle_controls,
        "episode_length_s": capture_episode_seconds(args),
        "device": args.device,
        "seed": args.seed,
        "headless": bool(args.headless),
        "asset_usd_sha256": MKII_RS05_USD_SHA256,
        "usd_override": None if args.usd is None else str(args.usd),
        "geometry": str(args.geometry),
        "geometry_extrema": str(args.geometry_extrema),
        "output": str(args.output),
        "source_paths": source_paths(),
        "standing_admission": False,
        "grading_command": (
            "python3 -c \"import json,sys;sys.path.insert(0,'.');"
            "from experiments.paper_walk.env import score_diagnostic;"
            f"print(json.dumps(score_diagnostic('{args.output}/standing'),indent=2))\""
        ),
        "required_before_execution": [
            "the Spark host is reachable and its reservation, GPU lock and process ownership are verified",
            "the selected USD matches the recorded asset hash",
            "the capture directory is new and its identity is recorded",
        ],
        "scope": "Recording evidence admits nothing; the unchanged scorer grades the result.",
    }


def run_capture(args) -> dict:
    """Start the simulator and record the capture. Isaac Sim is required."""

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=bool(args.headless), device=args.device)
    application = launcher.app
    sys.path[:0] = [path for path in source_paths() if path not in sys.path]
    import torch

    from hexapod_env.tasks.mkii_rs05.config import HexapodMkiiRs05FlatEnvCfg
    from hexapod_env.tasks.mkii_rs05.env import HexapodMkiiRs05Env

    cfg = HexapodMkiiRs05FlatEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.standing_only = True
    cfg.episode_length_s = capture_episode_seconds(args)
    if args.usd is not None:
        cfg.robot.spawn.usd_path = str(args.usd)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    environment = HexapodMkiiRs05Env(cfg)
    scope = ("Standing capture of the RS05 direct task; the unchanged scorer decides. "
             "No admission, learning or hardware result is implied.")
    try:
        environment.reset()
        environment.begin_diagnostic_capture(
            output / "standing", geometry_path=args.geometry,
            geometry_extrema_path=args.geometry_extrema, scope=scope,
        )
        action = torch.zeros(environment.num_envs, 18, device=environment.device)
        for _ in range(args.controls):
            environment.step(action)
    finally:
        environment.end_diagnostic_capture(scope=scope)
        environment.write_runtime_manifest(output / "runtime_manifest.json")
        environment.close()
        application.close()
    record = dict(launch_plan(args), execution="completed")
    (output / "RUN.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def main(argv=None) -> int:
    values = list(argv) if argv is not None else sys.argv[1:]
    args = prepare_arguments(values)
    if args.dry_run:
        print(json.dumps(launch_plan(args), indent=2))
        return 0
    print(json.dumps(run_capture(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
