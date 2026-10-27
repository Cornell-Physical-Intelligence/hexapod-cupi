"""Screen a checkpoint over a fixed-condition grid: one surrogate evaluation per cell.

Each cell runs ``locomotion.surrogate.evaluate --grid --cell`` in its own process under
``<output>/cells/<cell id>``, so any cell can be reproduced alone. ``grid_summary.json`` collects
the verdict of each probe in each cell: the relative-friction consensus for quiet and stop probes,
else the probe's own verdict. ``--continue`` finishes an interrupted sweep and keeps every
incomplete attempt under a fresh name.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCHEMA = "hexapod_surrogate_grid_sweep_v1"


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory.")
    parser.add_argument("--grid", choices=["table2_v1_grid"], default="table2_v1_grid")
    parser.add_argument("--cells", default="all", help="all, or comma-separated cell ids.")
    parser.add_argument(
        "--cases", default="focus", help="focus, all, or comma-separated learning probe ids."
    )
    parser.add_argument("--seed", type=int, help="Defaults to the checkpoint record's seed.")
    parser.add_argument(
        "--video", default="none", help="first, all, none or probe ids, in every cell."
    )
    parser.add_argument("--jobs", type=int, default=1, help="Cells that run at once.")
    parser.add_argument(
        "--threads", type=int, default=1, help="Torch threads of each cell process (1 to 6)."
    )
    parser.add_argument(
        "--continue",
        dest="resume",
        action="store_true",
        help="Reuse --output: keep finished cells and rename incomplete ones before rerunning them.",
    )
    parser.add_argument("--list", action="store_true", help="Print the cell ids and exit.")
    return parser


def cell_command(args, cell):
    command = [
        sys.executable,
        "-m",
        "locomotion.surrogate.evaluate",
        "--checkpoint",
        str(args.checkpoint),
        "--output",
        str(args.output / "cells" / cell),
        "--grid",
        args.grid,
        "--cell",
        cell,
        "--cases",
        args.cases,
        "--video",
        args.video,
        "--threads",
        str(args.threads),
    ]
    if args.seed is not None:
        command += ["--seed", str(args.seed)]
    return command


def run_process(command, log):
    """Run one cell process; its output goes to ``log``. Tests replace this function."""
    with open(log, "w") as stream:
        return subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode


def set_aside(directory):
    """Rename an incomplete attempt to the first free ``<name>.attempt_<n>``; nothing is deleted."""
    index = 1
    while directory.with_name(f"{directory.name}.attempt_{index:03d}").exists():
        index += 1
    target = directory.with_name(f"{directory.name}.attempt_{index:03d}")
    directory.rename(target)
    # Cell ids hold dots, so the log name appends its suffix rather than replacing one.
    log = directory.parent / (directory.name + ".log")
    if log.exists():
        log.rename(target.parent / (target.name + ".log"))
    return target.name


def probe_verdict(row):
    """The consensus verdict of a quiet or stop probe, else the probe's own pass or fail."""
    if "contact_consensus" in row:
        return row["contact_consensus"]["verdict"]
    return "pass" if row["pass"] else "fail"


def cell_row(cell, values, summary):
    friction, payload, strength = values
    row = {
        "cell": cell,
        "foot_friction": friction,
        "payload_mass_kg": payload,
        "motor_strength": strength,
    }
    if summary is None:
        row["status"] = "missing"
        return row
    # The probe suite stops at an acquisition failure; the cell then holds no complete screen.
    failure = summary.get("acquisition_failure")
    row["status"] = "evaluated" if failure is None else "acquisition_failure"
    if failure is not None:
        row["acquisition_failure"] = failure
    row["seed"] = summary["seed"]
    row["condition_draw"] = summary["condition"]["draw"]
    row["cases"] = {
        case["case_id"]: {
            "verdict": probe_verdict(case),
            "failed_bounds": case["failed_bounds"],
            "consensus_failed_bounds": case.get("contact_consensus", {}).get("failed_bounds"),
            "planar_error_mps": case["metrics"].get("planar_error_mps"),
            "mean_velocity_nav_mps": case.get("gait", {}).get("mean_velocity_nav_mps"),
            "max_applied_torque_nm": case["metrics"].get("max_applied_torque_nm"),
        }
        for case in summary["cases"]
    }
    return row


def main(argv=None):
    import locomotion.randomization as rnd
    from locomotion.surrogate.env import repository_state, sha

    args = build_parser().parse_args(argv)
    cells = rnd.grid_cells(args.grid)
    if args.list:
        for cell, _ in cells:
            print(cell)
        return 0
    if args.cells != "all":
        wanted = args.cells.split(",")
        known = dict(cells)
        if any(cell not in known for cell in wanted):
            raise ValueError("Unknown cell id; --list prints them")
        cells = [(cell, known[cell]) for cell in wanted]
    if args.jobs < 1:
        raise ValueError("--jobs needs at least one cell at a time")
    record = args.checkpoint.with_suffix(".json")
    if record.exists():
        expected = json.loads(record.read_text()).get("checkpoint_sha256")
        if expected is not None and sha(args.checkpoint) != expected:
            raise ValueError("Checkpoint bytes differ from the paired record")
    if args.output.exists() and not args.resume:
        raise ValueError("The output exists; pass --continue to finish that sweep")
    (args.output / "cells").mkdir(parents=True, exist_ok=args.resume)
    started = time.monotonic()
    set_aside_attempts, pending = {}, []
    for cell, _ in cells:
        directory = args.output / "cells" / cell
        if (directory / "surrogate_summary.json").exists():
            continue
        if directory.exists():
            set_aside_attempts.setdefault(cell, []).append(set_aside(directory))
        pending.append(cell)

    def run_cell(cell):
        code = run_process(cell_command(args, cell), args.output / "cells" / f"{cell}.log")
        print(json.dumps({"cell": cell, "exit_code": code}), flush=True)
        return cell, code

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        exit_codes = dict(pool.map(run_cell, pending))
    rows = []
    for cell, values in cells:
        path = args.output / "cells" / cell / "surrogate_summary.json"
        row = cell_row(cell, values, json.loads(path.read_text()) if path.exists() else None)
        if cell in exit_codes:
            row["exit_code"] = exit_codes[cell]
        if cell in set_aside_attempts:
            row["set_aside_attempts"] = set_aside_attempts[cell]
        rows.append(row)
    cases = sorted({case for row in rows for case in row.get("cases", {})})
    counts = {
        case: {
            verdict: sum(
                row.get("cases", {}).get(case, {}).get("verdict") == verdict for row in rows
            )
            for verdict in ("pass", "fail", "unresolved")
        }
        for case in cases
    }
    declaration = rnd.grid_declaration(args.grid)
    summary = {
        "schema": SCHEMA,
        "scope": "CPU MuJoCo surrogate screen at fixed conditions; design evidence. One trial per probe "
        "per cell. Native runs are the only acceptance evidence.",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha(args.checkpoint),
        "grid": args.grid,
        "grid_sha256": declaration["sha256"],
        "grid_declaration": json.loads(declaration["json"]),
        "cases_option": args.cases,
        "seed_option": args.seed,
        "seeds": sorted({row["seed"] for row in rows if "seed" in row}),
        "repository": repository_state(),
        "cells_complete": sum(row["status"] == "evaluated" for row in rows),
        "cells_with_acquisition_failure": [
            row["cell"] for row in rows if row["status"] == "acquisition_failure"
        ],
        "cells_requested": len(rows),
        "verdict_counts": counts,
        "cells": rows,
        "wall_seconds_this_invocation": time.monotonic() - started,
    }
    (args.output / "grid_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"cells_complete": summary["cells_complete"], "verdict_counts": counts}))
    return 0 if summary["cells_complete"] == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
