"""Judge recorded evaluation traces against the controller rows of the walking requirements.

``configs/walking_requirements.json`` owns two limits a fixed-command trace can measure:

- stopping distance: how far the body travels after the command becomes zero;
- drift per hold: how far the body strays from the path its command would trace over one hold.

The frozen scorer (``scoring.py``) keeps its gates and bytes; this module reads the same
``control_trace.npz`` files and reports against the requirements file and its hash. Each result
names the requirement set, the file's SHA-256 and the checkpoint the traces came from.

Run: ``python -m locomotion.evaluation.requirements <evaluation dir> [--set stage2_flat]``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REQUIREMENTS = ROOT / "configs/walking_requirements.json"
CONTROL_DT_S = 0.02


def load_requirements(path=REQUIREMENTS, name="stage2_flat"):
    """The named requirement set and the SHA-256 of the file that owns it."""
    data = json.loads(Path(path).read_text())
    return data["sets"][name], hashlib.sha256(Path(path).read_bytes()).hexdigest()


def planar_pose(root_pose_xyzw):
    """World x, y and yaw of root pose rows ``(x, y, z, qx, qy, qz, qw)``."""
    p = np.asarray(root_pose_xyzw, dtype=np.float64)
    x, y, z, w = p[:, 3], p[:, 4], p[:, 5], p[:, 6]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack((p[:, 0], p[:, 1], yaw), -1)


def ghost_path(start, commands, dt=CONTROL_DT_S):
    """Where the commands lead from ``start`` ``(x, y, yaw)``, one row per control, the start first.

    Commands are navigation-frame ``(forward, left, yaw rate)``; navigation (x, y) is body (-y, x),
    the arithmetic of ``omniwalk.mdp.destination.advance_destination``.
    """
    commands = np.asarray(commands, dtype=np.float64)
    out = np.empty((len(commands) + 1, 3))
    out[0] = start
    for i, (vx, vy, wz) in enumerate(commands):
        x, y, yaw = out[i]
        bx, by = vy, -vx
        middle = yaw + wz * dt / 2
        c, s = math.cos(middle), math.sin(middle)
        out[i + 1] = (x + (c * bx - s * by) * dt, y + (s * bx + c * by) * dt, yaw + wz * dt)
    return out


def hold_spans(commands):
    """``(first, last_exclusive, command)`` for each run of equal nonzero commands."""
    commands = np.asarray(commands, dtype=np.float64)
    spans, start = [], 0
    for i in range(1, len(commands) + 1):
        if i == len(commands) or np.any(commands[i] != commands[start]):
            if np.any(commands[start] != 0):
                spans.append((start, i, commands[start].copy()))
            start = i
    return spans


def drift_per_hold(trace):
    """Per hold: the largest and the final planar gap between the body and its ghost, in metres."""
    pose = planar_pose(trace["root_pose_xyzw"][:, 0, :])
    commands = np.asarray(trace["command"][:, 0, :], dtype=np.float64)
    rows = []
    for first, last, command in hold_spans(commands):
        ghost = ghost_path(pose[first], commands[first:last])
        gap = np.hypot(*(pose[first:last, :2] - ghost[: last - first, :2]).T)
        rows.append(
            {
                "first_control": int(first),
                "controls": int(last - first),
                "command": [float(v) for v in command],
                "max_gap_m": float(gap.max()),
                "end_gap_m": float(gap[-1]),
            }
        )
    return rows


def stopping_distance(trace, settle_s):
    """Planar travel after the first zero command that follows a nonzero one; None if there is no stop."""
    pose = planar_pose(trace["root_pose_xyzw"][:, 0, :])
    commands = np.asarray(trace["command"][:, 0, :], dtype=np.float64)
    moving = np.any(commands != 0, axis=1)
    stops = np.nonzero(moving[:-1] & ~moving[1:])[0] + 1
    if len(stops) == 0:
        return None
    k = int(stops[0])
    settle = min(len(pose) - 1, k + int(round(settle_s / CONTROL_DT_S)))
    travel = np.hypot(*(pose[k:, :2] - pose[k, :2]).T)
    return {
        "stop_control": k,
        "settle_distance_m": float(travel[settle - k]),
        "final_distance_m": float(travel[-1]),
        "max_distance_m": float(travel.max()),
    }


def judge(evaluation_dir, requirements=None, name="stage2_flat"):
    """Every trace under ``evaluation_dir`` against the controller rows of one requirement set."""
    rules, requirements_sha = load_requirements(requirements or REQUIREMENTS, name)
    limits = rules["controller"]
    settle_s = rules["stop"]["settle_s"]
    cases, checkpoints = [], set()
    for report_path in sorted(Path(evaluation_dir).glob("batch_*/report.json")):
        report = json.loads(report_path.read_text())
        checkpoints.add(report.get("checkpoint_sha256"))
        trace = np.load(report_path.with_name("control_trace.npz"))
        case = report["cases"][0]
        holds = drift_per_hold(trace)
        stop = stopping_distance(trace, settle_s)
        checks = {}
        walks = [
            h
            for h in holds
            if h["command"][2] == 0
            and abs(math.hypot(*h["command"][:2]) - limits["drift_speed_mps"]) < 1e-6
            and h["controls"] * CONTROL_DT_S >= limits["drift_hold_s"] - 1e-9
        ]
        if walks:
            worst = max(h["max_gap_m"] for h in walks)
            checks["drift_per_hold_m"] = {
                "value": worst,
                "bound": limits["drift_per_hold_max_m"],
                "pass": worst <= limits["drift_per_hold_max_m"],
            }
        if stop is not None:
            checks["stopping_distance_m"] = {
                "value": stop["final_distance_m"],
                "bound": limits["stopping_distance_max_m"],
                "pass": stop["final_distance_m"] <= limits["stopping_distance_max_m"],
            }
        cases.append(
            {
                "case_id": case["case_id"],
                "command": case["command"],
                "holds": holds,
                "stop": stop,
                "checks": checks,
                "pass": all(c["pass"] for c in checks.values()) if checks else None,
            }
        )
    judged = [c for c in cases if c["checks"]]
    return {
        "schema": "hexapod_walking_requirements_judgement_v1",
        "requirements_set": name,
        "requirements_sha256": requirements_sha,
        "checkpoint_sha256": sorted(c for c in checkpoints if c),
        "cases": cases,
        "judged_cases": len(judged),
        "pass": all(c["pass"] for c in judged) if judged else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("evaluation_dir")
    parser.add_argument("--set", default="stage2_flat")
    parser.add_argument("--requirements", default=str(REQUIREMENTS))
    args = parser.parse_args(argv)
    print(json.dumps(judge(args.evaluation_dir, args.requirements, args.set), indent=2))


if __name__ == "__main__":
    main()
