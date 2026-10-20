"""Build results.json in this folder from copies of the native evaluation folders.

Pass/fail and check values copy each probe's report.json; the walk measures come from its control_trace.npz:
the planar distance to the destination after the last control (the destination starts at the body and
follows the command exactly, as training builds it), the body-heading change and the path angle from 2 s
to the end, and the mean planar speed over the commanded 0.05 m/s.

Usage: python3 build_results.py <folder holding the evaluation attempts' run/standing/evaluation copies>
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
EVAL = Path(sys.argv[1])
CHAIN = [
    ("pr_model", "Learn to walk clumsily", "eval_v4_stride_c48_u2500_seed20260917_20261011_001"),
    ("destination_v1", "Get the optimal walk (the ghost) as input", "eval_dest_v1_c36_u3500_seed20260917_20261017_001"),
    ("destination_v2", "Tighten the reward for following the ghost", "eval_dest_v2_c36_u3500_seed20260917_20261018_001"),
    ("destination_v8", "Tighten the reward for following the ghost and solve heading and joint problems", "eval_dest_v8_c36_u3500_seed20260917_20261020_001"),
    ("destination_v9", "Small fine-tune to learn to stop", "eval_dest_v9_u600_seed20260917_20261020_001"),
]
OTHERS = [
    ("destination_v3", "eval_dest_v3_c36_u3500_seed20260917_20261019_001"),
    ("destination_v4", "eval_dest_v4_c36_u3500_seed20260917_20261019_002"),
    ("destination_v5", "eval_dest_v5_c36_u3500_seed20260917_20261019_001"),
    ("destination_v7", "eval_dest_v7_c36_u3500_seed20260917_20261020_001"),
    ("destination_v9_update300", "eval_dest_v9_u300_seed20260917_20261020_001"),
]


def yaw(p):
    x, y, z, w = p[..., 3], p[..., 4], p[..., 5], p[..., 6]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def walk(trace):
    pose = trace["root_pose_xyzw"][:, 0].astype(float)
    cmd = trace["command"][:, 0].astype(float)
    g = np.array([pose[0, 0], pose[0, 1], yaw(pose[0])])
    for i in range(len(pose) - 1):
        bx, by = cmd[i, 1], -cmd[i, 0]
        mid = g[2] + cmd[i, 2] * 0.01
        g[:2] += 0.02 * np.array([np.cos(mid) * bx - np.sin(mid) * by, np.sin(mid) * bx + np.cos(mid) * by])
        g[2] += cmd[i, 2] * 0.02
    heading = np.unwrap(yaw(pose[100:]))
    h0 = heading[0]
    body = np.array([cmd[0, 1], -cmd[0, 0]])
    want = np.array([np.cos(h0) * body[0] - np.sin(h0) * body[1], np.sin(h0) * body[0] + np.cos(h0) * body[1]])
    want /= np.linalg.norm(want)
    moved = pose[-1, :2] - pose[100, :2]
    cross = want[0] * moved[1] - want[1] * moved[0]
    return {
        "destination_distance_at_end_mm": round(float(np.hypot(*(pose[-1, :2] - g[:2])) * 1000), 3),
        "facing_change_2s_to_end_deg": round(float(np.degrees(heading[-1] - h0)), 3),
        "path_angle_2s_to_end_deg": round(float(np.degrees(np.arctan2(cross, want @ moved))), 3),
    }


def summarize(folder):
    probes, walks = {}, []
    for batch in sorted((EVAL / folder / "evaluation").glob("batch_*")):
        report = json.loads((batch / "report.json").read_text())
        result = report["results"][0]
        case = report["assigned_case_ids"][0].split(":")[1]
        checks = result["checks"]
        row = {
            "pass": bool(result["pass"]),
            "failed_bounds": {
                k: {"value": checks[k]["value"], "bound": checks[k]["bound"]} if k in checks else None
                for k in result.get("failed_bounds") or []
            },
        }
        if case.startswith("translate"):
            row.update(walk(np.load(batch / "control_trace.npz")))
            row["speed_over_command"] = round(float(np.linalg.norm(result["metrics"]["mean_velocity_mps"][:2]) / 0.05), 4)
            walks.append(row)
        elif case.startswith("yaw"):
            row["mean_yaw_rate_rad_s"] = round(result["metrics"]["mean_yaw_rate_rad_s"], 4)
        else:
            for k in ("max_joint_position_range_rad", "max_target_step_abs_p95_rad_per_20ms", "max_joint_velocity_rms_rad_s"):
                row[k] = {"value": round(checks[k]["value"], 5), "bound": checks[k]["bound"]}
        probes[case] = row
    ends = [w["destination_distance_at_end_mm"] for w in walks]
    facing = [abs(w["facing_change_2s_to_end_deg"]) for w in walks]
    path = [abs(w["path_angle_2s_to_end_deg"]) for w in walks]
    summary_file = EVAL / folder / "evaluation" / "summary.json"
    return {
        "evaluation_attempt": folder,
        "summary_sha256": hashlib.sha256(summary_file.read_bytes()).hexdigest() if summary_file.exists() else None,
        "probes_passed": sum(p["pass"] for p in probes.values()),
        "probes": len(probes),
        "walks": {
            "destination_distance_at_end_mm": {"mean": round(float(np.mean(ends)), 3), "median": round(float(np.median(ends)), 3), "worst": round(max(ends), 3)},
            "facing_change_deg": {"mean": round(float(np.mean(facing)), 3), "worst": round(max(facing), 3)},
            "path_angle_deg": {"mean": round(float(np.mean(path)), 3), "worst": round(max(path), 3)},
        },
        "probe_results": probes,
    }


results = {
    "schema": "hexapod_destination_lineage_results_v1",
    "scope": "Native Isaac learning probes on Spark, one robot per probe, the actor mean without sampling noise, one trial per probe. Pass/fail and check values copy each probe report; the walk measures are computed from each probe's control trace as the module docstring of build_results.py states.",
    "chain": [{"step": i, "experiment": k, "description": d, **summarize(f)} for i, (k, d, f) in enumerate(CHAIN)],
    "other_attempts": [{"experiment": k, **summarize(f)} for k, f in OTHERS if (EVAL / f / "evaluation").exists()],
}
(HERE / "results.json").write_text(json.dumps(results, indent=1) + "\n")
for s in results["chain"] + results["other_attempts"]:
    w = s["walks"]
    print(f"{s['experiment']:26s} {s['probes_passed']}/{s['probes']} end median {w['destination_distance_at_end_mm']['median']} worst {w['destination_distance_at_end_mm']['worst']} facing {w['facing_change_deg']['mean']}/{w['facing_change_deg']['worst']}")
