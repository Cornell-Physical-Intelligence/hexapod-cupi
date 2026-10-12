"""Collect the native training and learning-probe records of the yaw-gated stride reward into native_results.json.

The four training launches and four evaluations ran from source 1b4d67a1 on Spark under
/srv/cupi/hexapod/runs/julian. A standard-library program runs there over SSH, hashes each record
in place and summarizes the training metrics; the learning-probe traces of the walking probes come
back through a tar stream, their hashes must match, and numpy derives the heading drift. The script
starts no native work and adds no acceptance limit.

Run from the repository root:
  uv run python site/assets/yaw_gated_stride_native_20261012_001/collect.py <ssh host>
"""
import hashlib
import io
import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = "/srv/cupi/hexapod/runs/julian"
TRAINING = {
    "c24": ("ppo_v4_stride_c24_u2000_seed20260917_20261010_002", 2000),
    "c36": ("ppo_v4_stride_c36_u1333_seed20260917_20261010_001", 1333),
    "c48": ("ppo_v4_stride_c48_u1000_seed20260917_20261010_001", 1000),
    "c48_long": ("ppo_v4_stride_c48_u2500_seed20260917_20261011_001", 2500),
}
EVALUATIONS = {
    "c24": "eval_v4_stride_c24_u2000_seed20260917_20261011_001",
    "c36": "eval_v4_stride_c36_u1333_seed20260917_20261011_001",
    "c48": "eval_v4_stride_c48_u1000_seed20260917_20261011_001",
    "c48_long": "eval_v4_stride_c48_u2500_seed20260917_20261011_001",
}
FAILED = {"c24_permission": "ppo_v4_stride_c24_u2000_seed20260917_20261010_001"}
OUTPUT = Path(__file__).with_name("native_results.json")

REMOTE = r'''
import hashlib, json, sys
from pathlib import Path
root, training, evaluations, failed = Path(sys.argv[1]), *(json.loads(a) for a in sys.argv[2:5])
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
def pack(name):
    a = root / name
    exit_code = (a / "launcher.exitcode").read_text().strip() if (a / "launcher.exitcode").exists() else None
    return {"pack": name, "source_commit": (a / "SOURCE_COMMIT").read_text().strip(),
            "binding_sha256": (a / "BINDING_SHA256").read_text().strip(), "binding_file_sha256": sha(a / "binding.json"),
            "source_freeze_sha256": json.loads((a / "PACK.json").read_text())["source_freeze_sha256"],
            "launcher_exit_code": None if exit_code is None else int(exit_code),
            "command_args": json.loads((a / "binding.json").read_text())["command_args"]}
out = {"training": {}, "evaluations": {}, "failed": {}}
for key, (name, updates) in training.items():
    s = root / name / "run" / "standing"
    rows = [json.loads(line) for line in open(s / "metrics.jsonl")]
    quarter = updates // 4
    windows = [(1, quarter), (quarter + 1, 2 * quarter), (2 * quarter + 1, 3 * quarter), (3 * quarter + 1, updates)]
    def window(a, b):
        sel = [r for r in rows if a <= r["update"] <= b]
        return {"updates": [a, b], "terminations": sum(r["task"]["interval_metrics"]["terminations"] for r in sel),
                "mean_tracked_speed_ratio": sum(r["task"]["interval_gait"].get("tracked_speed_ratio") or 0 for r in sel) / len(sel)}
    final = s / f"checkpoint_update{updates:06d}"
    state = json.loads((s / "state.json").read_text())
    out["training"][key] = {**pack(name), "updates": updates, "rows": len(rows), "last_update": rows[-1]["update"],
        "transitions": rows[-1].get("transitions"), "state_status": state.get("status"), "state_errors": state.get("errors"),
        "metrics_sha256": sha(s / "metrics.jsonl"),
        "final_checkpoint": {"path": str(final.with_suffix(".pt").relative_to(root)), "sha256": sha(final.with_suffix(".pt")),
                             "declaration_sha256": sha(final.with_suffix(".json"))},
        "training_quarters": [window(a, b) for a, b in windows]}
for key, name in evaluations.items():
    e = root / name / "run" / "standing" / "evaluation"
    summary = json.loads((e / "summary.json").read_text())
    batches = {}
    for b in sorted(e.glob("batch_*")):
        case = json.loads((b / "declaration.json").read_text())["cases"][0]["case_id"]
        batches[case] = {"batch": b.name, "report_sha256": sha(b / "report.json"), "control_trace_sha256": sha(b / "control_trace.npz"),
                         "force_metrics_sha256": sha(b / "force_metrics.json")}
    probes = {}
    for r in summary["results"]:
        m = r["metrics"]
        probes[r["case_id"]] = {"pass": r["pass"], "failed_bounds": r.get("failed_bounds") or [],
            "mean_velocity_mps": m.get("mean_velocity_mps"), "mean_yaw_rate_rad_s": m.get("mean_yaw_rate_rad_s"),
            "planar_error_mps": m.get("planar_error_mps"), "yaw_error_rad_s": m.get("yaw_error_rad_s"), **batches[r["case_id"]]}
    out["evaluations"][key] = {**pack(name), "summary_sha256": sha(e / "summary.json"),
        "stage2_complete": summary["stage2_complete"], "failed_probe_cases": summary["failed_probe_cases"],
        "passed": sum(p["pass"] for p in probes.values()), "probes": probes}
for key, name in failed.items():
    job = json.loads((root / name / "run" / "jobs" / "standing.json").read_text())
    out["failed"][key] = {**pack(name), "error": job.get("error"), "status": job.get("status")}
print(json.dumps(out))
'''


def ssh(host, args, stdin=None):
    # The remote shell splits its command line, so each argument travels quoted.
    command = " ".join(shlex.quote(a) for a in args)
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, command], input=stdin, check=True, capture_output=True).stdout


def heading(pose):
    x, y, z, w = pose[:, 3:7].T
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def drift(trace):
    """Heading change and path angle from 2 s to the end of a walking probe, in degrees; positive is to the left."""
    t = trace["time_s"].reshape(-1)
    pose = trace["root_pose_xyzw"].reshape(len(t), -1)[:, :7][t >= 2]
    command = trace["command"].reshape(len(t), -1)[0]
    h = heading(pose)
    body = np.array([command[1], -command[0]])  # navigation (x, y) is body (y, -x)
    c, s = np.cos(h[0]), np.sin(h[0])
    wanted = np.array([c * body[0] - s * body[1], s * body[0] + c * body[1]])
    wanted /= np.linalg.norm(wanted)
    moved = pose[-1, :2] - pose[0, :2]
    cross = wanted[0] * moved[1] - wanted[1] * moved[0]
    return {"facing_drift_deg": float(np.degrees(h[-1] - h[0])),
            "path_angle_deg": float(np.degrees(np.arctan2(cross, wanted @ moved))),
            "off_course_m": float(cross), "speed_along_command_ratio": float((wanted @ moved) / (.05 * (t[-1] - 2)))}


def main(host):
    remote = json.loads(ssh(host, ["python3", "-", ROOT, json.dumps(TRAINING), json.dumps(EVALUATIONS), json.dumps(FAILED)],
                            REMOTE.encode()))
    for key, record in remote["evaluations"].items():
        walks = {case: p for case, p in record["probes"].items() if case.startswith("learning:translate")}
        paths = [f"{record['pack']}/run/standing/evaluation/{p['batch']}/control_trace.npz" for p in walks.values()]
        archive = tarfile.open(fileobj=io.BytesIO(ssh(host, ["tar", "-C", ROOT, "-cf", "-", *paths])))
        for (case, probe), path in zip(walks.items(), paths):
            data = archive.extractfile(path).read()
            if hashlib.sha256(data).hexdigest() != probe["control_trace_sha256"]:
                raise ValueError("Trace copy differs from its Spark hash: " + path)
            probe["derived"] = drift(np.load(io.BytesIO(data)))
        facing = [abs(p["derived"]["facing_drift_deg"]) for p in walks.values()]
        path_angles = [abs(p["derived"]["path_angle_deg"]) for p in walks.values()]
        record["walk_summary"] = {"facing_drift_mean_deg": float(np.mean(facing)), "facing_drift_max_deg": float(np.max(facing)),
                                  "path_angle_mean_deg": float(np.mean(path_angles)), "path_angle_max_deg": float(np.max(path_angles))}
    result = {"schema": "hexapod_yaw_gated_stride_native_v1", "native_started_by_collection": False,
              "remote_root": ROOT, "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": ("Four native training launches and their 13 learning probes, seed 20260917, one deterministic trial "
                        "per probe. Learning probes are diagnostics; Stage 2 remains incomplete and needs transition, torque, "
                        "contact and visual acceptance. Derived drift values come from the retained traces and set no limit."),
              **remote}
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUTPUT, {k: v["passed"] for k, v in result["evaluations"].items()})


if __name__ == "__main__":
    main(sys.argv[1])
