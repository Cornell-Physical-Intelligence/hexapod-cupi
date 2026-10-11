"""Compare the final bounded-mean evaluation with the original focus probes."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

import numpy as np

NEW = "/srv/cupi/hexapod/runs/james/ppo_bounded_mean_20261002_001_evaluate/run/standing/evaluation"
OLD = "/srv/cupi/hexapod/runs/james/flat_pilot_20261002_001/ppo_mlp_video_001/run/standing/evaluation"
ALLOWED = ("control_trace.npz", "report.json", "declaration.json", "rollout.mp4", "force_metrics.json")
SNAPSHOT_NAMES = tuple(f"batch_{i:03d}" for i in range(13))
CHECKPOINTS = {
    "bounded_mean": "8e0ff06ece3ed348129afba633dab1ab35dbd4002a1c68189c6b8538334451c3",
    "original": "6c56d8fa2db9e7537f1127c4953e45fb5fbb7a812a591611339d06d9c93ee922",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def remote_json(code):
    return json.loads(subprocess.check_output(["ssh", "spark", "python3 -c " + shlex.quote(code)]))


def fetch():
    DATA.mkdir(exist_ok=True)
    record = {"observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "batches": {}}
    for group, root, names in (("bounded_mean", NEW, SNAPSHOT_NAMES), ("original", OLD, ("batch_000", "batch_001", "batch_002"))):
        record["batches"][group] = {}
        for name in names:
            remote = root + "/" + name
            code = "from pathlib import Path; import hashlib,json; p=Path(" + repr(remote) + "); r=json.loads((p/'report.json').read_text()); assert r['acquisition_complete'] is True and r['failure'] is None; names=" + repr(ALLOWED) + "; print(json.dumps({'report':r,'hashes':{n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in names if (p/n).exists()}}))"
            remote_record = remote_json(code)
            destination = DATA / group / name
            if destination.exists():
                assert all(sha(destination / n) == digest for n, digest in remote_record["hashes"].items())
            else:
                destination.mkdir(parents=True)
                command = "tar -C " + shlex.quote(remote) + " -cf - " + " ".join(map(shlex.quote, remote_record["hashes"]))
                process = subprocess.Popen(["ssh", "spark", command], stdout=subprocess.PIPE)
                with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                    archive.extractall(destination, filter="data")
                assert process.wait() == 0
                assert all(sha(destination / n) == digest for n, digest in remote_record["hashes"].items())
            record["batches"][group][name] = {"remote_directory": remote, "sha256": remote_record["hashes"]}
    # This records only the declared suite; it reads no growing native capture.
    declaration = remote_json("from pathlib import Path; import json; print(Path(" + repr(NEW + "/allocation.json") + ").read_text())")
    record["declared_selected_cases"] = declaration["selected_case_ids"]
    (DATA / "snapshot.json").write_text(json.dumps(record, indent=2) + "\n")


def analyze(group, batch, inventory, neutral, joint_names):
    path = DATA / group / batch
    assert all(sha(path / name) == digest for name, digest in inventory["sha256"].items())
    report, declaration = read(path / "report.json"), read(path / "declaration.json")
    assert report["acquisition_complete"] and report["failure"] is None
    assert report["checkpoint_sha256"] == declaration["checkpoint_sha256"] == CHECKPOINTS[group]
    assert report["files"]["control_trace.npz"] == sha(path / "control_trace.npz")
    assert report["files"]["declaration.json"] == sha(path / "declaration.json")
    assert report["files"]["force_metrics.json"] == sha(path / "force_metrics.json")
    if "rollout.mp4" in inventory["sha256"]:
        assert report["files"]["rollout.mp4"] == sha(path / "rollout.mp4")
    with np.load(path / "control_trace.npz", allow_pickle=False) as bundle:
        trace = {key: bundle[key] for key in bundle.files}
    n = report["controls"]
    assert len(trace["time_s"]) == n and n == declaration["cases"][0]["controls"]
    assert np.allclose(trace["time_s"], (np.arange(n) + 1) * .02, atol=1e-12, rtol=0)
    mask = (trace["time_s"] > 2 + 1e-9) & (trace["time_s"] <= 20 + 1e-9)
    actions = trace["policy_action"][mask, 0].astype(np.float64)
    targets = trace["joint_target_rad"][mask, 0].astype(np.float64)
    target_feature = (targets - neutral) / .35
    per_joint = {}
    for j, name in enumerate(joint_names):
        per_joint[name] = {
            "action_mean": float(actions[:, j].mean()),
            "action_min": float(actions[:, j].min()), "action_max": float(actions[:, j].max()),
            "action_near_bound_fraction": float((np.abs(actions[:, j]) >= .95).mean()),
            "raw_action_outside_bounds_fraction": float((np.abs(actions[:, j]) > 1).mean()),
            "target_min_rad": float(targets[:, j].min()), "target_max_rad": float(targets[:, j].max()),
            "target_span_rad": float(np.ptp(targets[:, j])),
            "target_offset_from_neutral_mean_rad": float((targets[:, j] - neutral[j]).mean()),
            "target_near_action_envelope_fraction": float((np.abs(target_feature[:, j]) >= .95).mean()),
        }
    after = trace["amp_state_after"][mask, 0]
    native_linear_nav = np.stack((-after[:, 37], after[:, 36], after[:, 38]), -1).astype(np.float64)
    command = trace["command"][mask, 0].astype(np.float64)
    speed = np.linalg.norm(command[:, :2], axis=1)
    moving = speed > 1e-8
    along = np.sum(native_linear_nav[:, :2] * command[:, :2], axis=1) / np.maximum(speed, 1e-8)
    result = report["results"][0]
    return {
        "case_id": result["case_id"], "remote_directory": inventory["remote_directory"], "input_sha256": inventory["sha256"],
        "source_sha256": declaration["source_sha256"], "checkpoint_sha256": declaration["checkpoint_sha256"],
        "model_sha256": declaration["model_sha256"], "numerical_pass": result["pass"],
        "failed_bounds": result.get("failed_bounds", []), "human_label": "pending",
        "recorded_controls": n, "scored_action_controls": int(mask.sum()), "action_window_seconds": {"greater_than": 2, "less_than_or_equal": 20},
        "video": {"available": (path / "rollout.mp4").exists(), "frames": report["video_frames"], "remote_path": inventory["remote_directory"] + "/rollout.mp4" if (path / "rollout.mp4").exists() else None},
        "numeric_metrics": result.get("metrics", {}),
        "training_frame_along_command_speed_mps": float(along[moving].mean()) if moving.any() else None,
        "raw_action_abs_max": float(np.abs(actions).max()),
        "raw_action_outside_bounds_fraction": float((np.abs(actions) > 1).mean()),
        "action_near_bound_fraction_across_joint_controls": float((np.abs(actions) >= .95).mean()),
        "realized_target_near_action_envelope_fraction": float((np.abs(target_feature) >= .95).mean()),
        "mean_per_joint_target_span_rad": float(np.ptp(targets, axis=0).mean()),
        "max_per_joint_target_span_rad": float(np.ptp(targets, axis=0).max()),
        "per_joint": per_joint,
        "force_summary": read(path / "force_metrics.json"),
    }


def main():
    global DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--workspace", type=Path, required=True,
                        help="External directory for completed evaluation captures")
    parser.add_argument("--stance", type=Path, required=True,
                        help="Admitted stance metadata from the recorded input bundle")
    parser.add_argument("--output", type=Path,
                        help="Receipt path; defaults to WORKSPACE/evaluation_comparison.json")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    DATA = workspace / "evaluation_comparison_inputs"
    output = args.output or workspace / "evaluation_comparison.json"
    if output.exists():
        raise FileExistsError("Use a fresh receipt path: " + str(output))
    if args.fetch:
        fetch()
    inventory = read(DATA / "snapshot.json")
    stance_path = args.stance
    assert sha(stance_path) == "830cb07c0fdb3d80af82476d8e6e88f25440cfd2e9255ac1e16327ac826d259c"
    neutral = np.asarray(read(stance_path)["nominal_joint_position_rad"], dtype=np.float32).astype(np.float64)
    joint_names = [leg + "_" + joint for leg in ("lf", "lm", "lr", "rf", "rm", "rr") for joint in ("coxa_yaw", "femur_pitch", "tibia_pitch")]
    results = {group: [analyze(group, batch, item, neutral, joint_names) for batch, item in batches.items()] for group, batches in inventory["batches"].items()}
    present = {row["case_id"] for row in results["bounded_mean"]}
    missing = [name for name in inventory["declared_selected_cases"] if name not in present]
    assert not missing and len(results["bounded_mean"]) == 13
    assert len({row["model_sha256"] for rows in results.values() for row in rows}) == 1
    tracking = [row for row in results["bounded_mean"] if ":translate_" in row["case_id"] or ":yaw_" in row["case_id"]]
    assert len(tracking) == 10
    error = sum(row["numeric_metrics"]["planar_error_mps"] / .05 if ":translate_" in row["case_id"]
                else row["numeric_metrics"]["yaw_error_rad_s"] / .2 for row in tracking) / 10
    paired = []
    old_by_id = {row["case_id"]: row for row in results["original"]}
    for row in results["bounded_mean"]:
        if row["case_id"] in old_by_id:
            before = old_by_id[row["case_id"]]
            keys = ("raw_action_abs_max", "raw_action_outside_bounds_fraction", "action_near_bound_fraction_across_joint_controls", "realized_target_near_action_envelope_fraction", "mean_per_joint_target_span_rad", "max_per_joint_target_span_rad", "training_frame_along_command_speed_mps")
            paired.append({"case_id": row["case_id"], "original": {key: before[key] for key in keys}, "bounded_mean": {key: row[key] for key in keys}})
    receipt = {
        "schema": "hexapod_final_evaluation_comparison_v1", "analysis_sha256": sha(__file__),
        "snapshot_sha256": sha(DATA / "snapshot.json"), "stance_sha256": sha(stance_path),
        "snapshot_observed_utc": inventory["observed_utc"], "bounded_snapshot_batches": list(SNAPSHOT_NAMES),
        "methods": {
            "files": "Fetch only complete batch reports and their pinned trace, declaration, video and force summary. Read no growing native capture.",
            "actions": "The trace records deterministic inference actions. Count |action| >= 0.95 and |action| > 1 over joint-controls, then measure recorded targets. These are distinct from stochastic training sample fractions.",
            "targets": "Report per-joint recorded target minimum, maximum and span in radians. Normalize target displacement from the admitted neutral pose by the unchanged 0.35 rad action scale for envelope occupancy.",
            "window": "Use control endpoints 2 < t <= 20 s for action and target descriptors. Preserve the evaluator's own metric windows; do not rename those metrics as this action window.",
            "labels": "Leave human labels pending. A failed numerical screen cannot count as walking; missing probes and pending review block the primary pilot decision.",
            "scope": "The bounded-mean run covers all 13 probes. The original run covers forward, quiet-20s and forward-to-stop. Compare only those matched cases; retain missing original directions.",
            "tracking": "Compute E from eight planar errors divided by 0.05 m/s and two yaw errors divided by 0.20 rad/s, with equal case weight. Use the evaluator's unchanged controls 100 through 999. Preserve quiet and stop as separate diagnostics.",
            "loads": "Force and torque descriptors include failed motion. Failed motion screens prevent the matched-behavior load comparison against the walking tripod; no load pass is inferred.",
        },
        "missing_selected_cases": missing,
        "missing_original_cases": [name for name in inventory["declared_selected_cases"] if name not in old_by_id],
        "bounded_mean_E": error, "original_E": None, "primary_W": None,
        "results": results, "matched_cases": paired, "native_started": False,
    }
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"matched_cases": paired, "missing": missing, "numeric_results": [{"case": r["case_id"], "pass": r["numerical_pass"], "failed": r["failed_bounds"], "metrics": r["numeric_metrics"]} for r in results["bounded_mean"]]}, indent=2))


if __name__ == "__main__":
    main()
