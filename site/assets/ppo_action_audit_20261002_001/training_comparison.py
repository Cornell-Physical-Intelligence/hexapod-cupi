"""Compare retained training windows without starting native computation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOTS = {
    "original": "/srv/cupi/hexapod/runs/james/flat_pilot_20261002_001/ppo_mlp",
    "bounded_mean": "/srv/cupi/hexapod/runs/james/ppo_bounded_mean_20261002_001",
}
FILES = ["run/standing/metrics.jsonl", "run/standing/state.json", "run/standing/force_metrics.json",
         "binding.json", "PACK.json", "source/FREEZE_SHA256.json"]
SOURCE_FILES = ["locomotion/ppo.py", "locomotion/train.py", "locomotion/task.py", "locomotion/task_v2.py", "locomotion/env.py", "locomotion/env_config.py"]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch():
    DATA.mkdir(exist_ok=True)
    for label, remote in ROOTS.items():
        destination = DATA / label
        if destination.exists():
            continue
        destination.mkdir()
        names = FILES + ["source/" + name for name in SOURCE_FILES]
        if label == "bounded_mean":
            names.append("source/locomotion/action_distribution.py")
        remote_code = "import hashlib,json,pathlib; p=pathlib.Path(" + repr(remote) + "); names=" + repr(names) + "; print(json.dumps({n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in names}))"
        hashes = json.loads(subprocess.check_output(["ssh", "spark", "python3 -c " + shlex.quote(remote_code)]))
        command = "tar -C " + shlex.quote(remote) + " -cf - " + " ".join(map(shlex.quote, names))
        process = subprocess.Popen(["ssh", "spark", command], stdout=subprocess.PIPE)
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            archive.extractall(destination, filter="data")
        assert process.wait() == 0
        assert all(sha(destination / name) == digest for name, digest in hashes.items())
        (destination / "remote_file_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
        print("Retrieved", label, flush=True)


def avg(items, key, denominator="environment_controls"):
    count = sum(item[denominator] for item in items)
    return sum(item[key] * item[denominator] for item in items if item[key] is not None) / count if count else None


def actions(rows):
    available = [row["actions"] for row in rows if "actions" in row]
    if not available:
        return {"available": False, "reason": "Original source recorded no rollout action-bound telemetry."}
    assert len(available) == len(rows)
    assert all(a["mean_near_bound_threshold"] == .95 for a in available)
    count = sum(a["environment_controls"] for a in available)
    joint_names = list(available[0]["per_joint"])
    per_joint = {}
    for joint in joint_names:
        per_joint[joint] = {
            key: sum(a["per_joint"][joint][key] * a["environment_controls"] for a in available) / count
            for key in ("raw_sample_outside_bounds_fraction", "mean_near_bound_fraction")
        }
        per_joint[joint]["mean_abs_max"] = max(a["per_joint"][joint]["mean_abs_max"] for a in available)
    return {
        "available": True, "scope": "rollouts collected before the associated optimizer update",
        "environment_controls": count,
        "raw_sample_outside_bounds_fraction": avg(available, "raw_sample_outside_bounds_fraction"),
        "mean_near_bound_threshold": .95,
        "mean_near_bound_fraction_across_joint_controls": sum(v["mean_near_bound_fraction"] for v in per_joint.values()) / len(joint_names),
        "mean_abs_max": max(a["mean_abs_max"] for a in available), "per_joint": per_joint,
    }


def window(rows):
    samples = [row["task"]["interval_metrics"] for row in rows]
    count = sum(s["environment_controls"] for s in samples)
    moving = sum(s["moving_command_rows"] for s in samples)
    quiet = sum(s["zero_command_rows"] for s in samples)
    requested_moving = sum(s["requested_planar_speed_mps"] * s["environment_controls"] for s in samples) / moving
    projected_moving = avg(samples, "moving_signed_command_direction_speed_mps", "moving_command_rows")
    components = {name: sum(s["reward_components"][name]["sum"] for s in samples) / count for name in samples[0]["reward_components"]}
    classes = {}
    for name in samples[0]["command_classes"]:
        groups = [s["command_classes"][name] for s in samples]
        n = sum(s["environment_controls"] for s in groups)
        classes[name] = {"environment_controls": n, "time_fraction": n / count}
        for key in ("task_reward_mean", "requested_planar_speed_mps", "nonfoot_event_fraction"):
            classes[name][key] = avg(groups, key)
        classes[name]["moving_signed_command_direction_speed_mps"] = avg(groups, "moving_signed_command_direction_speed_mps", "moving_command_rows")
        classes[name]["reward_component_means"] = {
            component: sum(g["reward_component_means"][component] * g["environment_controls"] for g in groups if g["reward_component_means"][component] is not None) / n if n else None
            for component in components
        }
    rms_by_joint = [math.sqrt(sum(s["zero_command_rows"] * s["zero_hold_joint_rate_rms_rad_s"][joint] ** 2 for s in samples if s["zero_command_rows"]) / quiet) for joint in range(18)] if quiet else None
    slew = [s["actual_target_slew"] for s in samples]
    valid = sum(s["environment_controls"] for s in slew)
    slew_by_joint = {name: {
        "at_limit_fraction": sum(s["by_joint"][name]["at_limit_rows"] for s in slew) / valid,
        "beyond_limit_rows": sum(s["by_joint"][name]["beyond_limit_rows"] for s in slew),
    } for name in slew[0]["by_joint"]}
    terminations = sum(s["terminations"] for s in samples)
    reasons = {key: sum(s["termination_reasons"]["rows"][key] for s in samples) for key in samples[0]["termination_reasons"]["rows"]}
    return {
        "updates_inclusive": [rows[0]["update"], rows[-1]["update"]], "update_count": len(rows),
        "environment_controls": count, "moving_controls": moving, "quiet_controls": quiet,
        "reward_mean": sum(s["task_reward_sum"] for s in samples) / count, "reward_component_means": components,
        "moving_signed_command_direction_speed_mps": projected_moving,
        "moving_requested_speed_mps": requested_moving,
        "moving_projection_to_requested_ratio": projected_moving / requested_moving,
        "achieved_planar_speed_mps_all_controls": avg(samples, "achieved_planar_speed_mps"),
        "absolute_navigation_velocity_error_mps_by_axis": [sum(s["absolute_navigation_velocity_error_mps"][axis] * s["environment_controls"] for s in samples) / count for axis in range(2)],
        "absolute_yaw_error_rad_s": avg(samples, "absolute_yaw_error_rad_s"),
        "requested_saturation_fraction": avg(samples, "requested_saturation_fraction"),
        "nonfoot_event_fraction": sum(s["nonfoot_event_rows"] for s in samples) / count,
        "terminations": terminations, "terminations_per_1000_controls": terminations * 1000 / count,
        "truncations": sum(s["truncations"] for s in samples), "termination_reason_counts": reasons,
        "termination_flag_mismatch_rows": sum(s["termination_reasons"]["termination_flag_mismatch_rows"] for s in samples),
        "quiet_all_joint_rate_rms_rad_s": math.sqrt(sum(r * r for r in rms_by_joint) / 18) if rms_by_joint else None,
        "quiet_worst_joint_rate_rms_rad_s": max(rms_by_joint) if rms_by_joint else None,
        "quiet_by_joint_rate_rms_rad_s": rms_by_joint,
        "actual_target_slew": {"valid_controls": valid, "mean_joint_at_limit_fraction": sum(v["at_limit_fraction"] for v in slew_by_joint.values()) / 18, "per_joint": slew_by_joint},
        "mean_action_std_mean": sum(row["mean_action_std"] for row in rows) / len(rows),
        "learning_rate_first_last": [rows[0]["learning_rate"], rows[-1]["learning_rate"]],
        "command_classes": classes, "actions": actions(rows),
    }


def force_summary(value):
    windows = {}
    for name, data in value["windows"].items():
        metrics = data["metrics"]
        applied = {k: v for k, v in metrics.items() if k.startswith("applied_abs_torque_nm:")}
        requested = {k: v for k, v in metrics.items() if k.startswith("requested_abs_torque_nm:")}
        windows[name] = {
            "samples_across_replicas": data["samples_across_replicas"],
            "total_vertical_support_n": metrics["total_vertical_support_n"],
            "applied_worst_joint_rms_nm": max(v["rms"] for v in applied.values()),
            "applied_peak_nm": max(v["peak"] for v in applied.values()),
            "requested_peak_nm": max(v["peak"] for v in requested.values()),
        }
    return {"status": value["status"], "physics_steps": value["physics_steps"], "windows": windows}


def analyze(label):
    root = DATA / label
    hashes = read(root / "remote_file_hashes.json")
    assert all(sha(root / name) == digest for name, digest in hashes.items())
    binding, pack = read(root / "binding.json"), read(root / "PACK.json")
    freeze = read(root / "source/FREEZE_SHA256.json")
    assert binding["source_freeze_sha256"] == sha(root / "source/FREEZE_SHA256.json")
    assert pack["binding_sha256"] == sha(root / "binding.json")
    for name in SOURCE_FILES + (["locomotion/action_distribution.py"] if label == "bounded_mean" else []):
        assert freeze[name] == sha(root / "source" / name)
    state = read(root / "run/standing/state.json")
    assert state["status"] == "completed" and not state["errors"] and state["updates"] == 2000 and state["transitions"] == 6144000
    assert state["identity"]["config"]["num_envs"] == 128 and state["identity"]["seed"] == 20260917 and state["identity"]["reward_version"] == "2"
    with (root / "run/standing/metrics.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 2000
    for i, row in enumerate(rows, 1):
        assert row["update"] == i and row["transitions"] == i * 3072
        assert row["task"]["interval_metrics"]["environment_controls"] == 3072
    return {
        "remote_root": ROOTS[label], "input_sha256": hashes,
        "source_freeze_sha256": binding["source_freeze_sha256"], "source_sha256": {name: freeze[name] for name in SOURCE_FILES},
        "identity": state["identity"], "completion": {key: state[key] for key in ("status", "updates", "transitions", "wall_seconds")},
        "early_200": window(rows[:200]), "late_200": window(rows[-200:]),
        "latest_rollout_actions": rows[-1].get("actions"),
        "force_history_summary": force_summary(read(root / "run/standing/force_metrics.json")),
    }


def main():
    global DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--workspace", type=Path, required=True,
                        help="External directory for retained training inputs")
    parser.add_argument("--output", type=Path,
                        help="Receipt path; defaults to WORKSPACE/training_comparison.json")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    DATA = workspace / "training_comparison_inputs"
    output = args.output or workspace / "training_comparison.json"
    if output.exists():
        raise FileExistsError("Use a fresh receipt path: " + str(output))
    if args.fetch:
        fetch()
    runs = {label: analyze(label) for label in ROOTS}
    for filename in ("locomotion/task.py", "locomotion/task_v2.py", "locomotion/env.py", "locomotion/env_config.py"):
        assert runs["original"]["source_sha256"][filename] == runs["bounded_mean"]["source_sha256"][filename]
    receipt = {
        "schema": "hexapod_matched_training_windows_v1", "analysis_sha256": sha(__file__),
        "methods": {
            "windows": "Updates 1–200 and 1801–2000. Each contains 614,400 environment-controls. Use interval_metrics, never overlapping cumulative prefixes.",
            "weighting": "Pool reward sums and count-weighted means. Weight moving projection by moving-command rows and requested speed by all controls before dividing by moving rows. Pool command classes by class counts.",
            "quiet_rms": "Pool per-joint squared RMS weighted by zero-command counts, then take square root. Compute overall and worst joint only after pooling. Quiet rows include transitions and reset transients.",
            "terminations": "Counts per 1,000 controls; reasons can overlap, and physical termination can coincide with timeout. These are not episode failure probabilities.",
            "actions": "Raw Gaussian samples may exceed [−1, 1] although their learned means use tanh. Fractions count sampled joint-controls, not robots. Telemetry describes rollout data before its optimizer update. Original run lacks these fields.",
            "forces": "Final summaries pool the full changing training history. They cannot give first/last-window loads, matched achieved behavior or the flat_pilot_v1 load decision.",
            "limits": "Training has stochastic exploration and changing commands. Projection is not a gait label or held-out tracking score. These results cannot determine final-checkpoint evaluation, pilot E/W, or causal performance across seeds.",
        },
        "reward_and_physics_sources_match": True, "native_started": False, "runs": runs,
    }
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    concise = {}
    for label, run in runs.items():
        concise[label] = {}
        for name in ("early_200", "late_200"):
            w = run[name]
            concise[label][name] = {key: w[key] for key in ("reward_mean", "moving_signed_command_direction_speed_mps", "moving_requested_speed_mps", "moving_projection_to_requested_ratio", "absolute_yaw_error_rad_s", "terminations_per_1000_controls", "quiet_all_joint_rate_rms_rad_s", "reward_component_means", "actions")}
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
