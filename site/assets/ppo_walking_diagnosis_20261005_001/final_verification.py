"""Verify one retained native PPO training attempt and its two evaluations, then write both verification records.

The script reads the launch, exit, cleanup, state, metric and checkpoint records of the training attempt, then the summary,
batch reports and native 400 Hz captures of its probe evaluation and its stop-video evaluation. It compares each recorded sha256
with the retained bytes, checks the force and torque arrays, copies the probe verdicts and binds each published video to a batch.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REMOTE, MEDIA = "/srv/cupi/hexapod/runs/james/", "site/assets/media/"
STANDING, EVALUATION = "run/standing/", "run/standing/evaluation/"
RECORDS = ("PACK.json", "binding.json", "preparation.json", "run/cleanup.json", "run/jobs/standing.json",
           STANDING + "state.json", STANDING + "ppo_config.json", "run/launch_binding.json", "launcher.exitcode")
GOALS = {"forward": "learning:translate_0.05_0deg", "quiet_20s": "learning:quiet_20s", "quiet_32s": "learning:quiet_32s",
         "forward_to_stop": "learning:forward_0.05_to_stop"}
# Each evaluation suffix maps to the label of its published video and the goal probe that its camera records.
VIDEOS = {"_evaluate": ("forward", "forward"), "_evaluate_stop_video": ("stop", "forward_to_stop")}
PEAKS = ("distal_force_world_n", "nonfoot_force_world_n", "computed_torque_nm", "applied_torque_nm")
BOUNDS = ("joint_bound_violation_steps", "speed_bound_violation_steps", "maximum_applied_nm", "maximum_requested_nm")
# Each pin is the sha256 of the sorted JSON map from each file that the script checks in one attempt to its sha256.
PINS = {
    "ppo_v4_settle_seed20260917_20261006_001": "13a86abcdb9938cb89aabea708952a438189b47823ff51a526aa16f750fb2f58",
    "ppo_v4_settle_seed20260917_20261006_001_evaluate": "c97800090fb55544a1f5ecc424a197a75b835187ef88b9b53f69279f4ff55951",
    "ppo_v4_settle_seed20260917_20261006_001_evaluate_stop_video":
        "7d2fa8edcc2e018f644d611c2a9ed009d8cf4b5705d1875b3382bd1f861c0da3",
    "ppo_v4_settle_seed20260918_20261006_001": "2685abd48c29968d5223dd1f0dc0f16c6328b1376a0f5b9f64dc69d39ea91813",
    "ppo_v4_settle_seed20260918_20261006_001_evaluate": "0cf7a31e1a0d885ebf7bf11712a19612376156567027bda5f4e747e3d60cb011",
    "ppo_v4_settle_seed20260918_20261006_001_evaluate_stop_video":
        "50fb0c16310804af2e3ee79f58008abbfaabdbdaffc288051b1f18bd7c500b2d"}
METHODS = {
    "scope": "One training attempt, one seed and two retained evaluations of its final checkpoint with one trial per probe. "
             "The records verify retained bytes and copy the evaluator's verdicts. They add no acceptance limit, the script "
             "starts no native run and each summary must report Stage 2 as incomplete.",
    "hashes": "The script copies records, traces, 400 Hz arrays, the video and its first frame through tar streams and hashes "
              "the copies. Frozen source files, input files, checkpoint weights and contacts.jsonl stay on Spark, where "
              "sha256sum hashes them. A workspace file caches those digests, so a fresh workspace hashes Spark's bytes again. "
              "Each pin covers the sha256 of each file that the script checks in one attempt.",
    "launch": "Each attempt must hold launcher exit code 0, a completed job and state with exit code 0 and no errors, and a "
              "cleanup record whose inspections all report the owned container absent. binding.json must match PACK.json, "
              "preparation.json and the launch copy. One source freeze hash must appear in the pack, binding, preparation, job "
              "and state. Each frozen source file must match PACK.json and each input file must match binding.json.",
    "training": "Transitions must equal updates * robots * steps per robot. Each saved update must hold a sidecar that names "
                "it, its transitions, the state's identity and its weights' sha256. metrics.jsonl must hold one row per update "
                "in order. maximum_action_mean_magnitude and before_update_kl_max are the largest actions.mean_abs_max and "
                "policy_update.before.kl_max. force_metrics.json must hold vertical support and each joint's applied torque.",
    "evaluation": "Each evaluation must hold the training source hashes. The binding, allocation, each report and each result "
                  "must name the final checkpoint's sha256. The summary must embed the allocation and equal the state's copy. "
                  "Batch i must report selected case i alone with a result equal to the summary's. report.json and "
                  "capture.json pin each other file of the batch, and force_metrics.json must name the same capture files. "
                  "Earlier records repeat each result in the state and the batch. These records hold it once, in the summary.",
    "native_arrays": "Over a batch's substep files, sequence must run from 0 to the capture's step count minus 1 and equal "
                     "control_index * decimation + substep_index. explicit_counter must equal the capture's initial counter + "
                     "sequence + 1, each initial counter must equal the sum of the earlier batches' steps, and time_s must "
                     "equal (sequence + 1) * physics_dt. Each float array in each substep file and in the control trace must "
                     "be finite. native_peaks holds the largest absolute element of each contact force and motor torque array, "
                     "and the torque peaks must equal the capture's maxima.",
    "verdicts": "goal_probes copies pass and failed_bounds for the four goal probes from each evaluation that selected them. A "
                "failed probe leaves verified true and sets all_goal_probes_pass false. Each published video and poster must "
                "match the sha256 that the report of the video case holds for rollout.mp4 and first_frame.png.",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(source):
    """Decode a JSON file or a JSON text line. The int parser raises ValueError on a NaN or infinity constant."""
    return json.loads(source.read_text() if isinstance(source, Path) else source, parse_constant=int)


class Attempt:
    """Hold one retained attempt, its workspace copy and the sha256 of each file that the script checks."""

    def __init__(self, name, workspace, enabled):
        self.name, self.local, self.enabled, self.files = name, workspace / "inputs" / name, enabled, {}

    def fetch(self, names):
        """Copy the missing named files from Spark through one tar stream, then hash each local copy."""
        missing = [name for name in names if self.enabled and not (self.local / name).exists()]
        if missing:
            self.local.mkdir(parents=True, exist_ok=True)
            command = "tar -C " + shlex.quote(REMOTE + self.name) + " -cf - " + " ".join(map(shlex.quote, missing))
            process = subprocess.Popen(["ssh", "spark", command], stdout=subprocess.PIPE)
            with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                archive.extractall(self.local, filter="data")
            assert process.wait() == 0
        self.files.update({name: sha(self.local / name) for name in names})
        return [self.local / name for name in names]

    def remote(self, names):
        """Return the sha256 of files that stay on Spark, from one sha256sum call that the workspace caches."""
        cache = self.local / "remote_sha256.json"
        known = read(cache) if cache.exists() else {}
        if self.enabled and not set(names) <= set(known):
            command = "cd " + shlex.quote(REMOTE + self.name) + " && sha256sum -- " + " ".join(map(shlex.quote, names))
            lines = subprocess.run(["ssh", "spark", command], check=True, capture_output=True, text=True).stdout
            known.update({line[66:]: line[:64] for line in lines.splitlines()})
            cache.write_text(json.dumps(known, indent=1) + "\n")
        self.files.update({name: known[name] for name in names})
        return {name: known[name] for name in names}


def launch(attempt, mode):
    """Check the launch, source, input, exit and cleanup records of one attempt and return the fields they share."""
    paths = attempt.fetch(RECORDS)
    pack, binding, preparation, cleanup, job, state, config = map(read, paths[:7])
    files, freeze, root = attempt.files, pack["source_freeze_sha256"], REMOTE + attempt.name
    exitcode, arguments = int(paths[8].read_text()), binding["command_args"]
    assert exitcode == job["exit_code"] == 0 and job["status"] == state["status"] == "completed" and state["errors"] == []
    assert mode == pack["mode"] == binding["mode"] == job["native_mode"] == state["mode"]
    assert (pack["remote_root"], binding["source"], binding["output"]) == (root, root + "/source", root + "/run")
    assert files["binding.json"] == files["run/launch_binding.json"] == pack["binding_sha256"] == preparation["binding_sha256"]
    assert freeze == binding["source_freeze_sha256"] == preparation["source_freeze_sha256"] == job["source_freeze_sha256"]
    assert freeze == state["runtime_binding"]["runtime_tree_sha256"] and config == state["identity"]["ppo_config"]
    absent = bool(cleanup["inspections"]) and all(row == {"absent": True} for row in cleanup["inspections"])
    assert absent and cleanup["cleanup_checked"] and job["cleanup_checked"] and job["contact_data_completeness"]["passed"]
    assert (cleanup["container_id"], cleanup["container_name"]) == (job["container_id"], job["container_name"])
    source = {name: digest for name, digest in pack["files"].items() if name != "binding.json"}
    assert pack["files"]["binding.json"] == files["binding.json"] and source["source/FREEZE_SHA256.json"] == freeze
    assert attempt.remote([*source, *binding["input_files"]]) == {**source, **binding["input_files"]}
    fields = {"attempt": root, "binding_sha256": files["binding.json"], "source_freeze_sha256": freeze,
              "source_files_verified": len(source), "input_hashes_verified": len(binding["input_files"]),
              "launcher_exitcode": exitcode, "container_absent": absent, "job": job, "cleanup": cleanup,
              "state": {key: value for key, value in state.items() if key not in ("identity", "evaluation")}}
    return fields, pack, binding, preparation, state, source, lambda flag: arguments[arguments.index(flag) + 1]


def training(attempt):
    """Check the completed training attempt and return its record fields, state and frozen source hashes."""
    fields, pack, binding, preparation, state, source, option = launch(attempt, "train")
    metrics, loads = attempt.fetch([STANDING + "metrics.jsonl", STANDING + "force_metrics.json"])
    identity, updates, robots, files = state["identity"], state["updates"], int(option("--num-envs")), attempt.files
    config, per_update = identity["ppo_config"], robots * identity["ppo_config"]["num_steps_per_env"]
    assert updates == int(option("--updates")) == pack["updates"] and state["transitions"] == updates * per_update
    assert identity["seed"] == int(option("--seed")) == pack["seed"] == config["seed"]
    saved = range(config["save_interval"], updates + 1, config["save_interval"])
    names = [STANDING + "checkpoint_update%06d" % update for update in saved]
    weights, pairs = attempt.remote([name + ".pt" for name in names]), []
    for update, name, path in zip(saved, names, attempt.fetch([name + ".json" for name in names])):
        sidecar = read(path)
        assert (sidecar["updates"], sidecar["transitions"], sidecar["identity"]) == (update, update * per_update, identity)
        assert sidecar["checkpoint_sha256"] == weights[name + ".pt"]
        pairs.append({"update": update, "checkpoint_sha256": weights[name + ".pt"], "sidecar_sha256": files[name + ".json"]})
    assert saved[-1] == updates and pairs[-1]["checkpoint_sha256"] == state["checkpoint_sha256"]
    action = divergence = 0.0
    with metrics.open() as stream:
        for index, line in enumerate(stream, 1):
            row = read(line)
            assert row["update"] == index and row["transitions"] == index * per_update
            action = max(action, row["actions"]["mean_abs_max"])
            divergence = max(divergence, row["policy_update"]["before"]["kl_max"])
    force = read(loads)
    steps, window = updates * per_update // robots * identity["physics_config"]["decimation"], force["windows"]["full_training"]
    loads = ["total_vertical_support_n", *("applied_abs_torque_nm:" + joint for joint in identity["config"]["joint_names"])]
    assert index == updates and action <= 1 and force["status"] == "available" and force["physics_steps"] == steps
    assert window["samples_across_replicas"] == steps * robots and all(key in window["metrics"] for key in loads)
    record = {**fields, "source_commit": preparation["source_commit"], "seed": identity["seed"], "updates": updates,
              "reward_version": identity["reward_version"], "input_files": binding["input_files"],
              "transitions": state["transitions"], "metrics_sha256": files[STANDING + "metrics.jsonl"], "metrics_finite": True,
              "maximum_action_mean_magnitude": action, "before_update_kl_max": divergence,
              "force_metrics_sha256": files[STANDING + "force_metrics.json"], "checkpoint_pairs_verified": len(pairs),
              "checkpoint_pairs": pairs, "force_metrics": force}
    return record, state, source


def arrays(folder, controls, capture, physics):
    """Check the order and finiteness of one 400 Hz capture and its control trace, then return the force and torque peaks."""
    count, peaks = 0, dict.fromkeys(PEAKS, 0.0)
    finite = lambda group, rows: all(len(item) == rows and (item.dtype.kind != "f" or np.isfinite(item).all()) for item in group)
    for name in capture["substep_files"]:
        with np.load(folder / "native400hz" / name, allow_pickle=False) as data:
            step = count + 1 + np.arange(len(data["explicit_counter"]))
            order = [data["explicit_counter"] - capture["initial_counter"], data["sequence"] + 1,
                     data["control_index"] * physics["decimation"] + data["substep_index"] + 1]
            assert (np.array(order) == step).all() and finite([data[key] for key in data.files], len(step))
            assert np.allclose(data["time_s"], step * physics["physics_dt"], rtol=0, atol=1e-9)
            peaks = {key: max(peak, float(np.abs(data[key]).max())) for key, peak in peaks.items()}
            count = int(step[-1])
    with np.load(folder / "control_trace.npz", allow_pickle=False) as data:
        assert finite([data[key] for key in data.files], controls)
    assert count == capture["steps"] == capture["final_counter"] - capture["initial_counter"]
    assert [[peaks["applied_torque_nm"]], [peaks["computed_torque_nm"]]] == [capture[key] for key in BOUNDS[2:]]
    return peaks


def evaluation(attempt, trained, state, trained_source, goal):
    """Check one completed evaluation of the final checkpoint and return its record fields."""
    fields, _, binding, preparation, own, source, option = launch(attempt, "evaluate")
    summary, allocation = map(read, attempt.fetch([EVALUATION + "summary.json", EVALUATION + "allocation.json"]))
    checkpoint, physics, seed = state["checkpoint_sha256"], state["identity"]["physics_config"], state["identity"]["seed"]
    weights = REMOTE + trained + "/" + STANDING + "checkpoint_update%06d.pt" % state["updates"]
    assert checkpoint == option("--checkpoint-sha256") == allocation["checkpoint_sha256"] == binding["input_files"][weights]
    assert (preparation["training_attempt"], preparation["checkpoint"], source) == (REMOTE + trained, weights, trained_source)
    assert summary["allocation"] == allocation and own["evaluation"] == summary and own["identity"]["physics_config"] == physics
    cases, results = allocation["selected_case_ids"], summary["results"]
    assert [row["case_id"] for row in results] == cases and summary["missing_selected_probe_cases"] == []
    assert summary["failed_probe_cases"] == [row["case_id"] for row in results if not row["pass"]]
    assert summary["acquisition_failure"] is None and not summary["allocation_limit_reached"] and not summary["stage2_complete"]
    bases = [EVALUATION + "batch_%03d/" % index for index in range(len(cases))]
    attempt.remote([base + "native400hz/contacts.jsonl" for base in bases])
    paths = attempt.fetch([base + name for base in bases for name in ("report.json", "native400hz/capture.json")])
    rows = list(zip(bases, cases, results, map(read, paths[0::2]), map(read, paths[1::2])))
    attempt.fetch([base + name for base, _, _, report, capture in rows for name in (
        *report["files"], *("native400hz/" + item for item in capture["files"] if item != "contacts.jsonl"))])
    batches, files, verified = [], attempt.files, 0
    for base, case, result, report, capture in rows:
        native = {"native400hz/" + name: digest for name, digest in capture["files"].items()}
        substeps = {name: digest for name, digest in native.items() if "substeps_" in name}
        assert all(files[base + name] == digest for name, digest in {**report["files"], **native}.items())
        assert sorted(capture["files"]) == sorted(["contacts.jsonl", "initial_state.json", *capture["substep_files"]])
        force = read(attempt.local / base / "force_metrics.json")
        assert force["source_files"] == {"declaration.json": report["files"]["declaration.json"],
                                         "native400hz/capture.json": files[base + "native400hz/capture.json"], **substeps}
        assert (report["assigned_case_ids"], report["results"], report["checkpoint_sha256"]) == ([case], [result], checkpoint)
        assert report["acquisition_complete"] and report["failure"] is None and report["native_capture_failure"] is None
        assert report["recorded_physics_steps"] == capture["steps"] == physics["decimation"] * report["controls"]
        assert capture["initial_counter"] == sum(batch["native_steps"] for batch in batches)
        assert result["recorded_controls"] == report["controls"] and result["lineage"]["checkpoint_sha256"] == checkpoint
        assert result["native_capture_complete"] and capture["failure"] is None and report["seed"] == allocation["seed"] == seed
        assert force["status"] == "available" and force["capture_failure"] is None and force["checkpoint_sha256"] == checkpoint
        assert [(row["case_id"], row["recorded_samples"], row["requested_window_complete"]) for row in force["cases"]] == [
            (case, capture["steps"], True)] and force["physics_hz"] == round(1 / physics["physics_dt"])
        assert ("rollout.mp4" in report["files"]) == ("first_frame.png" in report["files"]) == (case == GOALS[goal])
        verified += 2 + len(report["files"]) + len(native)
        batches.append({
            "batch": base[-10:-1], "case_id": case, "pass": result["pass"], "failed_bounds": result["failed_bounds"],
            "report_sha256": files[base + "report.json"], "capture_sha256": files[base + "native400hz/capture.json"],
            "files": report["files"], "native_steps": capture["steps"], "native_bounds": {key: capture[key] for key in BOUNDS},
            "native_files": {name: capture["files"][name] for name in ("contacts.jsonl", "initial_state.json")},
            "native_peaks": arrays(attempt.local / base, report["controls"], capture, physics), "force_metrics": force})
    index = cases.index(GOALS[goal])
    report, video = rows[index][3], {"case_id": GOALS[goal], "batch": batches[index]["batch"], "published": {}}
    assert allocation["record_video"] and allocation["video_case_id"] == report["video_case_id"] == GOALS[goal]
    assert summary["video_files"] == [video["batch"] + "/rollout.mp4"] and report["video_frames"] > 0
    return {**fields, "eval_scope": option("--eval-scope"), "checkpoint_sha256": checkpoint,
            "summary_sha256": files[EVALUATION + "summary.json"], "allocation_sha256": files[EVALUATION + "allocation.json"],
            "summary": summary, "video": {**video, "frames": report["video_frames"]}, "batches": batches,
            "unique_files_hash_verified": verified, "native_force_arrays_finite_and_sequential": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True, help="Training attempt name under " + REMOTE)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Directory for both records. The default is the workspace.")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--publish-media", action="store_true", help="Copy each absent video and poster into " + MEDIA)
    args = parser.parse_args()
    label, date = re.fullmatch(r"ppo_v4_(.+)_(\d{8})_\d{3}", args.attempt).groups()
    outputs = [(args.output or args.workspace) / (label + "_%s_verification.json" % kind) for kind in ("training", "evaluation")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Use a fresh output path")
    attempts = [Attempt(args.attempt + suffix, args.workspace, args.fetch) for suffix in ("", *VIDEOS)]
    record, state, source = training(attempts[0])
    rows = [evaluation(item, args.attempt, state, source, goal) for item, (_, goal) in zip(attempts[1:], VIDEOS.values())]
    pins = {item.name: hashlib.sha256(json.dumps(item.files, sort_keys=True).encode()).hexdigest() for item in attempts}
    if {name: PINS.get(name) for name in pins} != pins:
        raise SystemExit("PINS differ from the retained files. Measured pins: " + json.dumps(pins, indent=4))
    for item, row in zip(attempts, (record, *rows)):
        row.update(inventory_sha256=pins[item.name], inventory_files=len(item.files))
    for item, row, (kind, _) in zip(attempts[1:], rows, VIDEOS.values()):
        video, hashes = row["video"], row["batches"][int(row["video"]["batch"][-3:])]["files"]
        for name, ending in (("rollout.mp4", ".mp4"), ("first_frame.png", "_poster.png")):
            origin, target = EVALUATION + video["batch"] + "/" + name, MEDIA + "ppo_v4_%s_%s_%s" % (label, kind, date) + ending
            if args.publish_media and not (ROOT / target).exists():
                (ROOT / target).write_bytes((item.local / origin).read_bytes())
            assert sha(ROOT / target) == hashes[name], "Published media differs: " + target
            video["published"][target] = {"source": REMOTE + item.name + "/" + origin, "sha256": hashes[name],
                                          "bytes": (ROOT / target).stat().st_size}
    goals = [{"goal": goal, "case_id": case, "evaluation": row["attempt"], "batch": batch["batch"], "pass": batch["pass"],
              "failed_bounds": batch["failed_bounds"]}
             for goal, case in GOALS.items() for row in rows for batch in row["batches"] if batch["case_id"] == case]
    assert {row["goal"] for row in goals} == set(GOALS)
    shared = {"verified": True, "native_started_by_analysis": False, "checked_utc": datetime.now(timezone.utc).isoformat(),
              "analysis_sha256": sha(__file__), "training_attempt": args.attempt, "methods": METHODS}
    verdict = {"all_goal_probes_pass": all(row["pass"] for row in goals), "goal_probes": goals,
               "unique_files_hash_verified": sum(row["unique_files_hash_verified"] for row in rows),
               "native_force_arrays_finite_and_sequential": True}
    records = [
        {"schema": "hexapod_ppo_training_verification_v1", **shared, **record},
        {"schema": "hexapod_ppo_evaluation_verification_v1", **shared, "checkpoint_sha256": state["checkpoint_sha256"],
         "seed": state["identity"]["seed"], "update": state["updates"], **verdict, "evaluations": rows}]
    for path, item in zip(outputs, records):
        path.write_text(json.dumps(item, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"updates": record["updates"], "transitions": record["transitions"], **verdict, "pins": pins,
                      "probes": [[batch["case_id"], batch["pass"]] for row in rows for batch in row["batches"]]}))


if __name__ == "__main__":
    main()
