"""Verify one retained training attempt of the controlled reward v4 comparison and its 13-probe evaluations.

The script reads the launch, exit, cleanup, state, metric and checkpoint records of the training attempt, then the summary,
batch reports and native 400 Hz captures of each evaluation. It compares each recorded sha256 with the retained bytes,
checks sequence continuity and requested-window completeness, and recomputes each probe's contact-normal force and motor
torque summary from the hash-checked raw chunks with locomotion/force_metrics.py. It starts no native process.
Run it from the repository root with PYTHONPATH set to that root.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

import numpy as np

from locomotion import force_metrics

REMOTE = "/srv/cupi/hexapod/runs/james/"
STANDING, EVALUATION = "run/standing/", "run/standing/evaluation/"
RECORDS = ("PACK.json", "binding.json", "preparation.json", "run/cleanup.json", "run/jobs/standing.json",
           STANDING + "state.json", STANDING + "ppo_config.json", "run/launch_binding.json", "launcher.exitcode")
EXPECTED_CONTROLS = {"omni_static": 1000, "quiet_stand": 1000, "stage2_long_quiet": 1600, "stop_to_stand": 1050}
PEAKS = ("distal_force_world_n", "nonfoot_force_world_n", "computed_torque_nm", "applied_torque_nm")
BOUNDS = ("joint_bound_violation_steps", "speed_bound_violation_steps", "maximum_applied_nm", "maximum_requested_nm")
RELATIVE_TOLERANCE = 1e-9
# Each pin is the sha256 of the sorted JSON map from each file that the script checks in one attempt to its sha256.
PINS = {
    "ppo_v4_omni_seed20260917_20261007_001": "ca38dd6be326c5d8afc46bd101c8bf808f2acc5b681a7fa9b30b91361a2d11ca",
    "ppo_v4_omni_seed20260917_20261007_001_evaluate": "5a72ac18bbd0ccc712f0f3e3f6a12a475b7754af44a95f656512ea3b9f4a4798",
    "ppo_v4_omni_seed20260918_20261007_001": "abc7bbaebfd1dac11b495aaf0da217b38f7e64fcbddee7058f6761387a1b5019",
    "ppo_v4_omni_seed20260918_20261007_001_evaluate": "e3aa528e58ee1db831d1099f7c44857a11299fc263ac7d2a77ceb0cb71626fce",
    "ppo_v4_omni5k_B_seed20260917_20261007_001": "778f975621d591be05b1707c1c8871a073f8361b7fb05ffeb67129d411b47360",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u002000": "45f8c006c1ab93d012bec8296e047f9fc1b5e96a1c072a19f813959b8c4f7f19",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u003500": "2fa380ed329162b12224dd30f7189c2c322f368615ad506fb3811dd6656156e1",
    "ppo_v4_omni5k_B_seed20260917_20261007_001_evaluate_u005000": "a612305674b4633b3d209986d5227e5514223d658d02e161f63354e79383ab5d",
}
METHODS = {
    "scope": "One training attempt and its retained 13-probe evaluations, one deterministic trial per probe (actor mean "
             "without sampling noise). The record verifies bytes and copies the evaluator's verdicts; it adds no acceptance "
             "limit and each summary must report Stage 2 as incomplete.",
    "hashes": "Records, traces and 400 Hz arrays come to the workspace through tar streams and the script hashes the copies. "
              "Frozen source files, input files, checkpoint weights, contacts.jsonl and videos stay on Spark, where sha256sum "
              "hashes them; a workspace file caches those digests. Each pin covers the sha256 of each checked file of one "
              "attempt.",
    "launch": "Each attempt must hold launcher exit code 0, a completed job and state with no errors, and a cleanup record "
              "whose inspections all report the owned container absent. binding.json must match PACK.json, preparation.json "
              "and the launch copy; one source freeze hash must appear in each record.",
    "training": "metrics.jsonl must hold one row per update in order with transitions = update * 128 * 24. Each saved update "
                "must hold a sidecar that names it, its transitions, the state's identity and its weights' sha256. The "
                "identity must record training_updates and the exploration decay length of the pack. force_metrics.json "
                "must hold available training loads over every physics step.",
    "continuity": "Over a batch's substep files, sequence must run from 0 to the capture's step count minus 1 and equal "
                  "control_index * 8 + substep_index; explicit_counter must equal the capture's initial counter + sequence + 1; "
                  "each initial counter must equal the sum of the earlier batches' steps; time_s must equal (sequence + 1) * "
                  "physics_dt. Every float array must be finite.",
    "completeness": "Each probe must record its profile's full control count, the capture must hold eight physics steps per "
                    "control, and the load summary must mark the requested window complete.",
    "loads": "locomotion/force_metrics.py report_for recomputes each probe's load summary from the substep chunks after it "
             "checks each chunk's sha256 against capture.json. The recomputed cases must equal the retained "
             "force_metrics.json within a relative tolerance of 1e-9. Forces are contact-normal forces and exclude "
             "tangential friction; requested torque (before the effort ceiling) and applied torque stay separate fields.",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(source):
    return json.loads(source.read_text() if isinstance(source, Path) else source, parse_constant=int)


class Attempt:
    """Hold one retained attempt, its workspace copy and the sha256 of each file that the script checks."""

    def __init__(self, name, workspace, enabled):
        self.name, self.local, self.enabled, self.files = name, workspace / "inputs" / name, enabled, {}

    def fetch(self, names):
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
        cache = self.local / "remote_sha256.json"
        known = read(cache) if cache.exists() else {}
        if self.enabled and not set(names) <= set(known):
            self.local.mkdir(parents=True, exist_ok=True)
            command = "cd " + shlex.quote(REMOTE + self.name) + " && sha256sum -- " + " ".join(map(shlex.quote, names))
            lines = subprocess.run(["ssh", "spark", command], check=True, capture_output=True, text=True).stdout
            known.update({line[66:]: line[:64] for line in lines.splitlines()})
            cache.write_text(json.dumps(known, indent=1) + "\n")
        self.files.update({name: known[name] for name in names})
        return {name: known[name] for name in names}


def launch(attempt, mode):
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
    fields = {"attempt": root, "source_commit": preparation["source_commit"], "binding_sha256": files["binding.json"],
              "source_freeze_sha256": freeze, "arguments": arguments, "source_files_verified": len(source),
              "input_hashes_verified": len(binding["input_files"]), "input_files": binding["input_files"],
              "launcher_exitcode": exitcode, "container_absent": absent,
              "job": {key: job[key] for key in ("container_id", "container_name", "status", "exit_code",
                                                "other_gpu_processes_seen") if key in job},
              "wall_seconds": state.get("wall_seconds")}
    return fields, pack, binding, preparation, state, source, lambda flag: arguments[arguments.index(flag) + 1]


def training(attempt):
    fields, pack, binding, preparation, state, source, option = launch(attempt, "train")
    metrics, loads = attempt.fetch([STANDING + "metrics.jsonl", STANDING + "force_metrics.json"])
    identity, updates, files = state["identity"], state["updates"], attempt.files
    config, per_update = identity["ppo_config"], 128 * identity["ppo_config"]["num_steps_per_env"]
    assert updates == int(option("--updates")) == pack["updates"] and state["transitions"] == updates * per_update
    assert identity["seed"] == int(option("--seed")) == pack["seed"] == config["seed"]
    if "--extended-updates" in binding["command_args"]:
        assert identity["training_updates"] == updates and pack["extended_updates"] is True
    if "action_std_decay_updates" in pack:
        assert config["exploration"]["action_std_decay_updates"] == pack["action_std_decay_updates"]
    saved = range(config["save_interval"], updates + 1, config["save_interval"])
    names = [STANDING + "checkpoint_update%06d" % update for update in saved]
    weights, pairs = attempt.remote([name + ".pt" for name in names]), []
    for update, name, path in zip(saved, names, attempt.fetch([name + ".json" for name in names])):
        sidecar = read(path)
        assert (sidecar["updates"], sidecar["transitions"], sidecar["identity"]) == (update, update * per_update, identity)
        assert sidecar["checkpoint_sha256"] == weights[name + ".pt"]
        pairs.append({"update": update, "checkpoint_sha256": weights[name + ".pt"], "sidecar_sha256": files[name + ".json"]})
    assert saved[-1] == updates and pairs[-1]["checkpoint_sha256"] == state["checkpoint_sha256"]
    with metrics.open() as stream:
        for index, line in enumerate(stream, 1):
            row = read(line)
            assert row["update"] == index and row["transitions"] == index * per_update
    force = read(loads)
    steps = updates * per_update // 128 * identity["physics_config"]["decimation"]
    window = force["windows"]["full_training"]
    assert index == updates and force["status"] == "available" and force["physics_steps"] == steps
    assert window["samples_across_replicas"] == steps * 128
    evaluated = {str(pair["update"]): pair for pair in pairs if pair["update"] in (2000, 3500, 5000)}
    record = {**fields, "seed": identity["seed"], "updates": updates, "reward_version": identity["reward_version"],
              "reward_options": identity.get("reward_options"), "training_updates": identity.get("training_updates"),
              "exploration": config.get("exploration"), "transitions": state["transitions"],
              "metrics_sha256": files[STANDING + "metrics.jsonl"], "metrics_rows": index,
              "force_metrics_sha256": files[STANDING + "force_metrics.json"], "checkpoint_pairs_verified": len(pairs),
              "checkpoint_pairs_sha256": hashlib.sha256(json.dumps(pairs, sort_keys=True).encode()).hexdigest(),
              "evaluated_checkpoints": evaluated,
              "training_loads": {"scope": force["scope"], "full_training": window}}
    return record, state, source, {pair["update"]: pair["checkpoint_sha256"] for pair in pairs}


def arrays(folder, controls, capture, physics):
    """Check the order and finiteness of one 400 Hz capture and its control trace; return force and torque peaks."""
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
    return {"contact_normal_distal_n": peaks["distal_force_world_n"], "contact_normal_nonfoot_n": peaks["nonfoot_force_world_n"],
            "requested_torque_nm": peaks["computed_torque_nm"], "applied_torque_nm": peaks["applied_torque_nm"]}


def difference(left, right):
    """Largest relative difference between two JSON values of the same shape; None when the shapes differ."""
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return None
        values = [difference(left[key], right[key]) for key in left]
        return None if None in values else max(values, default=0.)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return None
        values = [difference(a, b) for a, b in zip(left, right)]
        return None if None in values else max(values, default=0.)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool):
        return abs(left - right) / max(abs(left), abs(right), 1e-12)
    return 0. if left == right else None


def loads(force):
    """Compact load summary of one probe's full trial: contact-normal force per foot, requested and applied torque."""
    window = force["cases"][0]["windows"]["full_trial"]
    motors = window["motor_torque"]
    return {"total_vertical_support_n_mean": window["total_vertical_support_n"]["mean"],
            "contact_normal_force_per_foot_mean_n": {leg: row["normal_resultant_magnitude_n"]["mean"]
                                                     for leg, row in window["per_foot"].items()},
            "contact_fraction_per_foot": {leg: row["contact_fraction"] for leg, row in window["per_foot"].items()},
            "requested_torque": {key: motors["requested"][key] for key in ("worst_joint", "worst_joint_rms_nm",
                                                                           "mean_abs_across_joints_nm")},
            "applied_torque": {key: motors["applied"][key] for key in ("worst_joint", "worst_joint_rms_nm",
                                                                       "mean_abs_across_joints_nm")}}


def evaluation(attempt, trained, state, trained_source, checkpoints):
    fields, pack, binding, preparation, own, source, option = launch(attempt, "evaluate")
    summary, allocation = map(read, attempt.fetch([EVALUATION + "summary.json", EVALUATION + "allocation.json"]))
    physics, seed = state["identity"]["physics_config"], state["identity"]["seed"]
    weights = preparation["checkpoint"]
    update = int(weights.rsplit("checkpoint_update", 1)[1][:6])
    checkpoint = checkpoints[update]
    assert weights == REMOTE + trained + "/" + STANDING + "checkpoint_update%06d.pt" % update
    assert checkpoint == option("--checkpoint-sha256") == allocation["checkpoint_sha256"] == binding["input_files"][weights]
    assert preparation["training_attempt"] == REMOTE + trained and source == trained_source
    assert summary["allocation"] == allocation and own["evaluation"] == summary and own["identity"]["physics_config"] == physics
    cases, results = allocation["selected_case_ids"], summary["results"]
    assert len(cases) == 13 and [row["case_id"] for row in results] == cases and summary["missing_selected_probe_cases"] == []
    assert summary["failed_probe_cases"] == [row["case_id"] for row in results if not row["pass"]]
    assert summary["acquisition_failure"] is None and not summary["allocation_limit_reached"] and not summary["stage2_complete"]
    bases = [EVALUATION + "batch_%03d/" % index for index in range(len(cases))]
    paths = attempt.fetch([base + name for base in bases for name in ("report.json", "native400hz/capture.json")])
    rows = list(zip(bases, cases, results, map(read, paths[0::2]), map(read, paths[1::2])))
    stay = {"rollout.mp4", "first_frame.png"}
    attempt.remote([base + name for base, _, _, report, _ in rows for name in report["files"] if name in stay]
                   + [base + "native400hz/contacts.jsonl" for base in bases])
    attempt.fetch([base + name for base, _, _, report, capture in rows for name in (
        *(item for item in report["files"] if item not in stay),
        *("native400hz/" + item for item in capture["files"] if item != "contacts.jsonl"))])
    batches, files, verified, largest = [], attempt.files, 0, 0.
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
        controls = EXPECTED_CONTROLS[result["profile"]]
        assert report["controls"] == result["recorded_controls"] == controls and result["checks"]["complete_requested_window"]["status"] == "pass"
        assert report["recorded_physics_steps"] == capture["steps"] == physics["decimation"] * controls
        assert capture["initial_counter"] == sum(batch["native_steps"] for batch in batches)
        assert result["native_capture_complete"] and capture["failure"] is None and report["seed"] == allocation["seed"] == seed
        assert force["status"] == "available" and force["capture_failure"] is None and force["checkpoint_sha256"] == checkpoint
        assert [(row["case_id"], row["recorded_samples"], row["expected_samples"], row["requested_window_complete"])
                for row in force["cases"]] == [(case, capture["steps"], capture["steps"], True)]
        recomputed = force_metrics.report_for(attempt.local / base)
        gap = difference(json.loads(json.dumps(recomputed["cases"])), force["cases"])
        assert gap is not None and gap <= RELATIVE_TOLERANCE, (case, gap)
        assert recomputed["source_files"] == force["source_files"]
        largest = max(largest, gap)
        verified += 2 + len(report["files"]) + len(native)
        batches.append({
            "batch": base[-10:-1], "case_id": case, "pass": result["pass"], "failed_bounds": result["failed_bounds"],
            "report_sha256": files[base + "report.json"], "capture_sha256": files[base + "native400hz/capture.json"],
            "force_metrics_sha256": files[base + "force_metrics.json"], "native_steps": capture["steps"],
            "controls": controls, "requested_window_complete": True,
            "native_bounds": {key: capture[key] for key in BOUNDS}, "native_peaks": arrays(attempt.local / base, controls,
                                                                                         capture, physics),
            "recomputed_load_relative_difference": gap, "loads_full_trial": loads(force)})
    return {**fields, "update": update, "eval_scope": option("--eval-scope"), "checkpoint_sha256": checkpoint,
            "summary_sha256": files[EVALUATION + "summary.json"], "allocation_sha256": files[EVALUATION + "allocation.json"],
            "passes": sum(batch["pass"] for batch in batches), "failed_probe_cases": summary["failed_probe_cases"],
            "batches": batches, "unique_files_hash_verified": verified,
            "largest_recomputed_load_relative_difference": largest,
            "native_force_arrays_finite_and_sequential": True, "requested_windows_complete": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True, help="Training attempt name under " + REMOTE)
    parser.add_argument("--evaluation", action="append", required=True,
                        help="Evaluation attempt name under " + REMOTE + "; repeat for each checkpoint")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh output path")
    attempts = [Attempt(name, args.workspace, args.fetch) for name in (args.attempt, *args.evaluation)]
    record, state, source, checkpoints = training(attempts[0])
    rows = [evaluation(item, args.attempt, state, source, checkpoints) for item in attempts[1:]]
    pins = {item.name: hashlib.sha256(json.dumps(item.files, sort_keys=True).encode()).hexdigest() for item in attempts}
    if {name: PINS.get(name) for name in pins} != pins:
        raise SystemExit("PINS differ from the retained files. Measured pins: " + json.dumps(pins, indent=4))
    for item, row in zip(attempts, (record, *rows)):
        row.update(inventory_sha256=pins[item.name], inventory_files=len(item.files))
    receipt = {"schema": "hexapod_reward_v4_comparison_verification_v1", "verified": True,
               "native_started_by_analysis": False, "checked_utc": datetime.now(timezone.utc).isoformat(),
               "analysis_sha256": sha(__file__), "force_metrics_sha256": sha(force_metrics.__file__),
               "methods": METHODS, "training": record, "evaluations": rows, "stage2_complete": False,
               "unique_files_hash_verified": sum(row["unique_files_hash_verified"] for row in rows),
               "largest_recomputed_load_relative_difference": max(row["largest_recomputed_load_relative_difference"]
                                                                  for row in rows)}
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"attempt": args.attempt, "updates": record["updates"], "pins": pins,
                      "evaluations": {row["update"]: row["passes"] for row in rows},
                      "largest_recomputed_load_relative_difference": receipt["largest_recomputed_load_relative_difference"]}))


if __name__ == "__main__":
    main()
