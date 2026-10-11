"""Record every native attempt of the PPO walking diagnosis sprint in launch order.

The script reads the retained record files of each Spark attempt whose name holds 20261004, 20261005 or 20261006. It
records the source, learner flags, reward fields, exit state and training speed of each training attempt, and the
checkpoint, scope and four goal cases of each evaluation. It tabulates the goal cases of the final learner path at
update 2000. Failed, interrupted, staged and running attempts stay in the record. It starts no native process.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

REMOTE = "/srv/cupi/hexapod/runs/james/"
DATES = ("20261004", "20261005", "20261006")
BASE = ("PACK.json", "binding.json", "preparation.json")
EXIT, CLEANUP, JOB, STATE = "launcher.exitcode", "run/cleanup.json", "run/jobs/standing.json", "run/standing/state.json"
TASK, METRICS = "run/standing/task/task_definition.json", "run/standing/metrics.jsonl"
SUMMARY, CHECKPOINT = "run/standing/evaluation/summary.json", "run/standing/checkpoint_update%06d.json"
FILES = {"train": (EXIT, CLEANUP, JOB, STATE, TASK, METRICS), "evaluate": (EXIT, CLEANUP, JOB, STATE, SUMMARY),
         "probe": (EXIT, CLEANUP, JOB, STATE)}
KINDS = {"train": "training", "evaluate": "evaluation", "probe": "probe"}
# The files of an IN_PROGRESS attempt can change. Its pin covers BASE alone, the script reads no metrics.jsonl from it,
# and the script tolerates a missing exit code for it.
IN_PROGRESS = ()
FINAL, GOAL_UPDATE = "ppo_v4_settle_seed20260917_20261006_001", 2000  # FINAL holds the final reward configuration
GOAL_ATTEMPTS = ("ppo_v4_action_wall_20261005_001", "ppo_v4_vnoise_seed20260917_20261005_001",
                 "ppo_v4_vnoise_seed20260918_20261005_001", "ppo_v4_vnoise_seed20260919_20261005_001",
                 "ppo_v4_restload_seed20260917_20261005_001", "ppo_v4_restload_seed20260918_20261005_001",
                 "ppo_v4_settle_seed20260917_20261006_001", "ppo_v4_settle_seed20260918_20261006_001")
CASES = {"forward": "learning:translate_0.05_0deg", "quiet_20s": "learning:quiet_20s",
         "quiet_32s": "learning:quiet_32s", "forward_to_stop": "learning:forward_0.05_to_stop"}
CASE_METRICS = ("planar_error_mps", "yaw_error_rad_s", "tilt_rms_degrees", "computed_demand_over_rating_fraction",
                "max_joint_velocity_rms_rad_s", "max_target_step_abs_p95_rad_per_20ms", "max_joint_position_range_rad")
REWARD_FIELDS = ("action_limit_weight", "swing_travel_full", "planar_rate_weight", "over_rating_weight",
                 "quiet_contact_weight", "quiet_load_n", "quiet_strain_weight", "quiet_grace_controls")
SPEED, WINDOW = "moving_signed_command_direction_speed_mps", 400
# Each pin is the sha256 of one attempt's manifest. The manifest holds one line "<sha256>  <path>" for each file that the
# script reads from the attempt, in sorted path order, as sha256sum prints them from the attempt directory.
PINS = {
    "ppo_reward_v1_control_20261004_001": "4041957d528288a786ce287c1f1574214ea71792d9a7d405ae1c736861b447b8",
    "ppo_reward_v1_control_20261004_001_evaluate": "4bd59f1f27aa291bf9f87b0d5ed4f4e7e4bcc164a1e842fc431cb87a64cba1ad",
    "ppo_noise_probe_20261004_001": "2120d324d09154bef579dd64adaa7489724993e8b26db19d3004b4a9708dec5b",
    "ppo_v2_corrected_path_20261004_001": "81d0faad291864919d0f3b11dc4cd3ea230b9a2297a71a92d14802eaedbec6ca",
    "ppo_v2_corrected_path_20261004_001_evaluate": "9a87cd6d49f9c520af9b0a74b255333699d05aebdffb8d71e424e98620634337",
    "ppo_v1_corrected_path_20261004_001": "e80a7160dc079237119ea4f984d070833cf5dd2f2b50bcf89a116a1b3cdf7f4c",
    "ppo_v1_corrected_path_20261004_001_evaluate": "f67af2ed7d3300734bb1b2ce5e30670deb7bacba2cbe1313808e19ab9d081dea",
    "ppo_v4_draft_std015_20261004_001": "5a2f044ae3aa66b7146557422b453f05f79868a93cff39c46587e3f1ef480541",
    "ppo_v4_draft_std015_20261004_001_evaluate": "b556aaef245375fa5710b9267f54d4117c4a784a5233a00e7505c688efc2035b",
    "ppo_v4_draft_std008_20261004_001": "20a4182a5e0cc0b6c6cacdb836c9b28789f9870e8dae3cb4d63e22680ff35f20",
    "ppo_v4_draft_std008_20261004_001_evaluate": "a0aa751559c38c0d8c8f999a71f0e0dabccfa2cbbb8c524b7eb64eee2ccdb3f2",
    "ppo_v4_clock_20261004_001": "840e9c2e45535f12bcd716487a4f3e342dd2e7c00da785ec4cb47f50637e129e",
    "ppo_v4_clock_20261004_001_evaluate": "f9cbd5fe9cfb084d4978b1dc73b6d819010ebb87609d531a0db7970be01a13ec",
    "ppo_noise_probe_20261004_002": "5a04471c6c37b2f33e59bd0cd47393e463a2ff7984eed3f8768ee8771eb5b160",
    "ppo_v4_forward_20261004_001": "1e01ae840e86c68cb1a0fb8806468c4eb8630a1287226c09416bfc60c9a41981",
    "ppo_v4_forward_20261004_001_evaluate": "adf34b98908cf1f5c052e8d189cd4e811ab27735eefb2f5812084cff4165a458",
    "ppo_v4_forward_20261004_001_evaluate_stop_video": "8e2fc0a1b9f8cec58af5004989e9522095453e4c9a2f832bf8350fed9fa95b37",
    "ppo_v4_forward_20261004_001_evaluate_u0600": "2947711435691a2f295259ec3c60a41626a4ca546cf5c0cbc719d2659b2e5fa8",
    "ppo_v4_forward_20261004_001_evaluate_u1000": "90c56342f0629a3aa5be1a05e2d18dce304086940f78b56575a5a8ddf7644871",
    "ppo_v4_forward_20261004_001_evaluate_u1400": "b3f3382b9fd7ab3cbbeeca0cc953136bf9d6f98ccb92d265a0d25b55517814a7",
    "ppo_v4_action_wall_20261005_001": "2a53b66607263622aa2f47c9f6b3077fbd1f07781d67999b44da7f3bb31641c5",
    "ppo_v4_action_wall_20261005_001_evaluate_stop_video": "fef2d9e0fd9c21ff71c7e16bede0cef0c1ae10b0cac415089705ea07ace02f94",
    "ppo_v4_seed20260917_20261005_001": "61acb3652062dac239555178e49a49e73824a754cc987decc1d6a68d1bdf4510",
    "ppo_v4_smooth_seed20260917_20261005_001": "a685dff198f207bc21683b17d533341636f66d07280ef1e81603b53d0cf6975e",
    "ppo_v4_action_wall_20261005_001_evaluate_u1500": "811d5cd2e4016c2e2570bf45b8f7523bd68804b16487768c592f89096160b59d",
    "ppo_v4_vnoise_seed20260917_20261005_001": "2ae381b21e8346ded70e59cf81ac55eec981038337c45322f13d220898594c86",
    "ppo_v4_vnoise_seed20260917_20261005_001_evaluate": "5b8a1ffe241269805043164849a2cb74e5be48f68973f5a3f46db499b20d6547",
    "ppo_v4_vnoise_seed20260917_20261005_001_evaluate_stop_video": "028935f879d252f242fbb0537411f6c4002e87a1742cf24dd472e8f779f732b7",
    "ppo_v4_vnoise_seed20260918_20261005_001": "10fbd2f58aca128abfac7f8cad426cd35853cad0e24de92d292b472584f55ff8",
    "ppo_v4_vnoise_seed20260918_20261005_001_evaluate": "0c8fa5b41fa4e42b4508a220a8cf428bfc0e9bf4072ad4aa5bae95983142b721",
    "ppo_v4_vnoise_seed20260918_20261005_001_evaluate_stop_video": "48b0f49748d2ff4b3b09fbecc19190673f9a05f9fc401ff407ad20911dcf7571",
    "ppo_v4_vnoise_seed20260919_20261005_001": "9a880859c6e47f45ce6d7e32a8f0b83e5180864d96581fe1da8af289f5cd538b",
    "ppo_v4_vnoise_seed20260919_20261005_001_evaluate": "68c28922dac0ef26b60b1310805b4212918411a6c5617bb7cb082307fdf964ac",
    "ppo_v4_vnoise_seed20260919_20261005_001_evaluate_stop_video": "9413d21742dfec9db632ac85f82a10610e121a9cf86eec46ada92ddc0aa357dd",
    "ppo_v4_restload_seed20260917_20261005_001": "c6e0049c9881bd9e3680f43137efad1b7151f751f947cf392217d13de2792561",
    "ppo_v4_restload_seed20260917_20261005_001_evaluate": "eb5832ade593373a5b4b50040fbe5e11bb67e1a05c7570e08cae8faaef6cd0cc",
    "ppo_v4_restload_seed20260917_20261005_001_evaluate_stop_video": "19c6b32956ac4d9cdd64391191ca50c032815ad14740c0ba9604d36c0eace593",
    "ppo_v4_restload_seed20260918_20261005_001": "b4a1bcb15eb0b283c24caf75db1ee273b28eba8df16db6d1b391cbf8a2e06101",
    "ppo_v4_restload_seed20260918_20261005_001_evaluate": "a9d8a7d7b2e28265f2a2fb9fdfeec54a5672cd56b18ce56fc6ad3f8a858ba5ea",
    "ppo_v4_restload_seed20260918_20261005_001_evaluate_stop_video": "0affc00067f5d49b149a92e6ddee550d19d23b2c11c0e014deb253a009b6d7db",
    "ppo_v4_settle_seed20260917_20261006_001": "c28191f9c0cfd919d244cca73f141f060cefa1e6d594e08c048fbaf6f0baa041",
    "ppo_v4_settle_seed20260917_20261006_001_evaluate": "58c5cfaf0a48832848286d6707b52edf1d6e13db38739e9da3b17bb778b698b8",
    "ppo_v4_settle_seed20260917_20261006_001_evaluate_stop_video": "cf4da4dc0f7a437ddbf28e2a0a98283966232c082fd61e0044be110d810a087e",
    "ppo_v4_settle_seed20260918_20261006_001": "cd79eae95f06a43c47478b8675d9ddb8acd5e52aba7d69fb579ee41f9a5c5a9d",
    "ppo_v4_settle_seed20260918_20261006_001_evaluate": "bf38006a3ec557b22fdbfcb46a023c38a47fd717aea0ffc6fecf554f0e0b2ced",
    "ppo_v4_settle_seed20260918_20261006_001_evaluate_stop_video":
        "60173757600e65ab15be7a6eb6a25b1b4f29d9d98a563b19f6e4a99e8b10b289",
}
METHODS = {
    "scope": "One row per attempt directory under the remote directory whose name holds one of the name_filter dates. "
             "The script copies values from the retained files and runs no policy. Each training attempt has one seed.",
    "listing": "The script saves ls -1 of the remote directory once. Each pinned attempt must appear in it. "
               "outside_ledger names each listed directory that matches the filter and has no pin.",
    "order": "Rows sort by started_unix of run/jobs/standing.json, the launcher's job record. An attempt without a job "
             "record has not launched; it sorts last and its state reads staged.",
    "pins": "For each attempt the script hashes each file that it reads and lists the hashes under sha256. The "
            "manifest holds one line per file, sha256, two spaces and path, in sorted path order, as sha256sum prints "
            "it from the attempt directory. The sha256 of the manifest must equal the attempt's pin.",
    "in_progress": "A row with in_progress true can still change on Spark. Its pin covers PACK.json, binding.json and "
                   "preparation.json. The row holds the other files as the workspace copied them, with their sha256, "
                   "and gives no training speed. A fresh workspace can read a later state.",
    "state": "state repeats the job record's status, or staged without one. launcher_exit_code reads launcher.exitcode "
             "and cleanup_container_absent the last inspection of run/cleanup.json. state_status, updates_completed, "
             "wall_seconds and last_checkpoint read run/standing/state.json, which a stopped attempt leaves at running.",
    "source": "source_commit reads preparation.json. binding_sha256 is the hash of binding.json and must equal the "
              "value in PACK.json and preparation.json. The three files must agree on source_freeze_sha256. An "
              "evaluation's preparation.json names its training attempt and no commit, so the row repeats the training "
              "attempt's commit after the two source freezes match.",
    "learner_flags": "The command_args of binding.json after the --standing-admission option and its value.",
    "reward": "reward_version reads PACK.json and reward_label reads task_definition.json. reward_fields copies eight "
              "values from reward.config of task_definition.json; null means the file holds no such field. A reward "
              "version 1 file holds no reward.config. differs_from_final names the fields whose value differs from the "
              "final_configuration_attempt. An attempt without task_definition.json has null in both places.",
    "training_speed": "Read task.interval_metrics.command_classes['linear_0.05']." + SPEED + " from the last 400 rows "
                      "of metrics.jsonl, or from each row when the file holds fewer. Average the values with equal "
                      "weight and skip null. rows_without_command_classes counts rows that hold no command_classes. "
                      "The rows come from training rollouts with exploration noise.",
    "evaluations": "checkpoint_update and checkpoint_sha256 read the evaluation's binding.json and must match the "
                   "training attempt's checkpoint record. eval_scope reads --eval-scope. cases_selected counts the "
                   "allocation's selected_case_ids, and passing counts results with pass true over results. Each "
                   "evaluation has one trial per case with the actor mean and no sampling noise.",
    "cases": "Each goal case copies pass and failed_bounds. mean_forward_velocity_mps is mean_velocity_mps[0]. Each "
             "other metric reads the result's metrics, or the value of the check with that name when the metrics hold "
             "no such key. Null means the result holds neither. In the stop case locomotion/evaluation.py compares the "
             "rest velocity with the preceding 0.05 m/s command, so planar_error_mps reads near 0.05, with no bound.",
    "goal_probes": "For each attempt of the final learner path, take each goal case from the first evaluation in launch "
                   "order at update 2000 that holds the case. evaluations lists the pass value of each such "
                   "evaluation. Null means no retained evaluation at update 2000 holds the case.",
    "limits": "The record establishes no Stage 2 result. A pass is one trial of one seed and measures no spread.",
}


def sha(source):
    return hashlib.sha256(source if isinstance(source, bytes) else Path(source).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def option(arguments, flag):
    return arguments[arguments.index(flag) + 1] if flag in arguments else None


def files(name, mode):
    """Name the files that the script reads from one attempt of the given mode, beyond BASE."""
    return tuple(item for item in FILES[mode] if item != METRICS or name not in IN_PROGRESS)


def fetch(names, inputs):
    """Copy the missing named files from Spark through a tar stream. Spark skips each name that it does not hold."""
    missing = [name for name in names if not (inputs / name).exists()]
    if missing:
        inputs.mkdir(parents=True, exist_ok=True)
        command = "tar -C " + shlex.quote(REMOTE) + " --ignore-failed-read -cf - " + " ".join(map(shlex.quote, missing))
        process = subprocess.Popen(["ssh", "spark", command], stdout=subprocess.PIPE)
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            archive.extractall(inputs, filter="data")
        assert process.wait() == 0


def speed(path):
    """Average the 0.05 m/s class speed over the last WINDOW rows of one metrics file."""
    lines = path.read_bytes().splitlines()
    rows = [json.loads(line) for line in lines[-WINDOW:]]
    assert [row["update"] for row in rows] == list(range(len(lines) - len(rows) + 1, len(lines) + 1))
    classes = [row["task"]["interval_metrics"].get("command_classes") for row in rows]
    values = [row["linear_0.05"][SPEED] for row in classes if row is not None and row["linear_0.05"][SPEED] is not None]
    return {"metrics_rows": len(lines), "first_update": rows[0]["update"], "last_update": rows[-1]["update"],
            "rows_without_command_classes": sum(row is None for row in classes), "updates_with_speed": len(values),
            "linear_0.05_speed_mps": sum(values) / len(values) if values else None}


def case(result):
    """Copy the verdict and the named metrics of one summary result."""
    metrics, checks = result.get("metrics", {}), result.get("checks", {})
    return {"pass": result["pass"], "failed_bounds": result["failed_bounds"],
            "mean_forward_velocity_mps": (metrics.get("mean_velocity_mps") or [None])[0],
            **{key: metrics[key] if key in metrics else checks.get(key, {}).get("value") for key in CASE_METRICS}}


def record(name, inputs, checkpoints, final):
    """Check one attempt's pin and build its row from the files that the workspace holds."""
    folder = inputs / name
    pack, binding, preparation = (read(folder / item) for item in BASE)
    mode, arguments = binding["mode"], binding["command_args"]
    held = [item for item in sorted(BASE + files(name, mode) + checkpoints) if (folder / item).exists()]
    hashes = {item: sha(folder / item) for item in held}
    manifest = ["%s  %s\n" % (digest, item) for item, digest in hashes.items() if name not in IN_PROGRESS or item in BASE]
    assert sha("".join(manifest).encode()) == PINS[name], name
    job, state, cleanup = (read(folder / item) if item in hashes else {} for item in (JOB, STATE, CLEANUP))
    assert hashes["binding.json"] == pack["binding_sha256"] == preparation["binding_sha256"] and pack["mode"] == mode
    assert pack["source_freeze_sha256"] == preparation["source_freeze_sha256"] == binding["source_freeze_sha256"]
    assert pack["seed"] == int(option(arguments, "--seed")) and (EXIT in hashes or name in IN_PROGRESS)
    row = {"name": name, "kind": KINDS[mode], "state": job.get("status", "staged"), "in_progress": name in IN_PROGRESS,
           "started_unix": job.get("started_unix"), "finished_unix": job.get("finished_unix"),
           "launcher_exit_code": int((folder / EXIT).read_text()) if EXIT in hashes else None,
           "job_error": job.get("error"), "state_status": state.get("status"),
           "cleanup_container_absent": cleanup["inspections"][-1]["absent"] if cleanup else None,
           "source_commit": preparation.get("source_commit"), "source_freeze_sha256": pack["source_freeze_sha256"],
           "binding_sha256": hashes["binding.json"], "seed": pack["seed"]}
    if mode == "train":
        task = read(folder / TASK) if TASK in hashes else None
        fields = task and {key: task.get("reward", {}).get("config", {}).get(key) for key in REWARD_FIELDS}
        row.update({
            "reward_version": pack["reward_version"], "reward_label": task and task["reward_version"],
            "learner_flags": arguments[arguments.index("--standing-admission") + 2:], "reward_fields": fields,
            "differs_from_final": fields and [key for key in REWARD_FIELDS if fields[key] != final[key]],
            "updates_requested": int(option(arguments, "--updates")), "updates_completed": state.get("updates"),
            "wall_seconds": state.get("wall_seconds"), "last_checkpoint_sha256": state.get("checkpoint_sha256"),
            "last_checkpoint": state.get("checkpoint") and Path(state["checkpoint"]).name,
            "training_speed": speed(folder / METRICS) if METRICS in hashes else None})
    elif mode == "evaluate":
        parent, update = Path(preparation["training_attempt"]).name, int(option(arguments, "--updates"))
        digest, saved = option(arguments, "--checkpoint-sha256"), read(inputs / parent / (CHECKPOINT % update))
        summary = read(folder / SUMMARY) if SUMMARY in hashes else {"results": [], "allocation": {}}
        results = {result["case_id"]: result for result in summary["results"]}
        assert preparation["checkpoint"] == REMOTE + parent + "/" + (CHECKPOINT % update)[:-4] + "pt"
        assert binding["input_files"][preparation["checkpoint"]] == digest == saved["checkpoint_sha256"]
        assert saved["updates"] == update and summary["allocation"].get("checkpoint_sha256") in (None, digest)
        row.update({
            "training_attempt": parent, "checkpoint_update": update, "checkpoint_sha256": digest,
            "eval_scope": option(arguments, "--eval-scope"),
            "cases_selected": len(summary["allocation"].get("selected_case_ids", [])), "results": len(results),
            "passing": sum(result["pass"] for result in results.values()),
            "cases": {label: case(results[case_id]) for label, case_id in CASES.items() if case_id in results},
            "video_files": summary.get("video_files")})
    else:
        row["probe"] = {key: state["probe"][key] for key in ("controls", "controls_completed", "terminations")}
    return {**row, "sha256": hashes}


def goal(parent, rows):
    """Tabulate the goal cases of one training attempt from its evaluations at GOAL_UPDATE."""
    wanted = (parent["name"], GOAL_UPDATE)
    chosen = [row for row in rows if (row.get("training_attempt"), row.get("checkpoint_update")) == wanted]
    verdicts = {}
    for label in CASES:
        held = {row["name"]: row["cases"][label] for row in chosen if label in row["cases"]}
        first = next(iter(held.values()), None)
        verdicts[label] = {"pass": first and first["pass"], "failed_bounds": first and first["failed_bounds"],
                           "evaluations": {key: value["pass"] for key, value in held.items()}}
    labels = lambda value: [label for label in CASES if verdicts[label]["pass"] is value]
    return {"training_attempt": parent["name"], "seed": parent["seed"], "state": parent["state"], "update": GOAL_UPDATE,
            **verdicts, "passed": labels(True), "failed": labels(False), "not_run": labels(None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    output = args.output or args.workspace / "run_ledger.json"
    if output.exists():
        raise FileExistsError("Use a fresh output path")
    inputs, listing = args.workspace / "inputs", args.workspace / "inputs/listing.txt"
    if args.fetch:
        fetch([name + "/" + item for name in PINS for item in BASE], inputs)
    if args.fetch and not listing.exists():
        with listing.open("xb") as stream:
            subprocess.run(["ssh", "spark", "ls -1 " + shlex.quote(REMOTE)], stdout=stream, check=True)
    names = [name for name in listing.read_text().split() if any(date in name for date in DATES)]
    assert set(PINS) <= set(names) and set(IN_PROGRESS + GOAL_ATTEMPTS + (FINAL,)) <= set(PINS)
    # An evaluation names its training attempt and update, so the training attempt's checkpoint record joins its files.
    wanted, checkpoints = [], {name: () for name in PINS}
    for name in PINS:
        binding = read(inputs / name / "binding.json")
        wanted += [name + "/" + item for item in files(name, binding["mode"])]
        if binding["mode"] == "evaluate":
            parent = Path(read(inputs / name / "preparation.json")["training_attempt"]).name
            saved = CHECKPOINT % int(option(binding["command_args"], "--updates"))
            checkpoints[parent] = tuple(sorted({*checkpoints[parent], saved}))
    if args.fetch:
        fetch(wanted + [name + "/" + item for name, items in checkpoints.items() for item in items], inputs)
    final = read(inputs / FINAL / TASK)["reward"]["config"]
    rows = [record(name, inputs, checkpoints[name], final) for name in PINS]
    rows.sort(key=lambda row: (row["started_unix"] is None, row["started_unix"] or 0, row["name"]))
    by_name = {row["name"]: row for row in rows}
    for row in rows:
        if row["kind"] == "evaluation":
            parent = by_name[row["training_attempt"]]
            assert row["source_freeze_sha256"] == parent["source_freeze_sha256"] and row["seed"] == parent["seed"]
            row["source_commit"] = parent["source_commit"]
    receipt = {"schema": "hexapod_ppo_run_ledger_v1", "analysis_sha256": sha(__file__), "remote_directory": REMOTE,
               "name_filter": DATES, "in_progress": IN_PROGRESS, "final_configuration_attempt": FINAL,
               "final_reward_fields": {key: final.get(key) for key in REWARD_FIELDS}, "cases": CASES,
               "listing": {"sha256": sha(listing), "matching_names": len(names),
                           "outside_ledger": sorted(set(names) - set(PINS))},
               "counts": {kind: sum(row["kind"] == kind for row in rows) for kind in KINDS.values()}, "methods": METHODS,
               "attempts": [{"launch_index": index, **row} for index, row in enumerate(rows, 1)],
               "goal_probes": [goal(by_name[name], rows) for name in GOAL_ATTEMPTS], "native_started_by_analysis": False}
    output.write_text(json.dumps(receipt, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"listing": receipt["listing"], "goal_probes": {row["training_attempt"]: {
        label: row[label]["pass"] for label in CASES} for row in receipt["goal_probes"]}}, indent=1))


if __name__ == "__main__":
    main()
