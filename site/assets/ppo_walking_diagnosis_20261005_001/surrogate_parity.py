"""Evaluate retained native checkpoints on the CPU surrogate and record both simulators side by side.

The script reads four native PPO checkpoints of two learner designs and the native evaluation
summary of each. It evaluates each checkpoint with locomotion.surrogate.evaluate on the forward,
quiet and stop probes and records the native and the surrogate screen of each probe. For the
action-wall update-2000 checkpoint it also records the stop probe under scaled joint-velocity inputs
and under a forced two-control action mean. The surrogate result is design evidence. Native runs
are the only acceptance evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
REMOTE = "/srv/cupi/hexapod/runs/james/"
STANDING = "run/standing/"
EVALUATION = STANDING + "evaluation/"
CASES = {"forward": "learning:translate_0.05_0deg", "quiet_20s": "learning:quiet_20s",
         "forward_to_stop": "learning:forward_0.05_to_stop"}
METRICS = ("mean_velocity_mps", "planar_error_mps", "yaw_error_rad_s", "max_joint_velocity_rms_rad_s",
           "max_target_step_abs_p95_rad_per_20ms")
WALL, VNOISE = "ppo_v4_action_wall_20261005_001", "ppo_v4_vnoise_seed%d_20261005_001"
# Each entry pins the sha256 of the six remote files that the script reads for one checkpoint.
CHECKPOINTS = {
    "wall_u1500": {"attempt": WALL, "update": 1500, "evaluation": WALL + "_evaluate_u1500", "pins": {
        "checkpoint_update001500.pt": "3ff97e3dd7bb425358310fbbdb973dd89b9c00b10ba76bf7041a9ca359f519f2",
        "checkpoint_update001500.json": "a5ac3a6cad37cb4a7e06768249d03c1d7980524e2d1abf9752d7ff3787325c83",
        "ppo_config.json": "c425bcc9ee5ed1b7e70e460f56a66672a0fd47787940eaea00793808c6aa9161",
        "binding.json": "ca183307d94a33f3fdd7420b5b3a29664ee68193b72bf99fd68598ab4e4f2529",
        "summary.json": "8689fd94f564985276d724ace323439aef04b981f87640e7d782b0afe3fde55f",
        "allocation.json": "64c3569a08cff048bcec7fa8a64fccab2df5f23bf05f8d3c855bd09b22158634"}},
    "wall_u2000": {"attempt": WALL, "update": 2000, "evaluation": WALL + "_evaluate_stop_video", "pins": {
        "checkpoint_update002000.pt": "fe3eb69ef3ee4675d49373e8803a0f4207a387c85be4abf80ddcc163ece30371",
        "checkpoint_update002000.json": "8fa2c812fe31f7f50abdcd029d6bafcba878fd1afc7f7c58b32847fff15f2859",
        "ppo_config.json": "c425bcc9ee5ed1b7e70e460f56a66672a0fd47787940eaea00793808c6aa9161",
        "binding.json": "cd5b8fdeb370f03714d3295da7b85685374b226f84d8d3b1fb4cf83f97005c43",
        "summary.json": "835dc7c3bc7bc2df4ceec08b7fc21d7c0024d7a680586c0f5dfe67c52e3a499e",
        "allocation.json": "a987c286624aaba15d8d08218a15f83d36bf3e40e3af0fc942ce00582ebccee9"}},
    "vnoise_seed20260917_u2000": {"attempt": VNOISE % 20260917, "update": 2000, "evaluation": VNOISE % 20260917 + "_evaluate", "pins": {
        "checkpoint_update002000.pt": "54bf83cb4b5cfea042df32db5a9476685be43401ba9a9d0cc58de8c6adcab023",
        "checkpoint_update002000.json": "abf345747bacf0070a94a2b9d35de0ace96c72171ae22eda4f74dc1085c915f2",
        "ppo_config.json": "d535d43b793926b9bce27be8b5556fe0e45bda605f7aaa15a20a4aee91870afd",
        "binding.json": "dad00365c1c40c390aef38cf9d59b84e472bba49d44c5392b92b2edfc906bbca",
        "summary.json": "8ea3979af989bf2b636ff53a024c3a8916edc91f8008a0a10139fa72f5cf5fb4",
        "allocation.json": "aa010d7bbed2189132b93038fe8a0f9d26f3a1f95c9a07b0c51a362911488d6d"}},
    "vnoise_seed20260918_u2000": {"attempt": VNOISE % 20260918, "update": 2000, "evaluation": VNOISE % 20260918 + "_evaluate", "pins": {
        "checkpoint_update002000.pt": "9015680780e43b6b9b8cfbd4de8b27b4e377233577b86663af218a09486443a3",
        "checkpoint_update002000.json": "4b1c04610efd16c31d4f1564d4ef0b73b1f4ab495dd9f882163dd53951836b19",
        "ppo_config.json": "e431648260d980704989177c768b6563f2b00915c7a02ac5710287b62db1016e",
        "binding.json": "a710b4172582c4ff4a16c2628ca5d3ec8c8bb980a495a0a2775dd5227f30c3d0",
        "summary.json": "f27b1b14098f840a5b801496e195b208ed9554479c6f3d082800a5935eeebef3",
        "allocation.json": "13763055ef312dafb898c8dcf2493a07251426721ae136575e5ecdff57850b8a"}},
}
# Stop-probe stress runs of the action-wall update-2000 checkpoint: (velocity-input gain, forced two-control mean).
STRESS_CHECKPOINT = "wall_u2000"
STRESS = ((.5, False), (.75, False), (.9, False), (1., False), (1.25, False), (1.5, False), (1., True), (2., True))
REST_CONTROLS = 300
METHODS = {
    "scope": "One native evaluation and one surrogate evaluation per checkpoint, each with the actor mean and "
             "no sampling noise. Each probe has one trial per simulator, so the record measures no trial-to-trial spread.",
    "native": "Copy pass, failed_bounds, recorded_controls and the named metrics from the native summary.json. "
              "A null metric means the evaluator wrote no value for that probe.",
    "surrogate": "Run python -m locomotion.surrogate.evaluate with the native allocation seed, without video. The "
                 "evaluator builds the policy input and action as the evaluation branch of locomotion/train.py does and "
                 "calls locomotion.evaluate.run_learning_probe_suite, so the probe schedule and the scorer are the "
                 "native code. The evaluation computes no training reward.",
    "pass": "The surrogate pass and failed_bounds come from the default contact model.",
    "consensus": "Quiet and stop probes run under two more contact models: a pyramidal friction cone and friction 0.9. "
                 "The consensus is pass or fail where the three runs agree and unresolved otherwise. The forward probe "
                 "runs once and holds no consensus.",
    "difference": "Surrogate value minus native value for each metric that both simulators wrote.",
    "disagreements": "One entry for each probe whose pass state or failed-bound set differs between native and the "
                     "surrogate default contact model, and for each probe whose consensus is unresolved.",
    "agreement": "Counts over the probes. pass_equal and failed_bounds_equal compare native with the surrogate default "
                 "contact model, in total and by case. The static counts cover the quiet and stop probes: the probes "
                 "that fail on each simulator, the probes whose consensus is pass or fail, and the share of those whose "
                 "consensus equals the native pass state.",
    "stress": "The stop probe alone, with the policy's 90 joint-velocity inputs multiplied by the gain. A forced mean "
              "sends the environment the mean of each action and the previous one. Native ran neither option on this "
              "checkpoint, so the stress rows hold no native value.",
    "rest_action": "Over the last 300 controls of a surrogate stop or quiet trace: the largest standard deviation of "
                   "one joint's policy action, and the lag-1 autocorrelation of that joint's action. A value near -1 "
                   "marks an action that alternates on consecutive controls.",
    "limits": "The surrogate replaces rigid-body dynamics and floor contact with MuJoCo. Static-pose checks depend on "
              "friction-locked loads that differ between simulators; locomotion/surrogate/README.md lists the known gaps.",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def fetch(remote, names, destination):
    """Copy the missing named files from one Spark directory through a tar stream."""
    missing = [name for name in names if not (destination / name).exists()]
    if not missing:
        return
    destination.mkdir(parents=True, exist_ok=True)
    command = "tar -C " + shlex.quote(remote) + " -cf - " + " ".join(map(shlex.quote, missing))
    process = subprocess.Popen(["ssh", "spark", command], stdout=subprocess.PIPE)
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        archive.extractall(destination, filter="data")
    assert process.wait() == 0


def screen(result):
    """Return the pass state and the named metrics of one probe result."""
    metrics = result.get("metrics", {})
    return {"pass": result["pass"], "failed_bounds": sorted(result["failed_bounds"]),
            **{key: metrics.get(key) for key in METRICS}}


def rest_action(trace):
    """Measure the policy action over the last controls of one surrogate control trace."""
    with np.load(trace, allow_pickle=False) as data:
        tail = data["policy_action"][-REST_CONTROLS:, 0].astype(np.float64)
    deviation = tail.std(axis=0)
    joint = int(deviation.argmax())
    centered = tail[:, joint] - tail[:, joint].mean()
    lag = float(np.corrcoef(centered[:-1], centered[1:])[0, 1]) if deviation[joint] > 1e-9 else None
    return {"action_std_max": float(deviation[joint]), "joint_index": joint, "lag1_autocorrelation": lag}


def evaluate(checkpoint, output, seed, cases, *, gain=1., mean2=False, threads=2):
    """Run the surrogate evaluator in a fresh process and return its summary."""
    command = [sys.executable, "-B", "-m", "locomotion.surrogate.evaluate", "--checkpoint", str(checkpoint),
               "--output", str(output), "--video", "none", "--seed", str(seed), "--cases", ",".join(cases),
               "--velocity-input-gain", repr(gain), "--threads", str(threads)] + (["--action-mean2"] if mean2 else [])
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    Path(str(output) + ".log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("Surrogate evaluation failed: " + result.stderr[-2000:])
    return read(output / "surrogate_summary.json")


def surrogate_screen(summary, row, index, output):
    """Return one surrogate probe row with its consensus, joint margin and rest action."""
    value = screen({**row, "failed_bounds": row["failed_bounds"] or []})
    value["recorded_controls"] = row["gait"]["controls"]
    value["consensus"] = row.get("contact_consensus", {}).get("verdict")
    value["consensus_verdicts"] = row.get("contact_consensus", {}).get("verdicts")
    value["joint_limit_margin_min_rad"] = row.get("joint_limit_margin", {}).get("minimum_rad")
    if row["case_id"] != CASES["forward"]:
        value["rest_action"] = rest_action(output / "evaluation" / ("batch_%03d" % index) / "control_trace.npz")
    assert summary["selected_case_ids"][index] == row["case_id"]
    return value


def difference(native, surrogate):
    """Subtract each native metric from the surrogate metric where both hold a value."""
    result = {}
    for key in METRICS:
        a, b = native[key], surrogate[key]
        if a is not None and b is not None:
            result[key] = (np.asarray(b) - np.asarray(a)).tolist()
    return result


def analyze(label, entry, workspace, do_fetch, threads):
    """Check one checkpoint's records and evaluate it on the surrogate."""
    attempt, update, evaluation, pins = entry["attempt"], entry["update"], entry["evaluation"], entry["pins"]
    stem = "checkpoint_update%06d" % update
    files = [STANDING + stem + ".pt", STANDING + stem + ".json", STANDING + "ppo_config.json"]
    records = ["binding.json", EVALUATION + "summary.json", EVALUATION + "allocation.json"]
    source, native = workspace / "inputs" / attempt, workspace / "inputs" / evaluation
    if do_fetch:
        fetch(REMOTE + attempt, files, source)
        fetch(REMOTE + evaluation, records, native)
    hashes = {**{REMOTE + attempt + "/" + item: sha(source / item) for item in files},
              **{REMOTE + evaluation + "/" + item: sha(native / item) for item in records}}
    assert {Path(item).name: digest for item, digest in hashes.items()} == pins
    checkpoint = source / files[0]
    record, config = read(source / files[1]), read(source / files[2])
    binding, summary, allocation = (read(native / item) for item in records)
    digest = sha(checkpoint)
    arguments = binding["command_args"]
    option = lambda flag: arguments[arguments.index(flag) + 1]
    assert record["checkpoint_sha256"] == digest == option("--checkpoint-sha256") and record["updates"] == update
    assert record["identity"]["ppo_config"] == config and int(option("--updates")) == update
    assert binding["input_files"][REMOTE + attempt + "/" + files[0]] == digest
    assert summary["allocation"] == allocation and allocation["checkpoint_sha256"] == digest
    seed = allocation["seed"]
    assert seed == int(option("--seed")) == record["identity"]["seed"]
    results = {row["case_id"]: row for row in summary["results"]}
    assert all(results[case]["lineage"]["checkpoint_sha256"] == digest for case in CASES.values())
    output = workspace / "evaluations" / label
    mine = evaluate(checkpoint, output, seed, list(CASES.values()), threads=threads)
    assert mine["checkpoint_sha256"] == digest and mine["seed"] == seed and mine["acquisition_failure"] is None
    assert mine["wrapper_options"] == {"observation_scaling": "none", "command_segments": "continuous",
                                       **config.get("environment_wrapper", {})}
    current = mine["identity"]["source_files"]
    cases = {}
    for index, (name, case) in enumerate(CASES.items()):
        a = {**screen(results[case]), "recorded_controls": results[case]["recorded_controls"]}
        b = surrogate_screen(mine, mine["cases"][index], index, output)
        cases[name] = {"case_id": case, "native": a, "surrogate": b, "difference": difference(a, b),
                       "pass_equal": a["pass"] == b["pass"], "failed_bounds_equal": a["failed_bounds"] == b["failed_bounds"]}
    row = {"label": label, "training_attempt": attempt, "update": update, "seed": seed, "native_evaluation": evaluation,
           "native_eval_scope": option("--eval-scope"), "checkpoint_sha256": digest, "sha256": hashes,
           "wrapper_options": mine["wrapper_options"], "policy_path": mine["policy_path"],
           "kernel_files_changed_since_training": sorted(
               name for name, value in record["identity"]["source_files"].items() if current.get(name) != value),
           "cases": cases}
    return row, mine, checkpoint, seed


def stress(checkpoint, seed, workspace, base, identity, threads):
    """Run the stop probe of one checkpoint under each velocity-input gain and forced mean."""
    rows = []
    for gain, mean2 in STRESS:
        output = workspace / "evaluations" / ("%s_stop_gain%s%s" % (STRESS_CHECKPOINT, gain, "_mean2" if mean2 else ""))
        summary = evaluate(checkpoint, output, seed, [CASES["forward_to_stop"]], gain=gain, mean2=mean2, threads=threads)
        assert summary["policy_path"]["velocity_input_gain"] == gain and summary["policy_path"]["action_mean2_forced"] == mean2
        assert summary["identity"] == identity
        row = {"velocity_input_gain": gain, "action_mean2": mean2,
               **surrogate_screen(summary, summary["cases"][0], 0, output)}
        if gain == 1. and not mean2:
            # The stop probe alone must repeat the stop probe of the three-probe run.
            row["equals_three_probe_run"] = all(row[key] == base[key] for key in base)
        rows.append(row)
    return rows


def disagreements(rows):
    """List each probe where the surrogate differs from native, and each unresolved consensus."""
    found = []
    for row in rows:
        for name, case in row["cases"].items():
            a, b = case["native"], case["surrogate"]
            entry = {"checkpoint": row["label"], "case": name, "native_pass": a["pass"], "surrogate_pass": b["pass"],
                     "surrogate_consensus": b["consensus"],
                     "failed_in_native_alone": sorted(set(a["failed_bounds"]) - set(b["failed_bounds"])),
                     "failed_in_surrogate_alone": sorted(set(b["failed_bounds"]) - set(a["failed_bounds"]))}
            kinds = ([] if case["pass_equal"] else ["pass"]) + ([] if case["failed_bounds_equal"] else ["failed_bounds"])
            if b["consensus"] == "unresolved":
                kinds.append("consensus_unresolved")
            if kinds:
                found.append({"kinds": kinds, **entry})
    return found


def agreement(rows):
    """Count the probes where the surrogate equals native, in total, by case and for the static probes."""
    count = lambda cases: {"probes": len(cases), "pass_equal": sum(case["pass_equal"] for case in cases),
                           "failed_bounds_equal": sum(case["failed_bounds_equal"] for case in cases)}
    static = [case for row in rows for name, case in row["cases"].items() if name != "forward"]
    resolved = [case for case in static if case["surrogate"]["consensus"] in ("pass", "fail")]
    return {**count([case for row in rows for case in row["cases"].values()]),
            "by_case": {name: count([row["cases"][name] for row in rows]) for name in CASES},
            "static": {"probes": len(static), "native_failures": sum(not case["native"]["pass"] for case in static),
                       "surrogate_failures": sum(not case["surrogate"]["pass"] for case in static),
                       "consensus_resolved": len(resolved),
                       "consensus_pass_where_native_fails": sum(
                           case["surrogate"]["consensus"] == "pass" and not case["native"]["pass"] for case in resolved),
                       "consensus_equal_native": sum(
                           (case["surrogate"]["consensus"] == "pass") == case["native"]["pass"] for case in resolved)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    args.workspace.mkdir(parents=True, exist_ok=True)
    output = args.output or args.workspace / "surrogate_parity.json"
    if output.exists() or (args.workspace / "evaluations").exists():
        raise FileExistsError("Use a fresh output path and a workspace without an evaluations directory")
    rows, identities, stress_rows = [], [], None
    for label, entry in CHECKPOINTS.items():
        row, summary, checkpoint, seed = analyze(label, entry, args.workspace, args.fetch, args.threads)
        rows.append(row)
        identities.append({key: summary[key] for key in ("schema", "backend", "contact_model", "identity", "repository")})
        if label == STRESS_CHECKPOINT:
            stress_rows = stress(checkpoint, seed, args.workspace, row["cases"]["forward_to_stop"]["surrogate"],
                                 summary["identity"], args.threads)
    assert all(value == identities[0] for value in identities)
    receipt = {"schema": "hexapod_surrogate_parity_v1", "analysis_sha256": sha(__file__),
               "cases": CASES, "metrics": list(METRICS), "methods": METHODS,
               "surrogate": identities[0],
               "agreement": agreement(rows),
               "disagreements": disagreements(rows), "checkpoints": rows,
               "stop_probe_stress": {"checkpoint": STRESS_CHECKPOINT, "rows": stress_rows},
               "native_started_by_analysis": False}
    output.write_text(json.dumps(receipt, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"agreement": receipt["agreement"], "disagreements": receipt["disagreements"],
                      "stress": [{key: row[key] for key in ("velocity_input_gain", "action_mean2", "pass", "consensus",
                                                            "failed_bounds")} for row in stress_rows]}))


if __name__ == "__main__":
    main()
