"""Compare completed normalization trials with the retained action audit."""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "ppo_action_audit_20261002_001"
HELPER_SHA = "d10a1d601b5da0d4f267639a7b4dd8f2e6a88116fc475c58882225be306f3152"
PRIOR_SHA = "dbb10c2e2c9e405405e353ab86181bf15b7982a65a41f0fc9e8342c0acc6f4ed"
REMOTE = "/srv/cupi/hexapod/runs/james/ppo_no_running_norm_20261003_001"
CHECKPOINT = "d255216e14d7e29a9f62055da0615ac0381b8c995e85a3ded43519f3f3968180"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--stance", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    helper_path = PRIOR / "evaluation_comparison.py"
    import hashlib
    assert hashlib.sha256(helper_path.read_bytes()).hexdigest() == HELPER_SHA
    spec = importlib.util.spec_from_file_location("retained_action_audit", helper_path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    assert audit.sha(PRIOR / "evaluation_comparison.json") == PRIOR_SHA
    prior = audit.read(PRIOR / "evaluation_comparison.json")
    verification = audit.read(HERE / "no_norm_evaluation_verification.json")
    training = audit.read(HERE / "no_norm_training_verification.json")
    assert verification["verified"] and verification["container_absent"]
    assert verification["checkpoint_sha256"] == CHECKPOINT
    assert training["verified"] and training["updates"] == 2000
    args.workspace.mkdir(parents=True, exist_ok=True)
    output = args.output or args.workspace / "normalization_evaluation.json"
    if output.exists():
        raise FileExistsError("Use a fresh output path")
    audit.DATA = args.workspace / "inputs"
    audit.CHECKPOINTS["no_normalization"] = CHECKPOINT
    inventory = {}
    for batch in verification["batches"]:
        name = batch["batch"]
        remote = REMOTE + "_evaluate/run/standing/evaluation/" + name
        hashes = {key: value for key, value in batch["files"].items() if key in audit.ALLOWED}
        hashes["report.json"] = batch["report_sha256"]
        hashes["native400hz/capture.json"] = batch["capture_sha256"]
        destination = audit.DATA / "no_normalization" / name
        missing = [name for name in hashes if not (destination / name).exists()]
        if args.fetch and missing:
            destination.mkdir(parents=True, exist_ok=True)
            command = "tar -C " + shlex.quote(remote) + " -cf - " + " ".join(map(shlex.quote, missing))
            process = subprocess.Popen(["ssh", "spark", command], stdout=subprocess.PIPE)
            with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                archive.extractall(destination, filter="data")
            assert process.wait() == 0
        assert all(audit.sha(destination / name) == digest for name, digest in hashes.items())
        inventory[name] = {"remote_directory": remote, "sha256": hashes}
    assert audit.sha(args.stance) == "830cb07c0fdb3d80af82476d8e6e88f25440cfd2e9255ac1e16327ac826d259c"
    neutral = audit.np.asarray(audit.read(args.stance)["nominal_joint_position_rad"], dtype=audit.np.float32).astype(audit.np.float64)
    joints = [leg + "_" + joint for leg in ("lf", "lm", "lr", "rf", "rm", "rr")
              for joint in ("coxa_yaw", "femur_pitch", "tibia_pitch")]
    current = [audit.analyze("no_normalization", name, item, neutral, joints)
               for name, item in inventory.items()]
    for row, name in zip(current, inventory):
        capture = audit.read(audit.DATA / "no_normalization" / name / "native400hz/capture.json")
        row["native_bounds"] = {key: capture[key] for key in (
            "joint_bound_violation_steps", "speed_bound_violation_steps", "maximum_applied_nm", "maximum_requested_nm")}
    assert len(current) == 13 and len({row["case_id"] for row in current}) == 13
    assert {row["model_sha256"] for row in current} == {row["model_sha256"] for row in prior["results"]["bounded_mean"]}
    moving = [row for row in current if ":translate_" in row["case_id"] or ":yaw_" in row["case_id"]]
    assert len(moving) == 10
    error = sum(row["numeric_metrics"]["planar_error_mps"] / .05 if ":translate_" in row["case_id"]
                else row["numeric_metrics"]["yaw_error_rad_s"] / .2 for row in moving) / 10
    metrics_path = args.workspace / "training_metrics.jsonl"
    if args.fetch and not metrics_path.exists():
        with metrics_path.open("xb") as stream:
            subprocess.run(["ssh", "spark", "cat " + shlex.quote(REMOTE + "/run/standing/metrics.jsonl")],
                           stdout=stream, check=True)
    assert audit.sha(metrics_path) == training["metrics_sha256"]
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert [row["update"] for row in rows] == list(range(1, 2001))
    before_kl = [row["policy_update"]["before"]["kl_mean"] for row in rows]
    previous = {row["case_id"]: row for row in prior["results"]["bounded_mean"]}
    comparison = [{"case_id": row["case_id"],
                   "empirical_pass": previous[row["case_id"]]["numerical_pass"],
                   "none_pass": row["numerical_pass"], "none_failed_bounds": row["failed_bounds"]}
                  for row in current]
    receipt = {
        "schema": "hexapod_normalization_native_comparison_v1",
        "analysis_sha256": audit.sha(__file__), "retained_analysis_sha256": HELPER_SHA,
        "prior_comparison_sha256": PRIOR_SHA,
        "verification_sha256": audit.sha(HERE / "no_norm_evaluation_verification.json"),
        "training_verification_sha256": audit.sha(HERE / "no_norm_training_verification.json"),
        "methods": {**prior["methods"],
                    "scope": "Compare all 13 completed bounded-mean probes with and without running normalization. Retain the original run's missing probes and pending human labels.",
                    "training": "Read all 2000 complete metric rows after matching the verified file hash. Report pre-update Gaussian divergence from the collected rollout.",
                    "causality": "One matched seed and budget test this configuration. They do not isolate all learning failure causes or establish walking under another reward."},
        "budget": {"num_envs": 128, "updates": 2000, "transitions": 6144000, "seed": 20260917, "reward_version": "2"},
        "before_update_kl_mean_max": max(before_kl),
        "before_update_kl_mean_mean": sum(before_kl) / len(before_kl),
        "empirical_E": prior["bounded_mean_E"], "none_E": error,
        "original_E": None, "primary_W": None,
        "missing_original_cases": prior["missing_original_cases"],
        "comparison": comparison, "results": {"no_normalization": current},
        "native_started_by_analysis": False,
    }
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"none_E": error, "before_update_kl_mean_max": max(before_kl),
                      "passed": sum(row["numerical_pass"] for row in current), "probes": len(current)}))


if __name__ == "__main__":
    main()
