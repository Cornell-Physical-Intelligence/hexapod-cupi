#!/usr/bin/env python3
"""Verify captured release/CI/startup evidence without granting training admission."""
import argparse
import hashlib
import json
from pathlib import Path


COMMIT = "1239159c185cd504c359bb20e98bde9986acbbb4"
PROBE_SHA256 = "2a85e74824a8fc28f8f74907e1324e94db5fa2728681d6685115dfe9bb35b9b2"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(folder):
    folder = Path(folder)
    def read(name):
        return json.loads((folder / name).read_text())
    remote = read("remote_inventory.json")
    local = read("local_capture.json")
    for inventory in (remote["chosen_files"], local["local_originals"]):
        for relative, identity in inventory.items():
            path = folder / relative
            if path.stat().st_size != identity["bytes"] or digest(path) != identity["sha256"]:
                raise ValueError("Captured original differs: " + relative)
    staging = read("staging.json")
    release = read("local_release/hexapod_coordination_control_release.json")
    if staging["commit"] != COMMIT or any(staging.get(k) != v for k, v in release.items()):
        raise ValueError("Local release and remote staging differ")
    tests = (folder / "local_release/hexapod_coordination_control_tests.log").read_text()
    if "Ran 906 tests in 70.128s\n\nOK\n" not in tests:
        raise ValueError("Expected completed local test result is absent")
    ci = read("ci_run_34005889807.json")
    if (ci["databaseId"] != 34005889807 or ci["headSha"] != COMMIT
            or ci["status"] != "completed" or ci["conclusion"] != "success"
            or digest(folder / "ci_run_34005889807.json") != local["ci_query"]["output_sha256"]):
        raise ValueError("CI identity or conclusion differs")
    launch = read("campaign_launch_20260906T021507Z.json")
    if (launch["source_commit"] != COMMIT
            or launch["source_contract_sha256"] != release["functional_sha256"]
            or digest(folder / "launch_reviewed_campaign_v1.py") != launch["bootstrap_sha256"]):
        raise ValueError("Launch source or captured bootstrap differs")
    report = read("probe/report.json")
    supervisor = read("probe/supervisor.json")
    if (digest(folder / "probe/report.json") != PROBE_SHA256
            or report["pass"] is not True or report["errors"] != []
            or report["num_envs"] != 1 or report["steps_completed"] != 100
            or report["physics_substeps"] != 1600 or report["solver_iterations"] != [64, 16]
            or report["simulation_training_admission"] is not False or report["hardware_admission"] is not False):
        raise ValueError("Probe identity, coverage or scope differs")
    if (supervisor["source_commit"] != COMMIT or supervisor["contract"] != report["contract"]
            or report["contract"]["sha256"] != release["functional_sha256"]
            or supervisor["cleanup"] != "removed_exact_id" or supervisor["supervisor_exit_code"] != 0
            or supervisor["final_container_exit_code"] != 0
            or supervisor["source_identity_unchanged_at_finish"] is not True):
        raise ValueError("Probe source or cleanup is unverified")
    source = folder / "probe/source.SHA256SUMS"
    lines = source.read_text().splitlines()
    captured = dict((line.split("  ", 1)[1], line.split("  ", 1)[0]) for line in lines)
    if (digest(source) != supervisor["source_manifest"]["sha256"]
            or len(lines) != supervisor["source_manifest"]["files"]
            or any(captured.get(path) != sha for path, sha in report["contract"]["files"].items())):
        raise ValueError("Captured source manifest differs from functional identity")
    if supervisor["coordination_protocol"] != "canonical_share_status_v2":
        raise ValueError("Probe did not use the expected coordination protocol")
    for phase in ("coordination_initial", "coordination_before_barrier", "coordination_latest"):
        state = supervisor[phase]
        if state["status"] != "NONE" or state["semantic_token"] != "HEXAPOD_SHARE_STATUS=NONE":
            raise ValueError("Unexpected captured probe coordination state")
    snapshot = read("campaign_progress_snapshot.json")
    progress = snapshot["selected_fields"]
    if (progress["source_commit"] != COMMIT or progress["active_phase"] != "nominal"
            or [(p["name"], p["state"]) for p in progress["phases"]] != [("probe", "passed"), ("nominal", "running")]):
        raise ValueError("Captured campaign stage differs")
    return {
        "schema": "hexapod.coordination_control_release_evidence.v1",
        "captured_originals_verified": True, "source_commit": COMMIT,
        "functional_sha256": release["functional_sha256"],
        "pipeline_manifest_sha256": release["manifest_sha256"],
        "local_tests": {"passed": 906, "seconds": 70.128},
        "ci": ci,
        "probe": {"report_sha256": PROBE_SHA256, "pass": True, "num_envs": 1,
                  "controls_completed": 100, "physics_substeps": 1600,
                  "windows": report["windows"], "exact_cleanup_and_source_unchanged": True},
        "coordination_protocol": supervisor["coordination_protocol"],
        "probe_coordination_status": "NONE",
        "campaign_stage_at_capture": {"captured_utc": snapshot["captured_utc"], "phase": "nominal",
                                      "started_ppo": False},
        "simulation_training_admission": False, "hardware_admission": False,
        "limits": ["Startup probe only; full standing/driven qualification and PPO are not established.",
                   "This capture does not test a live change of coordination prose or control status.",
                   "Source archives remain remote; archive bytes were not downloaded or independently rehashed."],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(verify(args.folder), indent=2, sort_keys=True) + "\n")
