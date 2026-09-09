#!/usr/bin/env python3
"""Verify a completed failing primary result without changing its grade."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(folder):
    folder = Path(folder)
    inventory = json.loads((folder / "remote_inventory.json").read_text())
    for path, record in inventory["files"].items():
        if digest(folder / path) != record["sha256"]:
            raise ValueError("Captured original differs: " + path)
    campaign = json.loads((folder / "campaign.json").read_text())
    run = next((folder / "nominal").glob("hexapod-fourbar-validate-*"))
    report = json.loads((run / "report.json").read_text())
    supervisor = json.loads((run / "supervisor.json").read_text())
    if (report["pass"] is not False or len(report["errors"]) != 3
            or report["num_envs"] != 32 or report["steps_completed"] != 1000
            or report["steps_requested"] != 1000 or report["driven_steps"] != 2400
            or report["physics_substeps"] != 54400 or report["solver_iterations"] != [64, 16]
            or report["terminated_count"] != 0 or report["truncated_count"] != 0
            or report["simulation_training_admission"] is not False or report["hardware_admission"] is not False):
        raise ValueError("Unexpected primary coverage, result or scope")
    if (campaign["state"] != "failed" or [p["name"] for p in campaign["phases"]] != ["probe", "nominal"]
            or supervisor["cleanup"] != "removed_exact_id" or supervisor["supervisor_exit_code"] != 1
            or supervisor["final_container_exit_code"] != 0 or supervisor["validator_report_status"] != "rejected"
            or supervisor["source_identity_unchanged_at_finish"] is not True
            or report["contract"] != supervisor["contract"] or report["contract"] != campaign["contract"]):
        raise ValueError("Campaign terminal state, source or cleanup differs")
    lines = (run / "source.SHA256SUMS").read_text().splitlines()
    captured = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in lines}
    if (digest(run / "source.SHA256SUMS") != supervisor["source_manifest"]["sha256"]
            or len(lines) != supervisor["source_manifest"]["files"]
            or any(captured.get(path) != sha for path, sha in report["contract"]["files"].items())):
        raise ValueError("Source manifest differs from functional identity")
    for name, count in (("startup", 3200), ("settled", 12800), ("driven", 38400)):
        window = report["windows"][name]
        if window["substeps"] != count:
            raise ValueError("Incorrect window sample count")
        for members, count_key, squared_key, rms_key in (
            (12, "passive_velocity_relation_samples", "passive_velocity_relation_squared_sum", "rms_passive_velocity_relation_error_rad_s"),
            (6, "closure_relative_point_velocity_samples", "closure_relative_point_velocity_squared_norm_sum", "rms_closure_relative_point_velocity_m_s"),
        ):
            population = count * 32 * members
            if window[count_key] != population or not math.isclose(
                    math.sqrt(window[squared_key] / population), window[rms_key], rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError("Velocity RMS population/arithmetic differs")
    failures = {}
    for kind, field in (("individual", "individual_motor_positive_minus_negative_rad"),
                        ("group", "driven_positive_minus_negative_response_rad")):
        response = report[field]
        if set(response) != set(report["active_motor_names"]) or len(response) != 18:
            raise ValueError("Incomplete motor response coverage")
        failures[kind] = {name: value for name, value in response.items() if value <= .005}
    return {
        "schema": "hexapod.completed_motion_failure_evidence.v1",
        "primary_report_sha256": digest(run / "report.json"),
        "source_commit": supervisor["source_commit"], "functional_sha256": report["contract"]["sha256"],
        "primary_pass": False, "primary_errors": report["errors"],
        "coverage": {"num_envs": 32, "standing_controls": 1000, "driven_controls": 2400,
                     "physics_substeps_per_environment": 54400, "environment_physics_samples": 1740800},
        "windows": report["windows"], "failed_direction_checks": failures,
        "direction_requirement_rad": "> 0.005", "source_and_exact_cleanup_verified": True,
        "velocity_rms_arithmetic_verified": True,
        "container_exit_code": 0, "supervisor_exit_code": 1,
        "simulation_training_admission": False, "hardware_admission": False,
        "later_campaign_phases_started": False,
        "limits": ["Aggregate metrics do not identify the failing environment or event sequence.",
                   "No cause is inferred from the direction responses, torque demand or closure maxima.",
                   "No source archive or raw state trace was copied; archive bytes remain remote."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(verify(args.folder), indent=2, sort_keys=True) + "\n")
