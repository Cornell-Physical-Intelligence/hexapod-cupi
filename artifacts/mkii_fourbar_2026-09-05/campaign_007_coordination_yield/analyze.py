#!/usr/bin/env python3
"""Derive an interruption summary; never manufacture a missing validator report."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def progress_records(text):
    rows = []
    pattern = re.compile(r"^(\S+) FOURBAR_STANDING step=(\d+)/(\d+) metrics=(\{.*\})$")
    for number, line in enumerate(text.splitlines(), 1):
        match = pattern.match(line)
        if match:
            stamp, step, requested, payload = match.groups()
            metrics = json.loads(payload)
            step, requested = int(step), int(requested)
            startup_controls = max(1, requested // 5)
            window = "startup" if step <= startup_controls else "settled"
            expected = (step if window == "startup" else step - startup_controls) * 16
            if metrics["substeps"] != expected:
                raise ValueError("Logged measurement population differs from its cumulative window")
            rows.append({"line": number, "utc": stamp, "control_step": step,
                         "requested_standing_controls": requested, "window": window,
                         "metrics": metrics})
    return rows


def analyze(folder):
    folder = Path(folder)
    inventory = json.loads((folder / "remote_inventory.json").read_text())
    for relative, metadata in inventory["files"].items():
        if "sha256" in metadata and digest(folder / relative) != metadata["sha256"]:
            raise ValueError("Downloaded original differs from remote inventory: " + relative)
    campaign = json.loads((folder / "campaign.json").read_text())
    nominal = next((folder / "nominal").glob("hexapod-fourbar-validate-*"))
    probe = next((folder / "probe").glob("hexapod-fourbar-validate-*"))
    if (nominal / "report.json").exists() or inventory["nominal_primary_report_exists"]:
        raise ValueError("This interruption artifact expects the primary nominal report to be absent")
    if any(inventory[name] for name in ("refined_directory_exists", "scratch_directory_exists", "full_directory_exists")):
        raise ValueError("Unexpected later campaign phase exists")
    if campaign["state"] != "failed" or [p["name"] for p in campaign["phases"]] != ["probe", "nominal"]:
        raise ValueError("Unexpected campaign phase sequence")
    supervisors = {}
    for label, run in (("probe", probe), ("nominal", nominal)):
        supervisor = json.loads((run / "supervisor.json").read_text())
        if (supervisor["cleanup"] != "removed_exact_id"
                or supervisor["source_identity_unchanged_at_finish"] is not True
                or supervisor["contract"] != campaign["contract"]):
            raise ValueError("Unverified cleanup or source contract")
        manifest = run / "source.SHA256SUMS"
        lines = manifest.read_text().splitlines()
        if digest(manifest) != supervisor["source_manifest"]["sha256"] or len(lines) != supervisor["source_manifest"]["files"]:
            raise ValueError("Captured source hash list differs from supervisor")
        captured = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in lines}
        if any(captured.get(path) != sha for path, sha in campaign["contract"]["files"].items()):
            raise ValueError("Functional source files differ from captured hash list")
        supervisors[label] = supervisor
    n = supervisors["nominal"]
    if (n["error"] != "Coordination changed; owned job yielded compute"
            or n["pause_requested"] != "shared coordination file changed"
            or n["supervisor_exit_code"] != 1 or n["final_container_exit_code"] != 137
            or n["validator_report_present"] is not False):
        raise ValueError("Unexpected nominal termination mechanism")
    primary = json.loads((probe / "report.json").read_text())
    if (primary["pass"] is not True or primary["errors"] != []
            or primary["num_envs"] != 1 or primary["steps_completed"] != 100
            or primary["physics_substeps"] != 1600
            or digest(probe / "report.json") != campaign["phases"][0]["report_sha256"]):
        raise ValueError("Probe result or report identity differs")
    log = (nominal / "container.log").read_text()
    rows = progress_records(log)
    if not rows:
        raise ValueError("No nominal progress was captured")
    last = rows[-1]
    def mtime(path):
        return inventory["files"][str(path.relative_to(folder))]["mtime_utc"]
    return {
        "schema": "hexapod.campaign_coordination_yield_evidence.v1",
        "classification": "coordination_yield_with_incomplete_nominal_physics_result",
        "simulation_training_admission": False, "hardware_admission": False,
        "source_commit": campaign["source_commit"], "functional_contract_sha256": campaign["contract"]["sha256"],
        "remote_campaign": campaign["campaign"], "remote_capture_utc": inventory["captured_utc"],
        "probe": {"report_sha256": digest(probe / "report.json"), "pass": True,
                  "num_envs": 1, "controls_completed": 100, "physics_substeps": 1600,
                  "windows": primary["windows"]},
        "nominal_primary_report_present": False, "nominal_full_physics_result": "unknown_incomplete",
        "nominal_last_logged_progress": last,
        "nominal_logged_progress": rows,
        "nominal_completed_physics_substeps_per_env_lower_bound": last["control_step"] * 16,
        "nominal_completed_control_steps_exact": None,
        "driven_progress_records_present": "FOURBAR_DRIVEN" in log,
        "later_campaign_phases_started": False,
        "source_and_cleanup": {label: {k: value[k] for k in ("source_manifest", "coordination_sha256",
            "source_identity_unchanged_at_finish", "cleanup", "container_id", "final_container_exit_code",
            "supervisor_exit_code", "validator_report_status")} for label, value in supervisors.items()},
        "chronology": [
            {"utc": campaign["initial_resource_gate"]["evidence"]["time_utc"], "event": "Campaign resource gate passed", "time_basis": "recorded event"},
            {"utc": mtime(probe / "supervisor.json"), "event": "Probe supervisor completion record written", "time_basis": "remote filesystem modification time"},
            {"utc": n["gates"][0]["time_utc"], "event": "Nominal first resource gate passed", "time_basis": "recorded event"},
            {"utc": last["utc"], "event": "Last flushed nominal standing progress", "time_basis": "Docker log timestamp"},
            {"utc": n["last_runtime_gate"]["time_utc"], "event": "Last successful nominal runtime resource gate", "time_basis": "recorded event"},
            {"utc": mtime(nominal / "stop_requested"), "event": "Coordination-change stop marker written", "time_basis": "remote filesystem modification time"},
            {"utc": mtime(nominal / "supervisor.json"), "event": "Failed nominal supervisor record with exact cleanup written", "time_basis": "remote filesystem modification time"},
            {"utc": mtime(folder / "campaign.json"), "event": "Campaign failure record written", "time_basis": "remote filesystem modification time"},
        ],
        "limits": [
            "No final nominal report is synthesized. The last progress record establishes a lower bound, not the exact stop step.",
            "No driven phase is evidenced; incomplete standing cannot establish the required full physical result.",
            "The launcher records a coordination hash change, not the editor identity, intent or replacement file contents.",
            "File modification timestamps are preserved separately from explicitly recorded event timestamps.",
            "Large source archives remain remote and were not downloaded or independently rehashed.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(analyze(args.folder), indent=2, sort_keys=True) + "\n")
