#!/usr/bin/env python3
"""Check preserved takeover identities/actions; never send a process signal."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(folder):
    folder = Path(folder)
    def read(path):
        return json.loads((folder / path).read_text())
    inventory = read("remote_inventory.json")
    for relative, identity in inventory["files"].items():
        if hashlib.sha256((folder / relative).read_bytes()).hexdigest() != identity["sha256"]:
            raise ValueError("Captured original differs: " + relative)
    audit = read("cpu_cleanup_audit_20260906T030929Z.json")
    takeover = read("takeover.json")
    before = {row["pid"]: row for row in takeover["verified_before"]}
    audited = {row["pid"]: row for row in audit["processes"] if row.get("kind") == "known_weather"}
    if len(before) != 21 or set(before) != set(audited):
        raise ValueError("Expected exact audited weather process set")
    for pid, row in before.items():
        if any(row[key] != audited[pid][key] for key in ("start_ticks", "pgid", "argv", "uid")):
            raise ValueError("Pre-stop identity differs from audit")
    signals = [action for action in takeover["actions"] if "signal" in action]
    services = [action for action in takeover["actions"] if "service" in action]
    if len(signals) != 12 or len(services) != 1:
        raise ValueError("Unexpected action count")
    affected = set()
    for action in signals:
        pid = action["pid"]
        if pid not in before or action["start_ticks"] != before[pid]["start_ticks"] or action["signal"] != "SIGTERM":
            raise ValueError("Signal action identity differs")
        if action["target"] == "process_group":
            if before[pid]["pgid"] != pid:
                raise ValueError("Group action did not name its leader")
            members = set(audited[pid]["group_members"])
            if not members <= before.keys():
                raise ValueError("Audited group contains an unverified process")
            affected |= members
        elif action["target"] == "process":
            affected.add(pid)
        else:
            raise ValueError("Unknown signal target kind")
    if affected != set(before) - {1530836} or affected & set(audit["protected_pids"]):
        raise ValueError("Signal scope differs or includes protected hexapod processes")
    service = services[0]
    if (service["service"] != "corrdiff-radar-year2021-cPZhN6.service" or service["action"] != "stop"
            or service["exit_code"] != 0 or takeover["remaining"] != [] or takeover["outputs_deleted"] is not False
            or "ActiveState=inactive" not in takeover["service_after"] or "MainPID=0" not in takeover["service_after"]):
        raise ValueError("Stop completion or output-preservation record differs")
    guard, launch = read("guard/status.json"), read("guard/launch.json")
    if guard["state"] != "reserved" or guard["pid"] != 1772925 or launch["pid"] != 1772925:
        raise ValueError("Unexpected captured reservation identity")
    return {"schema": "hexapod.authorized_cpu_takeover_evidence.v1",
        "started_utc": takeover["started_utc"], "finished_utc": takeover["finished_utc"],
        "authorization_record": takeover["authorization"], "audited_process_identities": 21,
        "nonservice_processes_covered": 20, "sigterm_actions": 12, "service_stop_actions": 1,
        "affected_nonservice_pids": sorted(affected), "remaining_recorded": [],
        "protected_hexapod_pids_excluded": audit["protected_pids"],
        "service_after": takeover["service_after"], "outputs_deleted_recorded": False,
        "guard_snapshot": guard, "raw_original_hashes_and_pre_stop_identities_verified": True,
        "limits": ["This is a historical capture, not a promise that no new weather jobs will start.",
                   "Output preservation is recorded by the takeover; no exhaustive before/after filesystem audit was performed.",
                   "The reservation expiry is a recorded ceiling, not a guarantee it stays active until that time.",
                   "The CPU stop occurred after the completed campaign failure; no physics cause or remedy is inferred."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(verify(args.folder), indent=2, sort_keys=True) + "\n")
