"""Bounded cooperative GPU reservation for one exact existing campaign.

This holds the weather scheduler's existing lock; it never signals workloads,
changes the shared coordination note, or launches a GPU process.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve(strict=True)
    initial = json.loads(campaign.read_text())
    if initial.get("state") != "running" or initial.get("schema") != "hexapod.fourbar_campaign.v1":
        raise SystemExit("Expected one already-running owned physical campaign")
    interrupted = False

    def interrupt(_number, _frame):
        nonlocal interrupted
        interrupted = True

    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, interrupt)

    def record(state, **values):
        payload = {"state": state, "pid": os.getpid(), "campaign": str(campaign),
                   "campaign_pid": initial["pid"], "lock": "/opt/wx/gpu.lock",
                   "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   **values}
        temporary = args.status.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(args.status)
        print(json.dumps(payload), flush=True)

    with open("/opt/wx/gpu.lock", "r") as descriptor:
        record("waiting")
        acquisition_deadline = time.monotonic() + 90
        while not interrupted and not args.release.exists():
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= acquisition_deadline:
                    record("failed", reason="90-second lock handover timeout")
                    return 1
                time.sleep(.05)
        else:
            record("released", reason="interrupted before acquisition")
            return 0
        duration = 10 * 3600
        deadline = time.monotonic() + duration
        expires = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=duration)
        record("reserved", expires_utc=expires.isoformat())
        reason = "ten-hour maximum reached"
        while time.monotonic() < deadline:
            if interrupted or args.release.exists():
                reason = "explicit release"
                break
            try:
                state = json.loads(campaign.read_text())
            except (OSError, ValueError):
                state = {}
            if state.get("state") in {"complete", "failed", "paused"}:
                reason = "owned campaign " + state["state"]
                break
            try:
                command = Path(f"/proc/{initial['pid']}/cmdline").read_bytes()
            except FileNotFoundError:
                reason = "owned campaign process exited"
                break
            if b"run-mkii-fourbar-campaign" not in command:
                reason = "owned campaign PID identity changed"
                break
            time.sleep(2)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        record("released", reason=reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
