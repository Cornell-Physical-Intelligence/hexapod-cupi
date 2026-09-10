#!/usr/bin/env python3
"""Wait behind the active GPU job, then stop its coordinator between jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("campaign", type=Path)
args = p.parse_args()
fd = os.open("/opt/wx/gpu.lock", os.O_RDONLY)
try:
    fcntl.flock(fd, fcntl.LOCK_EX)
    # Acquiring the campaign's first lock proves its prior run_job finally
    # completed (including the owned container cleanup). Keep it until the
    # coordinator has observed the request. No running learner is signalled.
    (args.campaign / "stop.request").write_text("Hold for baseline evaluation and actual policy video before the other 48 sizes.\n")
    (args.campaign / "pilot_hold.json").write_text(json.dumps({
        "time": time.time(), "reason": "User requested baseline end-to-end video gate",
        "method": "acquired shared GPU lock after active job released it"}, indent=2))
    for _ in range(90):
        report = json.loads((args.campaign / "campaign.json").read_text())
        if report.get("status") in ("stopped", "failed", "completed"):
            break
        time.sleep(1)
finally:
    os.close(fd)
