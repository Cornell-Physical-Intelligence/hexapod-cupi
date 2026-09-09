"""Bounded priority lock across owned diagnostics and one following campaign.

Only holds the existing scheduler lock. Explicit release or selected campaign
termination releases it; ten hours is the absolute maximum. No workloads signalled.
"""
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=True)
stop = False

def interrupt(*_):
    global stop
    stop = True

for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(sig, interrupt)

def record(state, **extra):
    value = dict(state=state, pid=os.getpid(), lock='/opt/wx/gpu.lock',
                 utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra)
    temporary = root / 'status.partial'
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(root/'status.json')
    print(json.dumps(value), flush=True)

with open('/opt/wx/gpu.lock', 'r') as lock:
    record('waiting')
    fcntl.flock(lock, fcntl.LOCK_EX)
    deadline = time.monotonic()+36000
    expires = datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=10)
    record('reserved', expires_utc=expires.isoformat())
    reason = 'ten-hour maximum'
    while time.monotonic() < deadline:
        if stop or (root/'release').exists():
            reason = 'explicit release'
            break
        selection = root/'campaign_path.txt'
        if selection.exists():
            campaign = Path(selection.read_text().strip())
            try:
                state = json.loads(campaign.read_text())
                if state.get('schema') != 'hexapod.fourbar_campaign.v1':
                    raise ValueError('Wrong campaign schema')
                if state.get('state') in ('complete', 'failed', 'paused'):
                    reason = 'selected owned campaign '+state['state']
                    break
                argv = Path(f"/proc/{state['pid']}/cmdline").read_bytes()
                if b'run-mkii-fourbar-campaign' not in argv:
                    raise ValueError('Selected campaign PID no longer matches')
            except (OSError, ValueError, KeyError):
                reason = 'selected campaign unavailable'
                break
        time.sleep(2)
    fcntl.flock(lock, fcntl.LOCK_UN)
    record('released', reason=reason)
