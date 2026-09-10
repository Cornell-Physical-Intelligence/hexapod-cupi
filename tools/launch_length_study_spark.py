#!/usr/bin/env python3
"""Bounded, exclusive per-job launcher for the Isaac morphology comparison.

Run on spark-e26c. Never stops unrelated work, queues a hidden successor, or
creates a persistent GPU reservation. Exit 75 means the shared GPU is busy.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import uuid


def command(args):
    return subprocess.check_output(args, text=True, timeout=20).strip()


def resources():
    processes = command(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"])
    memory = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")}
    return processes, memory["MemAvailable"]


def preflight():
    processes, available = resources()
    containers = command(["docker", "ps", "--format", "{{.Names}} {{.Image}}"])
    if processes or any("isaac-lab" in line or "hexapod-rl" in line for line in containers.splitlines()):
        raise BlockingIOError(f"Shared GPU is occupied; no launch.\n{processes}\n{containers}")
    if available < 16 * 1024**3:
        raise BlockingIOError("Less than 16 GiB host memory available; no launch")
    return {"available_memory_bytes": available, "containers": containers, "compute_processes": processes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isaaclab", type=Path, default=Path("/home/orionh/IsaacLab"))
    parser.add_argument("--limit", type=int, choices=(1, 49), default=49)
    parser.add_argument("--timeout-seconds", type=int, default=2400)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    if not (source / "robot/hexapod_mkii_length_study/manifest.json").is_file():
        parser.error("Source snapshot has no study manifest")
    if args.timeout_seconds <= 0:
        parser.error("Timeout must be positive")
    if args.output.exists():
        parser.error("Use a new output directory to preserve previous run evidence")
    locks = []
    identity = None
    process = None
    report = None
    try:
        # This is a job-scoped collision lock, acquired nonblocking and released
        # on every exit. Opening the existing weather lock read-only needs no
        # permission changes and works with Linux flock(LOCK_EX).
        for path in ("/opt/wx/gpu.lock", "/tmp/hexapod-isaac-gpu.lock"):
            fd = os.open(path, os.O_RDONLY)
            locks.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = preflight()
        if args.preflight_only:
            print(json.dumps(snapshot, indent=2))
            return 0
        args.output.mkdir(parents=True)
        name = "hexapod-length-study-" + uuid.uuid4().hex[:12]
        report = {"container_name": name, "status": "starting", "started_unix": time.time(), "preflight": snapshot,
                  "limit": args.limit, "source": str(source), "training": False}
        def save():
            (args.output / "launcher.json").write_text(json.dumps(report, indent=2) + "\n")
        save()
        cli = ["docker", "compose", "--env-file", "docker/.env.base", "-f", "docker/docker-compose.yaml", "--profile", "base", "run", "--rm", "--no-deps", "--name", name,
               "-w", "/workspace/length-study-output", "-e", "PYTHONDONTWRITEBYTECODE=1",
               "-e", "PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/length-study/tools",
               "-v", f"{source}:/workspace/length-study:rw", "-v", f"{args.output.resolve()}:/workspace/length-study-output:rw",
               "--entrypoint", "/workspace/isaaclab/_isaac_sim/python.sh", "isaac-lab-base",
               "/workspace/length-study/tools/simulate_length_study.py", "--headless", "--enable_cameras",
               "--package", "/workspace/length-study/robot/hexapod_mkii_length_study",
               "--output", "/workspace/length-study-output", "--limit", str(args.limit),
               "--steps", "1000", "--capture-step", "500"]
        deadline = time.monotonic() + args.timeout_seconds
        with (args.output / "run.log").open("w") as log:
            process = subprocess.Popen(cli, cwd=args.isaaclab, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if identity is None:
                    inspected = subprocess.run(["docker", "inspect", "--format", "{{.Id}}", name], text=True, capture_output=True, timeout=20)
                    if inspected.returncode == 0:
                        identity = inspected.stdout.strip()
                        report.update(container_id=identity, status="running")
                        save()
                if time.monotonic() >= deadline:
                    raise TimeoutError("Bounded comparison reached its wall-clock deadline")
                processes, available = resources()
                if identity:
                    top = subprocess.run(["docker", "top", identity, "-eo", "pid"], text=True, capture_output=True, timeout=20)
                    if top.returncode == 0:
                        owned_pids = set(top.stdout.splitlines()[1:])
                        owned_pids = {pid.strip() for pid in owned_pids}
                        competing = [line for line in processes.splitlines() if line.split(",")[0].strip() not in owned_pids]
                        if competing:
                            raise BlockingIOError("A competing CUDA process appeared; stopping only this comparison")
                if available < 16 * 1024**3:
                    raise MemoryError("Available memory fell below 16 GiB; stopping only this comparison")
                time.sleep(5)
        state_path = args.output / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        complete = (process.returncode == 0 and state.get("status") == "completed"
                    and state.get("physics_steps_completed") == 1000
                    and (args.output / "isaac_length_study.png").is_file())
        # Kit can exit zero after an early Python exception. Require recorded
        # completion and the actual captured frame, not just the process code.
        report.update(status="completed" if complete else "failed", exit_code=process.returncode, finished_unix=time.time())
        save()
        return 0 if complete else 1
    except BlockingIOError as exc:
        if report is not None:
            report.update(status="blocked", reason=str(exc), finished_unix=time.time())
            save()
        print(f"BUSY: {exc}", flush=True)
        return 75
    except Exception as exc:
        if report is not None:
            report.update(status="failed", reason=str(exc), finished_unix=time.time())
            save()
        raise
    finally:
        # Cleanup is restricted to the immutable ID created by this invocation.
        if process is not None and process.poll() is None:
            if identity:
                subprocess.run(["docker", "stop", "--time", "20", identity], timeout=30, capture_output=True)
            else:
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for fd in reversed(locks):
            os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
