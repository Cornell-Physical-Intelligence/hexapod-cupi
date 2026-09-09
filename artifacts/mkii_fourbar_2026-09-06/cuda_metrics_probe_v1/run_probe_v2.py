#!/usr/bin/env python3
"""One bounded CUDA arithmetic comparison; never starts Isaac Sim or PPO.

Uses the frozen training supervisor's compute/ownership checks and its exact
container environment. The caller must wait until the current validation exits.
No long-lived reservation, scheduling, source edits, or automatic retries.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import time
from types import SimpleNamespace
import uuid

EXPECTED_SOURCE = "d6fbd6e9b5cf499dd8603613d6977ff226f443fb15f149bbe320fc11e15d5b54"
EXPECTED_PROBE = "3ffd4ae9dbc235ae2193a923b0fac1d6cb088153dc24cd50e959445b269e72aa"


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    source, probe, output = (p.resolve() for p in (args.source_dir, args.probe, args.output))
    if output.exists():
        raise ValueError("Output must be fresh")
    if hashlib.sha256(probe.read_bytes()).hexdigest() != EXPECTED_PROBE:
        raise ValueError("Unreviewed probe bytes")
    host = SimpleNamespace(**runpy.run_path(str(source / "isaaclab/deploy/run-mkii-fourbar")))
    contract = host.identity(source)
    if contract["sha256"] != EXPECTED_SOURCE:
        raise ValueError("Unexpected frozen source identity")
    owner = uuid.uuid4().hex
    name = "hexapod-cuda-math-" + owner[:12]
    opts = SimpleNamespace(mode="validate", steps=1000, solver_multiplier=2,
        asset_model="mkii_fourbar_v5", environment_layout="coincident_flat_origin_v1", num_envs=32)
    argv = host.compose_argv(source, output, name, owner, opts)
    entrypoint = argv.index("--entrypoint")
    argv[entrypoint:entrypoint] = ["-v", f"{probe}:/workspace/cpu_batch_probe.py:ro",
        "-e", "OMP_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1"]
    python_index = argv.index("/workspace/isaaclab/_isaac_sim/python.sh")
    argv[python_index:] = [argv[python_index], "/workspace/cpu_batch_probe.py",
        "--source-dir", "/workspace/hexapod", "--device", "cuda:0",
        "--report", "/workspace/validation_artifacts/report.json"]
    if not args.execute:
        print(json.dumps({"execution": "not_started", "argv": argv,
            "source_identity": contract, "timeout_seconds": 900}, indent=2))
        return 0
    descriptors = []
    container = None
    report = {"execution": "preflight", "owner": owner, "name": name,
        "source_identity": contract, "probe_sha256": EXPECTED_PROBE,
        "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "purpose": "synthetic CUDA arithmetic equivalence and timing, not physics admission",
        "argv": argv, "gates": [], "persistent_reservation": False,
        "timeout_seconds": 900, "cleanup": "not_required", "exit_code": None}
    def interrupted(signum, _frame):
        raise RuntimeError(f"Interrupted by signal {signum}")
    try:
        for lock in (host.SHARED_GPU_LOCK, host.LOCK):
            # Existing weather-owned file is readable but not writable to this
            # account. Linux flock permits an exclusive lock on a read-only FD;
            # never change the shared file's mode, owner, or contents.
            flags = os.O_RDONLY if lock == host.SHARED_GPU_LOCK else os.O_RDWR | os.O_CREAT
            descriptor = os.open(lock, flags, 0o666)
            descriptors.append(descriptor)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, interrupted)
        report["coordination_initial"] = host.read_coordination_control()
        host.require_coordination_none(report["coordination_initial"])
        report["gates"].append(host.resource_gate())
        host.atomic_json(output / "supervisor.json", report)
        result = host.command(argv, cwd=host.LAB, timeout=60, check=False)
        container = host.inspect_container(name)
        if not host.check_identity(container, name, owner):
            raise RuntimeError("Cannot establish created container ownership")
        report["container_id"] = container["id"]
        if result.returncode:
            raise RuntimeError(f"Container creation exited {result.returncode}")
        report["gates"].append(host.resource_gate(owned_container=container))
        host.require_unchanged_source(source, contract, "before CUDA comparison")
        if hashlib.sha256(probe.read_bytes()).hexdigest() != EXPECTED_PROBE:
            raise RuntimeError("Probe changed before launch")
        host.require_coordination_none(host.read_coordination_control())
        (output / "admitted").touch(exist_ok=False)
        deadline = time.monotonic() + 900
        report["execution"] = "running"
        report["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        host.atomic_json(output / "supervisor.json", report)
        print("CUDA_METRICS_STARTED " + str(output), flush=True)
        while True:
            current = host.inspect_container(container["id"])
            if not host.check_identity(current, name, owner):
                raise RuntimeError("Owned container identity changed")
            if not current["running"]:
                report["exit_code"] = int(current["exit_code"])
                if report["exit_code"]:
                    raise RuntimeError(f"Probe exited {report['exit_code']}")
                result = json.loads((output / "report.json").read_text())
                if result.get("pass") is not True or result.get("device") != "cuda:0":
                    raise RuntimeError("CUDA comparison did not pass")
                host.require_unchanged_source(source, contract, "after CUDA comparison")
                report["execution"] = "finished"
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("CUDA comparison exceeded 900 seconds")
            host.require_coordination_none(host.read_coordination_control())
            report["last_runtime_gate"] = host.resource_gate(owned_container=current, allow_owned_gpu=True)
            host.atomic_json(output / "supervisor.json", report)
            time.sleep(5)
    except Exception as error:
        report.update(execution="failed", error=str(error), exit_code=1)
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        try:
            if container is None:
                candidate = host.inspect_container(name)
                if host.check_identity(candidate, name, owner):
                    container = candidate
            if container is not None:
                current = host.inspect_container(container["id"])
                if not host.check_identity(current, name, owner):
                    raise RuntimeError("Cleanup ownership check failed")
                if current["running"]:
                    host.command(["docker", "stop", "--time", "20", current["id"]], timeout=40)
                logs = host.command(["docker", "logs", current["id"]], timeout=30, check=False)
                (output / "container.stdout.log").write_text(logs.stdout)
                (output / "container.stderr.log").write_text(logs.stderr)
                host.command(["docker", "rm", current["id"]], timeout=30)
                report["cleanup"] = "owned_container_removed"
        except Exception as error:
            report.update(execution="failed", cleanup="failed", cleanup_error=str(error), exit_code=1)
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if output.is_dir():
            host.atomic_json(output / "supervisor.json", report)
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    print(json.dumps({key: report.get(key) for key in ("execution", "exit_code", "cleanup", "error")}), flush=True)
    return report["exit_code"] or 0


if __name__ == "__main__":
    sys.exit(main())
