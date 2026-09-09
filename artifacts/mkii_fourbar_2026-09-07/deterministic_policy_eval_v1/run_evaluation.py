#!/usr/bin/env python3
"""Bounded host launcher for deterministic flat-policy evaluation, without rendering."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid

from evaluation_common import (SCHEMA,CAPTURE_HELPER_SHA256,helpers,verify_inputs,tool_identity,
    schedule_descriptor,validate_report)

COMMON = None

def digest(path):return COMMON.digest(path)
def environment_layout(source,runtime):return COMMON.environment_layout(source,runtime)
def load_module(path,name):return COMMON.load_module(path,name)
def require_same_inputs(before,after):return COMMON.require_same_inputs(before,after)
def write_json(path,value):return COMMON.write_json(path,value)


SOURCE = "/workspace/hexapod"
OUTPUT = "/workspace/validation_artifacts"
CAPTURE_TOOLS = "/workspace/policy_evaluation_tools"
FROZEN_HELPERS = "/workspace/frozen_capture_helpers"
INPUTS = "/workspace/evaluation_inputs"


def mount_path(value, *, exists=True):
    path = Path(value).expanduser().resolve(strict=exists)
    if any(character in str(path) for character in (":", "\n", "\r", "\x00")):
        raise ValueError("Docker bind paths cannot contain colons or control line breaks")
    return path


def compose_argv(guard, source, output, tools, name, owner, args, runtime):
    relative = Path(runtime["usd_path_relative"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Admitted USD path must remain inside frozen source")
    mounts = [(source, SOURCE, "ro"), (output, OUTPUT, "rw"), (tools, CAPTURE_TOOLS, "ro"),
              (args.checkpoint, INPUTS+"/checkpoint.pt", "ro"),
              (Path(str(args.checkpoint)+".json"), INPUTS+"/checkpoint.pt.json", "ro"),
              (args.admission, INPUTS+"/admission.json", "ro"),
              (args.training_report, INPUTS+"/training_report.json", "ro"),
              (args.capture_tools_dir, FROZEN_HELPERS, "ro")]
    mount_args = [value for host, target, mode in mounts for value in ("-v", f"{host}:{target}:{mode}")]
    environment = [value for key in ("HEXAPOD_USD_PATH", "HEXAPOD_MKII_FOURBAR_USD_PATH")
                   for value in ("-e", f"{key}={SOURCE}/{relative.as_posix()}")]
    environment += ["-e", f"HEXAPOD_MKII_ENVIRONMENT_LAYOUT={args.environment_layout}"]
    barrier = f'while [ ! -f {OUTPUT}/admitted ]; do sleep 0.1; done; exec "$@"'
    return ["docker", "compose", "--env-file", "docker/.env.base", "-f", "docker/docker-compose.yaml",
        "--profile", "base", "run", "--no-deps", "-d", "-T", "--name", name,
        "--label", f"{guard.OWNER_LABEL}={owner}", "-w", OUTPUT,
        "-e", f"PYTHONPATH={SOURCE}/isaaclab", "-e", "PYTHONDONTWRITEBYTECODE=1",
        *environment, *mount_args, "--entrypoint", "/bin/bash", "isaac-lab-base", "-c", barrier,
        "hexapod-policy-evaluation-barrier", "/workspace/isaaclab/_isaac_sim/python.sh",
        f"{CAPTURE_TOOLS}/evaluate_policy.py", *shlex.split(guard.TELEMETRY_ARGUMENT),
        "--source-dir", SOURCE, "--checkpoint", INPUTS+"/checkpoint.pt", "--admission", INPUTS+"/admission.json",
        "--training-report", INPUTS+"/training_report.json", "--output-dir", OUTPUT,
        "--capture-tools-dir", FROZEN_HELPERS, "--viz", "none", "--device", "cuda:0"]


def main(argv=None):
    global COMMON
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source-dir", "capture-tools-dir", "checkpoint", "admission", "training-report", "output-root"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--timeout-seconds", type=int, default=1800)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    args.capture_tools_dir = mount_path(args.capture_tools_dir)
    COMMON = helpers(args.capture_tools_dir)
    if not 60 <= args.timeout_seconds <= 1800:
        p.error("Evaluation timeout must be between 60 and 1800 seconds")
    source = mount_path(args.source_dir)
    args.checkpoint, args.admission, args.training_report = [mount_path(path) for path in
        (args.checkpoint, args.admission, args.training_report)]
    output_root = mount_path(args.output_root, exists=False)
    if output_root.is_relative_to(source) or source.is_relative_to(output_root):
        p.error("Evaluation output and frozen source must be separate directories")
    tools = mount_path(Path(__file__).parent)
    verified = verify_inputs(COMMON, source, args.checkpoint, args.admission, args.training_report)
    args.environment_layout = environment_layout(source, verified["runtime_manifest"])
    tools_before = tool_identity(tools)
    guard_path = source / "isaaclab/deploy/run-mkii-fourbar"
    guard_hash = digest(guard_path)
    guard = load_module(guard_path, "evaluation_frozen_guard")
    for name in ("read_coordination_control", "require_coordination_none", "resource_gate", "inspect_container",
                 "check_identity", "require_unchanged_source", "snapshot_source"):
        if not callable(getattr(guard, name, None)):
            raise ValueError("Frozen source lacks the reviewed coordination/ownership supervisor API")
    owner = uuid.uuid4().hex
    name = "hexapod-policy-evaluation-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + owner[:8]
    output = output_root / name
    cmd = compose_argv(guard, source, output, tools, name, owner, args, verified["runtime_manifest"])
    if args.dry_run:
        print(json.dumps({"execution": "not_started", "argv": cmd, "output": str(output),
                          "input_sha256": verified["input_sha256"], "timeout_seconds": args.timeout_seconds}, indent=2))
        return 0
    for required in (guard.LAB / "docker/.env.base", guard.LAB / "docker/docker-compose.yaml"):
        if not required.is_file():
            raise ValueError("Required Compose configuration is absent")
    # Compose alone reads .env.base. Never print its contents or expanded environment.
    # The shared WX file belongs to another project; never chmod or reopen it
    # writable. Both locks belong to this actual supervised job only.
    shared_fd = os.open('/opt/wx/gpu.lock', os.O_RDONLY)
    if shared_fd == 9:
        relocated = fcntl.fcntl(shared_fd, fcntl.F_DUPFD_CLOEXEC, 10)
        os.close(shared_fd); shared_fd = relocated
    try:
        fcntl.flock(shared_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fd = os.open(guard.LOCK, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            if lock_fd != 9:
                os.dup2(lock_fd, 9); os.close(lock_fd)
            fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(9); raise
    except BaseException:
        os.close(shared_fd); raise
    container = None
    exit_code = 1
    report = {"execution": "preflight", "pass": False, "container_name": name, "owner": owner,
              "source": str(source), "contract": verified["contract"], "input_sha256": verified["input_sha256"],
              "evaluation_tools_sha256": tools_before, "guard_script_sha256": guard_hash,
              "environment_layout": args.environment_layout,
              "cleanup": "not_required", "timeout_seconds": args.timeout_seconds, "gates": [], "argv": cmd,
              "frozen_capture_helper_sha256": CAPTURE_HELPER_SHA256}
    deadline = time.monotonic() + args.timeout_seconds
    old_handlers = {}
    def stop(signum, _frame):
        raise guard.Blocked(f"Evaluation supervisor interrupted by signal {signum}")
    try:
        output.mkdir(parents=True, exist_ok=False)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            old_handlers[signum] = signal.signal(signum, stop)
        report["coordination_initial"] = guard.read_coordination_control()
        guard.require_coordination_none(report["coordination_initial"])
        report["gates"].append(guard.resource_gate())
        report["source_manifest"] = guard.snapshot_source(source, output / "source.SHA256SUMS")
        # Small exact inputs are preserved; checkpoint bytes stay in their immutable original directory.
        for src, target in ((args.admission, "admission.json"), (args.training_report, "training_report.json"),
                            (Path(str(args.checkpoint)+".json"), "checkpoint.pt.json")):
            (output / target).write_bytes(src.read_bytes())
        write_json(output / "request.json", {"schema": SCHEMA, "contract": verified["contract"], "input_sha256": verified["input_sha256"],
            "evaluation_tools_sha256": tools_before, "schedule": schedule_descriptor(),
            "frozen_capture_helper_sha256": CAPTURE_HELPER_SHA256,
            "checkpoint_original_path": str(args.checkpoint), "guard_script_sha256": guard_hash})
        write_json(output / "supervisor.json", report)
        created = guard.command(cmd, cwd=guard.LAB, timeout=60, check=False)
        container = guard.inspect_container(name)
        if not guard.check_identity(container, name, owner):
            raise guard.Blocked("Cannot establish exact owned evaluation container")
        report["container_id"] = container["id"]
        if created.returncode:
            raise guard.Blocked("Evaluation container creation failed")
        restart = guard.command(["docker", "inspect", "--format", "{{.HostConfig.RestartPolicy.Name}}", container["id"]])
        if restart.stdout.strip() != "no":
            raise guard.Blocked("Evaluation container must have no automatic restart")
        report["gates"].append(guard.resource_gate(owned_container=container))
        time.sleep(1)
        report["gates"].append(guard.resource_gate(owned_container=container))
        require_same_inputs(verified, verify_inputs(COMMON, source, args.checkpoint, args.admission, args.training_report))
        if tool_identity(tools) != tools_before or digest(guard_path) != guard_hash:
            raise guard.Blocked("Evaluation implementation changed before admission")
        report["coordination_before_barrier"] = guard.read_coordination_control()
        guard.require_coordination_none(report["coordination_before_barrier"])
        (output / "admitted").touch(exist_ok=False)
        report["execution"] = "running"
        write_json(output / "supervisor.json", report)
        print(f"DETERMINISTIC_POLICY_EVAL_STARTED output={output} container_id={container['id']}", flush=True)
        next_gate = 0.
        while True:
            current = guard.inspect_container(container["id"])
            if not guard.check_identity(current, name, owner):
                raise guard.Blocked("Owned evaluation container identity disappeared or changed")
            if not current["running"]:
                report["container_exit_code"] = current["exit_code"]
                if current["exit_code"] != 0:
                    raise guard.Blocked(f"Evaluation process exited with status {current['exit_code']}")
                exit_code = 0
                break
            if time.monotonic() >= deadline:
                raise guard.Blocked("Evaluation execution timeout")
            if time.monotonic() >= next_gate:
                report["coordination_latest"] = guard.read_coordination_control()
                if report["coordination_latest"]["status"] != "NONE":
                    (output / "stop_requested").write_text("Evaluation yields to shared compute request.\n")
                    guard.require_coordination_none(report["coordination_latest"])
                report["last_runtime_gate"] = guard.resource_gate(owned_container=current, allow_owned_gpu=True)
                write_json(output / "supervisor.json", report)
                next_gate = time.monotonic() + 5
            time.sleep(1)
    except BaseException as error:
        report["execution"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        exit_code = 1
    finally:
        for signum in old_handlers:
            signal.signal(signum, signal.SIG_IGN)
        try:
            if container is None:
                container = guard.inspect_container(name)
            if guard.check_identity(container, name, owner):
                current = guard.inspect_container(container["id"])
                if not guard.check_identity(current, name, owner):
                    raise guard.Blocked("Evaluation cleanup identity mismatch")
                if current["running"]:
                    guard.command(["docker", "stop", "--time", "25", container["id"]], timeout=30, check=False)
                with (output / "container.log").open("w") as stream:
                    subprocess.run(["docker", "logs", "--timestamps", container["id"]], stdout=stream,
                                   stderr=subprocess.STDOUT, timeout=20, check=True)
                current = guard.inspect_container(container["id"])
                if not guard.check_identity(current, name, owner) or current["running"]:
                    raise guard.Blocked("Owned evaluation did not stop cleanly")
                report["final_container_exit_code"] = current["exit_code"]
                guard.command(["docker", "rm", container["id"]], timeout=15)
                report["cleanup"] = "removed_exact_id"
            elif container is not None:
                raise guard.Blocked("Evaluation cleanup refused foreign identity")
        except BaseException as error:
            report["cleanup"] = "FAILED"
            report["cleanup_error"] = f"{type(error).__name__}: {error}"
            exit_code = 1
        try:
            require_same_inputs(verified, verify_inputs(COMMON, source, args.checkpoint, args.admission, args.training_report))
            report["source_and_inputs_unchanged"] = True
            if tools_before != tool_identity(tools) or guard_hash != digest(guard_path):
                raise ValueError("Evaluation implementation changed while running")
            report["evaluation_report_present"] = (output / "report.json").is_file()
            if exit_code == 0:
                result = validate_report(output / "report.json", verified, tools_before)
                if result.get("evaluation_tools_sha256") != tools_before:
                    raise ValueError("Recorder implementation identity differs from launcher")
                report["evaluation_artifacts_verified"] = True
        except BaseException as error:
            report["report_error"] = f"{type(error).__name__}: {error}"
            exit_code = 1
        finally:
            report["pass"] = exit_code == 0
            report["execution"] = "finished" if exit_code == 0 else "failed"
            report["supervisor_exit_code"] = exit_code
            try:
                if output.is_dir():
                    write_json(output / "supervisor.json", report)
            finally:
                fcntl.flock(9, fcntl.LOCK_UN)
                os.close(9)
                os.close(shared_fd)
                for signum, handler in old_handlers.items():
                    signal.signal(signum, handler)
    print(f"DETERMINISTIC_POLICY_EVAL_FINISHED exit={exit_code} output={output}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
