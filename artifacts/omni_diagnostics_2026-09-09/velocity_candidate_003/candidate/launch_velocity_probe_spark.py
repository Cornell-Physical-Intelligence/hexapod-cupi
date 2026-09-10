#!/usr/bin/env python3
"""Bounded new-action validate/probe only; no automatic walking training."""
import argparse


import fcntl


import hashlib


import json


import os


from pathlib import Path


import shutil


import signal


import subprocess


import sys


import time


import uuid


from launch_length_training_spark import live_competitors, save, verified_source


from launch_length_study_spark import preflight, resources

VARIANT = "f050_t060"
RUNTIME_TREE = "abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280"
COORDINATION = Path("/home/orionh/SPARK_COMPUTE_COORDINATION.md")
LINEAGE = "c_serial_omni_target_velocity_v1"


def check_source(source):
    verified_source(source)
    runtime = json.loads(subprocess.check_output([sys.executable, str(source / "tools/c_study_runtime.py"),
        "--repo-root", str(source)], text=True, timeout=30))
    if runtime.get("runtime_tree_sha256") != RUNTIME_TREE or runtime.get("python_files_verified") != 16:
        raise ValueError("Pinned C-study runtime required")
    plan = json.loads((source / "robot/hexapod_mkii_length_study/training_plan.json").read_text())
    if plan.get("validation_num_envs") != 32 or plan.get("validation_control_steps") != 1000:
        raise ValueError("Exact full standing gate required")
    omni = plan.get("omni", {})
    if (plan.get("reference_controller") or omni.get("architecture") != LINEAGE
            or omni.get("velocity_candidate") != {"profile": "formal_004", "max_acceleration_rad_s2": 8.0}):
        raise ValueError("This bounded launcher is only the explicit formal_004/8 candidate")
    if omni.get("initial_exploration_std") != .005 or omni.get("sampled_calibration_control_steps") != 1000:
        raise ValueError("Candidate 002 requires explicit std 0.005 and 20 s sampled calibration")
    return runtime


def command(source, output, name, phase):
    if phase not in ("flat", "probe"):
        raise ValueError("Only fresh standing and schema/exploration probes are authorized here")
    argv = ["docker", "compose", "--env-file", "docker/.env.base", "-f", "docker/docker-compose.yaml",
        "--profile", "base", "run", "--rm", "--no-deps", "--name", name, "-w", "/outputs",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/hexapod/tools:/workspace/hexapod/isaaclab",
        "-v", f"{source}:/workspace/hexapod:ro", "-v", f"{output}:/outputs:rw",
        "-v", f"{output / 'inputs/study'}:/study:{'rw' if phase == 'flat' else 'ro'}",
        "--entrypoint", "/workspace/isaaclab/_isaac_sim/python.sh", "isaac-lab-base",
        "/workspace/hexapod/tools/train_velocity_candidate.py", "--mode", "validate" if phase == "flat" else "probe",
        "--package", "/study", "--variant", VARIANT, "--stance-index", "0", "--output", f"/outputs/{phase}",
        "--headless", "--device", "cuda:0", "--info",
        "--kit_args=--/exts/omni.kit.telemetry/skipDeferredStartup=true --/app/extensions/excluded/4=omni.kit.telemetry"]
    if phase == "probe":
        argv += ["--admission", "/outputs/flat/admission.json"]
    return argv


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hashes(path):
    return {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob("*")) if p.is_file()}


def owned_container(name, identity=None):
    """Recover immutable identity even if Docker client exits before first poll."""
    found = subprocess.run(["docker", "inspect", "--format", "{{.Id}} {{.Name}} {{.State.Running}}",
                            identity or name], text=True, capture_output=True, timeout=20)
    if found.returncode:
        return None
    fields = found.stdout.strip().split()
    if len(fields) != 3 or fields[1] != "/" + name or (identity and fields[0] != identity):
        raise RuntimeError("Container identity mismatch; do not signal it")
    return fields[0], fields[2] == "true"


def run_owned(args, phase):
    locks, process, identity = [], None, None
    name = "hexapod-velocity-probe-" + uuid.uuid4().hex
    report = dict(status="starting", phase=phase, container_name=name, started_unix=time.time(),
                  deadline_seconds=600, existing_checkpoint_loaded=False, walking_training_started=False,
                  stand_only_update_cap=(2 if phase == "probe" else 0))
    report_path = args.output / "jobs" / (phase + ".json")
    try:
        if (args.output / "stop.request").exists():
            raise InterruptedError("Stop requested before acquiring a new job")
        if digest(COORDINATION) != args.coordination_sha256:
            raise InterruptedError("Coordination note changed; return ownership for review")
        report["coordination_sha256"] = args.coordination_sha256
        for path in ("/opt/wx/gpu.lock", "/tmp/hexapod-isaac-gpu.lock"):
            fd = os.open(path, os.O_RDONLY)
            locks.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report["preflight"] = preflight()
        verified_source(args.source)
        save(report_path, report)
        deadline = time.monotonic() + 600
        with (args.output / "logs" / (phase + ".log")).open("w") as log:
            process = subprocess.Popen(command(args.source, args.output, name, phase), cwd=args.isaaclab,
                                       stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                owned = owned_container(name, identity)
                if owned:
                    identity = owned[0]
                    report.update(status="running", container_id=identity)
                    save(report_path, report)
                if (args.output / "stop.request").exists():
                    raise InterruptedError("Stop requested")
                if digest(COORDINATION) != args.coordination_sha256:
                    raise InterruptedError("Coordination changed during standing; yielding owned job")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Standing phase exceeded ten-minute bound")
                processes, available = resources()
                if available < 16 * 1024**3:
                    raise MemoryError("Less than 16 GiB available")
                if identity:
                    top = subprocess.run(["docker", "top", identity, "-eo", "pid"], text=True,
                                         capture_output=True, timeout=20)
                    if top.returncode == 0:
                        owned_pids = {v.strip() for v in top.stdout.splitlines()[1:]}
                        competitors = live_competitors(processes, owned_pids, identity)
                        if competitors:
                            report["competitors"] = competitors
                            raise RuntimeError("Unrelated CUDA process appeared; yielding this owned job")
                time.sleep(5)
        state_path = args.output / phase / "state.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        if process.returncode != 0 or state.get("status") != "completed" or (args.output / phase / "failure.json").exists():
            raise RuntimeError(f"{phase} standing did not complete; inspect its state and log")
        if state.get("runtime_binding", {}).get("runtime_tree_sha256") != RUNTIME_TREE:
            raise RuntimeError("Runtime identity missing from standing evidence")
        report.update(status="completed", exit_code=process.returncode)
        return state
    except Exception as exc:
        report.update(status="stopped" if isinstance(exc, InterruptedError) else "failed", error=repr(exc))
        raise
    finally:
        # No Docker client process-state condition: client exit does not mean
        # its container exited. Name recovery is bounded and identity-checked.
        try:
            owned = owned_container(name, identity) if process is not None else None
            if owned and owned[1]:
                identity = owned[0]
                subprocess.run(["docker", "stop", "--time", "20", identity], timeout=30, check=True, capture_output=True)
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            # Catch container creation racing the Docker-client termination.
            remaining = owned_container(name, identity) if process is not None else None
            if remaining and remaining[1]:
                subprocess.run(["docker", "stop", "--time", "20", remaining[0]], timeout=30, check=True, capture_output=True)
            report["cleanup_checked"] = True
        finally:
            report.update(finished_unix=time.time(), container_id=identity)
            save(report_path, report)
            for fd in reversed(locks):
                os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isaaclab", type=Path, default=Path("/home/orionh/IsaacLab"))
    args = parser.parse_args()
    args.source, args.output = args.source.resolve(), args.output.resolve()
    if args.output.exists():
        parser.error("Use a fresh output; no automatic retry or checkpoint resume")
    runtime = check_source(args.source)
    args.coordination_sha256 = digest(COORDINATION)
    args.output.mkdir(parents=True)
    for directory in ("inputs", "jobs", "logs"):
        (args.output / directory).mkdir()
    signal.signal(signal.SIGTERM, lambda *_: (args.output / "stop.request").touch())
    signal.signal(signal.SIGINT, lambda *_: (args.output / "stop.request").touch())
    package = args.output / "inputs/study"
    report = dict(status="preparing", runtime_binding=runtime, started_unix=time.time(),
        source=str(args.source), source_manifest_sha256=digest(args.source / "campaign_source_hashes.json"),
        action_lineage=LINEAGE, stage2_complete=False, walking_training_started=False)
    save(args.output / "campaign.json", report)
    try:
        shutil.copytree(args.source / "robot/hexapod_mkii_length_study", package)
        before = tree_hashes(package)
        save(args.output / "inputs/study_before_flat.sha256.json", before)
        run_owned(args, "flat")
        admission = json.loads((args.output / "flat/admission.json").read_text())
        gate = admission.get("gate", {})
        if (not gate.get("passed") or gate.get("num_envs") != 32 or gate.get("control_steps") != 1000
                or admission.get("plan_sha256") != digest(package / "training_plan.json")
                or admission.get("variant") != VARIANT or admission.get("stance_index") != 0):
            raise RuntimeError("Fresh exact-plan full standing gate is missing or rejected")
        after = tree_hashes(package)
        changed = [p for p in sorted(set(before) | set(after)) if before.get(p) != after.get(p)]
        if set(changed) - {"training_usd/f050_t060/f050_t060.usda"}:
            raise RuntimeError("Standing changed something besides its new writable USD root")
        save(args.output / "inputs/study_after_flat.sha256.json", after)
        report.update(status="flat_admitted", flat_admission_sha256=digest(args.output / "flat/admission.json"),
            plan_sha256=admission["plan_sha256"], writable_asset_changes=changed)
        save(args.output / "campaign.json", report)
        check_source(args.source)
        run_owned(args, "probe")
        if tree_hashes(package) != after:
            raise RuntimeError("Probe changed the read-only admitted study asset")
        calibration = json.loads((args.output / "probe/calibration.json").read_text())
        runner_smoke = json.loads((args.output / "probe/runner_smoke.json").read_text())
        if not calibration.get("passed") or not runner_smoke.get("passed"):
            raise RuntimeError("Exploration or runner smoke rejected; no walking training")
        check_source(args.source)
        report.update(status="completed", calibration_sha256=digest(args.output / "probe/calibration.json"),
            runner_smoke_sha256=digest(args.output / "probe/runner_smoke.json"),
            source_unchanged=True, admitted_asset_unchanged=True,
            scope="New-action standing/exploration and two-update stand-only runner smoke; no walking qualification")
    except Exception as exc:
        report.update(status="stopped" if isinstance(exc, InterruptedError) else "failed", error=repr(exc))
        raise
    finally:
        report["finished_unix"] = time.time()
        save(args.output / "campaign.json", report)


if __name__ == "__main__":
    main()
