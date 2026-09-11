#!/usr/bin/env python3
"""Two bounded standing checks; no policy, training, queue or forecast management."""

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
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

from experiments.c_length_study.tools.launch_length_training_spark import live_competitors, save, verified_source
from experiments.c_length_study.tools.launch_length_study_spark import preflight, resources
from experiments.terrain.tools.terrain_contact_evidence import audit_contact_log

VARIANT = "f050_t060"
FIXTURE = "train_ramp_1103"
RUNTIME_TREE = "abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280"
CATALOG = "artifacts/terrain_readiness_2026-09-09/terrain_catalog.json"
FIXTURE_ADMISSION = "artifacts/terrain_readiness_2026-09-09/runtime/terrain_fixture_smoke_003/fixtures/validation.json"
COORDINATION = Path("/home/orionh/SPARK_COMPUTE_COORDINATION.md")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hashes(path):
    return {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob("*")) if p.is_file()}


def check_source(source):
    verified_source(source)
    runtime = json.loads(subprocess.check_output([sys.executable, str(source / "tools/c_study_runtime.py"),
        "--repo-root", str(source)], text=True, timeout=30))
    if runtime.get("runtime_tree_sha256") != RUNTIME_TREE or runtime.get("python_files_verified") != 16:
        raise ValueError("Pinned 16-file C runtime is required")
    package = source / "robot/hexapod_mkii_length_study"
    plan = json.loads((package / "training_plan.json").read_text())
    if plan.get("validation_num_envs") != 32 or plan.get("validation_control_steps") != 1000:
        raise ValueError("The exact 32-environment/1000-step flat gate is required")
    if not plan.get("omni") or plan.get("reference_controller") or plan["omni"].get("diagnostics"):
        raise ValueError("Use the exact full omni plan, without a diagnostic controller")
    fixture = json.loads((source / FIXTURE_ADMISSION).read_text())
    if (fixture.get("status") != "completed" or not fixture.get("all_fixture_smokes_passed")
            or fixture.get("catalog_sha256") != digest(source / CATALOG)
            or not any(r.get("id") == FIXTURE and r.get("passed") for r in fixture.get("runtime_rows", []))):
        raise ValueError("Original catalog and passed attempt003 fixture evidence required")
    return runtime


def command(source, output, name, phase):
    if phase not in ("flat", "terrain"):
        raise ValueError("Only two standing phases exist")
    common = ["docker", "compose", "--env-file", "docker/.env.base", "-f", "docker/docker-compose.yaml",
        "--profile", "base", "run", "--rm", "--no-deps", "--name", name, "-w", "/outputs",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/hexapod/tools:/workspace/hexapod/isaaclab",
        "-v", f"{source}:/workspace/hexapod:ro", "-v", f"{output}:/outputs:rw",
        "-v", f"{output / 'inputs/study'}:/study:{'rw' if phase == 'flat' else 'ro'}",
        "--entrypoint", "/workspace/isaaclab/_isaac_sim/python.sh", "isaac-lab-base"]
    if phase == "flat":
        common += ["/workspace/hexapod/experiments/c_length_study/tools/train_length_study.py", "--mode", "validate"]
    else:
        common += ["/workspace/hexapod/experiments/terrain/tools/validate_terrain_robot.py", "--admission", "/outputs/flat/admission.json",
            "--fixture-admission", "/workspace/hexapod/" + FIXTURE_ADMISSION,
            "--catalog", "/workspace/hexapod/" + CATALOG, "--fixture-id", FIXTURE, "--steps", "1000"]
    return common + ["--package", "/study", "--variant", VARIANT, "--stance-index", "0",
        "--output", f"/outputs/{phase}", "--headless", "--device", "cuda:0", "--info",
        "--kit_args=--/exts/omni.kit.telemetry/skipDeferredStartup=true --/app/extensions/excluded/4=omni.kit.telemetry"]


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
    name = "hexapod-terrain-standing-" + uuid.uuid4().hex
    report = dict(status="starting", phase=phase, container_name=name, started_unix=time.time(),
                  deadline_seconds=600, no_policy_loaded=True)
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
        contact_audit = audit_contact_log(args.output / "logs" / (phase + ".log"))
        save(args.output / "jobs" / (phase + "_contact_data_audit.json"), contact_audit)
        report["contact_data_completeness"] = contact_audit
        if not contact_audit["passed"]:
            raise RuntimeError(f"{phase} has incomplete contact/friction data; standing is not admitted")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isaaclab", type=Path, default=Path("/home/orionh/IsaacLab"))
    args = parser.parse_args()
    args.source, args.output = args.source.resolve(), args.output.resolve()
    if args.output.exists():
        parser.error("Use a fresh output; neither phase resumes automatically")
    runtime = check_source(args.source)
    args.coordination_sha256 = digest(COORDINATION)
    args.output.mkdir(parents=True)
    for directory in ("inputs", "jobs", "logs"):
        (args.output / directory).mkdir()
    signal.signal(signal.SIGTERM, lambda *_: (args.output / "stop.request").touch())
    signal.signal(signal.SIGINT, lambda *_: (args.output / "stop.request").touch())
    package = args.output / "inputs/study"
    report = dict(status="preparing", runtime_binding=runtime, started_unix=time.time(),
        ready_for_terrain_training=False, policy_training_started=False, source=str(args.source),
        source_manifest_sha256=digest(args.source / "campaign_source_hashes.json"), fixture=FIXTURE)
    save(args.output / "campaign.json", report)
    try:
        shutil.copytree(args.source / "robot/hexapod_mkii_length_study", package)
        before = tree_hashes(package)
        save(args.output / "inputs/study_before_flat.sha256.json", before)
        flat = run_owned(args, "flat")
        admission = json.loads((args.output / "flat/admission.json").read_text())
        gate = admission.get("gate", {})
        if (not gate.get("passed") or gate.get("num_envs") != 32 or gate.get("control_steps") != 1000
                or admission.get("plan_sha256") != digest(package / "training_plan.json")
                or admission.get("variant") != VARIANT or admission.get("stance_index") != 0):
            raise RuntimeError("Fresh exact-plan full standing gate is missing or rejected")
        after = tree_hashes(package)
        changed = [p for p in sorted(set(before) | set(after)) if before.get(p) != after.get(p)]
        if set(changed) - {"training_usd/f050_t060/f050_t060.usda"}:
            raise RuntimeError("Flat validator modified something besides its fresh writable USD root")
        save(args.output / "inputs/study_after_flat.sha256.json", after)
        report.update(status="flat_admitted", flat_admission_sha256=digest(args.output / "flat/admission.json"),
                      writable_asset_changes=changed, plan_sha256=admission["plan_sha256"])
        save(args.output / "campaign.json", report)
        check_source(args.source)
        terrain = run_owned(args, "terrain")
        if tree_hashes(package) != after:
            raise RuntimeError("Terrain standing changed its read-only admitted asset")
        check_source(args.source)
        report.update(status="completed", terrain_state_sha256=digest(args.output / "terrain/state.json"),
                      terrain_standing_passed=True, source_unchanged=True, admitted_asset_unchanged=True)
    except Exception as exc:
        report.update(status="stopped" if isinstance(exc, InterruptedError) else "failed", error=repr(exc))
        raise
    finally:
        report["finished_unix"] = time.time()
        save(args.output / "campaign.json", report)


if __name__ == "__main__":
    main()
