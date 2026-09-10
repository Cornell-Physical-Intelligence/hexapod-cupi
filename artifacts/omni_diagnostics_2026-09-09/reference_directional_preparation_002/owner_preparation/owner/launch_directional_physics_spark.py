#!/usr/bin/env python3
"""Fresh standing then three previously unmeasured named low-speed directional reference cases; no PPO."""
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
from terrain_contact_evidence import audit_contact_log
from solver_comparison import PROTOCOL as SOLVER_PROTOCOL
from directional_contract import PROTOCOL, CASES, check_wave

VARIANT = "f050_t060"
GEOMETRY_REFERENCE = "robot/hexapod_mkii_length_study/candidate_c_reference.json"
RUNTIME_TREE = "abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280"
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
    from types import SimpleNamespace
    from screen_contract import preflight as screen_preflight
    package = source / "robot/hexapod_mkii_length_study"
    screen_preflight(SimpleNamespace(package=package, variant=VARIANT, stance_index=0,
        output=source.parent / ("uncreated-preflight-"+uuid.uuid4().hex), mode="standing",
        num_envs=32, steps=1000, admission=None, geometry_reference=source / GEOMETRY_REFERENCE), source)
    check_wave(source)
    return runtime


def command(source, output, name, phase):
    if phase != "standing" and phase not in CASES:
        raise ValueError("Only fresh standing and three previously unmeasured named directional cases exist")
    common = ["docker", "compose", "--env-file", "docker/.env.base", "-f", "docker/docker-compose.yaml",
        "--profile", "base", "run", "--rm", "--no-deps", "--name", name, "-w", "/outputs",
        "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONUNBUFFERED=1",
        "-e", "PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/hexapod/tools:/workspace/hexapod/isaaclab",
        "-v", f"{source}:/workspace/hexapod:ro", "-v", f"{output}:/outputs:rw",
        "-v", f"{output / 'inputs/study'}:/study:ro",
        "--entrypoint", "/workspace/isaaclab/_isaac_sim/python.sh", "isaac-lab-base",
        "/workspace/hexapod/tools/" + ("run_reference_physics.py" if phase == "standing" else "run_directional_physics.py"), "--mode", "standing" if phase == "standing" else "directional",
        "--num-envs", "32" if phase == "standing" else "1",
        "--steps", "1000" if phase == "standing" else "2400",
        "--geometry-reference", "/workspace/hexapod/" + GEOMETRY_REFERENCE]
    if phase != "standing":
        common += ["--admission", "/outputs/standing/admission.json", "--case", phase]
    return common + ["--package", "/study", "--variant", VARIANT, "--stance-index", "0",
        "--output", f"/outputs/{phase}", "--headless", "--device", "cuda:0", "--info",
        "--kit_args=--/exts/omni.kit.telemetry/skipDeferredStartup=true --/app/extensions/excluded/4=omni.kit.telemetry"]


def owned_container(name, identity=None):
    """Recover immutable identity even if Docker client exits before first poll."""
    found = subprocess.run(["docker", "inspect", "--format", "{{.Id}} {{.Name}} {{.State.Running}}",
                            identity or name], text=True, capture_output=True, timeout=20)
    if found.returncode:
        if 'no such object' in found.stderr.lower() or 'no such container' in found.stderr.lower():
            return None
        raise RuntimeError('Container inspection failed; ownership/absence is unknown: '+found.stderr.strip())
    fields = found.stdout.strip().split()
    if len(fields) != 3 or fields[1] != "/" + name or (identity and fields[0] != identity):
        raise RuntimeError("Container identity mismatch; do not signal it")
    return fields[0], fields[2] == "true"


def run_owned(args, phase):
    locks, process, identity = [], None, None
    name = "hexapod-reference-physics-" + uuid.uuid4().hex
    report = dict(status="starting", phase=phase, container_name=name, started_unix=time.time(),
                  deadline_seconds=600, app_ready_deadline_seconds=90, no_policy_loaded=True)
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
        app_ready_deadline = time.monotonic() + 90
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
                if time.monotonic() >= app_ready_deadline:
                    if 'REFERENCE_SCREEN_APP_READY' not in (args.output / 'logs' / (phase+'.log')).read_text(errors='replace'):
                        report['startup_failure_kind'] = 'no_reference_AppReady_by90s'
                        raise TimeoutError('No AppReady by90s; preserve startup log and45s traceback')
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
        except Exception as cleanup_error:
            report['cleanup_checked'] = False
            report['cleanup_error'] = repr(cleanup_error)
            report['cleanup_requires_owner_review'] = True
            raise
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
    if args.output == args.source or args.source in args.output.parents:
        parser.error('Output must be outside the immutable source')
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
        stage2_complete=False, terrain_qualified=False, policy_training_started=False, source=str(args.source),
        source_manifest_sha256=digest(args.source / "campaign_source_hashes.json"), automatic_continuation=False, solver_comparison=SOLVER_PROTOCOL, directional_protocol=PROTOCOL)
    save(args.output / "campaign.json", report)
    try:
        shutil.copytree(args.source / "robot/hexapod_mkii_length_study", package)
        before = tree_hashes(package)
        save(args.output / "inputs/study_before.sha256.json", before)
        standing = run_owned(args, "standing")
        admission = json.loads((args.output / "standing/admission.json").read_text())
        gate = admission.get("gate", {})
        identity = admission.get("identity", {})
        if (admission.get("status") != "completed" or not gate.get("passed")
                or gate.get("num_envs") != 32 or gate.get("control_steps") != 1000
                or not gate.get("all_replica_quiet", {}).get("passed")
                or identity.get("plan_sha256") != digest(package / "training_plan.json")
                or identity.get("source_manifest_sha256") != report["source_manifest_sha256"]
                or identity.get("variant") != VARIANT or identity.get("stance_index") != 0):
            raise RuntimeError("Fresh exact-source zero-residual standing gate is missing or rejected")
        if tree_hashes(package) != before:
            raise RuntimeError("Standing changed its read-only asset")
        report.update(status="standing_admitted", standing_admission_sha256=digest(args.output / "standing/admission.json"),
                      identity=identity)
        save(args.output / "campaign.json", report)
        check_source(args.source)
        comparison = json.loads((args.output / "standing/solver_comparison.json").read_text())
        if (identity.get("solver_comparison") != SOLVER_PROTOCOL or comparison.get("protocol") != SOLVER_PROTOCOL
                or comparison.get("actual_scene_attributes", {}).get("physxScene:enableExternalForcesEveryIteration") is not True):
            raise RuntimeError("Fresh standing solver/source identity mismatch")
        case_states = {}
        for case in CASES:
            check_source(args.source)
            result = run_owned(args, case)
            case_identity = result.get('identity', {})
            if (case_identity.get('standing_identity') != identity or case_identity.get('directional_protocol') != PROTOCOL
                    or case_identity.get('case') != case or case_identity.get('requested_forward_left_yaw') != list(CASES[case])
                    or case_identity.get('standing_admission_sha256') != digest(args.output / 'standing/admission.json')
                    or result.get('control_steps') != 2400 or result.get('gate', {}).get('passed') is not True):
                raise RuntimeError('Directional result differs from its exact named-case/fresh standing contract')
            actual_solver = json.loads((args.output / case / 'solver_comparison.json').read_text())
            if actual_solver.get('protocol') != SOLVER_PROTOCOL or actual_solver.get('actual_scene_attributes') != comparison.get('actual_scene_attributes'):
                raise RuntimeError('Directional solver readback differs from fresh standing admission')
            if tree_hashes(package) != before:
                raise RuntimeError('Directional case changed its read-only admitted asset')
            case_states[case] = result
            report.update(status='running_directional_cases', completed_cases=list(case_states), untested_cases=[k for k in CASES if k not in case_states])
            save(args.output / 'campaign.json', report)
        check_source(args.source)
        report.update(status='completed', all_declared_reference_cases_passed=True, declared_case_count=len(CASES),
                      all_omni_directions_qualified=False, velocity_fidelity_qualified=False,
                      case_state_sha256={case:digest(args.output / case / 'state.json') for case in CASES},
                      source_unchanged=True, admitted_asset_unchanged=True)
    except Exception as exc:
        report.update(status="stopped" if isinstance(exc, InterruptedError) else "failed", error=repr(exc))
        raise
    finally:
        try:
            check_source(args.source)
            report['terminal_source_integrity'] = {'passed': True}
        except Exception as integrity_error:
            report['terminal_source_integrity'] = {'passed': False, 'error': repr(integrity_error)}
            report['status'] = 'failed'
        if package.exists():
            try:
                actual_assets = tree_hashes(package)
                before_path = args.output / 'inputs/study_before.sha256.json'
                original_assets = json.loads(before_path.read_text()) if before_path.exists() else None
                report['terminal_asset_integrity'] = {'passed': original_assets == actual_assets,
                    'files_checked': len(actual_assets), 'before_map_available': original_assets is not None}
                if original_assets != actual_assets:
                    report['status'] = 'failed'
            except Exception as integrity_error:
                report['terminal_asset_integrity'] = {'passed': False, 'error': repr(integrity_error)}
                report['status'] = 'failed'
        report["finished_unix"] = time.time()
        save(args.output / "campaign.json", report)
    if report['status'] != 'completed':
        raise RuntimeError('Terminal source/asset integrity failed; campaign not admitted')


if __name__ == "__main__":
    main()
