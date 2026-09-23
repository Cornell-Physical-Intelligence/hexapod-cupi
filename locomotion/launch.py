"""Launch one native allocation with explicit ownership and bounded cleanup."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from types import SimpleNamespace
import uuid

from . import reservation as guard

_INCOMPLETE = re.compile(r"incomplete\s+(?:contact|friction)\s+data", re.IGNORECASE)

REMOTE_ROOT = Path('/home/orionh/HEXAPOD_runs/restart_20260914')


def preflight():
    reservation = guard.verify_reservation()
    compute = guard.no_live_compute()
    processes, available = resources()
    guard.require(not processes and available >= 16 * 1024**3, 'GPU busy or host memory below 16 GiB')
    return {'reservation': reservation, 'compute': compute, 'available_memory_bytes': available}


def resources():
    processes = guard.call(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"])
    memory = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")}
    return processes, memory["MemAvailable"]

def live_competitors(processes, owned, identity, proc_root=Path("/proc")):
    """Reconcile nvidia-smi and Docker snapshots taken at different times.

    A CUDA PID may exit between the GPU query and docker top during Kit
    shutdown. Ignore vanished processes and independently confirm the cgroup
    for live processes omitted by the later Docker snapshot. All other live
    CUDA processes remain competitors, including unreadable cgroups.
    """
    competing=[]
    for line in processes.splitlines():
        pid=line.split(",")[0].strip()
        if pid in owned:
            continue
        try:
            cgroup=(proc_root/pid/"cgroup").read_text()
        except FileNotFoundError:
            continue
        except PermissionError:
            cgroup="unreadable"
        if identity not in cgroup:
            competing.append({"process":line,"cgroup":cgroup.strip()})
    return competing

def audit_contact_log(path):
    path = Path(path)
    data = path.read_bytes()  # A missing/unreadable log must never pass.
    lines = data.decode("utf-8", errors="replace").splitlines()
    matches = [line for line in lines if _INCOMPLETE.search(line)]
    capacities = sorted({int(value) for line in matches
                         for value in re.findall(r"maxContactDataCount\s*=\s*(\d+)", line)})
    return dict(log_sha256=hashlib.sha256(data).hexdigest(),
        incomplete_data_warning_count=len(matches), reported_capacities=capacities,
        first_warning=matches[0] if matches else None,
        last_warning=matches[-1] if matches else None,
        passed=not matches,
        scope="Required absence of reported contact/friction data truncation; raw physical gate must also pass")

def verify(binding, own):
    guard.require(binding['schema']=='hexapod_locomotion_launch_v1','Wrong launch schema')
    guard.require(binding['root_review_complete'] is True,'Root review incomplete')
    guard.require(binding['mode'] in ('diagnostic','replay','train','video','evaluate','tripod'),'Unsupported native mode')
    guard.require(type(binding['max_seconds']) is int and 120<=binding['max_seconds']<=7200,'Unbounded allocation')
    guard.require(binding.get('module') in ('locomotion.train', 'locomotion.priors.replay_native',
                                          'locomotion.tripod_evaluate'), 'Unsupported native entry point')
    guard.require((binding['mode'] == 'tripod') == (binding['module'] == 'locomotion.tripod_evaluate'),
                  'Prescribed-controller mode and entry must match')
    paths={k:guard.canonical_path(binding[k]) for k in ('source','asset','prior','geometry_source','output')}
    guard.require(paths['source']==own,'Launcher must belong to its bound source')
    for k in ('source','output','prior'):
        guard.require(REMOTE_ROOT in paths[k].parents,'Fresh restart path required')
    for k,v in paths.items():
        if k!='output':
            guard.require(v!=paths['output'] and v not in paths['output'].parents
                          and paths['output'] not in v.parents,'Output overlaps immutable input')
    guard.verify_tree(paths['source'],binding['source_freeze_sha256'])
    for name,digest in binding['input_files'].items(): guard.pinned_file(name,digest)
    guard.require(binding.get('stage2_complete') is False and binding.get('physical_admission') is False,
                  'Launch cannot grant qualification')
    guard.require(binding['command_args'] and all(isinstance(x,str) for x in binding['command_args']),
                  'Exact native arguments required')
    argv=binding['command_args']
    guard.require(not any(x=='--output' or x.startswith('--output=') for x in argv),'Output override rejected')
    for option,want in (('--mode',binding['mode']),('--source-freeze-sha256',binding['source_freeze_sha256']),
                        ('--asset','/asset'),('--model','/asset/source/model.json'),('--device','cuda:0')):
        guard.require(argv.count(option)==1 and not any(x.startswith(option+'=') for x in argv),
                      'Exact single native argument required: '+option)
        index=argv.index(option)
        guard.require(index+1<len(argv) and argv[index+1]==want,'Native argument differs: '+option)
    guard.require(argv.count('--headless')==1,'Explicit headless allocation required')
    if online_wandb(argv):
        guard.require(bool(os.environ.get('WANDB_API_KEY')),'Online W&B logging needs WANDB_API_KEY in the launcher environment')
    return paths

def online_wandb(argv):
    return any(a=='--wandb-mode' and b=='online' for a,b in zip(argv,argv[1:]))

def command(binding, paths, name):
    args=['docker','compose','--env-file','docker/.env.base','-f','docker/docker-compose.yaml','--profile','base',
          'run','--rm','--no-deps','--name',name,'-w','/output','-e','PYTHONDONTWRITEBYTECODE=1',
          '-e','PYTHONUNBUFFERED=1','-e','PYTHONPATH=/source:/workspace/isaaclab/source/isaaclab',
          '-v',str(paths['output'])+':/output:rw','-v',str(paths['source'])+':/source:ro',
          '-v',str(paths['asset'])+':/asset:ro','-v',str(paths['prior'])+':/prior:ro',
          '-v',str(paths['geometry_source'])+':/geometry_source:ro']
    if online_wandb(binding['command_args']):
        # Pass the key by name so its value never enters the command line or a record.
        args+=['-e','WANDB_API_KEY']
    for source,target in binding.get('extra_mounts',[]):
        guard.canonical_path(source)
        guard.require(target in ('/standing_one','/standing_batch','/admission','/checkpoint','/realized_prior'),'Unexpected read-only input mount')
        args+=['-v',source+':'+target+':ro']
    return args+['--entrypoint','/workspace/isaaclab/_isaac_sim/python.sh','isaac-lab-base',
                '-m',binding['module'],'--output','/output/standing',*binding['command_args']]

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

def run_owned(binding, paths):
    args = SimpleNamespace(source=paths["source"], output=paths["output"],
                           isaaclab=Path("/home/orionh/IsaacLab"),
                           coordination_sha256=guard.COORDINATION_SHA256)
    phase = "standing"
    locks, process, identity = [], None, None
    name = "hexapod-reference-physics-" + uuid.uuid4().hex
    report = dict(status="starting", phase=phase, container_name=name, started_unix=time.time(),
                  deadline_seconds=binding["max_seconds"], app_ready_deadline_seconds=90, no_policy_loaded=binding["mode"] in ("diagnostic", "replay", "tripod"),
                  native_mode=binding["mode"], source_freeze_sha256=binding["source_freeze_sha256"],
                  stage2_complete=False, physical_admission=False)
    report_path = args.output / "jobs" / (phase + ".json")
    try:
        if (args.output / "stop.request").exists():
            raise InterruptedError("Stop requested before acquiring a new job")
        if guard.sha(guard.COORDINATION) != args.coordination_sha256:
            raise InterruptedError("Coordination note changed; return ownership for review")
        report["coordination_sha256"] = args.coordination_sha256
        for path in ("/opt/wx/gpu.lock", "/tmp/hexapod-isaac-gpu.lock"):
            fd = os.open(path, os.O_RDONLY)
            locks.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report["preflight"] = preflight()
        guard.verify_tree(args.source, binding["source_freeze_sha256"])
        guard.save(report_path, report)
        deadline = time.monotonic() + binding["max_seconds"]
        app_ready_deadline = time.monotonic() + 90
        with (args.output / "logs" / (phase + ".log")).open("w") as log:
            process = subprocess.Popen(command(binding, paths, name), cwd=args.isaaclab,
                                       stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                owned = owned_container(name, identity)
                if owned:
                    identity = owned[0]
                    report.update(status="running", container_id=identity)
                    guard.save(report_path, report)
                if (args.output / "stop.request").exists():
                    raise InterruptedError("Stop requested")
                if guard.sha(guard.COORDINATION) != args.coordination_sha256:
                    raise InterruptedError("Coordination changed during standing; yielding owned job")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Native allocation exceeded its deadline")
                if time.monotonic() >= app_ready_deadline:
                    if 'REFERENCE_SCREEN_APP_READY' not in (args.output / 'logs' / (phase+'.log')).read_text(errors='replace'):
                        report['startup_failure_kind'] = 'no_reference_AppReady_by90s'
                        raise TimeoutError('No AppReady by90s; preserve startup log and45s traceback')
                guard.verify_policy_bytes()
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
        guard.save(args.output / "jobs" / (phase + "_contact_data_audit.json"), contact_audit)
        report["contact_data_completeness"] = contact_audit
        if not contact_audit["passed"]:
            raise RuntimeError(f"{phase} has incomplete contact/friction data; standing is not admitted")
        state_path = args.output / phase / "state.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        if process.returncode != 0 or state.get("status") != "completed" or (args.output / phase / "failure.json").exists():
            raise RuntimeError(f"{phase} standing did not complete; inspect its state and log")
        if state.get("runtime_binding", {}).get("runtime_tree_sha256") != binding["source_freeze_sha256"]:
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
            guard.save(report_path, report)
            for fd in reversed(locks):
                os.close(fd)

def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--bindings-sha256', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--cleanup-only', action='store_true')
    cli = parser.parse_args()
    guard.pinned_file(cli.bindings, cli.bindings_sha256)
    binding = guard.read(cli.bindings)
    paths = verify(binding, Path(__file__).resolve().parents[1])
    if cli.cleanup_only:
        with guard.both_locks():
            receipt = guard.cleanup_owned(paths['output'])
            receipt['reservation'] = guard.verify_reservation()
            receipt['resources'] = guard.no_live_compute()
            if paths['output'].exists():
                guard.save(paths['output']/'cleanup.json', receipt)
        print(json.dumps(receipt, indent=2))
        return
    guard.require(not paths['output'].exists(), 'Attempt output must be fresh')
    with guard.both_locks():
        snapshot = preflight()
    if cli.preflight_only:
        print(json.dumps({'preflight': snapshot, 'mode': binding['mode'], 'native_started': False}, indent=2))
        return
    paths['output'].mkdir(parents=True, exist_ok=False)
    for name in ('jobs', 'logs'):
        (paths['output']/name).mkdir()
    signal.signal(signal.SIGTERM, lambda *_: (paths['output']/'stop.request').touch())
    signal.signal(signal.SIGINT, lambda *_: (paths['output']/'stop.request').touch())
    guard.save(paths['output']/'launch_binding.json', binding)
    try:
        run_owned(binding, paths)
    finally:
        with guard.both_locks():
            cleanup = guard.cleanup_owned(paths['output'])
            cleanup['reservation'] = guard.verify_reservation()
            cleanup['resources'] = guard.no_live_compute()
            verify(binding, Path(__file__).resolve().parents[1])
            guard.save(paths['output']/'cleanup.json', cleanup)


if __name__ == '__main__':
    main()
