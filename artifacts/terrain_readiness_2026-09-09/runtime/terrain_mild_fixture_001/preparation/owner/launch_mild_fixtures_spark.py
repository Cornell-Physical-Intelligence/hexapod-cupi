#!/usr/bin/env python3
"""One bounded48-fixture geometry/ray/contact smoke; no robot, actor or training."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time,uuid
from pathlib import Path
from gpu_ownership_helpers import preflight,resources,live_competitors
from terrain_contact_evidence import audit_contact_log
from mild_fixture_contract import CATALOG,PROTOCOL,catalog_identity,validate_result,digest
COORDINATION=Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md')


def save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def verified_source(source):
    source=Path(source).resolve();mapping=json.loads((source/'campaign_source_hashes.json').read_text())
    actual={str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
    if actual!=set(mapping)|{'campaign_source_hashes.json'}:raise ValueError('Source contains extra or missing files')
    for relative,expected in mapping.items():
        path=source/relative
        if path.is_symlink() or not path.resolve().is_relative_to(source) or digest(path)!=expected:raise ValueError('Frozen source changed: '+relative)
    return mapping


def check_source(source):
    mapping=verified_source(source);entries,fixtures=catalog_identity(source)
    return {'source_manifest_sha256':digest(Path(source)/'campaign_source_hashes.json'),'source_files':len(mapping),
            'catalog_geometry_files':len(fixtures),'fixture_count':len(entries),'protocol':PROTOCOL}


def command(source,output,name,phase):
    if phase!='fixtures':raise ValueError('Only the exact48-fixture phase is permitted')
    return ['docker','compose','--env-file','docker/.env.base','-f','docker/docker-compose.yaml','--profile','base','run',
        '--rm','--no-deps','--name',name,'-w','/outputs','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
        '-e','PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/terrain-source/tools',
        '-v',f'{source}:/workspace/terrain-source:ro','-v',f'{output}:/outputs:rw',
        '--entrypoint','/workspace/isaaclab/_isaac_sim/python.sh','isaac-lab-base',
        '/workspace/terrain-source/tools/validate_terrain_fixtures.py','--catalog','/workspace/terrain-source/'+CATALOG,
        '--output','/outputs/fixtures','--steps','500','--dt','0.005','--headless','--device','cuda:0','--info',
        '--kit_args=--/exts/omni.kit.telemetry/skipDeferredStartup=true --/app/extensions/excluded/4=omni.kit.telemetry']


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
    name = "hexapod-mild-fixtures-" + uuid.uuid4().hex
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
                    raise InterruptedError("Coordination changed during fixtures; yielding owned job")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Fixture phase exceeded ten-minute bound")
                if time.monotonic() >= app_ready_deadline:
                    if 'TERRAIN_PHASE loading_geometry_helpers' not in (args.output / 'logs' / (phase+'.log')).read_text(errors='replace'):
                        report['startup_failure_kind'] = 'no_terrain_AppReady_by90s'
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
            raise RuntimeError(f"{phase} has incomplete contact/friction data; fixtures are not admitted")
        state_path = args.output / phase / "validation.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        if process.returncode != 0 or state.get("status") != "completed" or (args.output / phase / "failure.json").exists():
            raise RuntimeError(f"{phase} fixtures did not complete; inspect its state and log")
        report["fixture_gate"] = validate_result(state, args.source)
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--isaaclab',type=Path,default=Path('/home/orionh/IsaacLab'))
    args=parser.parse_args();args.source=args.source.resolve();args.output=args.output.resolve()
    if args.output==args.source or args.output.is_relative_to(args.source):parser.error('Output must be outside immutable source')
    if args.output.exists():parser.error('Use a fresh output; no automatic resume or retry')
    identity=check_source(args.source);args.coordination_sha256=digest(COORDINATION)
    args.output.mkdir(parents=True)
    for directory in ('jobs','logs','inputs'):(args.output/directory).mkdir()
    signal.signal(signal.SIGTERM,lambda *_:(args.output/'stop.request').touch())
    signal.signal(signal.SIGINT,lambda *_:(args.output/'stop.request').touch())
    report=dict(status='preparing',started_unix=time.time(),source=str(args.source),identity=identity,
                source_manifest_sha256=identity['source_manifest_sha256'],protocol=PROTOCOL,
                robot_validation_performed=False,ready_for_terrain_training=False,policy_training_started=False,
                no_actor_or_robot_asset_loaded=True,automatic_continuation=False)
    before=None
    save(args.output/'campaign.json',report)
    try:
        _,before=catalog_identity(args.source);save(args.output/'inputs/fixture_before.sha256.json',before)
        state=run_owned(args,'fixtures')
        gate=validate_result(state,args.source);check_source(args.source)
        _,after=catalog_identity(args.source)
        if after!=before:raise RuntimeError('Readonly catalog/geometry changed')
        report.update(status='completed',gate=gate,all48_fixture_smokes_passed=True,
                      fixture_validation_sha256=digest(args.output/'fixtures/validation.json'))
    except Exception as exc:
        report.update(status='stopped' if isinstance(exc,InterruptedError) else 'failed',error=repr(exc))
        raise
    finally:
        try:
            check_source(args.source);report['terminal_source_integrity']={'passed':True}
        except Exception as exc:
            report['terminal_source_integrity']={'passed':False,'error':repr(exc)};report['status']='failed'
        try:
            _,after=catalog_identity(args.source)
            report['terminal_fixture_integrity']={'passed':before is not None and before==after,'files_checked':len(after)}
            if before!=after:report['status']='failed'
        except Exception as exc:
            report['terminal_fixture_integrity']={'passed':False,'error':repr(exc)};report['status']='failed'
        log=args.output/'logs/fixtures.log'
        if log.exists():
            try:
                report['terminal_contact_log_audit']=audit_contact_log(log)
                if report['terminal_contact_log_audit']['passed'] is not True:report['status']='failed'
            except Exception as exc:
                report['terminal_contact_log_audit']={'passed':False,'error':repr(exc)};report['status']='failed'
        report['finished_unix']=time.time();save(args.output/'campaign.json',report)
        if report['status']=='failed' and sys.exc_info()[0] is None:
            raise RuntimeError('Terminal source/fixture/contact provenance failed; fixture campaign is rejected')


if __name__=='__main__':main()
