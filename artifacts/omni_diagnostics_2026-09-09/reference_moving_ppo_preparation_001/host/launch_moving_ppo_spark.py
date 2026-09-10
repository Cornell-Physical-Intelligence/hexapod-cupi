"""Fresh standing, calibrated ten-update moving PPO and matched immutable screens."""
from pathlib import Path
from types import SimpleNamespace
import argparse,fcntl,hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time,uuid
sys.dont_write_bytecode=True
SOURCE_MAP='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
SOURCE_HOST='9ebabaf254cc77fabad7d4759cc80fc26d7efd3912ea84bd55304594c88ed561'
RUNTIME_TREE='abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280'
COORDINATION=Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md')
PHASES=('standing','calibration','profile_32','train_10','evaluate_initial','evaluate_010','quiet_010')
MODE={'calibration':'calibrate'}
PHASE_SECONDS={p:(1800 if p=='train_10' else 600) for p in PHASES}
CONSUMER_FREEZE='dd49fd3e3639e453b067b9d4d67e5465db20865200c6a17616dd6936b527b842'
CONSUMER=None
PARENT=None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
digest=sha
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)
def verify_tree(root,manifest,bound):
    if sha(root/manifest)!=bound:raise ValueError('Wrong frozen input '+str(root))
    if any(p.is_symlink() for p in root.rglob('*')):raise ValueError('Symbolic input substitution')
    actual={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()};actual.pop(manifest)
    if actual!=read(root/manifest):raise ValueError('Changed or unlisted source '+str(root))
def verify_own_bundle(expected=None):
    p=Path(__file__).resolve().parent;bound=sha(p/'FREEZE_SHA256.json')
    if expected is not None and bound!=expected:raise ValueError('Host changed during campaign')
    verify_tree(p,'FREEZE_SHA256.json',bound);return bound

def consumer_args(args,phase,require_standing):
    return SimpleNamespace(source_root=args.source,run=args.run,device_run=args.device_run,bridge=args.bridge,
        observation_bundle=args.observation,package=args.output/'inputs/study' if require_standing else args.run/'inputs/study',
        standing=args.output/'standing/admission.json',output=args.output/phase,mode=MODE.get(phase,phase),decision_receipt=None)

def validate_inputs(args,phase='calibration',require_standing=False):
    global CONSUMER
    verify_own_bundle(getattr(args,'host_freeze_sha256',None))
    verify_tree(args.source,'campaign_source_hashes.json',SOURCE_MAP)
    verify_tree(args.consumer,'FREEZE_SHA256.json',CONSUMER_FREEZE)
    if sha(args.source/'tools/launch_reference_physics_spark.py')!=SOURCE_HOST:raise ValueError('Wrong source009 supervisor')
    if CONSUMER is None:
        spec=importlib.util.spec_from_file_location('_bound_moving_campaign_contract',args.consumer/'campaign_contract.py')
        CONSUMER=importlib.util.module_from_spec(spec);spec.loader.exec_module(CONSUMER)
    if Path(CONSUMER.__file__).resolve()!=args.consumer/'campaign_contract.py':raise ValueError('Wrong imported consumer')
    return CONSUMER.verify_inputs(consumer_args(args,phase,require_standing),require_standing=require_standing)

def command(args,name,phase):
    if phase not in PHASES:raise ValueError('Only declared bounded phases exist')
    if phase=='standing':return PARENT.command(args.source,args.output,name,phase)
    mounts=[(args.source,'/workspace/hexapod'),(args.run,'/qualified'),(args.device_run,'/device-proof'),
        (args.bridge,'/bridge'),(args.consumer,'/consumer'),(args.observation,'/observation'),(args.output/'inputs/study','/study')]
    cli=['docker','compose','--env-file','docker/.env.base','-f','docker/docker-compose.yaml','--profile','base',
        'run','--rm','--no-deps','--name',name,'-w','/outputs','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
        '-e','PYTHONPATH=/workspace/isaaclab/source/isaaclab:/workspace/hexapod/tools:/workspace/hexapod/isaaclab']
    for local,target in mounts:cli+=['-v',str(local)+':'+target+':ro']
    cli+=['-v',str(args.output)+':/outputs:rw','-v',str(args.output/'inputs')+':/outputs/inputs:ro']
    for prior in PHASES[:PHASES.index(phase)]:
        cli+=['-v',str(args.output/prior)+':/outputs/'+prior+':ro']
    cli+=['--entrypoint','/workspace/isaaclab/_isaac_sim/python.sh','isaac-lab-base','/consumer/run_moving_ppo.py',
        '--mode',MODE.get(phase,phase),'--source-root','/workspace/hexapod','--run','/qualified','--device-run','/device-proof',
        '--bridge','/bridge','--observation-bundle','/observation','--package','/study',
        '--standing','/outputs/standing/admission.json','--output','/outputs/'+phase]
    return cli+['--headless','--device','cuda:0','--info',
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
    name = "hexapod-reference-physics-" + uuid.uuid4().hex
    report = dict(status="starting", phase=phase, container_name=name, started_unix=time.time(),
                  deadline_seconds=PHASE_SECONDS[phase], app_ready_deadline_seconds=90, no_policy_loaded=phase == "standing")
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
        deadline = time.monotonic() + PHASE_SECONDS[phase]
        app_ready_deadline = time.monotonic() + 90
        with (args.output / "logs" / (phase + ".log")).open("w") as log:
            process = subprocess.Popen(command(args, name, phase), cwd=args.isaaclab,
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
                    raise TimeoutError(f"{phase} exceeded declared {PHASE_SECONDS[phase]} second bound")
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

def verify_completed(args, completed):
    for phase, hashes in completed.items():
        if tree_hashes(args.output/phase)!=hashes:raise ValueError('Immutable completed phase changed: '+phase)

def run_phases(args,baseline,report):
    expected=None;completed={}
    for phase in PHASES:
        if validate_inputs(args,require_standing=False)!=baseline:raise ValueError('Bound input identity changed')
        verify_completed(args,completed)
        if phase!='standing':
            identity=validate_inputs(args,phase,True)
            if expected is None:expected=identity
            if identity!=expected:raise ValueError('Standing or consumer identity changed between phases')
        PARENT.check_source(args.source)
        state=run_owned(args,phase)
        verify_completed(args,completed)
        if phase=='standing':
            CONSUMER.verify_standing(args.output/'standing/admission.json',args.output)
            accepted={'passed':True,'admission_sha256':sha(args.output/'standing/admission.json')}
        else:
            accepted=CONSUMER.validate_result(args.output/phase,identity)
            if (accepted.get('passed') is not True or accepted.get('mode')!=MODE.get(phase,phase)
                or accepted.get('phase')!=phase or accepted.get('source_identity')!=identity):
                raise ValueError('Consumer result validation or phase binding did not pass')
            if phase!='calibration':
                prior=('train_10','decision_010.pt') if phase in ('evaluate_010','quiet_010') else ('calibration','initial.pt')
                if state.get('input_checkpoint_sha256')!=sha(args.output/prior[0]/prior[1]):
                    raise ValueError('Phase loaded the wrong immutable checkpoint')
        completed[phase]=tree_hashes(args.output/phase)
        if phase!='standing' and accepted.get('files_sha256')!=completed[phase]:raise ValueError('Consumer raw evidence receipt differs')
        save(args.output/(phase+'_immutable.sha256.json'),completed[phase])
        save(args.output/(phase+'_accepted.json'),accepted)
        report.setdefault('accepted_phases',{})[phase]=accepted
        report.update(status='running',last_completed_phase=phase)
        if phase=='train_10':report['PPO_updates_completed']=10
        save(args.output/'campaign.json',report)
    verify_completed(args,completed)

def tree_hashes(p):return {str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file()}

def main():
    global PARENT,verified_source,resources,preflight,live_competitors,audit_contact_log
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','run','device-run','bridge','consumer','observation','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--isaaclab',type=Path,default=Path('/home/orionh/IsaacLab'));args=parser.parse_args()
    for key,v in vars(args).items():
        if isinstance(v,Path):setattr(args,key,v.resolve())
    inputs=[getattr(args,k) for k in ('source','run','device_run','bridge','consumer','observation')]+[Path(__file__).resolve().parent]
    if args.output.exists() or any(args.output==p or p in args.output.parents for p in inputs):raise ValueError('Fresh output outside immutable inputs required')
    args.host_freeze_sha256=verify_own_bundle();baseline=validate_inputs(args)
    sys.path.insert(0,str(args.source/'tools'))
    import launch_reference_physics_spark as parent
    if Path(parent.__file__).resolve()!=args.source/'tools/launch_reference_physics_spark.py':raise ValueError('Wrong parent supervisor')
    PARENT=parent;verified_source=parent.verified_source;resources=parent.resources;preflight=parent.preflight
    live_competitors=parent.live_competitors;audit_contact_log=parent.audit_contact_log
    runtime=PARENT.check_source(args.source);args.coordination_sha256=sha(COORDINATION)
    args.output.mkdir(parents=True)
    for name in ('jobs','logs','inputs'):(args.output/name).mkdir()
    shutil.copytree(args.run/'inputs/study',args.output/'inputs/study');assets=tree_hashes(args.output/'inputs/study')
    save(args.output/'inputs/study_before.sha256.json',assets)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:(args.output/'stop.request').touch())
    report=dict(status='preparing',started_unix=time.time(),runtime_binding=runtime,source_manifest_sha256=SOURCE_MAP,
        consumer_freeze_sha256=CONSUMER_FREEZE,host_freeze_sha256=args.host_freeze_sha256,baseline=baseline,
        PPO_updates_completed=0,Stage2_complete=False,scope='Ten moving PPO updates with 32 replicas and matched retention/quiet; no automatic25 or128 allocation')
    try:
        save(args.output/'campaign.json',report);run_phases(args,baseline,report)
        report.update(status='completed',ten_update_moving_pilot_and_matched_screens_passed=True)
    except Exception as error:report.update(status='failed',error=repr(error));raise
    finally:
        try:
            training_state=args.output/'train_10/state.json'
            if training_state.is_file():
                observed=read(training_state)
                report['training_reported_PPO_updates_completed']=observed.get('PPO_updates_completed')
                report['training_reported_policy_training_started']=observed.get('policy_training_started')
            completed={phase:read(args.output/(phase+'_immutable.sha256.json')) for phase in PHASES if (args.output/(phase+'_immutable.sha256.json')).is_file()}
            verify_completed(args,completed)
            if validate_inputs(args)!=baseline or tree_hashes(args.output/'inputs/study')!=assets:raise ValueError('Terminal source or input integrity failure')
            report['source_and_inputs_unchanged']=True
        except Exception as error:report.update(status='failed',source_and_inputs_unchanged=False,integrity_error=repr(error));raise
        finally:report['finished_unix']=time.time();save(args.output/'campaign.json',report)

if __name__=='__main__':main()
