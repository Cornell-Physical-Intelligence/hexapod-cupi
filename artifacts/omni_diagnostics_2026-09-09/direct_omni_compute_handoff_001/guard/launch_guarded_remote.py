"""External pause054 guard: bounded direct315 CAPS two-update smoke only."""
from pathlib import Path
from types import SimpleNamespace
import fcntl,hashlib,importlib.util,json,os,subprocess,sys,time
sys.dont_write_bytecode=True
BASE=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
SUPERVISOR_SOURCE=BASE/'reference_physics_source_009'
SOURCE=BASE/'direct_omni_train_source_001'
CONTRACT=BASE/'direct_omni_train_preparation_001'
CHECKPOINT=BASE/'omni_repair_003/branch_a/inputs/original.pt'
OUTPUT=BASE/'direct_omni_train_smoke_001'
PAUSE=BASE/'forecast_pause_054'
UNIT='hexapod-direct-omni-train-smoke-001-20260910.service'
HOST=BASE/'direct_omni_train_host_001/launch_train_spark.py'
SOURCE_SHA256='37b4b6d5d75f6d697a122fc2a1903f8f625f6a67d1727e1257e1533300e44da6'
SUPERVISOR_SHA256='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
CONTRACT_SHA256='9e591b93ae99ca3a76b7f7500ea20cc778da76c7f6b9b2c86fd22d31aceab3d7'
CHECKPOINT_SHA256='1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8'
HOST_SHA256='cbd7e76ab13717036c78ba86ccf9903f6c41a6f9f48954190aac9af3a8ab9d69'
HOST_FREEZE_SHA256='332fad29e0a0f88ba07012700f2a21c3a4f0453f3d71224da84fcc477e48394f'
PREVIOUS_UNIT='hexapod-direct-omni-cold-001-20260910.service'
PREVIOUS_INVOCATION='ae3b95c756db4aa4b639ee4230e8c958'
PRIOR_PINS={'direct_omni_cold_001/campaign.json': '2366c88ec70ec95dca54eeca97a5c5ce42535a4e232dd2b18bd1e5d85170a827', 'direct_omni_cold_001/jobs/standing.json': 'b2b45f21688adab6dfe056cdadaeed94216bc578896a4d1a9700b597bc41a158', 'direct_omni_cold_001/jobs/baseline.json': 'b41b7d8c0e50c75d10334ade2308fe6838ff3d40e37c54340ca1c87a25713a4d', 'forecast_pause_053/restored.json': '98a2b717cca638f6b2466d54edcbc6449a4e6635789d544d0949dc305527f603', 'forecast_pause_053/pause.json': '47f5ad662bad775e43ab95207de77720202d83197c87ef9624c55866331dc6b4'}

def call(args):return subprocess.check_output(args,text=True,timeout=30).strip()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require_final_bindings():
    for name,value in {'source':SOURCE_SHA256,'supervisor':SUPERVISOR_SHA256,'contract':CONTRACT_SHA256,
        'checkpoint':CHECKPOINT_SHA256,'host':HOST_SHA256,'host freeze':HOST_FREEZE_SHA256,**PRIOR_PINS}.items():
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise RuntimeError('Final reviewed binding pending: '+name)
    if len(PREVIOUS_INVOCATION)!=32 or any(c not in '0123456789abcdef' for c in PREVIOUS_INVOCATION):raise RuntimeError('Previous invocation not bound')
    if set(PRIOR_PINS)!={'direct_omni_cold_001/campaign.json','direct_omni_cold_001/jobs/standing.json','direct_omni_cold_001/jobs/baseline.json','forecast_pause_053/restored.json','forecast_pause_053/pause.json'}:raise RuntimeError('Exact completed cold owner/restoration receipts required')

def verify_frozen(root,expected):
    manifest=root/'FREEZE_SHA256.json'
    if sha(manifest)!=expected:raise RuntimeError('Wrong reviewed frozen bundle: '+str(root))
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):raise RuntimeError('Frozen input symbolic substitution')
    actual={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p!=manifest}
    if actual!=json.loads(manifest.read_text()):raise RuntimeError('Frozen source changed or has unlisted files: '+str(root))

def verify_previous_owner():
    previous=call(['systemctl','--user','show',PREVIOUS_UNIT,'-p','ActiveState','-p','SubState','-p','InvocationID'])
    fields=dict(line.split('=',1) for line in previous.splitlines() if '=' in line)
    if fields.get('ActiveState') not in ('inactive','failed'):raise RuntimeError('Previous owner still active')
    live=fields.get('InvocationID','')
    if live and live!=PREVIOUS_INVOCATION:raise RuntimeError('Previous owner invocation changed')
    for name,bound in PRIOR_PINS.items():
        if sha(BASE/name)!=bound:raise RuntimeError('Previous receipt changed: '+name)
    campaign=json.loads((BASE/'direct_omni_cold_001/campaign.json').read_text())
    if campaign.get('status')!='completed' or campaign.get('terminal_inputs_unchanged') is not True or campaign.get('baseline_complete') is not True:raise RuntimeError('Prior cold campaign not complete and immutable')
    previous_pause=json.loads((BASE/'forecast_pause_053/pause.json').read_text())
    restored=json.loads((BASE/'forecast_pause_053/restored.json').read_text())
    expected_timers={name for name,state in previous_pause['units'].items() if name.endswith('.timer') and 'ActiveState=active' in state}
    if previous_pause.get('unit')!=PREVIOUS_UNIT or set(restored.get('timers',[]))!=expected_timers or not restored.get('restored_unix'):raise RuntimeError('Prior exact timer restoration missing')
    for phase in ('standing','baseline'):
        job=json.loads((BASE/('direct_omni_cold_001/jobs/'+phase+'.json')).read_text())
        if job.get('status')!='completed' or job.get('phase')!=phase or job.get('cleanup_checked') is not True or not job.get('container_id') or not job.get('container_name'):raise RuntimeError('Prior cold job lacks exact completed cleanup proof')
        for identifier in (job['container_name'],job['container_id']):
            found=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',identifier],text=True,capture_output=True,timeout=20)
            if found.returncode==0 or not any(x in found.stderr.lower() for x in ('no such object','no such container')):raise RuntimeError('Previous container not proven absent: '+identifier)

def validate_train_inputs(train_host):
    if sha(SUPERVISOR_SOURCE/'campaign_source_hashes.json')!=SUPERVISOR_SHA256:raise RuntimeError('Changed supervisor source')
    if sha(SOURCE/'campaign_source_hashes.json')!=SOURCE_SHA256:raise RuntimeError('Changed native source')
    if CHECKPOINT.is_symlink() or sha(CHECKPOINT)!=CHECKPOINT_SHA256:raise RuntimeError('Wrong original checkpoint')
    if sha(HOST)!=HOST_SHA256:raise RuntimeError('Changed native host')
    verify_frozen(HOST.parent,HOST_FREEZE_SHA256);verify_frozen(CONTRACT,CONTRACT_SHA256)
    args=SimpleNamespace(source=SOURCE,checkpoint=CHECKPOINT,contract=CONTRACT,supervisor_source=SUPERVISOR_SOURCE,output=OUTPUT,isaaclab=Path('/home/orionh/IsaacLab'),host_freeze_sha256=HOST_FREEZE_SHA256,allocation='smoke',branch='caps',smoke=None)
    if train_host.selected_phases(args)!=('standing','train','final_constant','final_stop'):raise RuntimeError('Host changed bounded smoke phase order')
    identity=train_host.verify_inputs(args)
    if identity.get('source_manifest_sha256')!=SOURCE_SHA256 or identity.get('checkpoint_sha256')!=CHECKPOINT_SHA256:raise RuntimeError('Host admitted different native inputs')
    expected={'schema':'direct315_stand_stop_caps_native_v2','allocation':'smoke','branch':'caps','replicas':32,'controls_per_update':24,'updates':2,'caps':{'temporal_weight':.1,'spatial_weight':.1,'noise_scale':1.,'noise_seed':1157}}
    if identity.get('selection')!=expected or (identity.get('actor_width'),identity.get('critic_width'))!=(315,318):raise RuntimeError('Host selection differs from bounded two-update CAPS smoke')
    return identity

def main():
    if PAUSE.exists() or OUTPUT.exists():raise RuntimeError('Fresh pause/output names required')
    require_final_bindings()
    sys.path.insert(0,str(SUPERVISOR_SOURCE/'tools'))
    from launch_reference_physics_spark import check_source
    from launch_length_study_spark import preflight
    check_source(SUPERVISOR_SOURCE)
    if sha(HOST)!=HOST_SHA256:raise RuntimeError('Wrong reviewed host before import')
    spec=importlib.util.spec_from_file_location('train_host',HOST)
    train_host=importlib.util.module_from_spec(spec);spec.loader.exec_module(train_host)
    validate_train_inputs(train_host)
    verify_previous_owner()
    coordination = Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md').read_bytes()
    if hashlib.sha256(coordination).hexdigest() != '22c615d67c6c4dc14ab370b291f8eddd72f8475ad34118b91a4237a34ee05fab':
        raise RuntimeError('Coordination changed; review it before launch')
    locks, restorer, launched = [], None, False
    try:
        units = {name: call(['systemctl', '--user', 'show', name, '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID'])
            for name in ('stormscope-dispatch.timer', 'stormscope-scout.timer', 'stormscope-dispatch.service', 'stormscope-scout.service')}
        active_services = [name for name, value in units.items() if name.endswith('.service')
            and ('ActiveState=active' in value or 'ActiveState=activating' in value)]
        parents = {int(next(line.split('=', 1)[1] for line in units[name].splitlines() if line.startswith('MainPID=')))
                   for name in active_services}
        parents.discard(0)
        gpu = call(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'])
        for line in gpu.splitlines():
            pid = int(line.split(',', 1)[0]); seen = set()
            while pid > 1 and pid not in parents and pid not in seen:
                seen.add(pid)
                try:
                    status = Path(f'/proc/{pid}/status').read_text()
                except FileNotFoundError:
                    pid = 0
                    break
                pid = int(next(v.split()[1] for v in status.splitlines() if v.startswith('PPid:')))
            if pid != 0 and pid not in parents:
                raise RuntimeError('Unrelated CUDA process; no forecasting pause or candidate launch: ' + line)
        snapshot = {'compute_processes_before_authorized_pause': gpu,
            'containers': call(['docker', 'ps', '--format', '{{.ID}} {{.Names}} {{.Image}}']),
            'forecast_services_authorized_for_graceful_stop': active_services,
            'service_stop_semantics': {name: call(['systemctl','--user','show',name,'-p','KillMode','-p','KillSignal','-p','TimeoutStopUSec']) for name in active_services}}
        PAUSE.mkdir()
        record = dict(user_authorized_pause=True, created_unix=time.time(), units=units, unit=UNIT,
            output=str(OUTPUT), source=str(SOURCE), preflight=snapshot,
            coordination_sha256=hashlib.sha256(coordination).hexdigest(),
            reason='Bounded direct315 CAPS smoke: fresh standing, two updates with32 replicas, final constant and stop diagnostics; no pilot or automatic continuation',
            current_instruction='9 September C-study authorization supersedes older physical-model pause notes')
        (PAUSE / 'pause.json').write_text(json.dumps(record, indent=2) + '\n')
        restorer = PAUSE / 'resume_forecasting.py'
        restorer.write_text('''from pathlib import Path
import fcntl,json,subprocess,sys,time
p=Path(__file__).parent
r=json.loads((p/'pause.json').read_text())
if '--stop-owner' in sys.argv:
    subprocess.run(['systemctl','--user','stop',r['unit']],check=True,timeout=240)
with (p/'restore.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    if (p/'restored.json').exists():raise SystemExit(0)
    cleanup=[]
    for job in sorted((Path(r['output'])/'jobs').glob('*.json')):
        j=json.loads(job.read_text()); name=j.get('container_name'); identity=j.get('container_id')
        if not name:continue
        q=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',identity or name],text=True,capture_output=True,timeout=20)
        if q.returncode:
            if 'no such object' in q.stderr.lower() or 'no such container' in q.stderr.lower():continue
            raise RuntimeError('Owned container absence is unknown; keep forecasting paused: '+q.stderr.strip())
        fields=q.stdout.strip().split()
        if len(fields)!=3 or fields[1]!='/'+name or (identity and fields[0]!=identity):raise RuntimeError('Owned cleanup identity mismatch')
        if fields[2]=='true':subprocess.run(['docker','stop','--time','20',fields[0]],check=True,timeout=30,capture_output=True)
        after=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',fields[0]],text=True,capture_output=True,timeout=20)
        if after.returncode:
            if 'no such object' not in after.stderr.lower() and 'no such container' not in after.stderr.lower():raise RuntimeError('Post-stop container absence unknown')
        elif after.stdout.strip()!=fields[0]+' /'+name+' false':raise RuntimeError('Owned container not verified stopped')
        cleanup.append(fields[0])
    active=[name for name,state in r['units'].items() if name.endswith('.timer') and 'ActiveState=active' in state]
    if active:subprocess.run(['systemctl','--user','start',*active],check=True,timeout=30)
    result={'restored_unix':time.time(),'timers':active,'owned_cleanup_checked':cleanup}
    t=p/'restored.tmp';t.write_text(json.dumps(result,indent=2)+'\\n');t.replace(p/'restored.json')
''')
        # Arm recovery before mutating timer state. It stops only this exact
        # owner unit if needed, then restores only previously active timers.
        subprocess.run(['systemd-run', '--user', '--unit=hexapod-forecast-restore-054', '--on-active=45m',
            '/usr/bin/python3', str(restorer), '--stop-owner'], check=True)
        active = [name for name, state in units.items() if name.endswith('.timer') and 'ActiveState=active' in state]
        if active:
            subprocess.run(['systemctl', '--user', 'stop', *active], check=True, timeout=30)
        record['gracefully_stopped_services'] = active_services
        (PAUSE / 'pause.json').write_text(json.dumps(record, indent=2) + '\n')
        if active_services:
            subprocess.run(['systemctl', '--user', 'stop', *active_services], check=True, timeout=120)
        if any('ActiveState=active' in call(['systemctl', '--user', 'show', name, '-p', 'ActiveState'])
               or 'ActiveState=activating' in call(['systemctl', '--user', 'show', name, '-p', 'ActiveState'])
               for name in units if name.endswith('.service')):
            raise RuntimeError('Forecast service raced the timer pause; preserve it and restore timers')
        for path in ('/opt/wx/gpu.lock', '/tmp/hexapod-isaac-gpu.lock'):
            fd = os.open(path, os.O_RDONLY)
            locks.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record['post_pause_preflight'] = preflight()
        record['paused_unix'] = time.time()
        (PAUSE / 'pause.json').write_text(json.dumps(record, indent=2) + '\n')
    except Exception:
        if restorer is not None:
            subprocess.run(['/usr/bin/python3', str(restorer)], check=True, timeout=120)
        raise
    finally:
        for fd in reversed(locks):
            os.close(fd)
    # Known not yet launched: if immutable inputs changed during authorized
    # pause, restore directly without stopping a unit that has not been started.
    try:
        validate_train_inputs(train_host)
    except Exception:
        subprocess.run(['/usr/bin/python3',str(restorer)],check=True,timeout=120)
        raise
    try:
        cmd=['systemd-run','--user','--unit='+UNIT,'--property=RuntimeMaxSec=2520',
            '--property=TimeoutStopSec=180','--property=KillMode=process','--property=Environment=PYTHONDONTWRITEBYTECODE=1',
            '--property=ExecStopPost=/usr/bin/python3 '+str(restorer),'--property=Environment=PYTHONUNBUFFERED=1',
            '/usr/bin/python3',str(HOST),'--source',str(SOURCE),'--checkpoint',str(CHECKPOINT),
            '--contract',str(CONTRACT),'--supervisor-source',str(SUPERVISOR_SOURCE),'--output',str(OUTPUT),
            '--allocation','smoke','--branch','caps']
        subprocess.run(cmd,check=True,timeout=30)
        launched=True
        (PAUSE/'launch.json').write_text(json.dumps(dict(command=cmd,launched_unix=time.time()),indent=2)+'\n')
        print(json.dumps(dict(unit=UNIT,source=str(SOURCE),output=str(OUTPUT),pause=str(PAUSE))))
    finally:
        if not launched:
            subprocess.run(['/usr/bin/python3',str(restorer),'--stop-owner'],check=True,timeout=300)

if __name__=='__main__':main()
