"""Pause058: one explicitly selected completed-pilot progress recording."""
from pathlib import Path
from types import SimpleNamespace
import argparse,fcntl,hashlib,importlib.util,json,os,re,subprocess,sys,time
sys.dont_write_bytecode=True
BASE=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
SUPERVISOR_SOURCE=BASE/'reference_physics_source_009'
SOURCE=BASE/'direct_omni_train_source_002'
CONTRACT=BASE/'direct_omni_train_preparation_002'
ADAPTER=BASE/'direct_omni_preview_adapter_002'
OUTPUT=BASE/'direct_omni_preview_001'
PAUSE=BASE/'forecast_pause_058'
UNIT='hexapod-direct-omni-preview-001-20260910.service'
HOST=BASE/'direct_omni_preview_host_001/launch_preview_spark.py'
SOURCE_SHA256='64144b2c2466f17c809554ef53f82b736b4199136cc821451d7ac2788752bd2e'
SUPERVISOR_SHA256='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
CONTRACT_SHA256='20fbcd40a869b39d8e3143a6c9715ec62ada5cc6211eaf473f60f8b7167678cb'
ADAPTER_SHA256='2f2b807a8c4e265dc25270bceb21734e6f97c72a873f407eccc654b9480af0a4'
HOST_SHA256='6cc347096166902be89a9c7bde3ef36ca35ac97ea50e7ebf07fc7c260dccd1af'
HOST_FREEZE_SHA256='640ea3891d8e0d6e83f173f5bc5a2206ccdd2d0ef74c9a324d62ecda4fd9594e'
TRAIN_HOST_SHA256='19545653a00e635cc14ed9415f9799978da34674e84f6cad76eb42bd2a961ac4'
PREVIOUS_UNIT='hexapod-direct-omni-train-pilot-caps-001-20260910.service'
PREVIOUS_ROOT=BASE/'direct_omni_train_pilot_caps_001'
PHASES=('standing','initial_constant','initial_stop','train','final_constant','final_stop')
SELECTION=None;SELECTION_PATH=None;SELECTION_SHA256=None
PILOT=None;CHECKPOINT=None;CHECKPOINT_SHA256=None;CAMPAIGN_SHA256=None

def call(args):return subprocess.check_output(args,text=True,timeout=30).strip()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def valid_hash(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None

def bind_selection(argv=None):
    global SELECTION,SELECTION_PATH,SELECTION_SHA256,PILOT,CHECKPOINT,CHECKPOINT_SHA256,CAMPAIGN_SHA256
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--selection',type=Path,required=True);parser.add_argument('--selection-sha256',required=True)
    args=parser.parse_args(argv)
    if not valid_hash(args.selection_sha256) or args.selection.is_symlink() or sha(args.selection)!=args.selection_sha256:raise RuntimeError('Unbound/changed root selection receipt')
    data=json.loads(args.selection.read_text())
    if data.get('schema')!='direct_preview_selection_v1' or data.get('pilot_name') not in ('direct_omni_train_pilot_curriculum_001','direct_omni_train_pilot_caps_001'):raise RuntimeError('Only completed curriculum/CAPS pilot selection is allowed')
    if not valid_hash(data.get('checkpoint_sha256')) or not valid_hash(data.get('pilot_campaign_sha256')):raise RuntimeError('Selected final checkpoint/campaign binding pending')
    previous=data.get('previous_owner',{});attempted=previous.get('attempted_phases',[])
    if not re.fullmatch('[0-9a-f]{32}',str(previous.get('invocation',''))) or not attempted or attempted!=list(PHASES[:len(attempted)]):raise RuntimeError('Previous CAPS owner phase/invocation binding pending')
    expected={'direct_omni_train_pilot_caps_001/campaign.json','forecast_pause_057/pause.json','forecast_pause_057/restored.json'}|{'direct_omni_train_pilot_caps_001/jobs/'+phase+'.json' for phase in attempted}
    if set(previous.get('pins',{}))!=expected or not all(valid_hash(x) for x in previous['pins'].values()):raise RuntimeError('Exact previous CAPS owner receipt map required')
    SELECTION=data;SELECTION_PATH=args.selection.resolve();SELECTION_SHA256=args.selection_sha256
    PILOT=BASE/data['pilot_name'];CHECKPOINT=PILOT/'train/policy/final.pt';CHECKPOINT_SHA256=data['checkpoint_sha256'];CAMPAIGN_SHA256=data['pilot_campaign_sha256']

def require_final_bindings():
    for name,value in {'source':SOURCE_SHA256,'supervisor':SUPERVISOR_SHA256,'contract':CONTRACT_SHA256,'adapter':ADAPTER_SHA256,'host':HOST_SHA256,'host freeze':HOST_FREEZE_SHA256,'selected checkpoint':CHECKPOINT_SHA256,'selected campaign':CAMPAIGN_SHA256,'selection':SELECTION_SHA256}.items():
        if not valid_hash(value):raise RuntimeError('Final reviewed binding pending: '+name)
    if SELECTION is None or sha(SELECTION_PATH)!=SELECTION_SHA256:raise RuntimeError('Selection receipt changed')

def verify_frozen(root,expected):
    manifest=root/'FREEZE_SHA256.json'
    if sha(manifest)!=expected:raise RuntimeError('Wrong reviewed frozen bundle: '+str(root))
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):raise RuntimeError('Frozen input symbolic substitution')
    actual={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p!=manifest}
    if actual!=json.loads(manifest.read_text()):raise RuntimeError('Frozen source changed or has unlisted files: '+str(root))

def verify_previous_owner():
    previous=SELECTION['previous_owner']
    fields=dict(line.split('=',1) for line in call(['systemctl','--user','show',PREVIOUS_UNIT,'-p','ActiveState','-p','SubState','-p','InvocationID']).splitlines() if '=' in line)
    if fields.get('ActiveState') not in ('inactive','failed'):raise RuntimeError('Previous CAPS owner still active')
    if fields.get('InvocationID') and fields['InvocationID']!=previous['invocation']:raise RuntimeError('Previous CAPS invocation changed')
    for path,bound in previous['pins'].items():
        if sha(BASE/path)!=bound:raise RuntimeError('Previous receipt changed: '+path)
    campaign=json.loads((PREVIOUS_ROOT/'campaign.json').read_text())
    if campaign.get('status') not in ('completed','failed') or campaign.get('terminal_inputs_unchanged') is not True:raise RuntimeError('Previous CAPS campaign not terminal and immutable')
    if campaign.get('allocation')!='pilot' or campaign.get('branch')!='caps' or campaign.get('planned_phases')!=list(PHASES) or campaign.get('host_freeze_sha256')!=TRAIN_HOST_SHA256 or campaign.get('identity',{}).get('source_manifest_sha256')!=SOURCE_SHA256:raise RuntimeError('Previous CAPS lineage differs')
    pause=json.loads((BASE/'forecast_pause_057/pause.json').read_text());restored=json.loads((BASE/'forecast_pause_057/restored.json').read_text())
    timers={name for name,state in pause['units'].items() if name.endswith('.timer') and 'ActiveState=active' in state}
    if pause.get('unit')!=PREVIOUS_UNIT or pause.get('output')!=str(PREVIOUS_ROOT) or set(restored.get('timers',[]))!=timers or not restored.get('restored_unix'):raise RuntimeError('Previous exact timer restoration missing')
    jobs={p.stem for p in (PREVIOUS_ROOT/'jobs').glob('*.json') if not p.name.endswith('_contact_data_audit.json')}
    if jobs!=set(previous['attempted_phases']):raise RuntimeError('Previous job inventory changed')
    for phase in previous['attempted_phases']:
        job=json.loads((PREVIOUS_ROOT/('jobs/'+phase+'.json')).read_text())
        if job.get('phase')!=phase or job.get('status') not in ('completed','failed') or job.get('cleanup_checked') is not True or not job.get('container_name') or not job.get('container_id'):raise RuntimeError('Previous exact job cleanup incomplete')
        for identifier in (job['container_name'],job['container_id']):
            result=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',identifier],text=True,capture_output=True,timeout=20)
            if result.returncode==0 or not any(x in result.stderr.lower() for x in ('no such object','no such container')):raise RuntimeError('Previous container absence not proven: '+identifier)

def validate_preview_inputs(host):
    require_final_bindings()
    if sha(SUPERVISOR_SOURCE/'campaign_source_hashes.json')!=SUPERVISOR_SHA256 or sha(SOURCE/'campaign_source_hashes.json')!=SOURCE_SHA256:raise RuntimeError('Source changed')
    if sha(HOST)!=HOST_SHA256:raise RuntimeError('Preview host changed')
    verify_frozen(HOST.parent,HOST_FREEZE_SHA256);verify_frozen(CONTRACT,CONTRACT_SHA256);verify_frozen(ADAPTER,ADAPTER_SHA256)
    args=SimpleNamespace(source=SOURCE,contract=CONTRACT,supervisor_source=SUPERVISOR_SOURCE,adapter=ADAPTER,
        pilot=PILOT,checkpoint=CHECKPOINT,checkpoint_sha256=CHECKPOINT_SHA256,campaign_sha256=CAMPAIGN_SHA256,
        output=OUTPUT,isaaclab=Path('/home/orionh/IsaacLab'),host_freeze_sha256=HOST_FREEZE_SHA256)
    host.require_fresh_output(args)
    identity=host.verify_inputs(args)
    if identity.get('source_manifest_sha256')!=SOURCE_SHA256 or identity.get('checkpoint_sha256')!=CHECKPOINT_SHA256 or identity.get('pilot',{}).get('campaign_sha256')!=CAMPAIGN_SHA256 or identity.get('pilot',{}).get('allocation')!='pilot' or identity.get('pilot',{}).get('completed_updates')!=50:raise RuntimeError('Host admitted another selected pilot/checkpoint')
    return identity

def main(argv=None):
    bind_selection(argv)
    if PAUSE.exists() or OUTPUT.exists():raise RuntimeError('Fresh pause/output names required')
    require_final_bindings()
    sys.path.insert(0,str(SUPERVISOR_SOURCE/'tools'))
    from launch_reference_physics_spark import check_source
    from launch_length_study_spark import preflight
    check_source(SUPERVISOR_SOURCE)
    if sha(HOST)!=HOST_SHA256:raise RuntimeError('Wrong reviewed host before import')
    spec=importlib.util.spec_from_file_location('train_host',HOST)
    train_host=importlib.util.module_from_spec(spec);spec.loader.exec_module(train_host)
    validate_preview_inputs(train_host)
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
            selection_sha256=SELECTION_SHA256, selection=SELECTION,
            output=str(OUTPUT), source=str(SOURCE), preflight=snapshot,
            coordination_sha256=hashlib.sha256(coordination).hexdigest(),
            reason='One38s unqualified native003 direct-PPO progress recording from the explicitly selected completed pilot; no training or automatic continuation',
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
        subprocess.run(['systemd-run', '--user', '--unit=hexapod-forecast-restore-058', '--on-active=15m',
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
        validate_preview_inputs(train_host)
    except Exception:
        subprocess.run(['/usr/bin/python3',str(restorer)],check=True,timeout=120)
        raise
    try:
        cmd=['systemd-run','--user','--unit='+UNIT,'--property=RuntimeMaxSec=720',
            '--property=TimeoutStopSec=180','--property=KillMode=process','--property=Environment=PYTHONDONTWRITEBYTECODE=1',
            '--property=ExecStopPost=/usr/bin/python3 '+str(restorer),'--property=Environment=PYTHONUNBUFFERED=1',
            '/usr/bin/python3',str(HOST),'--source',str(SOURCE),'--checkpoint',str(CHECKPOINT),
            '--checkpoint-sha256',CHECKPOINT_SHA256,'--campaign-sha256',CAMPAIGN_SHA256,
            '--contract',str(CONTRACT),'--supervisor-source',str(SUPERVISOR_SOURCE),'--output',str(OUTPUT),
            '--adapter',str(ADAPTER),'--pilot',str(PILOT)]
        subprocess.run(cmd,check=True,timeout=30)
        launched=True
        (PAUSE/'launch.json').write_text(json.dumps(dict(command=cmd,launched_unix=time.time(),selection_sha256=SELECTION_SHA256),indent=2)+'\n')
        print(json.dumps(dict(unit=UNIT,source=str(SOURCE),output=str(OUTPUT),pause=str(PAUSE))))
    finally:
        if not launched:
            subprocess.run(['/usr/bin/python3',str(restorer),'--stop-owner'],check=True,timeout=300)

if __name__=='__main__':main()
