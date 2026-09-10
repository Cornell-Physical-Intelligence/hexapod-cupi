"""External pause057 guard: bounded matched CAPS pilot after curriculum cleanup."""
from pathlib import Path
from types import SimpleNamespace
import fcntl,hashlib,importlib.util,json,os,subprocess,sys,time
sys.dont_write_bytecode=True
BASE=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
SUPERVISOR_SOURCE=BASE/'reference_physics_source_009'
SOURCE=BASE/'direct_omni_train_source_002'
CONTRACT=BASE/'direct_omni_train_preparation_002'
CHECKPOINT=BASE/'omni_repair_003/branch_a/inputs/original.pt'
SMOKE=BASE/'direct_omni_train_smoke_002'
OUTPUT=BASE/'direct_omni_train_pilot_caps_001'
PAUSE=BASE/'forecast_pause_057'
UNIT='hexapod-direct-omni-train-pilot-caps-001-20260910.service'
HOST=BASE/'direct_omni_train_host_002/launch_train_spark.py'
SOURCE_SHA256='64144b2c2466f17c809554ef53f82b736b4199136cc821451d7ac2788752bd2e'
SUPERVISOR_SHA256='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
CONTRACT_SHA256='20fbcd40a869b39d8e3143a6c9715ec62ada5cc6211eaf473f60f8b7167678cb'
CHECKPOINT_SHA256='1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8'
HOST_SHA256='52c4b0bd944fe9e2f1fc0313199a2d214c53dbffa4d01700d7299c3c94275fa1'
HOST_FREEZE_SHA256='19545653a00e635cc14ed9415f9799978da34674e84f6cad76eb42bd2a961ac4'
SMOKE_UNIT='hexapod-direct-omni-train-smoke-002-20260910.service'
SMOKE_INVOCATION='6cd738f30a0b4f499026de5cbf8a5bb6'
SMOKE_PINS={'direct_omni_train_smoke_002/campaign.json': 'aa478e602caceb75dd488db4fdd5e275aed6ed966a5390ad9aa20c6d66738d6f', 'direct_omni_train_smoke_002/jobs/standing.json': '43afe6d3190e463ff88f4d13fdab9639f6d6b209caca9f2113f43e47a53a98a2', 'direct_omni_train_smoke_002/jobs/train.json': 'a527c95d2e27378f0a50a80f737e6d0152d6374ce2068670e731a28fb6273c6c', 'direct_omni_train_smoke_002/jobs/final_constant.json': '5afed704d8982a2641cc19ef80d5deab1453fd08edfff7d1f2da375b6a90afe7', 'direct_omni_train_smoke_002/jobs/final_stop.json': 'c448a874c47940c5f4c96b7b14cc49dafabf6819dd454368280c2c5c66bf89b5', 'forecast_pause_055/pause.json': '92197a1eb5692474f5cb197241076808a2653a1c4007a8833dde71b86bca27b1', 'forecast_pause_055/restored.json': '6060f4a2fe7760fe2421022a2c7d668d7fe545672facb9897610f374f2f11956'}


PREVIOUS_ROOT=BASE/'direct_omni_train_pilot_curriculum_001'
PREVIOUS_UNIT='hexapod-direct-omni-train-pilot-curriculum-001-20260910.service'
PREVIOUS_INVOCATION='d9ffdd9882ec48dc8417d36a4245b2c4'
PREVIOUS_PHASES=('standing','initial_constant','initial_stop','train','final_constant','final_stop')
PREVIOUS_ATTEMPTED_PHASES=('standing', 'initial_constant', 'initial_stop', 'train', 'final_constant', 'final_stop')
PRIOR_PINS={'direct_omni_train_pilot_curriculum_001/campaign.json': '736cd00183181823d6b560af5c9a8ba99f48575e0d0d7801405af4708cc51a65', 'direct_omni_train_pilot_curriculum_001/jobs/standing.json': '3772c6a065ce806e14a9734f0f31908d1ec15a4507bff0b639c99292e257054e', 'direct_omni_train_pilot_curriculum_001/jobs/initial_constant.json': '8c543b7445e4fef6a0bb09dd34686b2b0275e292c42820d6dda820904eb6814d', 'direct_omni_train_pilot_curriculum_001/jobs/initial_stop.json': 'c222168a47dd90263abec88fbbc54a0e479372e4e2aeba9f8beeef683bfe4a3c', 'direct_omni_train_pilot_curriculum_001/jobs/train.json': '5ee5fa6e5fc4d92de784544f711865e8be6e6f06b6fa47162fed900ba079539a', 'direct_omni_train_pilot_curriculum_001/jobs/final_constant.json': 'c0910fc42f3797ef692687409598f9ace7f913ee9bc049111ab4193f77356b9d', 'direct_omni_train_pilot_curriculum_001/jobs/final_stop.json': 'e05ac51e61a9ade44e1311fa3cc9e79cd594602b2eae3fab4e1faff162e50664', 'forecast_pause_056/pause.json': '2e8573ff1ec515d5687dcd6656b359717aa689b9f1f66284d177b77e3913c8c8', 'forecast_pause_056/restored.json': '70ddd88a7356a787dae1075b73ef5759c9ef0371113708d57ccf750e10740e4c'}

def call(args):return subprocess.check_output(args,text=True,timeout=30).strip()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require_final_bindings():
    for name,value in {'source':SOURCE_SHA256,'supervisor':SUPERVISOR_SHA256,'contract':CONTRACT_SHA256,
        'checkpoint':CHECKPOINT_SHA256,'host':HOST_SHA256,'host freeze':HOST_FREEZE_SHA256,**SMOKE_PINS,**PRIOR_PINS}.items():
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise RuntimeError('Final reviewed binding pending: '+name)
    if len(SMOKE_INVOCATION)!=32 or any(c not in '0123456789abcdef' for c in SMOKE_INVOCATION):raise RuntimeError('Previous invocation not bound')
    if set(SMOKE_PINS)!={'direct_omni_train_smoke_002/campaign.json',
        'direct_omni_train_smoke_002/jobs/standing.json','direct_omni_train_smoke_002/jobs/train.json',
        'direct_omni_train_smoke_002/jobs/final_constant.json','direct_omni_train_smoke_002/jobs/final_stop.json',
        'forecast_pause_055/restored.json','forecast_pause_055/pause.json'}:raise RuntimeError('Exact completed smoke owner/restoration receipts required')

    if len(PREVIOUS_INVOCATION)!=32 or any(c not in '0123456789abcdef' for c in PREVIOUS_INVOCATION):raise RuntimeError('Curriculum invocation not bound')
    if not PREVIOUS_ATTEMPTED_PHASES or tuple(PREVIOUS_ATTEMPTED_PHASES)!=PREVIOUS_PHASES[:len(PREVIOUS_ATTEMPTED_PHASES)]:raise RuntimeError('Curriculum attempted phase prefix not bound')
    expected={'direct_omni_train_pilot_curriculum_001/campaign.json','forecast_pause_056/pause.json','forecast_pause_056/restored.json'}|{'direct_omni_train_pilot_curriculum_001/jobs/'+p+'.json' for p in PREVIOUS_ATTEMPTED_PHASES}
    if set(PRIOR_PINS)!=expected:raise RuntimeError('Exact terminal curriculum owner/restoration inventory required')

def verify_frozen(root,expected):
    manifest=root/'FREEZE_SHA256.json'
    if sha(manifest)!=expected:raise RuntimeError('Wrong reviewed frozen bundle: '+str(root))
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):raise RuntimeError('Frozen input symbolic substitution')
    actual={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p!=manifest}
    if actual!=json.loads(manifest.read_text()):raise RuntimeError('Frozen source changed or has unlisted files: '+str(root))

def verify_smoke_owner():
    previous=call(['systemctl','--user','show',SMOKE_UNIT,'-p','ActiveState','-p','SubState','-p','InvocationID'])
    fields=dict(line.split('=',1) for line in previous.splitlines() if '=' in line)
    if fields.get('ActiveState') not in ('inactive','failed'):raise RuntimeError('Previous owner still active')
    live=fields.get('InvocationID','')
    if live and live!=SMOKE_INVOCATION:raise RuntimeError('Previous owner invocation changed')
    for name,bound in SMOKE_PINS.items():
        if sha(BASE/name)!=bound:raise RuntimeError('Previous receipt changed: '+name)
    campaign=json.loads((SMOKE/'campaign.json').read_text())
    if campaign.get('status')!='completed' or campaign.get('terminal_inputs_unchanged') is not True or campaign.get('bounded_campaign_complete') is not True:raise RuntimeError('Prior smoke campaign not complete and immutable')
    phases=('standing','train','final_constant','final_stop')
    if (campaign.get('allocation'),campaign.get('branch'),campaign.get('PPO_updates_completed'))!=('smoke','caps',2) or campaign.get('planned_phases')!=list(phases) or set(campaign.get('accepted_phases',{}))!=set(phases) or campaign.get('host_freeze_sha256')!=HOST_FREEZE_SHA256 or campaign.get('identity',{}).get('source_manifest_sha256')!=SOURCE_SHA256:raise RuntimeError('Prior smoke allocation/source/phase inventory differs')
    previous_pause=json.loads((BASE/'forecast_pause_055/pause.json').read_text())
    restored=json.loads((BASE/'forecast_pause_055/restored.json').read_text())
    expected_timers={name for name,state in previous_pause['units'].items() if name.endswith('.timer') and 'ActiveState=active' in state}
    if previous_pause.get('unit')!=SMOKE_UNIT or set(restored.get('timers',[]))!=expected_timers or not restored.get('restored_unix'):raise RuntimeError('Prior exact timer restoration missing')
    for phase in phases:
        job=json.loads((SMOKE/('jobs/'+phase+'.json')).read_text())
        if job.get('status')!='completed' or job.get('phase')!=phase or job.get('cleanup_checked') is not True or not job.get('container_id') or not job.get('container_name'):raise RuntimeError('Prior smoke job lacks exact completed cleanup proof')
        for identifier in (job['container_name'],job['container_id']):
            found=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',identifier],text=True,capture_output=True,timeout=20)
            if found.returncode==0 or not any(x in found.stderr.lower() for x in ('no such object','no such container')):raise RuntimeError('Previous container not proven absent: '+identifier)


def verify_previous_owner():
    # Successful smoke is admission. Curriculum is only a terminal ownership
    # dependency; no quality score or completed-training requirement is added.
    verify_smoke_owner()
    previous=call(['systemctl','--user','show',PREVIOUS_UNIT,'-p','ActiveState','-p','SubState','-p','InvocationID'])
    fields=dict(line.split('=',1) for line in previous.splitlines() if '=' in line)
    if fields.get('ActiveState') not in ('inactive','failed'):raise RuntimeError('Curriculum owner still active')
    live=fields.get('InvocationID','')
    if live and live!=PREVIOUS_INVOCATION:raise RuntimeError('Curriculum invocation changed')
    for name,bound in PRIOR_PINS.items():
        if sha(BASE/name)!=bound:raise RuntimeError('Curriculum receipt changed: '+name)
    campaign=json.loads((PREVIOUS_ROOT/'campaign.json').read_text())
    if campaign.get('status') not in ('completed','failed') or campaign.get('terminal_inputs_unchanged') is not True:raise RuntimeError('Curriculum not terminal with unchanged inputs')
    if campaign.get('allocation')!='pilot' or campaign.get('branch')!='curriculum' or campaign.get('planned_phases')!=list(PREVIOUS_PHASES) or campaign.get('host_freeze_sha256')!=HOST_FREEZE_SHA256 or campaign.get('identity',{}).get('source_manifest_sha256')!=SOURCE_SHA256:raise RuntimeError('Curriculum lineage or allocation differs')
    previous_pause=json.loads((BASE/'forecast_pause_056/pause.json').read_text());restored=json.loads((BASE/'forecast_pause_056/restored.json').read_text())
    expected_timers={name for name,state in previous_pause['units'].items() if name.endswith('.timer') and 'ActiveState=active' in state}
    if previous_pause.get('unit')!=PREVIOUS_UNIT or previous_pause.get('output')!=str(PREVIOUS_ROOT) or set(restored.get('timers',[]))!=expected_timers or not restored.get('restored_unix'):raise RuntimeError('Curriculum exact timer restoration missing')
    actual_jobs={p.stem for p in (PREVIOUS_ROOT/'jobs').glob('*.json') if not p.name.endswith('_contact_data_audit.json')}
    if actual_jobs!=set(PREVIOUS_ATTEMPTED_PHASES):raise RuntimeError('Curriculum attempted job inventory changed')
    for phase in PREVIOUS_ATTEMPTED_PHASES:
        job=json.loads((PREVIOUS_ROOT/('jobs/'+phase+'.json')).read_text())
        if job.get('status') not in ('completed','failed') or job.get('phase')!=phase or job.get('cleanup_checked') is not True or not job.get('container_id') or not job.get('container_name'):raise RuntimeError('Curriculum job lacks exact terminal cleanup proof')
        for identifier in (job['container_name'],job['container_id']):
            found=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',identifier],text=True,capture_output=True,timeout=20)
            if found.returncode==0 or not any(x in found.stderr.lower() for x in ('no such object','no such container')):raise RuntimeError('Curriculum container not proven absent: '+identifier)

def validate_train_inputs(train_host):
    if sha(SUPERVISOR_SOURCE/'campaign_source_hashes.json')!=SUPERVISOR_SHA256:raise RuntimeError('Changed supervisor source')
    if sha(SOURCE/'campaign_source_hashes.json')!=SOURCE_SHA256:raise RuntimeError('Changed native source')
    if CHECKPOINT.is_symlink() or sha(CHECKPOINT)!=CHECKPOINT_SHA256:raise RuntimeError('Wrong original checkpoint')
    if sha(HOST)!=HOST_SHA256:raise RuntimeError('Changed native host')
    verify_frozen(HOST.parent,HOST_FREEZE_SHA256);verify_frozen(CONTRACT,CONTRACT_SHA256)
    args=SimpleNamespace(source=SOURCE,checkpoint=CHECKPOINT,contract=CONTRACT,supervisor_source=SUPERVISOR_SOURCE,output=OUTPUT,isaaclab=Path('/home/orionh/IsaacLab'),host_freeze_sha256=HOST_FREEZE_SHA256,allocation='pilot',branch='caps',smoke=SMOKE)
    if train_host.selected_phases(args)!=('standing','initial_constant','initial_stop','train','final_constant','final_stop'):raise RuntimeError('Host changed bounded pilot phase order')
    identity=train_host.verify_inputs(args)
    if identity.get('source_manifest_sha256')!=SOURCE_SHA256 or identity.get('checkpoint_sha256')!=CHECKPOINT_SHA256:raise RuntimeError('Host admitted different native inputs')
    expected={'schema':'direct315_stand_stop_caps_native_v2','allocation':'pilot','branch':'caps','replicas':1024,'controls_per_update':24,'updates':50,'caps':{'temporal_weight':.1,'spatial_weight':.1,'noise_scale':1.,'noise_seed':1157}}
    if identity.get('selection')!=expected or (identity.get('actor_width'),identity.get('critic_width'))!=(315,318):raise RuntimeError('Host selection differs from bounded fifty-update CAPS pilot')
    if identity.get('smoke_campaign_sha256')!=SMOKE_PINS['direct_omni_train_smoke_002/campaign.json']:raise RuntimeError('Host admitted different prior smoke campaign')
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
            reason='Bounded direct315 matched CAPS pilot: fresh standing, original-policy constant and stop diagnostics,50 updates with1024 replicas and24 controls per update, final constant and stop diagnostics; no automatic continuation or Stage2 performance admission',
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
        subprocess.run(['systemd-run', '--user', '--unit=hexapod-forecast-restore-057', '--on-active=70m',
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
        cmd=['systemd-run','--user','--unit='+UNIT,'--property=RuntimeMaxSec=3720',
            '--property=TimeoutStopSec=180','--property=KillMode=process','--property=Environment=PYTHONDONTWRITEBYTECODE=1',
            '--property=ExecStopPost=/usr/bin/python3 '+str(restorer),'--property=Environment=PYTHONUNBUFFERED=1',
            '/usr/bin/python3',str(HOST),'--source',str(SOURCE),'--checkpoint',str(CHECKPOINT),
            '--contract',str(CONTRACT),'--supervisor-source',str(SUPERVISOR_SOURCE),'--output',str(OUTPUT),
            '--allocation','pilot','--branch','caps','--smoke',str(SMOKE)]
        subprocess.run(cmd,check=True,timeout=30)
        launched=True
        (PAUSE/'launch.json').write_text(json.dumps(dict(command=cmd,launched_unix=time.time()),indent=2)+'\n')
        print(json.dumps(dict(unit=UNIT,source=str(SOURCE),output=str(OUTPUT),pause=str(PAUSE))))
    finally:
        if not launched:
            subprocess.run(['/usr/bin/python3',str(restorer),'--stop-owner'],check=True,timeout=300)

if __name__=='__main__':main()
