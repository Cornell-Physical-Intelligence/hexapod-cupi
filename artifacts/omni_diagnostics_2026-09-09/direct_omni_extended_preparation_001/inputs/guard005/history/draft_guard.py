"""External pause061 guard: native005 smoke004 after exact quiet-priority pilot cleanup."""
from pathlib import Path
from types import SimpleNamespace
import fcntl,hashlib,importlib.util,json,os,subprocess,sys,time
sys.dont_write_bytecode=True
BASE=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
SUPERVISOR_SOURCE=BASE/'reference_physics_source_009'
SOURCE=BASE/'direct_omni_train_source_004'
CONTRACT=BASE/'direct_omni_train_preparation_004'
CHECKPOINT=BASE/'omni_repair_003/branch_a/inputs/original.pt'
OUTPUT=BASE/'direct_omni_train_smoke_004'
PAUSE=BASE/'forecast_pause_061'
UNIT='hexapod-direct-omni-train-smoke-004-20260910.service'
HOST=BASE/'direct_omni_train_host_004/launch_train_spark.py'
SOURCE_SHA256='PENDING'
SUPERVISOR_SHA256='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
CONTRACT_SHA256='PENDING'
CHECKPOINT_SHA256='1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8'
HOST_SHA256='PENDING'
HOST_FREEZE_SHA256='PENDING'
PREVIOUS_UNIT='hexapod-direct-omni-train-pilot-quiet-priority-001-20260910.service'
PREVIOUS_INVOCATION='721f1886478a4e518c7e0b741256413f'
PRIOR_PINS={'direct_omni_train_pilot_quiet_priority_001/campaign.json': '227a11ba67ed9f374cae4879171f222c296395813444ff494d3d57eaa4b81b79', 'direct_omni_train_pilot_quiet_priority_001/jobs/standing.json': '06e2d0278df5724d383b110275092bee7b4fd45a156f4f9aea8633528ebaa6f8', 'direct_omni_train_pilot_quiet_priority_001/jobs/initial_constant.json': 'e91f810b49ad050978209c6d3b5d20e7ca1af9503b88babb4c4f13d6a94fc3e2', 'direct_omni_train_pilot_quiet_priority_001/jobs/initial_stop.json': '468ad33e188022bb461e8fb6a3b2c6dc7208677262d3bea7e72a53826f6a38ed', 'direct_omni_train_pilot_quiet_priority_001/jobs/train.json': '6689bad9ba467b9a509acc427590181ba4705f1c3fd474fa77beea6187277858', 'direct_omni_train_pilot_quiet_priority_001/jobs/final_constant.json': '14f159d7cfcf87560c419ead7b25d1b234ffe40aa4f93ef90a41a0f323f3b044', 'direct_omni_train_pilot_quiet_priority_001/jobs/final_stop.json': 'a446bcf7bd53bfa04a7e1d4ae5cbc525511b5cbeeeef618a78e6444348ada575', 'forecast_pause_060/pause.json': 'ba45d3695d7e520ef6b2a44b80aa33026d98926bb5945f0c45a48c385d5bf1fd', 'forecast_pause_060/restored.json': '9dfb238a497e65cd31434acfce03203f88be907bf680d19df80e471047dc24bf'}

PREVIOUS_OUTPUT='direct_omni_train_pilot_quiet_priority_001'
PREVIOUS_PHASES=('standing','initial_constant','initial_stop','train','final_constant','final_stop')
PREVIOUS_SOURCE_SHA256='ef962070bd770d03553eb824fb6caccada7d6a06ac26d2f180bad068f0a9cc62'
PREVIOUS_NATIVE_FREEZE_SHA256='1ab678beeba6bf74978c210c4e38f13655e25728fe66e3007e3b1db5e19b6b20'
PREVIOUS_HOST_FREEZE_SHA256='df815208f340a9c8c0c2ce8c9fcfad7fd28791f745961ab714f96c0508454efc'
PREVIOUS_SELECTION={'schema':'direct315_quiet_priority_native_v3','allocation':'pilot','branch':'quiet_priority','replicas':1024,'controls_per_update':24,'updates':50,'caps':{'temporal_weight':.1,'spatial_weight':.1,'noise_scale':1.,'noise_seed':1157,'quiet_temporal_weight':1.0}}

def call(args):return subprocess.check_output(args,text=True,timeout=30).strip()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require_final_bindings():
    for name,value in {'source':SOURCE_SHA256,'supervisor':SUPERVISOR_SHA256,'contract':CONTRACT_SHA256,
        'checkpoint':CHECKPOINT_SHA256,'host':HOST_SHA256,'host freeze':HOST_FREEZE_SHA256,**PRIOR_PINS}.items():
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise RuntimeError('Final reviewed binding pending: '+name)
    if len(PREVIOUS_INVOCATION)!=32 or any(c not in '0123456789abcdef' for c in PREVIOUS_INVOCATION):raise RuntimeError('Previous invocation not bound')
    if set(PRIOR_PINS)!={PREVIOUS_OUTPUT+'/campaign.json', *(PREVIOUS_OUTPUT+'/jobs/'+phase+'.json' for phase in PREVIOUS_PHASES), 'forecast_pause_060/pause.json', 'forecast_pause_060/restored.json'}:raise RuntimeError('Exact nine pilot owner/restoration receipts required')

def verify_frozen(root,expected):
    manifest=root/'FREEZE_SHA256.json'
    if sha(manifest)!=expected:raise RuntimeError('Wrong reviewed frozen bundle: '+str(root))
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):raise RuntimeError('Frozen input symbolic substitution')
    actual={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p!=manifest}
    if actual!=json.loads(manifest.read_text()):raise RuntimeError('Frozen source changed or has unlisted files: '+str(root))

def verify_previous_owner():
    previous=call(['systemctl','--user','show',PREVIOUS_UNIT,'-p','ActiveState','-p','SubState','-p','InvocationID','-p','MainPID','-p','ExecMainStatus'])
    fields=dict(line.split('=',1) for line in previous.splitlines() if '=' in line)
    if fields.get('ActiveState')!='inactive' or fields.get('MainPID')!='0' or fields.get('ExecMainStatus')!='0':raise RuntimeError('Previous owner not completed and inactive')
    live=fields.get('InvocationID','')
    if live and live!=PREVIOUS_INVOCATION:raise RuntimeError('Previous owner invocation changed')
    for name,bound in PRIOR_PINS.items():
        if sha(BASE/name)!=bound:raise RuntimeError('Previous receipt changed: '+name)
    campaign=json.loads((BASE/PREVIOUS_OUTPUT/'campaign.json').read_text())
    if campaign.get('status')!='completed' or campaign.get('terminal_inputs_unchanged') is not True or campaign.get('bounded_campaign_complete') is not True or campaign.get('last_completed_phase')!='final_stop':raise RuntimeError('Prior pilot campaign not terminal and immutable')
    if campaign.get('planned_phases')!=list(PREVIOUS_PHASES) or set(campaign.get('accepted_phases',{}))!=set(PREVIOUS_PHASES) or campaign.get('allocation')!='pilot' or campaign.get('branch')!='quiet_priority' or campaign.get('PPO_updates_completed')!=50 or campaign.get('Stage2_complete') is not False or campaign.get('automatic_continuation') is not False:raise RuntimeError('Prior pilot scope differs')
    identity=campaign.get('identity',{})
    if campaign.get('host_freeze_sha256')!=PREVIOUS_HOST_FREEZE_SHA256 or identity.get('source_manifest_sha256')!=PREVIOUS_SOURCE_SHA256 or identity.get('selection')!=PREVIOUS_SELECTION or identity.get('checkpoint_sha256')!=CHECKPOINT_SHA256 or (identity.get('actor_width'),identity.get('critic_width'))!=(315,318):raise RuntimeError('Prior pilot source/host/selection identity differs')
    for relative,bound in [('direct_omni_train_preparation_003/FREEZE_SHA256.json',PREVIOUS_NATIVE_FREEZE_SHA256),('direct_omni_train_host_003/FREEZE_SHA256.json',PREVIOUS_HOST_FREEZE_SHA256)]:
        if sha(BASE/relative)!=bound:raise RuntimeError('Prior native/host freeze identity changed: '+relative)
    for phase,accepted in campaign['accepted_phases'].items():
        if accepted.get('phase')!=phase or accepted.get('source_manifest_sha256')!=PREVIOUS_SOURCE_SHA256 or accepted.get('Stage2_complete') is not False or accepted.get('passed' if phase=='standing' else 'complete') is not True:raise RuntimeError('Prior accepted phase lacks bounded completion: '+phase)
    if campaign['accepted_phases']['train'].get('updates_completed')!=50:raise RuntimeError('Prior train completion count differs')
    previous_pause=json.loads((BASE/'forecast_pause_060/pause.json').read_text())
    restored=json.loads((BASE/'forecast_pause_060/restored.json').read_text())
    expected_timers={name for name,state in previous_pause['units'].items() if name.endswith('.timer') and 'ActiveState=active' in state}
    if previous_pause.get('unit')!=PREVIOUS_UNIT or set(restored.get('timers',[]))!=expected_timers or not restored.get('restored_unix'):raise RuntimeError('Prior exact timer restoration missing')
    identifiers=[]
    for phase in PREVIOUS_PHASES:
        job=json.loads((BASE/PREVIOUS_OUTPUT/'jobs'/(phase+'.json')).read_text())
        if job.get('status')!='completed' or job.get('phase')!=phase or job.get('exit_code')!=0 or job.get('cleanup_checked') is not True or not job.get('container_id') or not job.get('container_name') or job.get('allocation')!='pilot' or job.get('branch')!='quiet_priority':raise RuntimeError('Prior phase lacks exact terminal cleanup proof: '+phase)
        identifiers.extend((job['container_name'],job['container_id']))
    if len(set(identifiers))!=12:raise RuntimeError('Prior six jobs do not have twelve distinct owned identifiers')
    for identifier in identifiers:
        found=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',identifier],text=True,capture_output=True,timeout=20)
        if found.returncode==0 or not any(x in found.stderr.lower() for x in ('no such object','no such container')):raise RuntimeError('Previous container not proven absent: '+identifier)

def validate_train_inputs(train_host):
    if sha(SUPERVISOR_SOURCE/'campaign_source_hashes.json')!=SUPERVISOR_SHA256:raise RuntimeError('Changed supervisor source')
    if sha(SOURCE/'campaign_source_hashes.json')!=SOURCE_SHA256:raise RuntimeError('Changed native source')
    if CHECKPOINT.is_symlink() or sha(CHECKPOINT)!=CHECKPOINT_SHA256:raise RuntimeError('Wrong original checkpoint')
    if sha(HOST)!=HOST_SHA256:raise RuntimeError('Changed native host')
    verify_frozen(HOST.parent,HOST_FREEZE_SHA256);verify_frozen(CONTRACT,CONTRACT_SHA256)
    args=SimpleNamespace(source=SOURCE,checkpoint=CHECKPOINT,contract=CONTRACT,supervisor_source=SUPERVISOR_SOURCE,output=OUTPUT,isaaclab=Path('/home/orionh/IsaacLab'),host_freeze_sha256=HOST_FREEZE_SHA256,allocation='smoke',branch='quiet_priority',smoke=None)
    if train_host.selected_phases(args)!=('standing','train','final_constant','final_stop'):raise RuntimeError('Host changed bounded smoke phase order')
    identity=train_host.verify_inputs(args)
    if identity.get('source_manifest_sha256')!=SOURCE_SHA256 or identity.get('checkpoint_sha256')!=CHECKPOINT_SHA256:raise RuntimeError('Host admitted different native inputs')
    expected={'schema':'direct315_extended_native_v4','allocation':'smoke','branch':'quiet_priority','replicas':32,'controls_per_update':24,'updates':2,'caps':{'temporal_weight':.1,'spatial_weight':.1,'noise_scale':1.,'noise_seed':1157,'quiet_temporal_weight':1.0}}
    if identity.get('selection')!=expected or (identity.get('actor_width'),identity.get('critic_width'))!=(315,318):raise RuntimeError('Host selection differs from bounded two-update quiet-priority smoke')
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
    if hashlib.sha256(coordination).hexdigest() != '35af5100e39980f55a3f5e19654e4550b3c3fb6da114b64fbe0d143817242cf3':
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
            reason='Bounded native005 smoke004: fresh standing, two quiet-priority updates with 32 replicas, final constant and stop diagnostics; prior 50-update pilot completion is cleanup ancestry only, no quality promotion or automatic extended training',
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
        subprocess.run(['systemd-run', '--user', '--unit=hexapod-forecast-restore-061', '--on-active=45m',
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
            '--allocation','smoke','--branch','quiet_priority']
        subprocess.run(cmd,check=True,timeout=30)
        launched=True
        (PAUSE/'launch.json').write_text(json.dumps(dict(command=cmd,launched_unix=time.time()),indent=2)+'\n')
        print(json.dumps(dict(unit=UNIT,source=str(SOURCE),output=str(OUTPUT),pause=str(PAUSE))))
    finally:
        if not launched:
            subprocess.run(['/usr/bin/python3',str(restorer),'--stop-owner'],check=True,timeout=300)

if __name__=='__main__':main()
