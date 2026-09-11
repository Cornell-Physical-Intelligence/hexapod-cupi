"""First fresh canonical PPO smoke guard; actual standing32 terminal bindings are pending."""
from pathlib import Path
from types import SimpleNamespace
import fcntl,hashlib,importlib.util,json,os,subprocess,sys,time
sys.dont_write_bytecode=True
BASE=Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910')
PRIOR_BASE=BASE
SOURCE=BASE/'ppo_source_002'
STANDING_SOURCE=BASE/'standing_source_003'
ASSET=BASE/'asset_001'
ADMISSION=BASE/'native_actuation_001/actuation'
SUPERVISOR_SOURCE=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_physics_source_009')
OUTPUT=BASE/'canonical_ppo_smoke_001'
PAUSE=BASE/'forecast_pause_012'
UNIT='hexapod-canonical-ppo-smoke-001-20260910.service'
HOST=BASE/'ppo_host_002/launch_ppo_spark.py'
SOURCE_SHA256='1e173946fe2548a82207791528f503ac6d12766fd5b50f25746e6edf92704617'
SUPERVISOR_SHA256='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
HOST_SHA256='f85a5200a99d14348b0ff418292575d7e78c4bef61d46eb0fea2eb9c4e344c4c'
HOST_FREEZE_SHA256='d314bf361da009e48e6180b5cc4596cb45e0de77d0be6c28baf2ccaf9fe89f19'
PREVIOUS_UNIT='hexapod-canonical-native-standing32-002-20260910.service'
PREVIOUS_INVOCATION='1a38495bda9f4fefa4e5585574aabc56'
PREVIOUS_ROOT=BASE/'native_standing32_002'
PREVIOUS_PAUSE=BASE/'forecast_pause_010'
PREVIOUS_HOST='c72c4dd63599f2c459c7b8eca7244528cb725094252c084f695f4aa44d8dba25'
PREVIOUS_SOURCE='e922ff1312b5106f957b384874ca5eb6ed4a9ed6566c0ef9a0602247a49670f0'
PREVIOUS_NAME=None
PREVIOUS_ID=None
PREVIOUS_AUDIT_SHA256=None
PRIOR_PINS={}

STANDING_ONE=BASE/'native_standing_004/standing'
STANDING_ONE_STATE_SHA256='318c0bd2500cc954b6eec5464d5a0feca1fc2d570c5dae2e65e0177f18475da0'
STANDING_ONE_INVENTORY_SHA256='43f799625273680b29d00e3b2f063896dc1591582414b000ef6ba345f5830524'
PREVIOUS_RESTORED_UNIX=None
STANDING_SOURCE_SHA256='e922ff1312b5106f957b384874ca5eb6ed4a9ed6566c0ef9a0602247a49670f0'
STANDING32=PREVIOUS_ROOT/'standing'
STANDING32_STATE_SHA256=None
STANDING32_INVENTORY_SHA256=None
BINDINGS=BASE/'ppo_bindings_001.json'
BINDINGS_SHA256=None

RESERVATION=BASE/'exclusive_reservation_001'
RESERVATION_PINS={'/home/orionh/HEXAPOD_runs/canonical_direct_20260910/exclusive_reservation_001/ACTIVE': 'd6135e574b033fbbbe716e7f6b34876a42f1897f560d0c97ab3ae4da33532dc2', '/home/orionh/HEXAPOD_runs/canonical_direct_20260910/exclusive_reservation_001/initial_receipt.json': 'b0e17c5e7f2ae397260441dc8728eec079608cdf372717536dafc22d179df175', '/home/orionh/HEXAPOD_runs/canonical_direct_20260910/exclusive_reservation_001/scheduler_before.json': '139c7eec8e336bbf2df68f480848416232535bdceecabd4a5f210697955fd4f1', '/home/orionh/.config/systemd/user/stormscope-dispatch.timer.d/90-hexapod-exclusive-reservation-20260910.conf': 'bae12ac17239c89273a5b3daa7349ba76d74a0608dc0495abe35000a537bdd26', '/home/orionh/.config/systemd/user/stormscope-dispatch.service.d/90-hexapod-exclusive-reservation-20260910.conf': 'bae12ac17239c89273a5b3daa7349ba76d74a0608dc0495abe35000a537bdd26', '/home/orionh/.config/systemd/user/stormscope-scout.timer.d/90-hexapod-exclusive-reservation-20260910.conf': 'bae12ac17239c89273a5b3daa7349ba76d74a0608dc0495abe35000a537bdd26', '/home/orionh/.config/systemd/user/stormscope-scout.service.d/90-hexapod-exclusive-reservation-20260910.conf': 'bae12ac17239c89273a5b3daa7349ba76d74a0608dc0495abe35000a537bdd26', '/home/orionh/.config/systemd/user/stormscope-monitor.timer.d/90-hexapod-exclusive-reservation-20260910.conf': 'bae12ac17239c89273a5b3daa7349ba76d74a0608dc0495abe35000a537bdd26', '/home/orionh/.config/systemd/user/stormscope-monitor.service.d/90-hexapod-exclusive-reservation-20260910.conf': 'bae12ac17239c89273a5b3daa7349ba76d74a0608dc0495abe35000a537bdd26', '/home/orionh/.config/systemd/user/stormscope-publish.timer.d/90-hexapod-exclusive-reservation-20260910.conf': 'bae12ac17239c89273a5b3daa7349ba76d74a0608dc0495abe35000a537bdd26', '/home/orionh/.config/systemd/user/stormscope-publish.service.d/90-hexapod-exclusive-reservation-20260910.conf': 'bae12ac17239c89273a5b3daa7349ba76d74a0608dc0495abe35000a537bdd26', '/home/orionh/.config/systemd/user/stormscope-verify.timer.d/90-hexapod-exclusive-reservation-20260910.conf': 'bae12ac17239c89273a5b3daa7349ba76d74a0608dc0495abe35000a537bdd26', '/home/orionh/.config/systemd/user/stormscope-verify.service.d/90-hexapod-exclusive-reservation-20260910.conf': 'bae12ac17239c89273a5b3daa7349ba76d74a0608dc0495abe35000a537bdd26'}

def call(args):return subprocess.check_output(args,text=True,timeout=30).strip()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def valid_hash(v):return isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v)
def require_final_bindings():
    for name,bound in {'source':SOURCE_SHA256,'standing source':STANDING_SOURCE_SHA256,'supervisor':SUPERVISOR_SHA256,'host':HOST_SHA256,'host freeze':HOST_FREEZE_SHA256,'previous audit':PREVIOUS_AUDIT_SHA256,'standing1 state':STANDING_ONE_STATE_SHA256,'standing1 inventory':STANDING_ONE_INVENTORY_SHA256,'standing32 state':STANDING32_STATE_SHA256,'standing32 inventory':STANDING32_INVENTORY_SHA256,'enabled bindings':BINDINGS_SHA256,**PRIOR_PINS,**RESERVATION_PINS}.items():
        if not valid_hash(bound):raise RuntimeError('Final reviewed binding pending: '+name)
    if not PRIOR_PINS or not isinstance(PREVIOUS_NAME,str) or not PREVIOUS_NAME or not valid_hash(PREVIOUS_ID):raise RuntimeError('Actual standing32 owned cleanup identities pending')
    if not isinstance(PREVIOUS_INVOCATION,str) or len(PREVIOUS_INVOCATION)!=32 or any(c not in '0123456789abcdef'for c in PREVIOUS_INVOCATION) or type(PREVIOUS_RESTORED_UNIX)not in(int,float)or PREVIOUS_RESTORED_UNIX<=0:raise RuntimeError('Actual standing32 invocation/restoration pending')

def verify_manifest(root,manifest,bound):
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):raise RuntimeError('Symbolic frozen source')
    if sha(root/manifest)!=bound:raise RuntimeError('Wrong frozen manifest')
    actual={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p!=root/manifest}
    if actual!=json.loads((root/manifest).read_text()):raise RuntimeError('Changed/unlisted frozen source')

def verify_previous_owner():
    audit_path=Path(__file__).resolve().parent/'inputs/previous_terminal_audit.json'
    if sha(audit_path)!=PREVIOUS_AUDIT_SHA256:raise RuntimeError('Prior admission audit changed')
    audit=json.loads(audit_path.read_text())
    if audit.get('audit_verified')is not True or audit.get('terminal_outcome')!='authentic_completed_standing' or audit.get('standing_completed')is not True or audit.get('expected_invocation')!=PREVIOUS_INVOCATION or audit.get('errors')!=[]:raise RuntimeError('Prior standing32 admission invalid')
    fields=dict(line.split('=',1)for line in call(['systemctl','--user','show',PREVIOUS_UNIT,'-p','ActiveState','-p','SubState','-p','InvocationID','-p','MainPID','-p','ExecMainStatus']).splitlines()if '='in line)
    if fields.get('ActiveState')!='inactive' or fields.get('MainPID')!='0' or fields.get('ExecMainStatus')!='0':raise RuntimeError('Previous completed owner state changed')
    if fields.get('InvocationID') and fields['InvocationID']!=PREVIOUS_INVOCATION:raise RuntimeError('Previous invocation changed')
    for name,bound in PRIOR_PINS.items():
        if sha(PRIOR_BASE/name)!=bound:raise RuntimeError('Previous admission/cleanup receipt changed: '+name)
    standing_manifest=json.loads((PREVIOUS_ROOT/'standing_immutable.sha256.json').read_text())
    if hashlib.sha256(json.dumps(standing_manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=STANDING32_INVENTORY_SHA256 or any(PRIOR_PINS.get(PREVIOUS_ROOT.name+'/standing/'+name)!=bound for name,bound in standing_manifest.items()):raise RuntimeError('Actual standing32 complete inventory binding differs')
    campaign=json.loads((PREVIOUS_ROOT/'campaign.json').read_text());identity=campaign.get('identity',{})
    if campaign.get('status')!='completed' or campaign.get('terminal_inputs_unchanged')is not True or campaign.get('planned_phases')!=['standing']or campaign.get('host_freeze_sha256')!=PREVIOUS_HOST or campaign.get('source_freeze_sha256')!=PREVIOUS_SOURCE or identity.get('inspector_freeze_sha256')!=STANDING_SOURCE_SHA256 or identity.get('num_envs')!=32:raise RuntimeError('Previous standing32 campaign identity changed')
    receipt=campaign.get('standing',{})
    if receipt.get('standing_pass')is not True or receipt.get('status')!='completed' or receipt.get('num_envs')!=32 or receipt.get('state_sha256')!=STANDING32_STATE_SHA256 or campaign.get('post_exit_original_inputs_reverified')is not True or campaign.get('post_exit_all_standing_payloads_inventoried')is not True:raise RuntimeError('Previous standing32 acceptance/seal changed')
    if identity.get('standing_one_state_sha256')!=STANDING_ONE_STATE_SHA256 or identity.get('standing_one_inventory_sha256')!=STANDING_ONE_INVENTORY_SHA256:raise RuntimeError('Standing32 did not consume exact standing1')
    native=json.loads((STANDING32/'state.json').read_text());session=json.loads((STANDING32/'session.json').read_text())
    if native.get('status')!='completed' or native.get('standing_pass')is not True or native.get('explicit_steps_completed')!=8000 or native.get('identity')!=identity or native.get('inputs_unchanged')is not True or native.get('errors')!=[] or native.get('physical_admission')is not False or native.get('training_allowed')is not False:raise RuntimeError('Previous raw standing32 state changed')
    if session.get('steps')!=8000 or session.get('captured_steps')!=8000 or session.get('controls')!=1000 or session.get('all_rows_recorded')is not True or session.get('reset_count')!=1 or session.get('failure')is not None:raise RuntimeError('Previous standing32 acquisition differs')
    pause=json.loads((PREVIOUS_PAUSE/'pause.json').read_text());restored=json.loads((PREVIOUS_PAUSE/'restored.json').read_text());timers={name for name,value in pause['units'].items()if name.endswith('.timer')and 'ActiveState=active'in value}
    if pause.get('unit')!=PREVIOUS_UNIT or pause.get('output')!=str(PREVIOUS_ROOT)or set(restored.get('timers',[]))!=timers or restored.get('restored_unix')!=PREVIOUS_RESTORED_UNIX:raise RuntimeError('Previous exact pause010 restoration missing')
    if restored.get('scope')!='per_job_snapshot_only' or restored.get('persistent_reservation_release_attempted')is not False or restored.get('reservation_path')!=str(RESERVATION):raise RuntimeError('Prior cleanup must preserve persistent reservation')
    names={p.stem for p in (PREVIOUS_ROOT/'jobs').glob('*.json')if not p.name.endswith('_contact_data_audit.json')}
    if names!={'standing'}:raise RuntimeError('Previous job inventory changed')
    job=json.loads((PREVIOUS_ROOT/'jobs/standing.json').read_text())
    if job.get('status')!='completed'or job.get('phase')!='standing'or job.get('exit_code')!=0 or job.get('cleanup_checked')is not True or job.get('container_name')!=PREVIOUS_NAME or job.get('container_id')!=PREVIOUS_ID:raise RuntimeError('Previous completed owned cleanup incomplete')
    for identifier in (PREVIOUS_NAME,PREVIOUS_ID):
        q=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',identifier],capture_output=True,text=True,timeout=20)
        if q.returncode==0 or not any(v in q.stderr.lower()for v in ('no such object','no such container')):raise RuntimeError('Previous container absence unknown: '+identifier)

def verify_reservation():
    for name,bound in RESERVATION_PINS.items():
        p=Path(name)
        if p.is_symlink() or not p.is_file() or sha(p)!=bound:raise RuntimeError('Persistent reservation missing/changed: '+name)
    for name in RESERVATION_PINS:
        if not name.endswith('.conf'):continue
        unit=Path(name).parent.name.removesuffix('.d')
        fields=dict(line.split('=',1)for line in call(['systemctl','--user','show',unit,'-p','DropInPaths','-p','NeedDaemonReload','-p','ActiveState']).splitlines()if '='in line)
        if fields.get('NeedDaemonReload')!='no' or name not in fields.get('DropInPaths','').split() or fields.get('ActiveState')not in ('inactive','failed'):raise RuntimeError('Reservation drop-in not loaded or covered scheduler active: '+unit)
    if json.loads((RESERVATION/'ACTIVE').read_text()).get('exclusive')is not True:raise RuntimeError('Exclusive reservation is not active')
    return {'path':str(RESERVATION),'marker_sha256':RESERVATION_PINS[str(RESERVATION/'ACTIVE')],'verified_files':len(RESERVATION_PINS),'reservation_active_at_preflight':True,'release_attempted':False}

def validate_ppo_inputs(host):
    require_final_bindings()
    verify_reservation()
    if sha(HOST)!=HOST_SHA256:raise RuntimeError('Canonical host changed')
    verify_manifest(HOST.parent,'FREEZE_SHA256.json',HOST_FREEZE_SHA256)
    verify_manifest(SOURCE,'FREEZE_SHA256.json',SOURCE_SHA256)
    verify_manifest(STANDING_SOURCE,'FREEZE_SHA256.json',STANDING_SOURCE_SHA256)
    if host.SOURCE_FREEZE!=SOURCE_SHA256 or host.STANDING_FREEZE!=STANDING_SOURCE_SHA256 or host.SUPERVISOR_MAP!=SUPERVISOR_SHA256 or host.EXPECTED_COORDINATION!='649ccda1bdef20bc967ae78f68a992cc8001f4ed7009f78cab5133cc67b29d8f':raise RuntimeError('Host pins do not match canonical PPO guard')
    if BINDINGS.is_symlink()or not BINDINGS.is_file()or sha(BINDINGS)!=BINDINGS_SHA256:raise RuntimeError('Exact external enabled PPO bindings changed/missing')
    binding=json.loads(BINDINGS.read_text())
    for key,value in {'standing_source_freeze_sha256':STANDING_SOURCE_SHA256,'standing1_state_sha256':STANDING_ONE_STATE_SHA256,'standing32_state_sha256':STANDING32_STATE_SHA256,'ready_for_native_dispatch':True}.items():
        if binding.get(key)!=value:raise RuntimeError('Wrong pending/source/standing PPO binding:'+key)
    args=SimpleNamespace(source=SOURCE,asset=ASSET,admission=ADMISSION,standing_source=STANDING_SOURCE,standing_one=STANDING_ONE,standing32=STANDING32,bindings=BINDINGS,supervisor_source=SUPERVISOR_SOURCE,output=OUTPUT,isaaclab=Path('/home/orionh/IsaacLab'),host_freeze_sha256=HOST_FREEZE_SHA256)
    host.require_fresh_output(args);identity=host.verify_inputs(args)
    if identity.get('schema')!='canonical_direct_drive_ppo_smoke_v1' or identity.get('phase')!='canonical_ppo_smoke' or identity.get('runtime_binding')!={'runtime_tree_sha256':SOURCE_SHA256,'scope':'fresh_canonical_405_408_two_update_smoke'}:raise RuntimeError('Wrong fresh canonical PPO scope')
    for key,value in {'training_integration_only':True,'quality_admitted':False,'Stage2_complete':False,'fresh_neutral_controls':1000,'fresh_neutral_substeps':8000,'policy_controls':48,'policy_substeps':384,'total_controls':1048,'total_substeps':8384}.items():
        if identity.get(key)!=value:raise RuntimeError('Wrong fixed PPO allocation:'+key)
    lineage=identity.get('policy_lineage',{});native=identity.get('native_identity',{})
    if lineage.get('standing1_state_sha256')!=STANDING_ONE_STATE_SHA256 or lineage.get('standing32_state_sha256')!=STANDING32_STATE_SHA256 or native.get('standing_one_inventory_sha256')!=STANDING_ONE_INVENTORY_SHA256:raise RuntimeError('Wrong exact same-source standing pair')
    return identity

def main():
    if PAUSE.exists() or OUTPUT.exists():raise RuntimeError('Fresh pause/output names required')
    require_final_bindings()
    sys.path.insert(0,str(SUPERVISOR_SOURCE/'tools'))
    from launch_length_study_spark import preflight
    verify_manifest(SUPERVISOR_SOURCE,'campaign_source_hashes.json',SUPERVISOR_SHA256)
    if sha(HOST)!=HOST_SHA256:raise RuntimeError('Wrong reviewed host before import')
    spec=importlib.util.spec_from_file_location('train_host',HOST)
    train_host=importlib.util.module_from_spec(spec);spec.loader.exec_module(train_host)
    validate_ppo_inputs(train_host)
    verify_previous_owner()
    coordination = Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md').read_bytes()
    if hashlib.sha256(coordination).hexdigest() != '649ccda1bdef20bc967ae78f68a992cc8001f4ed7009f78cab5133cc67b29d8f':
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
            output=str(OUTPUT), source=str(SOURCE), admission=str(ADMISSION), standing_source=str(STANDING_SOURCE), standing_one=str(STANDING_ONE), standing32=str(STANDING32), bindings=str(BINDINGS), preflight=snapshot,
            coordination_sha256=hashlib.sha256(coordination).hexdigest(),
            reservation_path=str(RESERVATION), restoration_scope='per_job_snapshot_only', persistent_reservation_release_attempted=False,
            reason='Fresh canonical32 PPO smoke: exact same-source standing1/32,1000 neutral controls then48 policy controls/two updates; fresh actor and no checkpoint input; no walking or quality admission',
            current_instruction='10 September user requests persistent exclusive HEXAPOD priority; preserve exact other-job progress and resume identities; this per-job guard never releases the reservation or bypasses actual CUDA ownership checks')
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
    result={'restored_unix':time.time(),'timers':active,'owned_cleanup_checked':cleanup,'scope':'per_job_snapshot_only','persistent_reservation_release_attempted':False,'reservation_path':r.get('reservation_path')}
    t=p/'restored.tmp';t.write_text(json.dumps(result,indent=2)+'\\n');t.replace(p/'restored.json')
''')
        # Arm recovery before mutating timer state. It stops only this exact
        # owner unit if needed, then restores only previously active timers.
        subprocess.run(['systemd-run', '--user', '--unit=hexapod-canonical-forecast-restore-012', '--on-active=15m',
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
        validate_ppo_inputs(train_host)
        verify_previous_owner()
    except Exception:
        subprocess.run(['/usr/bin/python3',str(restorer)],check=True,timeout=120)
        raise
    try:
        cmd=['systemd-run','--user','--unit='+UNIT,'--property=RuntimeMaxSec=720',
            '--property=TimeoutStopSec=180','--property=KillMode=process','--property=Environment=PYTHONDONTWRITEBYTECODE=1',
            '--property=ExecStopPost=/usr/bin/python3 '+str(restorer),'--property=Environment=PYTHONUNBUFFERED=1',
            '/usr/bin/python3',str(HOST),'--source',str(SOURCE),'--asset',str(ASSET),'--admission',str(ADMISSION),
            '--supervisor-source',str(SUPERVISOR_SOURCE),'--output',str(OUTPUT),'--standing-source',str(STANDING_SOURCE),'--standing-one',str(STANDING_ONE),'--standing32',str(STANDING32),'--bindings',str(BINDINGS)]
        subprocess.run(cmd,check=True,timeout=30)
        launched=True
        (PAUSE/'launch.json').write_text(json.dumps(dict(command=cmd,launched_unix=time.time(),native_source_freeze_sha256=SOURCE_SHA256),indent=2)+'\n')
        print(json.dumps(dict(unit=UNIT,source=str(SOURCE),output=str(OUTPUT),pause=str(PAUSE))))
    finally:
        if not launched:
            subprocess.run(['/usr/bin/python3',str(restorer),'--stop-owner'],check=True,timeout=300)

if __name__=='__main__':main()
