"""Canonical native standing32 guard; same-source completed standing1 is a required admission."""
from pathlib import Path
from types import SimpleNamespace
import fcntl,hashlib,importlib.util,json,os,subprocess,sys,time
sys.dont_write_bytecode=True
BASE=Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910')
PRIOR_BASE=BASE
SOURCE=BASE/'standing_source_002'
ASSET=BASE/'asset_001'
ADMISSION=BASE/'native_actuation_001/actuation'
SUPERVISOR_SOURCE=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_physics_source_009')
OUTPUT=BASE/'native_standing32_001'
PAUSE=BASE/'forecast_pause_008'
UNIT='hexapod-canonical-native-standing32-001-20260910.service'
HOST=BASE/'standing_host_002/launch_standing_spark.py'
SOURCE_SHA256='acb589708fcba8be2f6ef64171795889eca8b6583fdbb4bc2fc25d61717b0467'
SUPERVISOR_SHA256='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
HOST_SHA256='302aaf5da6d95f771832cc51ef5c002564cd8af77f9c19e65326ea04ea02f1bc'
HOST_FREEZE_SHA256='c533454d1699058a7b686e49436247bbb9c94bbeec0a91b01090c234f4576763'
PREVIOUS_UNIT='hexapod-canonical-native-standing-002-20260910.service'
PREVIOUS_INVOCATION='d9725b51027e44b59ffcb00a2b67763f'
PREVIOUS_ROOT=BASE/'native_standing_002'
PREVIOUS_PAUSE=BASE/'forecast_pause_007'
PREVIOUS_HOST='c533454d1699058a7b686e49436247bbb9c94bbeec0a91b01090c234f4576763'
PREVIOUS_SOURCE='acb589708fcba8be2f6ef64171795889eca8b6583fdbb4bc2fc25d61717b0467'
PREVIOUS_NAME='hexapod-reference-physics-a299ef0f35af417c9525da679a226f2a'
PREVIOUS_ID='7061d3d9147c629e0531caeac17684dd98039a66440755a3e01ff891a97afa9d'
PREVIOUS_AUDIT_SHA256='1feb59b4eace8724cf7909589c55e4478c7ecf6a1b0804d029902a0ed540a39b'
PRIOR_PINS={'native_standing_002/campaign.json': 'a6ac121a77906ad98fb0b4f3f084818bd6870daddfd24437c440c91dceed1271', 'native_standing_002/jobs/standing.json': 'bdcf80e62226088e0b03c78044d21d36c2004bacc4d7345432390f85109f0e56', 'native_standing_002/jobs/standing_contact_data_audit.json': 'cdc8cad57cb6b9b7ef605e5f0647df2bc62d4319bf53aafe58451a43ef066887', 'native_standing_002/logs/standing.log': 'c5a1b2890421cd8fae1c3b9a8dd5c62d6837065106a7ff78bf99e67952fbc14b', 'native_standing_002/standing/contact_view.json': '0ef9cbae7134b7ca5a2da23a0e95028324030fcf73cd558ea6504f78a759bde0', 'native_standing_002/standing/contacts.jsonl': '95ed8eb951dcf5aa56b79e4f43f53b3588d858b3a8af71e613aec79d9b09b216', 'native_standing_002/standing/control_trace.npz': 'd828c327ed3cf366eb16c23839c4d431943f40e593aac7cad62eec9841937323', 'native_standing_002/standing/initial_reset.json': '34b397b8eb1c7015ab9f31641bfcd14ef7ecc4f2cc76dc513ac1edcd802e2c97', 'native_standing_002/standing/memory_before_scene.json': '54ad96ac3d185ebe0b2e8208c63c0ad330c7adcec090d76fd915bc048bae8b3c', 'native_standing_002/standing/native_errors.json': '37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570', 'native_standing_002/standing/native_materials.json': 'aa1174dcab7dfeed253cadbf11c4a83075b9cc6db874df644b2a8491ee8087b7', 'native_standing_002/standing/native_readback.json': 'd37ab0a618739c5ac77a3f5e412f8dd558081a23f03b1ede1c98578251d6aca4', 'native_standing_002/standing/native_scene.json': '5342a8b913c38d6675817d62c1dedff4765a3f141178e206fa7ce7d76f47dbc9', 'native_standing_002/standing/resolved_stage.usda': '692efa7d5f1c6d4872c8af1e289bad25b73a2d32b8ca891bbcdce433c816c4de', 'native_standing_002/standing/runtime_api.json': '0e402d157689122e829aa4597033364a9168c3b8f78b9514b3a80918c009e237', 'native_standing_002/standing/sdf_readback.json': '0a92ec3d0cd7101a932b7e5617ec5dc2072e89c59e70c28464908867e074597a', 'native_standing_002/standing/sdk_readback.json': '1016fef5aa047bdd0e5684a9fc8acd599e95738601847ad4084f2be58c53c454', 'native_standing_002/standing/session.json': '997d600e96d1c5b4afe42a7bf0856cfb9cda1dd46935e05a18950eee6604ec8d', 'native_standing_002/standing/standing_report.json': 'd876970b4751da0a198eda4edeaf2e50ccecfb13610858ca647cdb7c18c1b201', 'native_standing_002/standing/state.json': '0d536d6906bfa8e755e971272bca98fed0e709db8f02651680cd95c04248cccd', 'native_standing_002/standing/substeps_000.npz': 'e2d09a8757e281207044c037bc91997a913c7749dc3e47406fcbdb388952400c', 'native_standing_002/standing/substeps_001.npz': '2bd8cd98a2f411fd7c05759969b3940310113577bbe812f494840017741fb1a6', 'native_standing_002/standing/substeps_002.npz': '4b732d547da314c08858ed05433fac68d68237ed7bc73a502c571f001d001927', 'native_standing_002/standing/substeps_003.npz': 'ab1a8b3afc830243c68cb81294df1ac688abd01eee257fe73705e3b44cc0d532', 'native_standing_002/standing/substeps_004.npz': '50b8bc7ef7ee6fb127950239f93a811e3fb784902d567b22cae0ca32c3f244a1', 'native_standing_002/standing/substeps_005.npz': '0fa8208e80e98492b52c2eb49d7ae201f4481a80d5c15d6f132109ba057a45d2', 'native_standing_002/standing/substeps_006.npz': 'c6d1d0915c73f573fb2fc75471c8aa581298225595f1b055aecbbed7274cdb31', 'native_standing_002/standing/substeps_007.npz': '3ded9f6715874010fac6f68f48a7df8ddba05849f456e19b3686366ee5da6a6c', 'native_standing_002/standing/substeps_008.npz': '5d2f5873f2175329d257daab77b978d068ce7f8f4acc3dbc688f66909d1a31aa', 'native_standing_002/standing/substeps_009.npz': '860abe5b5fba3fb5691c492529c3601f13138885b7644d327d050925918888e3', 'native_standing_002/standing/usd_readback.json': '9a3547d211a7fd456a897ccb54beda88cab027c00c6605532795f3eee01c67b8', 'native_standing_002/standing/warmup_native_state.json': 'ef36393b80071441f3832cfe810bd449069f5d8f1c4bb0cf827dad792b9aa70b', 'native_standing_002/standing_immutable.sha256.json': '557a942273b79d6aebd44b429478966bb67c53ead1196a5f141a63a90df4f5ac', 'forecast_pause_007/launch.json': '2a3101c888fedafb7214c3b5706a635f839c0b7c08d1aca661c9c4a7366a3a75', 'forecast_pause_007/pause.json': '86801daa066c9379851be072ffa56a0c0331685144f4bb3f26bf8d593ecf622e', 'forecast_pause_007/restore.lock': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'forecast_pause_007/restored.json': 'a5c7c52aed7faee85e6713d5ce01ac7bf09eb3460f7291d488760fdd1d487256', 'forecast_pause_007/resume_forecasting.py': '08c4b4cb564035f61731ba957e45cf14c28ceb27410eb4421f53fb3a4af7ae92'}

STANDING_ONE=PREVIOUS_ROOT/'standing'
STANDING_ONE_STATE_SHA256='0d536d6906bfa8e755e971272bca98fed0e709db8f02651680cd95c04248cccd'
STANDING_ONE_INVENTORY_SHA256='849c15536b0d6c3c7a408510608417e8732ced1fa28594bdb662c9fa35a9ebb1'
PREVIOUS_RESTORED_UNIX=1789077818.5469449

def call(args):return subprocess.check_output(args,text=True,timeout=30).strip()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def valid_hash(v):return isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v)
def require_final_bindings():
    for name,bound in {'source':SOURCE_SHA256,'supervisor':SUPERVISOR_SHA256,'host':HOST_SHA256,'host freeze':HOST_FREEZE_SHA256,'previous audit':PREVIOUS_AUDIT_SHA256,'standing1 state':STANDING_ONE_STATE_SHA256,'standing1 inventory':STANDING_ONE_INVENTORY_SHA256,**PRIOR_PINS}.items():
        if not valid_hash(bound):raise RuntimeError('Final reviewed binding pending: '+name)

def verify_manifest(root,manifest,bound):
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):raise RuntimeError('Symbolic frozen source')
    if sha(root/manifest)!=bound:raise RuntimeError('Wrong frozen manifest')
    actual={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p!=root/manifest}
    if actual!=json.loads((root/manifest).read_text()):raise RuntimeError('Changed/unlisted frozen source')

def verify_previous_owner():
    audit_path=Path(__file__).resolve().parent/'inputs/previous_terminal_audit.json'
    if sha(audit_path)!=PREVIOUS_AUDIT_SHA256:raise RuntimeError('Prior admission audit changed')
    audit=json.loads(audit_path.read_text())
    if audit.get('audit_verified')is not True or audit.get('terminal_outcome')!='authentic_completed_standing' or audit.get('standing_completed')is not True or audit.get('expected_invocation')!=PREVIOUS_INVOCATION or audit.get('errors')!=[]:raise RuntimeError('Prior standing1 admission invalid')
    fields=dict(line.split('=',1)for line in call(['systemctl','--user','show',PREVIOUS_UNIT,'-p','ActiveState','-p','SubState','-p','InvocationID','-p','MainPID','-p','ExecMainStatus']).splitlines()if '='in line)
    if fields.get('ActiveState')!='inactive' or fields.get('MainPID')!='0' or fields.get('ExecMainStatus')!='0':raise RuntimeError('Previous completed owner state changed')
    if fields.get('InvocationID') and fields['InvocationID']!=PREVIOUS_INVOCATION:raise RuntimeError('Previous invocation changed')
    for name,bound in PRIOR_PINS.items():
        if sha(PRIOR_BASE/name)!=bound:raise RuntimeError('Previous admission/cleanup receipt changed: '+name)
    campaign=json.loads((PREVIOUS_ROOT/'campaign.json').read_text());identity=campaign.get('identity',{})
    if campaign.get('status')!='completed' or campaign.get('terminal_inputs_unchanged')is not True or campaign.get('planned_phases')!=['standing']or campaign.get('host_freeze_sha256')!=PREVIOUS_HOST or campaign.get('source_freeze_sha256')!=PREVIOUS_SOURCE or identity.get('inspector_freeze_sha256')!=SOURCE_SHA256 or identity.get('num_envs')!=1:raise RuntimeError('Previous standing1 campaign identity changed')
    receipt=campaign.get('standing',{})
    if receipt.get('standing_pass')is not True or receipt.get('status')!='completed' or receipt.get('num_envs')!=1 or receipt.get('state_sha256')!=STANDING_ONE_STATE_SHA256 or campaign.get('post_exit_original_inputs_reverified')is not True or campaign.get('post_exit_all_standing_payloads_inventoried')is not True:raise RuntimeError('Previous standing1 acceptance/seal changed')
    native=json.loads((STANDING_ONE/'state.json').read_text());session=json.loads((STANDING_ONE/'session.json').read_text())
    if native.get('status')!='completed' or native.get('standing_pass')is not True or native.get('explicit_steps_completed')!=8000 or native.get('identity')!=identity or native.get('inputs_unchanged')is not True or native.get('errors')!=[] or native.get('physical_admission')is not False or native.get('training_allowed')is not False:raise RuntimeError('Previous raw standing1 state changed')
    if session.get('steps')!=8000 or session.get('captured_steps')!=8000 or session.get('controls')!=1000 or session.get('all_rows_recorded')is not True or session.get('reset_count')!=1 or session.get('failure')is not None:raise RuntimeError('Previous standing1 acquisition differs')
    pause=json.loads((PREVIOUS_PAUSE/'pause.json').read_text());restored=json.loads((PREVIOUS_PAUSE/'restored.json').read_text());timers={name for name,value in pause['units'].items()if name.endswith('.timer')and 'ActiveState=active'in value}
    if pause.get('unit')!=PREVIOUS_UNIT or pause.get('output')!=str(PREVIOUS_ROOT)or set(restored.get('timers',[]))!=timers or restored.get('restored_unix')!=PREVIOUS_RESTORED_UNIX:raise RuntimeError('Previous exact pause007 restoration missing')
    names={p.stem for p in (PREVIOUS_ROOT/'jobs').glob('*.json')if not p.name.endswith('_contact_data_audit.json')}
    if names!={'standing'}:raise RuntimeError('Previous job inventory changed')
    job=json.loads((PREVIOUS_ROOT/'jobs/standing.json').read_text())
    if job.get('status')!='completed'or job.get('phase')!='standing'or job.get('exit_code')!=0 or job.get('cleanup_checked')is not True or job.get('container_name')!=PREVIOUS_NAME or job.get('container_id')!=PREVIOUS_ID:raise RuntimeError('Previous completed owned cleanup incomplete')
    for identifier in (PREVIOUS_NAME,PREVIOUS_ID):
        q=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',identifier],capture_output=True,text=True,timeout=20)
        if q.returncode==0 or not any(v in q.stderr.lower()for v in ('no such object','no such container')):raise RuntimeError('Previous container absence unknown: '+identifier)

def validate_standing_inputs(host):
    require_final_bindings()
    if sha(HOST)!=HOST_SHA256:raise RuntimeError('Canonical host changed')
    verify_manifest(HOST.parent,'FREEZE_SHA256.json',HOST_FREEZE_SHA256)
    verify_manifest(SOURCE,'FREEZE_SHA256.json',SOURCE_SHA256)
    if host.SOURCE_FREEZE!=SOURCE_SHA256 or host.SUPERVISOR_MAP!=SUPERVISOR_SHA256:raise RuntimeError('Host pins do not match canonical guard')
    args=SimpleNamespace(source=SOURCE,asset=ASSET,admission=ADMISSION,supervisor_source=SUPERVISOR_SOURCE,output=OUTPUT,isaaclab=Path('/home/orionh/IsaacLab'),host_freeze_sha256=HOST_FREEZE_SHA256,num_envs=32,standing_one=STANDING_ONE)
    host.require_fresh_output(args);identity=host.verify_inputs(args)
    if identity.get('training_allowed') is not False or identity.get('physical_admission') is not False or not valid_hash(identity.get('runtime_binding',{}).get('runtime_tree_sha256')):raise RuntimeError('Only identified canonical coordinate/effort diagnosis is allocated')
    if identity.get('schema')!='canonical_native_standing_v1' or identity.get('runtime_binding',{}).get('scope')!='canonical_provisional_native_standing_only' or identity.get('steps')!=8000 or identity.get('num_envs')!=32 or identity.get('controls')!=1000:raise RuntimeError('Wrong diagnostic phase or scope')
    if identity.get('standing_one_state_sha256')!=STANDING_ONE_STATE_SHA256 or identity.get('standing_one_inventory_sha256')!=STANDING_ONE_INVENTORY_SHA256:raise RuntimeError('Wrong same-source standing1 admission')
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
    validate_standing_inputs(train_host)
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
            output=str(OUTPUT), source=str(SOURCE), admission=str(ADMISSION), standing_one=str(STANDING_ONE), preflight=snapshot,
            coordination_sha256=hashlib.sha256(coordination).hexdigest(),
            reason='Canonical provisional standing32:1000 controls/8000 explicit400Hz steps; exact asset, completed coordinate/effort and same-source standing1 evidence; no policy or broader physical admission',
            current_instruction='10 September canonical detailed direct-drive model is authoritative; explicit all-weather priority applies, unknown non-weather work remains protected')
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
        subprocess.run(['systemd-run', '--user', '--unit=hexapod-canonical-forecast-restore-008', '--on-active=15m',
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
        validate_standing_inputs(train_host)
        verify_previous_owner()
    except Exception:
        subprocess.run(['/usr/bin/python3',str(restorer)],check=True,timeout=120)
        raise
    try:
        cmd=['systemd-run','--user','--unit='+UNIT,'--property=RuntimeMaxSec=720',
            '--property=TimeoutStopSec=180','--property=KillMode=process','--property=Environment=PYTHONDONTWRITEBYTECODE=1',
            '--property=ExecStopPost=/usr/bin/python3 '+str(restorer),'--property=Environment=PYTHONUNBUFFERED=1',
            '/usr/bin/python3',str(HOST),'--source',str(SOURCE),'--asset',str(ASSET),'--admission',str(ADMISSION),
            '--supervisor-source',str(SUPERVISOR_SOURCE),'--output',str(OUTPUT),'--num-envs','32','--standing-one',str(STANDING_ONE)]
        subprocess.run(cmd,check=True,timeout=30)
        launched=True
        (PAUSE/'launch.json').write_text(json.dumps(dict(command=cmd,launched_unix=time.time(),native_source_freeze_sha256=SOURCE_SHA256),indent=2)+'\n')
        print(json.dumps(dict(unit=UNIT,source=str(SOURCE),output=str(OUTPUT),pause=str(PAUSE))))
    finally:
        if not launched:
            subprocess.run(['/usr/bin/python3',str(restorer),'--stop-owner'],check=True,timeout=300)

if __name__=='__main__':main()
