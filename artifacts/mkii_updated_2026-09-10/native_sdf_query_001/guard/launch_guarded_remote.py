"""Canonical native SDF query guard; prior coordinate/effort test supplies ownership lineage only."""
from pathlib import Path
from types import SimpleNamespace
import fcntl,hashlib,importlib.util,json,os,subprocess,sys,time
sys.dont_write_bytecode=True
BASE=Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910')
PRIOR_BASE=BASE
SOURCE=BASE/'query_source_002'
ASSET=BASE/'asset_001'
ADMISSION=BASE/'native_inspection_003/inspection'
SUPERVISOR_SOURCE=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_physics_source_009')
OUTPUT=BASE/'native_query_001'
PAUSE=BASE/'forecast_pause_005'
UNIT='hexapod-canonical-native-query-001-20260910.service'
HOST=BASE/'query_host_001/launch_query_spark.py'
SOURCE_SHA256='174a0c23eb351816abc9fea72acacaf8f40173baa40f73bbe69f70bc71e785f2'
SUPERVISOR_SHA256='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
HOST_SHA256='9737b48897f97e09cfae0da1453a3eee727092853e93f8994918db0d70f3853f'
HOST_FREEZE_SHA256='b919fc90924da8765fb32510730392e42b94b7b5fa41d832272a6912f60509ae'
PREVIOUS_UNIT='hexapod-canonical-native-actuation-001-20260910.service'
PREVIOUS_INVOCATION='c6d287881452428d90e8e432ae739c86'
PREVIOUS_ROOT=BASE/'native_actuation_001'
PREVIOUS_PAUSE=BASE/'forecast_pause_004'
PREVIOUS_HOST='0f9a54963ec3220d0625722e588bb17c3efd534b0c5ac04c1f482db69511163e'
PREVIOUS_SOURCE='1f02009cb22c1efbe4a0d0c7f7988d9d63b658d78e509ea444b7afafee6178c9'
PREVIOUS_NAME='hexapod-reference-physics-b9fe5bdb12364a908682c3adacadd6cf'
PREVIOUS_ID='9ec8f6e14a537a7491935c1952cff690521cb414130cc9cc51fc21c7b1c3176f'
PREVIOUS_AUDIT_SHA256='695583b751c1a768256ffce5ba6ae781aa2ca0e4735d73d95345466e4f17bd45'
PRIOR_PINS={'native_actuation_001/actuation/case_results.json': '89255b6b94b36134e08dfe17001a217f5d6a1833cb82bd630c26ef31b7c4be37', 'native_actuation_001/actuation/coordinate_results.json': 'e4ceae5d99e62f6549f018e24597066ee3a7e060b964c1b1dacd1117375683bb', 'native_actuation_001/actuation/experiment.jsonl': '45ea952bc7ae1eb251d6491dd2dbadf36e4e6d15c50d54dc4499b82dfbb9d552', 'native_actuation_001/actuation/experiment_summary.json': '60ae81953a0b381c7fdf4048781f9261372fa42885bf5c913788d8e9f3fa9dd7', 'native_actuation_001/actuation/mass_matrix_readback.json': 'e0f2961221e8203f4a2bbf5bd80b3462f3272ab386d23e1f698b7b2309401bb5', 'native_actuation_001/actuation/motion_diagnostic.json': 'd7e799c42e86e42c1a23fc1e8d07947409a695db9d0a2bd9c8a3467395106dd5', 'native_actuation_001/actuation/native_errors.json': '37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570', 'native_actuation_001/actuation/native_readback.json': 'd37ab0a618739c5ac77a3f5e412f8dd558081a23f03b1ede1c98578251d6aca4', 'native_actuation_001/actuation/native_scene.json': '64ca46b0c699cf8cc23bb5e549fd69550c298e54939fe03bdcfbaef342bf8f4d', 'native_actuation_001/actuation/reset_readbacks.jsonl': '5b91e068c38cb278e274669e2018a4ec23a0d8e5810da9b62d77faec08f7904c', 'native_actuation_001/actuation/resolved_after_stage.usda': '48aa034a13fe7f4d3c2f1e896fc6fbfd2387a1bf868f77b1e42db1a57522f5df', 'native_actuation_001/actuation/resolved_stage.usda': '48aa034a13fe7f4d3c2f1e896fc6fbfd2387a1bf868f77b1e42db1a57522f5df', 'native_actuation_001/actuation/runtime_api.json': 'fa0e49c910c02110c70c57ace66e717cc88bed46a9dcead9ca5b0a8c8b51f313', 'native_actuation_001/actuation/samples.json': '1d0202dab44cf41161e599f44753d87eab67e765935135bc39f360855111f597', 'native_actuation_001/actuation/sdf_readback.json': '1006fc8ac80b91ec0e7bc269dab999dc8f3b22b35801bf060967ff4a4cbe4f97', 'native_actuation_001/actuation/sdk_readback.json': '1016fef5aa047bdd0e5684a9fc8acd599e95738601847ad4084f2be58c53c454', 'native_actuation_001/actuation/simulation_config.json': '82217827c5d3969f7537282d724412cb16e1191bb5ef05bc461f222749f3b51f', 'native_actuation_001/actuation/state.json': '5a561006d0e63f2f69b7d4129836a33a55b5a9c5a2b576bdd91fb112fe845a8b', 'native_actuation_001/actuation/usd_readback.json': '9a3547d211a7fd456a897ccb54beda88cab027c00c6605532795f3eee01c67b8', 'native_actuation_001/actuation_immutable.sha256.json': '1ccd8f4ffaa2eadc1fd9c6202af8da965b0e9b95f39c585b68a3a64313ce2728', 'native_actuation_001/campaign.json': '32e74b3954b7ce2109784c0e819e7c0cf6470faec1f21eec5809987061b38650', 'native_actuation_001/jobs/actuation.json': '000bb465a8b7ab282999924fe18d78447ed630f48d195db6139579bd3dd5370c', 'native_actuation_001/jobs/actuation_contact_data_audit.json': 'f526be6160713c7f48befc8cb44833ce5f985ce317cf1900c4711d0a16119ee8', 'native_actuation_001/logs/actuation.log': 'eae846a240fb45cc8299bbdeb3133d9b08a445b25af14d6a4ae4715d33d648f0', 'forecast_pause_004/launch.json': 'd83f32cdf7be990f15e2efc1b06a04dd6d960c001ca6d6f1d17359acf624aa9a', 'forecast_pause_004/pause.json': 'd621bfd785c916e3f99adc2d12afa260fbee04c463488bec4027f534c8bfddc2', 'forecast_pause_004/restore.lock': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'forecast_pause_004/restored.json': '1198a1d141b309dcf86505e386a209f777097ca4728344e58e69a22b4023464e', 'forecast_pause_004/resume_forecasting.py': '08c4b4cb564035f61731ba957e45cf14c28ceb27410eb4421f53fb3a4af7ae92'}

def call(args):return subprocess.check_output(args,text=True,timeout=30).strip()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def valid_hash(v):return isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v)
def require_final_bindings():
    for name,bound in {'source':SOURCE_SHA256,'supervisor':SUPERVISOR_SHA256,'host':HOST_SHA256,'host freeze':HOST_FREEZE_SHA256,**PRIOR_PINS}.items():
        if not valid_hash(bound):raise RuntimeError('Final reviewed binding pending: '+name)

def verify_manifest(root,manifest,bound):
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):raise RuntimeError('Symbolic frozen source')
    if sha(root/manifest)!=bound:raise RuntimeError('Wrong frozen manifest')
    actual={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p!=root/manifest}
    if actual!=json.loads((root/manifest).read_text()):raise RuntimeError('Changed/unlisted frozen source')

def verify_previous_owner():
    # Completed coordinate/effort test is ownership lineage, never physical admission.
    audit_path=Path(__file__).resolve().parent/'inputs/previous_terminal_audit.json'
    if sha(audit_path)!=PREVIOUS_AUDIT_SHA256:raise RuntimeError('Prior completed-owner audit changed')
    audit=json.loads(audit_path.read_text())
    if audit.get('audit_verified') is not True or audit.get('terminal_outcome')!='authentic_completed_actuation' or audit.get('actuation_completed') is not True or audit.get('expected_invocation')!=PREVIOUS_INVOCATION:raise RuntimeError('Prior completion provenance invalid')
    fields=dict(line.split('=',1) for line in call(['systemctl','--user','show',PREVIOUS_UNIT,'-p','ActiveState','-p','SubState','-p','InvocationID','-p','MainPID','-p','ExecMainStatus']).splitlines() if '=' in line)
    if fields.get('ActiveState')!='inactive' or fields.get('MainPID')!='0' or fields.get('ExecMainStatus')!='0':raise RuntimeError('Previous completed actuation owner not inactive exit0')
    if fields.get('InvocationID') and fields['InvocationID']!=PREVIOUS_INVOCATION:raise RuntimeError('Previous inspection invocation changed')
    for name,bound in PRIOR_PINS.items():
        if sha(PRIOR_BASE/name)!=bound:raise RuntimeError('Previous completion/cleanup receipt changed: '+name)
    campaign=json.loads((PREVIOUS_ROOT/'campaign.json').read_text());identity=campaign.get('identity',{})
    if campaign.get('status')!='completed' or campaign.get('terminal_inputs_unchanged') is not True or campaign.get('post_exit_original_inputs_reverified') is not True or campaign.get('post_exit_all_actuation_payloads_inventoried') is not True or campaign.get('planned_phases')!=['actuation'] or campaign.get('host_freeze_sha256')!=PREVIOUS_HOST or campaign.get('source_freeze_sha256')!=PREVIOUS_SOURCE or identity.get('inspector_freeze_sha256')!=PREVIOUS_SOURCE:raise RuntimeError('Previous completed campaign identity changed')
    native=json.loads((PREVIOUS_ROOT/'actuation/state.json').read_text())
    if native.get('status')!='completed' or native.get('explicit_steps_completed')!=2064 or native.get('inputs_unchanged') is not True or native.get('identity')!=identity or native.get('physical_admission') is not False or native.get('training_allowed') is not False or native.get('errors')!=[]:raise RuntimeError('Previous native completion state changed')
    pause=json.loads((PREVIOUS_PAUSE/'pause.json').read_text());restored=json.loads((PREVIOUS_PAUSE/'restored.json').read_text())
    timers={name for name,value in pause['units'].items() if name.endswith('.timer') and 'ActiveState=active' in value}
    if pause.get('unit')!=PREVIOUS_UNIT or pause.get('output')!=str(PREVIOUS_ROOT) or set(restored.get('timers',[]))!=timers or restored.get('restored_unix')!=1789074862.1635034:raise RuntimeError('Previous exact pause004 restoration missing')
    names={p.stem for p in (PREVIOUS_ROOT/'jobs').glob('*.json') if not p.name.endswith('_contact_data_audit.json')}
    if names!={'actuation'}:raise RuntimeError('Previous job inventory changed')
    job=json.loads((PREVIOUS_ROOT/'jobs/actuation.json').read_text())
    if job.get('status')!='completed' or job.get('phase')!='actuation' or job.get('exit_code')!=0 or job.get('cleanup_checked') is not True or job.get('container_name')!=PREVIOUS_NAME or job.get('container_id')!=PREVIOUS_ID:raise RuntimeError('Previous completed owned cleanup incomplete')
    for identifier in (PREVIOUS_NAME,PREVIOUS_ID):
        q=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',identifier],capture_output=True,text=True,timeout=20)
        if q.returncode==0 or not any(v in q.stderr.lower() for v in ('no such object','no such container')):raise RuntimeError('Previous container absence unknown: '+identifier)

def validate_query_inputs(host):
    require_final_bindings()
    if sha(HOST)!=HOST_SHA256:raise RuntimeError('Canonical host changed')
    verify_manifest(HOST.parent,'FREEZE_SHA256.json',HOST_FREEZE_SHA256)
    verify_manifest(SOURCE,'FREEZE_SHA256.json',SOURCE_SHA256)
    if host.SOURCE_FREEZE!=SOURCE_SHA256 or host.SUPERVISOR_MAP!=SUPERVISOR_SHA256:raise RuntimeError('Host pins do not match canonical guard')
    args=SimpleNamespace(source=SOURCE,asset=ASSET,admission=ADMISSION,supervisor_source=SUPERVISOR_SOURCE,output=OUTPUT,isaaclab=Path('/home/orionh/IsaacLab'),host_freeze_sha256=HOST_FREEZE_SHA256)
    host.require_fresh_output(args);identity=host.verify_inputs(args)
    if identity.get('training_allowed') is not False or identity.get('physical_admission') is not False or not valid_hash(identity.get('runtime_binding',{}).get('runtime_tree_sha256')):raise RuntimeError('Only identified canonical coordinate/effort diagnosis is allocated')
    if identity.get('schema')!='canonical_native_sdf_query_v1' or identity.get('runtime_binding',{}).get('scope')!='canonical_native_sdf_query_only' or identity.get('steps')!=8:raise RuntimeError('Wrong diagnostic phase or scope')
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
    validate_query_inputs(train_host)
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
            output=str(OUTPUT), source=str(SOURCE), admission=str(ADMISSION), preflight=snapshot,
            coordination_sha256=hashlib.sha256(coordination).hexdigest(),
            reason='Canonical SDF acquisition:8 import steps and12 read-only query calls; no query step/state writes, no physical or training admission',
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
        subprocess.run(['systemd-run', '--user', '--unit=hexapod-canonical-forecast-restore-005', '--on-active=15m',
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
        validate_query_inputs(train_host)
        verify_previous_owner()
    except Exception:
        subprocess.run(['/usr/bin/python3',str(restorer)],check=True,timeout=120)
        raise
    try:
        cmd=['systemd-run','--user','--unit='+UNIT,'--property=RuntimeMaxSec=720',
            '--property=TimeoutStopSec=180','--property=KillMode=process','--property=Environment=PYTHONDONTWRITEBYTECODE=1',
            '--property=ExecStopPost=/usr/bin/python3 '+str(restorer),'--property=Environment=PYTHONUNBUFFERED=1',
            '/usr/bin/python3',str(HOST),'--source',str(SOURCE),'--asset',str(ASSET),'--admission',str(ADMISSION),
            '--supervisor-source',str(SUPERVISOR_SOURCE),'--output',str(OUTPUT)]
        subprocess.run(cmd,check=True,timeout=30)
        launched=True
        (PAUSE/'launch.json').write_text(json.dumps(dict(command=cmd,launched_unix=time.time(),native_source_freeze_sha256=SOURCE_SHA256),indent=2)+'\n')
        print(json.dumps(dict(unit=UNIT,source=str(SOURCE),output=str(OUTPUT),pause=str(PAUSE))))
    finally:
        if not launched:
            subprocess.run(['/usr/bin/python3',str(restorer),'--stop-owner'],check=True,timeout=300)

if __name__=='__main__':main()
