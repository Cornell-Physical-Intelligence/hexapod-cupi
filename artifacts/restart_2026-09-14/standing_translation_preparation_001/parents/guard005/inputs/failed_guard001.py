"""Canonical native SDF standing guard; prior coordinate/effort test supplies ownership lineage only."""
from pathlib import Path
from types import SimpleNamespace
import fcntl,hashlib,importlib.util,json,os,subprocess,sys,time
sys.dont_write_bytecode=True
BASE=Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910')
PRIOR_BASE=BASE
SOURCE=BASE/'standing_source_001'
ASSET=BASE/'asset_001'
ADMISSION=BASE/'native_actuation_001/actuation'
SUPERVISOR_SOURCE=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_physics_source_009')
OUTPUT=BASE/'native_standing_001'
PAUSE=BASE/'forecast_pause_006'
UNIT='hexapod-canonical-native-standing-001-20260910.service'
HOST=BASE/'standing_host_001/launch_standing_spark.py'
SOURCE_SHA256='f81f61d257cb92b65ad80f3b7d27c45497e66254092e244a32da02555b11e2a4'
SUPERVISOR_SHA256='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
HOST_SHA256='31c69e8df516a29fb6865e65989042cf9c07af37d6ff7a7d75b4be9b052577b6'
HOST_FREEZE_SHA256='30235851f0dc95ce0c36ac3ec6912db144a7d3d6a6ab21c023d70d7cfefc2a33'
PREVIOUS_UNIT='hexapod-canonical-native-query-001-20260910.service'
PREVIOUS_INVOCATION='8304b4418c734871a21cad8f2fe4184e'
PREVIOUS_ROOT=BASE/'native_query_001'
PREVIOUS_PAUSE=BASE/'forecast_pause_005'
PREVIOUS_HOST='b919fc90924da8765fb32510730392e42b94b7b5fa41d832272a6912f60509ae'
PREVIOUS_SOURCE='174a0c23eb351816abc9fea72acacaf8f40173baa40f73bbe69f70bc71e785f2'
PREVIOUS_NAME='hexapod-reference-physics-91ca7b3608ed48e280d02b3d83bff449'
PREVIOUS_ID='0e18d4a064880a09f84f677386aff978d4965149cca851262b4837c15a50b5b8'
PREVIOUS_AUDIT_SHA256='c9cac8e63733e3fdf4ba2c7fe2ec7b564f2b2bd91902eaa12a8892548134ef72'
PRIOR_PINS={'native_query_001/campaign.json': '62aa26a18d8456a8a50527f2a1cfa4afe790fd64b3e3436eba10a2706a9c476f', 'native_query_001/jobs/query.json': 'd75301d877aa3ae6eb209d6ce401c5ee0aa629f181968c8720fff02911fa97fc', 'native_query_001/jobs/query_contact_data_audit.json': '1a2be60a29973c8f85fa1e5f80653b762506dab2132501ea1a790f97b51f9fac', 'native_query_001/logs/query.log': 'e369329e1b330f6d427398ab52d148fcce6bd420f8dcb469fe255e160ab50c91', 'native_query_001/query/motion_diagnostic.json': 'd7e799c42e86e42c1a23fc1e8d07947409a695db9d0a2bd9c8a3467395106dd5', 'native_query_001/query/native_errors.json': '37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570', 'native_query_001/query/native_readback.json': 'd37ab0a618739c5ac77a3f5e412f8dd558081a23f03b1ede1c98578251d6aca4', 'native_query_001/query/native_scene.json': 'c316f73de2e5b065208091efbb8b598603e0db80bf3166697e7e194d06f57251', 'native_query_001/query/query/after.json': '07e020cf273efe826bd7e3027896264e76374b991b38c5a8a26ebf9a33a3ec02', 'native_query_001/query/query/before.json': '07e020cf273efe826bd7e3027896264e76374b991b38c5a8a26ebf9a33a3ec02', 'native_query_001/query/query/frame_hypotheses.json': '8513d71199c8524abc3a07b09e95aa6458365cb708bbb206f626d36c2a2c8421', 'native_query_001/query/query/query_0.json': '4a06251134ab2782a3646815497d2e75514a3af7a21d64d95dbb1080c8953e68', 'native_query_001/query/query/query_0.npz': '2efa1dd6c638b7b219611525a6e91b80ae947c010969896d410b16970677bc20', 'native_query_001/query/query/query_1.json': '9216bd4d07a4915ce422dbddda84c22c9bcd91cf4c3eb0b5a8cfdfa822220761', 'native_query_001/query/query/query_1.npz': '4f92e17078ca02b26fab74a3d44567994605cab483176ab3de45ee716debca9d', 'native_query_001/query/query/query_2.json': 'f89e5e5c92c7e7617723d8ea80797721ecac7c4e91208ec4c9338f5e485bd36e', 'native_query_001/query/query/query_2.npz': '1166fde7c70ac8981ee6e536f12bb79449517a7b323d64f280bcbe5fd42c26d5', 'native_query_001/query/query/query_3.json': '6df923731f02c79ffa5cfd7874e41f46ec0b1e634934ec125cc9f89ebcf379cc', 'native_query_001/query/query/query_3.npz': '51cf91b5a75b9102fbfc3c167056e2372878d173c86db23aa500232b452731a7', 'native_query_001/query/query/query_4.json': 'beb26cd9b75e32883b71bd643abd0c4dd7f10414cf5383b5696e5132edf02701', 'native_query_001/query/query/query_4.npz': '8120e67fe21da3a69e70380f777ead0a756064341d1d497737266a4d3e184f64', 'native_query_001/query/query/query_5.json': '78e3be71f1da1ad2ca286ba4f254afb58238892116ff8f348312a04247359a7e', 'native_query_001/query/query/query_5.npz': 'c12595194efa6493fdd3314039885508f78cce076372300c0d1a6bf036143f99', 'native_query_001/query/query/report.json': 'c5871d381ee1bb0dd56f38da471df772036ebd4736349a8c0cd1c2e0006706d4', 'native_query_001/query/query/state.json': '3a73ee557a6155a12ee7a10162f406d2dfa0b60ac199d7db71655b2d1d0b7af4', 'native_query_001/query/query/view_0.json': '67199b8da45d84496d11bc2e338af5a074c09e16987e24a49d75c5a8fe7c3514', 'native_query_001/query/query/view_1.json': '9e56d016ae5c556a1398009f470a9287308f4ae12f075fc4ec90d48e0e70e4b2', 'native_query_001/query/query/view_2.json': 'cb651e9e36f69d76663868470800c4f3861694779c9f134917fa705698a149d3', 'native_query_001/query/query/view_3.json': '7e170ab8e2fc951c41cd89b5b3be8f4334cd793b7bdf5013d41340d9e813efda', 'native_query_001/query/query/view_4.json': '8201f026742f23b9061466649a19b13e978660c5f10d7f607a45106227dce8ef', 'native_query_001/query/query/view_5.json': 'b3a0c469702c81c18e90638b10bdca4eef1d7f042f58066aaa58da19014672ab', 'native_query_001/query/resolved_after_stage.usda': '4ab2e348e6af0bb63769c23a0a532fd4ed94175d8462a5dedeed96c6a0f314ad', 'native_query_001/query/resolved_stage.usda': '4ab2e348e6af0bb63769c23a0a532fd4ed94175d8462a5dedeed96c6a0f314ad', 'native_query_001/query/runtime_api.json': 'fa0e49c910c02110c70c57ace66e717cc88bed46a9dcead9ca5b0a8c8b51f313', 'native_query_001/query/samples.json': '1d0202dab44cf41161e599f44753d87eab67e765935135bc39f360855111f597', 'native_query_001/query/sdf_readback.json': '1006fc8ac80b91ec0e7bc269dab999dc8f3b22b35801bf060967ff4a4cbe4f97', 'native_query_001/query/sdk_readback.json': '1016fef5aa047bdd0e5684a9fc8acd599e95738601847ad4084f2be58c53c454', 'native_query_001/query/simulation_config.json': '64a5dcab3e3733501e226826e186710f63f52dc0326ab9b391bd5c16bfb9b644', 'native_query_001/query/state.json': '7c3b3a018f7e390903ee8015b891a1c399199fe71e2e28288bb333cfa0c97d32', 'native_query_001/query/usd_readback.json': '9a3547d211a7fd456a897ccb54beda88cab027c00c6605532795f3eee01c67b8', 'native_query_001/query_immutable.sha256.json': 'ca20618ee866013aa2782d5eb755594e0bb5e537c5b83e5be27b894f7c2db38d', 'forecast_pause_005/launch.json': '90f8ebcbe80ede09492b9bc5be1dc6f756267f1c92e3614b99af3e0c5077fee8', 'forecast_pause_005/pause.json': 'afaca1c896e157c09cd407e5ef3683f0397b8db18c6bc0cba94aa62e99b073ea', 'forecast_pause_005/restore.lock': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'forecast_pause_005/restored.json': 'c3e7e9717e3038f238a9af51fc6a364eebebd8c2daaefb110005968517c90d1f', 'forecast_pause_005/resume_forecasting.py': '08c4b4cb564035f61731ba957e45cf14c28ceb27410eb4421f53fb3a4af7ae92'}

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
    # Completed SDF acquisition is ownership lineage, never physical admission.
    audit_path=Path(__file__).resolve().parent/'inputs/previous_terminal_audit.json'
    if sha(audit_path)!=PREVIOUS_AUDIT_SHA256:raise RuntimeError('Prior completed-owner audit changed')
    audit=json.loads(audit_path.read_text())
    if audit.get('audit_verified') is not True or audit.get('terminal_outcome')!='authentic_completed_query' or audit.get('query_completed') is not True or audit.get('expected_invocation')!=PREVIOUS_INVOCATION:raise RuntimeError('Prior completion provenance invalid')
    fields=dict(line.split('=',1) for line in call(['systemctl','--user','show',PREVIOUS_UNIT,'-p','ActiveState','-p','SubState','-p','InvocationID','-p','MainPID','-p','ExecMainStatus']).splitlines() if '=' in line)
    if fields.get('ActiveState')!='inactive' or fields.get('MainPID')!='0' or fields.get('ExecMainStatus')!='0':raise RuntimeError('Previous completed query owner not inactive exit0')
    if fields.get('InvocationID') and fields['InvocationID']!=PREVIOUS_INVOCATION:raise RuntimeError('Previous inspection invocation changed')
    for name,bound in PRIOR_PINS.items():
        if sha(PRIOR_BASE/name)!=bound:raise RuntimeError('Previous completion/cleanup receipt changed: '+name)
    campaign=json.loads((PREVIOUS_ROOT/'campaign.json').read_text());identity=campaign.get('identity',{})
    if campaign.get('status')!='completed' or campaign.get('terminal_inputs_unchanged') is not True or campaign.get('post_exit_original_inputs_reverified') is not True or campaign.get('post_exit_all_query_payloads_inventoried') is not True or campaign.get('planned_phases')!=['query'] or campaign.get('host_freeze_sha256')!=PREVIOUS_HOST or campaign.get('source_freeze_sha256')!=PREVIOUS_SOURCE or identity.get('inspector_freeze_sha256')!=PREVIOUS_SOURCE:raise RuntimeError('Previous completed campaign identity changed')
    native=json.loads((PREVIOUS_ROOT/'query/state.json').read_text())
    if native.get('status')!='completed' or native.get('explicit_steps_completed')!=8 or native.get('inputs_unchanged') is not True or native.get('identity')!=identity or native.get('physical_admission') is not False or native.get('training_allowed') is not False or native.get('errors')!=[]:raise RuntimeError('Previous native completion state changed')
    pause=json.loads((PREVIOUS_PAUSE/'pause.json').read_text());restored=json.loads((PREVIOUS_PAUSE/'restored.json').read_text())
    timers={name for name,value in pause['units'].items() if name.endswith('.timer') and 'ActiveState=active' in value}
    if pause.get('unit')!=PREVIOUS_UNIT or pause.get('output')!=str(PREVIOUS_ROOT) or set(restored.get('timers',[]))!=timers or restored.get('restored_unix')!=1789075815.8519926:raise RuntimeError('Previous exact pause005 restoration missing')
    names={p.stem for p in (PREVIOUS_ROOT/'jobs').glob('*.json') if not p.name.endswith('_contact_data_audit.json')}
    if names!={'query'}:raise RuntimeError('Previous job inventory changed')
    job=json.loads((PREVIOUS_ROOT/'jobs/query.json').read_text())
    if job.get('status')!='completed' or job.get('phase')!='query' or job.get('exit_code')!=0 or job.get('cleanup_checked') is not True or job.get('container_name')!=PREVIOUS_NAME or job.get('container_id')!=PREVIOUS_ID:raise RuntimeError('Previous completed owned cleanup incomplete')
    for identifier in (PREVIOUS_NAME,PREVIOUS_ID):
        q=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',identifier],capture_output=True,text=True,timeout=20)
        if q.returncode==0 or not any(v in q.stderr.lower() for v in ('no such object','no such container')):raise RuntimeError('Previous container absence unknown: '+identifier)

def validate_standing_inputs(host):
    require_final_bindings()
    if sha(HOST)!=HOST_SHA256:raise RuntimeError('Canonical host changed')
    verify_manifest(HOST.parent,'FREEZE_SHA256.json',HOST_FREEZE_SHA256)
    verify_manifest(SOURCE,'FREEZE_SHA256.json',SOURCE_SHA256)
    if host.SOURCE_FREEZE!=SOURCE_SHA256 or host.SUPERVISOR_MAP!=SUPERVISOR_SHA256:raise RuntimeError('Host pins do not match canonical guard')
    args=SimpleNamespace(source=SOURCE,asset=ASSET,admission=ADMISSION,supervisor_source=SUPERVISOR_SOURCE,output=OUTPUT,isaaclab=Path('/home/orionh/IsaacLab'),host_freeze_sha256=HOST_FREEZE_SHA256,num_envs=1,standing_one=None)
    host.require_fresh_output(args);identity=host.verify_inputs(args)
    if identity.get('training_allowed') is not False or identity.get('physical_admission') is not False or not valid_hash(identity.get('runtime_binding',{}).get('runtime_tree_sha256')):raise RuntimeError('Only identified canonical coordinate/effort diagnosis is allocated')
    if identity.get('schema')!='canonical_native_standing_v1' or identity.get('runtime_binding',{}).get('scope')!='canonical_provisional_native_standing_only' or identity.get('steps')!=8000 or identity.get('num_envs')!=1 or identity.get('controls')!=1000:raise RuntimeError('Wrong diagnostic phase or scope')
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
        gpu = call(['nvidia-smi', '--standing-compute-apps=pid,process_name', '--format=csv,noheader,nounits'])
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
            reason='Canonical provisional standing1:1000 controls/8000 explicit400Hz steps; exact asset and completed coordinate/effort evidence; no policy or broader physical admission',
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
        subprocess.run(['systemd-run', '--user', '--unit=hexapod-canonical-forecast-restore-006', '--on-active=15m',
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
            '--supervisor-source',str(SUPERVISOR_SOURCE),'--output',str(OUTPUT),'--num-envs','1']
        subprocess.run(cmd,check=True,timeout=30)
        launched=True
        (PAUSE/'launch.json').write_text(json.dumps(dict(command=cmd,launched_unix=time.time(),native_source_freeze_sha256=SOURCE_SHA256),indent=2)+'\n')
        print(json.dumps(dict(unit=UNIT,source=str(SOURCE),output=str(OUTPUT),pause=str(PAUSE))))
    finally:
        if not launched:
            subprocess.run(['/usr/bin/python3',str(restorer),'--stop-owner'],check=True,timeout=300)

if __name__=='__main__':main()
