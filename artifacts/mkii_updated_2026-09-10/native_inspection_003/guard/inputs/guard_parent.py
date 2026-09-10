"""Canonical native inspection guard. Prior failed canonical inspection supplies cleanup ancestry only."""
from pathlib import Path
from types import SimpleNamespace
import fcntl,hashlib,importlib.util,json,os,subprocess,sys,time
sys.dont_write_bytecode=True
BASE=Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910')
PRIOR_BASE=BASE
SOURCE=BASE/'inspection_source_002'
ASSET=BASE/'asset_001'
SUPERVISOR_SOURCE=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_physics_source_009')
OUTPUT=BASE/'native_inspection_002'
PAUSE=BASE/'forecast_pause_002'
UNIT='hexapod-canonical-native-inspection-002-20260910.service'
HOST=BASE/'inspection_host_002/launch_inspection_spark.py'
SOURCE_SHA256='a9a4dbe0e28c0ed85a5d5d6dd7788bdf8bc1b9154f676aa7262fc6eb4beac3f9'
SUPERVISOR_SHA256='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
HOST_SHA256='fb3a2ac8c9eea0fbd4eb67a38464e5684ae22c3546d6b55563791e9636240be8'
HOST_FREEZE_SHA256='3b0d22016e653b53fade516b5d55402cada4624b3c60296f27e044bf1083835e'
PREVIOUS_UNIT='hexapod-canonical-native-inspection-001-20260910.service'
PREVIOUS_INVOCATION='87a75ef9d130496a9e040b86b14367e9'
PREVIOUS_ROOT=BASE/'native_inspection_001'
PREVIOUS_PAUSE=BASE/'forecast_pause_001'
PREVIOUS_HOST='2d21500ae7273ecf70576b577772c38844915a05f0fe67b8e2c213137d9994a1'
PREVIOUS_SOURCE='c59ebd53bb221fd17b46d9fbc31e3df7240271bfd13d6f7269d7307876c7bfaa'
PREVIOUS_NAME='hexapod-reference-physics-9b1041b391234997a46e0565571d8fe7'
PREVIOUS_ID='e745693e7a4377eb13599c1f768b16508ff288715e91de1802067cb19beb3354'
PREVIOUS_AUDIT_SHA256='b402ad44c0f15b39d567f6685eaaf50f6082aea0831139ac57410a738e5a1e71'
PRIOR_PINS={'native_inspection_001/campaign.json': 'ae046c0f7916ff44ac39d3d0a29a9dac14b97c3af161af37c19f66e0bd0290d6', 'native_inspection_001/inspection/failure.json': '47371ee5f20c9e0b5a6e88415f6a807eae5a129c38de398e8da1abe06eca49b9', 'native_inspection_001/inspection/sdk_readback.json': '1016fef5aa047bdd0e5684a9fc8acd599e95738601847ad4084f2be58c53c454', 'native_inspection_001/inspection/state.json': 'f8e3ce6536505c4a216a5d3c7238ccc9f66ce2903e7bb96e2e0168d5268b0429', 'native_inspection_001/inspection/traceback.txt': '45a1ad33384ec3e6c78cbbf5ebbc24c365160e4206b4af64e7e3d437178ae613', 'native_inspection_001/jobs/inspection.json': 'cd9922d3c1293b6be3a28b72f8a53cc84a6381edc50d61dc8ffb458570a30e61', 'native_inspection_001/jobs/inspection_contact_data_audit.json': '0a7296ef2c2802373ae4756f4c75639370828ec67aaee350204c691babc07e41', 'native_inspection_001/logs/inspection.log': 'b1f998d59c586915ecf353d639733ee7690f316e4163ac887a817d9c8ec175b9', 'forecast_pause_001/launch.json': '5974fc19d292a2893b8005bb66630180d32de06deac636d12e68cf8087ce3fdf', 'forecast_pause_001/pause.json': '29714cff6231347a5681850400ee9869c2db20f7d0debe818e36f70f569b61bb', 'forecast_pause_001/restore.lock': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'forecast_pause_001/restored.json': 'cdc387d4b6306451026d56f86d9527cf7203d6b0e324b698fb30a7b83256cf43', 'forecast_pause_001/resume_forecasting.py': '08c4b4cb564035f61731ba957e45cf14c28ceb27410eb4421f53fb3a4af7ae92'}

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
    # A failed native setup is authentic terminal ancestry, never an admitted inspection.
    audit_path=Path(__file__).resolve().parent/'inputs/previous_terminal_audit.json'
    if sha(audit_path)!=PREVIOUS_AUDIT_SHA256:raise RuntimeError('Prior failed-owner audit changed')
    audit=json.loads(audit_path.read_text())
    if audit.get('audit_verified') is not True or audit.get('terminal_outcome')!='authentic_terminal_failure' or audit.get('expected_invocation')!=PREVIOUS_INVOCATION:raise RuntimeError('Prior failure provenance invalid')
    fields=dict(line.split('=',1) for line in call(['systemctl','--user','show',PREVIOUS_UNIT,'-p','ActiveState','-p','SubState','-p','InvocationID','-p','MainPID','-p','ExecMainStatus']).splitlines() if '=' in line)
    if fields.get('ActiveState') not in ('inactive','failed') or fields.get('MainPID')!='0' or fields.get('ExecMainStatus')!='1':raise RuntimeError('Previous failed inspection owner not terminal with recorded exit1')
    if fields.get('InvocationID') and fields['InvocationID']!=PREVIOUS_INVOCATION:raise RuntimeError('Previous inspection invocation changed')
    for name,bound in PRIOR_PINS.items():
        if sha(PRIOR_BASE/name)!=bound:raise RuntimeError('Previous failure/cleanup receipt changed: '+name)
    campaign=json.loads((PREVIOUS_ROOT/'campaign.json').read_text());identity=campaign.get('identity',{})
    if campaign.get('status')!='failed' or campaign.get('terminal_inputs_unchanged') is not True or campaign.get('planned_phases')!=['inspection'] or campaign.get('host_freeze_sha256')!=PREVIOUS_HOST or campaign.get('source_freeze_sha256')!=PREVIOUS_SOURCE or identity.get('inspector_freeze_sha256')!=PREVIOUS_SOURCE:raise RuntimeError('Previous failed campaign identity changed')
    native=json.loads((PREVIOUS_ROOT/'inspection/state.json').read_text())
    if native.get('status')!='failed' or native.get('explicit_steps_completed')!=0 or native.get('inputs_unchanged') is not True or native.get('identity')!=identity or native.get('physical_admission') is not False or native.get('training_allowed') is not False:raise RuntimeError('Previous native setup failure state changed')
    if not any("No module named 'omni.physics.tensors.impl'" in error for error in native.get('errors',[])):raise RuntimeError('Previous exact API lookup failure missing')
    pause=json.loads((PREVIOUS_PAUSE/'pause.json').read_text());restored=json.loads((PREVIOUS_PAUSE/'restored.json').read_text())
    timers={name for name,value in pause['units'].items() if name.endswith('.timer') and 'ActiveState=active' in value}
    if pause.get('unit')!=PREVIOUS_UNIT or pause.get('output')!=str(PREVIOUS_ROOT) or set(restored.get('timers',[]))!=timers or restored.get('restored_unix')!=1789071659.3738954:raise RuntimeError('Previous exact pause001 restoration missing')
    names={p.stem for p in (PREVIOUS_ROOT/'jobs').glob('*.json') if not p.name.endswith('_contact_data_audit.json')}
    if names!={'inspection'}:raise RuntimeError('Previous job inventory changed')
    job=json.loads((PREVIOUS_ROOT/'jobs/inspection.json').read_text())
    if job.get('status')!='failed' or job.get('phase')!='inspection' or job.get('cleanup_checked') is not True or job.get('container_name')!=PREVIOUS_NAME or job.get('container_id')!=PREVIOUS_ID:raise RuntimeError('Previous failed owned cleanup incomplete')
    for identifier in (PREVIOUS_NAME,PREVIOUS_ID):
        q=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',identifier],capture_output=True,text=True,timeout=20)
        if q.returncode==0 or not any(v in q.stderr.lower() for v in ('no such object','no such container')):raise RuntimeError('Previous container absence unknown: '+identifier)

def validate_inspection_inputs(host):
    require_final_bindings()
    if sha(HOST)!=HOST_SHA256:raise RuntimeError('Canonical host changed')
    verify_manifest(HOST.parent,'FREEZE_SHA256.json',HOST_FREEZE_SHA256)
    verify_manifest(SOURCE,'FREEZE_SHA256.json',SOURCE_SHA256)
    if host.SOURCE_FREEZE!=SOURCE_SHA256 or host.SUPERVISOR_MAP!=SUPERVISOR_SHA256:raise RuntimeError('Host pins do not match canonical guard')
    args=SimpleNamespace(source=SOURCE,asset=ASSET,supervisor_source=SUPERVISOR_SOURCE,output=OUTPUT,isaaclab=Path('/home/orionh/IsaacLab'),host_freeze_sha256=HOST_FREEZE_SHA256)
    host.require_fresh_output(args);identity=host.verify_inputs(args)
    if identity.get('training_allowed') is not False or identity.get('physical_admission') is not False or not valid_hash(identity.get('runtime_binding',{}).get('runtime_tree_sha256')):raise RuntimeError('Only identified canonical inspection is allocated')
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
    validate_inspection_inputs(train_host)
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
            reason='Canonical corrected-mass direct-drive native asset inspection only: zero gravity, eight explicit steps, no controller/gains/targets/training or physical admission',
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
        subprocess.run(['systemd-run', '--user', '--unit=hexapod-canonical-forecast-restore-002', '--on-active=15m',
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
        validate_inspection_inputs(train_host)
        verify_previous_owner()
    except Exception:
        subprocess.run(['/usr/bin/python3',str(restorer)],check=True,timeout=120)
        raise
    try:
        cmd=['systemd-run','--user','--unit='+UNIT,'--property=RuntimeMaxSec=720',
            '--property=TimeoutStopSec=180','--property=KillMode=process','--property=Environment=PYTHONDONTWRITEBYTECODE=1',
            '--property=ExecStopPost=/usr/bin/python3 '+str(restorer),'--property=Environment=PYTHONUNBUFFERED=1',
            '/usr/bin/python3',str(HOST),'--source',str(SOURCE),'--asset',str(ASSET),
            '--supervisor-source',str(SUPERVISOR_SOURCE),'--output',str(OUTPUT)]
        subprocess.run(cmd,check=True,timeout=30)
        launched=True
        (PAUSE/'launch.json').write_text(json.dumps(dict(command=cmd,launched_unix=time.time(),native_source_freeze_sha256=SOURCE_SHA256),indent=2)+'\n')
        print(json.dumps(dict(unit=UNIT,source=str(SOURCE),output=str(OUTPUT),pause=str(PAUSE))))
    finally:
        if not launched:
            subprocess.run(['/usr/bin/python3',str(restorer),'--stop-owner'],check=True,timeout=300)

if __name__=='__main__':main()
