"""Canonical native standing32 guard; same-source completed standing1 is a required admission."""
from pathlib import Path
from types import SimpleNamespace
import fcntl,hashlib,importlib.util,json,os,subprocess,sys,time
sys.dont_write_bytecode=True
BASE=Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910')
PRIOR_BASE=BASE
SOURCE=BASE/'standing_source_005'
ASSET=BASE/'asset_001'
ADMISSION=BASE/'native_actuation_001/actuation'
SUPERVISOR_SOURCE=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909/reference_physics_source_009')
OUTPUT=BASE/'native_standing32_003'
PAUSE=BASE/'forecast_pause_015'
UNIT='hexapod-canonical-native-standing32-003-20260910.service'
HOST=BASE/'standing_host_007/launch_standing_spark.py'
SOURCE_SHA256='c87f2d340c935cd947c7aed3f3dcd7c9a9965b6e19d6726546abb560b27ab131'
SUPERVISOR_SHA256='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
HOST_SHA256='0235b5cb7116a842acf2617917a6c6257e7ab5891012283e3944931f27037475'
HOST_FREEZE_SHA256='7085776db928bd600ee3e8aff48af433eee1f35e758ce6221b34e4f2cfe50577'
PREVIOUS_UNIT='hexapod-canonical-native-standing-006-20260910.service'
PREVIOUS_INVOCATION='64a373e301c24b70aa3790c18aca147c'
PREVIOUS_ROOT=BASE/'native_standing_006'
PREVIOUS_PAUSE=BASE/'forecast_pause_014'
PREVIOUS_HOST='7085776db928bd600ee3e8aff48af433eee1f35e758ce6221b34e4f2cfe50577'
PREVIOUS_SOURCE='c87f2d340c935cd947c7aed3f3dcd7c9a9965b6e19d6726546abb560b27ab131'
PREVIOUS_NAME='hexapod-reference-physics-b77957dd91c84d19bf6f65f84239d415'
PREVIOUS_ID='da34f5531eda481c07ee3295748e64d54f94921a9c45474f113d2515d1cc7305'
PREVIOUS_AUDIT_SHA256='dd0ab8a3a38514bc9f2540496bf2b75268ccdb4123320f7fd8e319d949513c2b'
PRIOR_PINS={'native_standing_006/campaign.json': '99428249f9458c5f507af75d0a55872ed887dcdec35848eb4ccada41871b5297', 'native_standing_006/jobs/standing.json': '6e42cba7f76ffc3371eba0349fe699137f61580701a5f18fdbd94caaa6ef139e', 'native_standing_006/jobs/standing_contact_data_audit.json': 'e244ecd39c0c78557c6faf3fce992cefc5324dd14e5de7056ba7857e05467c3a', 'native_standing_006/logs/standing.log': '6a21dfe8e9042b55f110a9c9a58ff703915f262a967d0ff7fcf523dcdbf18526', 'native_standing_006/standing/contact_view.json': '0ef9cbae7134b7ca5a2da23a0e95028324030fcf73cd558ea6504f78a759bde0', 'native_standing_006/standing/contacts.jsonl': '29d628cf39f126645e78e8781fc7e2c97ca4ff431f51c0b84d15023dd8989706', 'native_standing_006/standing/control_trace.npz': 'decab118d383e463867befbd4a64d8f93a8b582af99afa3837ac2fd32e1395e3', 'native_standing_006/standing/initial_reset.json': '86bb78472900f9be68a3f5317cea4c3d3fff601ed4e68fe88e0402538323bd54', 'native_standing_006/standing/legacy_friction_readback.json': 'a58ccee28c4d7e32eb158de994eeef213dd8136f287f66c6a929e9653add9b2a', 'native_standing_006/standing/memory_before_scene.json': 'c0d3c6a6427167d047af92dc2b512b1d5e6477cca85425c72974cb8743dbf8c9', 'native_standing_006/standing/native_errors.json': '37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570', 'native_standing_006/standing/native_materials.json': 'aa1174dcab7dfeed253cadbf11c4a83075b9cc6db874df644b2a8491ee8087b7', 'native_standing_006/standing/native_readback.json': 'd37ab0a618739c5ac77a3f5e412f8dd558081a23f03b1ede1c98578251d6aca4', 'native_standing_006/standing/native_scene.json': '5342a8b913c38d6675817d62c1dedff4765a3f141178e206fa7ce7d76f47dbc9', 'native_standing_006/standing/resolved_stage.usda': '7c6db61b764d512c70da79bce66840b9603a1296a20aee7f7bcde9ec5a11178a', 'native_standing_006/standing/runtime_api.json': '0e402d157689122e829aa4597033364a9168c3b8f78b9514b3a80918c009e237', 'native_standing_006/standing/sdf_readback.json': '0a92ec3d0cd7101a932b7e5617ec5dc2072e89c59e70c28464908867e074597a', 'native_standing_006/standing/sdk_readback.json': '1016fef5aa047bdd0e5684a9fc8acd599e95738601847ad4084f2be58c53c454', 'native_standing_006/standing/session.json': 'b68f6915f1347e07d9515901a8c0e15a8874b00e0b2d53d14ba748a7a63b28ca', 'native_standing_006/standing/solver_readback.json': 'd279eb1e9b7b80f3b4eff8191982ff3225b744f97ce3c0e02bcaf8be463a6969', 'native_standing_006/standing/standing_report.json': 'b82865ef990cbb6839b2834a31de92310883b1a079b7969077327da20cb4762e', 'native_standing_006/standing/state.json': '33920a4af296ed124643fe61a16f4840c19949b5f6e434327d7cd7ea9953cd3f', 'native_standing_006/standing/substeps_000.npz': 'c9fa208bbd56bbb4c8c9f7699de187907dc0685d35e065257dbaf877031c91f9', 'native_standing_006/standing/substeps_001.npz': '576f962289ba4fbe340b118365ab068fd1cdd2853f4b8cdd6e420f9beae72131', 'native_standing_006/standing/substeps_002.npz': 'b9344c42d514f269a7b5fe9069cac31f831b72628220ca986e730739ff29c79f', 'native_standing_006/standing/substeps_003.npz': 'b36ffe88aa5a68c7d245e90dca95ea2858e9af532f2cd7ccc9411e4bc29d6dcb', 'native_standing_006/standing/substeps_004.npz': 'c1379b22e7eac0a07641e41e3596942756507a75f13d121bb87bcaa7b24320f5', 'native_standing_006/standing/substeps_005.npz': 'e5c113ea0cc1f6a77ed651340501e34152b5cb259b224cd54376da0693d99628', 'native_standing_006/standing/substeps_006.npz': '96f55e1173fa54bbc48ad2c547f19aad397d9fc5c178ba10809a5ab7c1c69e5a', 'native_standing_006/standing/substeps_007.npz': '20b4070d99c182269867ca2fa8b71cb0b92372f5079cc8d86b944a34b847fa2f', 'native_standing_006/standing/substeps_008.npz': '58a15c90fd9169a673c9cf72bb8bf81cd20df29ca64b2c3a185f34fd893ad2a7', 'native_standing_006/standing/substeps_009.npz': 'e8dc44aa8421141cf1f052be5cd2cc6f844d882593608543acee1ef3f55c6551', 'native_standing_006/standing/usd_readback.json': '9a3547d211a7fd456a897ccb54beda88cab027c00c6605532795f3eee01c67b8', 'native_standing_006/standing/warmup_native_state.json': 'ef36393b80071441f3832cfe810bd449069f5d8f1c4bb0cf827dad792b9aa70b', 'native_standing_006/standing_immutable.sha256.json': 'c577d4449b69d22f09412600aea3da8e2c6d4caf7f8c6bef93615d127f47760f', 'forecast_pause_014/launch.json': '1693a97815c21fb826fd7347add8940261889e0f6c4acb25c039cb3987644c88', 'forecast_pause_014/pause.json': 'e52f30976275f7292cad7bab6c1ee0aff9e9d147b29671c17af49da5db891a71', 'forecast_pause_014/restore.lock': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'forecast_pause_014/restored.json': '3e9d717ce0a287f5f2898d3e61686eed2afbc7e9811273e5a54aecb2d2a544a8', 'forecast_pause_014/resume_forecasting.py': 'daffe2b2397c31b37145b193c29c947c4162123799365323f27acda0218a5d90'}

STANDING_ONE=PREVIOUS_ROOT/'standing'
STANDING_ONE_STATE_SHA256='33920a4af296ed124643fe61a16f4840c19949b5f6e434327d7cd7ea9953cd3f'
STANDING_ONE_INVENTORY_SHA256='877e26ab3fedae042d5e024c11116bc1b808fcb1aa539233eb88009d4a193afc'
PREVIOUS_RESTORED_UNIX=1789084703.6548698

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
    for name,bound in {'source':SOURCE_SHA256,'supervisor':SUPERVISOR_SHA256,'host':HOST_SHA256,'host freeze':HOST_FREEZE_SHA256,'previous audit':PREVIOUS_AUDIT_SHA256,'standing1 state':STANDING_ONE_STATE_SHA256,'standing1 inventory':STANDING_ONE_INVENTORY_SHA256,**PRIOR_PINS,**RESERVATION_PINS}.items():
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
    if pause.get('unit')!=PREVIOUS_UNIT or pause.get('output')!=str(PREVIOUS_ROOT)or set(restored.get('timers',[]))!=timers or restored.get('restored_unix')!=PREVIOUS_RESTORED_UNIX:raise RuntimeError('Previous exact pause014 restoration missing')
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

def validate_standing_inputs(host):
    require_final_bindings()
    verify_reservation()
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
            output=str(OUTPUT), source=str(SOURCE), admission=str(ADMISSION), standing_one=str(STANDING_ONE), preflight=snapshot,
            coordination_sha256=hashlib.sha256(coordination).hexdigest(),
            reservation_path=str(RESERVATION), restoration_scope='per_job_snapshot_only', persistent_reservation_release_attempted=False,
            reason='Canonical provisional standing32:1000 controls/8000 explicit400Hz steps; exact asset, completed coordinate/effort and same-source standing1 evidence; no policy or broader physical admission',
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
        subprocess.run(['systemd-run', '--user', '--unit=hexapod-canonical-forecast-restore-015', '--on-active=25m',
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
        cmd=['systemd-run','--user','--unit='+UNIT,'--property=RuntimeMaxSec=1320',
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
