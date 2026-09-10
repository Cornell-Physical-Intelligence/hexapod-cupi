"""Standard-library provenance contract for an unqualified direct315 preview."""
from pathlib import Path
import hashlib,importlib.util,json,re,sys

HERE=Path(__file__).resolve().parent
SOURCE='aa70659d33243f1e28e9c71b1a7f9d33341642db770fd7356192573a26fe904e'
PLAN='b2286828148c89fb34624d38beaf84b389584f7bda4d2648512e017da4b50acb'
NATIVE='0bdd2abf3811e68ac59e4fbc325eb5d2bb96afd2183eb84c4d37faeb86ff9e9f'
HOST='2db7e44ec40543753a6d1325f6d5dc9ed81be2948931b2ca670deffda68ce959'
LEGACY='abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280'
ORIGINAL='1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8'
DT=.02;FPS=25;STEPS=1900;FRAMES=950
SEGMENTS=(('QUIET',4.,(0.,0.,0.)),('FORWARD',6.,(.10,0.,0.)),('STOP FORWARD',4.,(0.,0.,0.)),
          ('STRAFE LEFT',6.,(0.,.10,0.)),('STOP STRAFE',4.,(0.,0.,0.)),
          ('FORWARD LEFT ARC',6.,(.10,0.,.20)),('STOP ARC',8.,(0.,0.,0.)))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,data):
    path=Path(path);temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)
def tree(root):return {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
def tree_digest(mapping):return hashlib.sha256(json.dumps(mapping,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def verify_tree(root,manifest,bound):
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):raise ValueError('Symlink input')
    if sha(root/manifest)!=bound:raise ValueError('Wrong immutable manifest')
    actual=tree(root);actual.pop(manifest)
    if actual!=read(root/manifest):raise ValueError('Changed or unlisted immutable input')
    return actual
def verify_self():
    bound=sha(HERE/'FREEZE_SHA256.json');verify_tree(HERE,'FREEZE_SHA256.json',bound);return bound
def verify_source(root):
    mapping=verify_tree(root,'campaign_source_hashes.json',SOURCE)
    if len(mapping)!=599 or sha(root/'robot/hexapod_mkii_length_study/training_plan.json')!=PLAN:raise ValueError('Wrong native source/plan')
    return mapping
def schedule():
    rows=[]
    for index,(name,seconds,command) in enumerate(SEGMENTS):
        for _ in range(round(seconds/DT)):
            rows.append({'segment':name,'segment_index':index,'time_start_s':len(rows)*DT,'requested_command':list(command)})
    if len(rows)!=STEPS:raise ValueError('Recording schedule changed')
    return rows

def load_native(root):
    previous=list(sys.path);cached=sys.modules.pop('direct_config',None)
    try:
        sys.path.insert(0,str(root))
        spec=importlib.util.spec_from_file_location('_preview_native_contract',root/'direct_contract.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        if Path(sys.modules['direct_config'].__file__).resolve()!=root/'direct_config.py':raise ValueError('Wrong native config import')
        return module
    finally:
        sys.path[:]=previous;sys.modules.pop('direct_config',None)
        if cached is not None:sys.modules['direct_config']=cached

def verify_pilot(args,contract):
    if not re.fullmatch('[0-9a-f]{64}',HOST):raise ValueError('Selected host005 binding pending')
    if any(p.is_symlink() for p in args.pilot.rglob('*')):raise ValueError('Symlink selected campaign')
    campaign=read(args.pilot/'campaign.json');identity=campaign.get('identity',{})
    if (campaign.get('status')!='completed' or campaign.get('terminal_inputs_unchanged') is not True
        or campaign.get('host_freeze_sha256')!=HOST or identity.get('source_manifest_sha256')!=SOURCE
        or identity.get('plan_sha256')!=PLAN or identity.get('checkpoint_sha256')!=ORIGINAL
        or identity.get('actor_width')!=315 or identity.get('critic_width')!=318):raise ValueError('Unmatched completed native campaign')
    allocation=campaign.get('allocation');branch=campaign.get('branch')
    if allocation!='extended' or branch!='caps':raise ValueError('Preview requires completed CAPS500 extended allocation; smoke and pilots are not eligible')
    selected=contract.selection('train',allocation,branch,None,None)
    if identity.get('selection')!=selected or selected.get('updates')!=500:raise ValueError('Wrong campaign training selection')
    if (campaign.get('bounded_campaign_complete') is not True or campaign.get('last_completed_phase')!='final_stop' or campaign.get('PPO_updates_completed')!=500 or identity.get('schema')!='direct315_extended_native_v4' or campaign.get('Stage2_complete') is not False):raise ValueError('Incomplete or wrong extended500 source/schema scope')
    phases=('standing','initial_constant','initial_stop','train','final_constant','final_stop')
    if campaign.get('planned_phases')!=list(phases) or set(campaign.get('accepted_phases',{}))!=set(phases):raise ValueError('Incomplete selected campaign')
    learned=contract.validate_result(args.pilot/'train','train',identity)
    bound=learned['checkpoint_sha256']
    if not re.fullmatch('[0-9a-f]{64}',args.checkpoint_sha256) or bound!=args.checkpoint_sha256 or sha(args.checkpoint)!=bound:raise ValueError('Selected checkpoint differs from trained final receipt')
    for phase in phases:
        result=contract.validate_result(args.pilot/phase,phase,identity,expected_checkpoint_sha256=bound)
        if result!=campaign['accepted_phases'][phase]:raise ValueError('Accepted phase receipt changed: '+phase)
        recorded=read(args.pilot/(phase+'_immutable.sha256.json'))
        if tree(args.pilot/phase)!=recorded:raise ValueError('Completed phase tree changed: '+phase)
    if sha(args.admission)!=sha(args.pilot/'standing/admission.json'):raise ValueError('Standing admission alias differs')
    return {'campaign_sha256':sha(args.pilot/'campaign.json'),'training_receipt_sha256':sha(args.pilot/'train/training_receipt.json'),
            'standing_admission_sha256':sha(args.admission),'identity':identity,'allocation':allocation,'branch':branch,
            'completed_updates':selected['updates'],'pilot_tree_sha256':tree_digest(tree(args.pilot))}

def verify_inputs(args,*,check_self=True):
    if args.seed!=7057:raise ValueError('Preview seed is explicitly fixed to7057')
    if check_self:adapter=verify_self()
    else:adapter=None
    source=verify_source(args.source_root);verify_tree(args.native_contract,'FREEZE_SHA256.json',NATIVE)
    native=load_native(args.native_contract);pilot=verify_pilot(args,native)
    plan=read(args.source_root/'robot/hexapod_mkii_length_study/training_plan.json')
    if (pilot['identity'].get('schema')!=native.SCHEMA
        or pilot['identity'].get('overrides')!=plan['omni']['overrides']
        or pilot['identity'].get('diagnostic_options')!=plan['omni']['diagnostics']):raise ValueError('Campaign dynamics/observation profile differs')
    for protected in (args.source_root,args.native_contract,args.pilot,args.checkpoint.parent,HERE):
        if args.output==protected or protected in args.output.parents:raise ValueError('Output overlaps immutable input')
    return {'kind':'native_direct315_progress_not_qualification','source_manifest_sha256':SOURCE,'plan_sha256':PLAN,
            'native_contract_freeze_sha256':NATIVE,'adapter_freeze_sha256':adapter,'checkpoint_sha256':args.checkpoint_sha256,
            'checkpoint_path':str(args.checkpoint),'pilot':pilot,'source_payloads':len(source),'seed':args.seed,
            'rendering_overrides':{'num_envs':1,'width':1280,'height':720,'enable_cameras':True,'render_mode':'rgb_array'},
            'stage2_complete':False,'qualification_performed':False,'policy_training_started':False,'pose_forcing':False,
            'planned_control_steps':STEPS,'planned_frames':FRAMES,'fps':FPS,'playback_rate':1.0,'dt_s':DT}

def legacy_binding(source):
    expected=read(HERE/'inputs/legacy_runtime.json')
    normalized={k.removeprefix('isaaclab/hexapod_rl/'):v for k,v in expected.items()}
    if tree_digest(normalized)!=LEGACY or len(expected)!=16:raise ValueError('Wrong legacy runtime inventory')
    origins={}
    for name,module in list(sys.modules.items()):
        if name=='hexapod_rl' or name.startswith('hexapod_rl.'):
            p=Path(module.__file__).resolve()
            if not p.is_relative_to(source/'isaaclab/hexapod_rl'):raise ValueError('Wrong legacy import origin')
            key=p.relative_to(source).as_posix()
            if key not in expected or sha(p)!=expected[key]:raise ValueError('Unpinned legacy module')
            origins[name]={'path':str(p),'sha256':expected[key]}
    if 'hexapod_rl.env' not in origins:raise ValueError('Actual legacy environment not loaded')
    return {'runtime_tree_sha256':LEGACY,'python_files_verified':16,'imported_origins':origins,
            'scope':'Legacy inference runtime; rendering overrides and newly trained checkpoint are separate identities.'}

def native_inference_origins(source):
    manifest=read(HERE/'inputs/native_source_manifest.json');result={}
    for name in ('direct_config','omni_flat_env','omni_flat_math','omni_diagnostics','omni_path_demo'):
        module=sys.modules.get(name);path=Path(getattr(module,'__file__','')).resolve()
        expected=source/'tools'/ (name+'.py')
        if path!=expected or sha(path)!=manifest['tools/'+name+'.py']:
            raise ValueError('Unpinned native inference import: '+name)
        result[name]={'path':str(path),'sha256':sha(path)}
    return result
