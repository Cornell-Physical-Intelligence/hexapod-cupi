#!/usr/bin/env python3
"""One fresh canonical two-update PPO integration; actual standing1/32 required."""
import argparse, hashlib, importlib, importlib.util, json, signal, sys, time
from pathlib import Path
from types import SimpleNamespace

SOURCE_FREEZE='cc62938666b7b394e119fe05664767b2a265928ca46c8e15afd97312611f5d63'
STANDING_FREEZE='acb589708fcba8be2f6ef64171795889eca8b6583fdbb4bc2fc25d61717b0467'
SUPERVISOR_MAP='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
SUPERVISOR_CODE='9ebabaf254cc77fabad7d4759cc80fc26d7efd3912ea84bd55304594c88ed561'
SELECTOR_SHA256='d11edd466902e2122d6cec7cd1b0c4399b566a6ed1598d6bfe911317197353e4'
ADMISSION_STATE='5a561006d0e63f2f69b7d4129836a33a55b5a9c5a2b576bdd91fb112fe845a8b'
COORDINATION=Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md')
EXPECTED_COORDINATION='35af5100e39980f55a3f5e19654e4550b3c3fb6da114b64fbe0d143817242cf3'
PHASE='canonical_ppo_smoke'
SCHEMA='canonical_direct_drive_ppo_smoke_v1'
CONTRACT=None
PARENT=None
INPUT_ARGS=('source','asset','admission','standing_source','standing_one','standing32','bindings','supervisor_source')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,data):
    p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');q.replace(p)
def tree(p):return {f.relative_to(p).as_posix():sha(f)for f in sorted(p.rglob('*'))if f.is_file()}
def valid_hash(h):return isinstance(h,str)and len(h)==64 and all(c in '0123456789abcdef'for c in h)
def verify_tree(root,manifest,bound):
    if not valid_hash(bound):raise ValueError('Final source binding pending')
    if root.is_symlink()or any(p.is_symlink()for p in root.rglob('*')):raise ValueError('Symbolic immutable input')
    if sha(root/manifest)!=bound:raise ValueError('Wrong immutable manifest:'+str(root))
    actual=tree(root);actual.pop(manifest)
    if actual!=read(root/manifest):raise ValueError('Changed/unlisted immutable input:'+str(root))
def verify_own_bundle(expected=None):
    p=Path(__file__).resolve().parent;bound=sha(p/'FREEZE_SHA256.json')
    if expected is not None and expected!=bound:raise ValueError('Host changed during PPO integration')
    verify_tree(p,'FREEZE_SHA256.json',bound);return bound

def require_fresh_output(args):
    output=args.output.resolve()
    if args.output.exists()or args.output.is_symlink():raise ValueError('Fresh PPO output required')
    for p in [*(getattr(args,k).resolve()for k in INPUT_ARGS),Path(__file__).resolve().parent]:
        if output==p or p in output.parents or output in p.parents:raise ValueError('Output overlaps immutable input')

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def load_contract(source):
    source=source.resolve();package=source/'canonical_direct_ppo'
    def verify_cached():
        for name,module in list(sys.modules.items()):
            if name=='canonical_direct_ppo'or name.startswith('canonical_direct_ppo.'):
                origin=getattr(module,'__file__',None)
                if origin is None or not Path(origin).resolve().is_relative_to(package):raise ValueError('Foreign cached canonical PPO module:'+name)
    verify_cached();before=sys.path[:]
    try:
        sys.path.insert(0,str(source));module=importlib.import_module('canonical_direct_ppo.native_contract');verify_cached()
    finally:sys.path[:]=before
    if Path(module.__file__).resolve()!=package/'native_contract.py':raise ValueError('Wrong canonical contract import origin')
    return module

def native_args(args):
    return SimpleNamespace(asset=args.asset,admission=args.admission,standing_source=args.standing_source,
        standing_one=args.standing_one,standing32=args.standing32,bindings=args.bindings,output=args.output/PHASE)

def validate_identity(identity,args):
    if identity.get('schema')!=SCHEMA or identity.get('phase')!=PHASE:raise ValueError('Wrong canonical integration schema/phase')
    if identity.get('runtime_binding')!={'runtime_tree_sha256':SOURCE_FREEZE,'scope':'fresh_canonical_405_408_two_update_smoke'}:raise ValueError('Wrong fresh PPO source identity')
    for k,v in {'fresh_neutral_controls':1000,'fresh_neutral_substeps':8000,'policy_controls':48,'policy_substeps':384,'total_controls':1048,'total_substeps':8384,'training_integration_only':True,'quality_admitted':False,'Stage2_complete':False}.items():
        if identity.get(k)!=v:raise ValueError('Unexpected fixed integration allocation:'+k)
    native=identity.get('native_identity',{});lineage=identity.get('policy_lineage',{});protocol=identity.get('protocol',{})
    if native.get('runtime_binding')!={'runtime_tree_sha256':STANDING_FREEZE,'scope':'canonical_provisional_native_standing_only'}:raise ValueError('Wrong exact standing runtime')
    for k,v in {'num_envs':32,'controls':1000,'steps':8000,'dt':.0025,'control_dt':.02,'substeps_per_control':8,'settle_controls':200,'ground':True,'gravity':[0.,0.,-9.81],'actuation_state_sha256':ADMISSION_STATE,'physical_admission':False,'training_allowed':False}.items():
        if native.get(k)!=v:raise ValueError('Unexpected native standing setup:'+k)
    for k,v in {'replicas':32,'controls_per_update':24,'updates':2,'total_controls':48,'transitions':1536,'actor_width':405,'critic_width':408,'control_dt':.02,'physics_dt':.0025,'substeps':8,'checkpoint_input':None,'auto_reset':False,'automatic_continuation':False,'walking_objective_adopted':False,'command':'allzero48controls; smoke-only'}.items():
        if k not in protocol or protocol[k]!=v:raise ValueError('Unexpected fixed PPO protocol:'+k)
    if lineage.get('consumer_source_freeze_sha256')!=SOURCE_FREEZE or lineage.get('standing_source_freeze_sha256')!=STANDING_FREEZE or lineage.get('fresh_checkpoint_only')is not True:raise ValueError('Wrong fresh checkpoint lineage')
    for field in ['standing1_state_sha256','standing32_state_sha256','servo_sha256','geometry_sha256','adapter_sha256','rsl_source_map_sha256']:
        if not valid_hash(lineage.get(field)):raise ValueError('Pending actual admission/learner binding:'+field)
    if identity.get('bindings_sha256')!=sha(args.bindings):raise ValueError('External bindings identity changed')
    return identity

def verify_inputs(args):
    global CONTRACT
    if not valid_hash(SOURCE_FREEZE)or not valid_hash(STANDING_FREEZE):raise ValueError('Native/source binding pending; no allocation')
    verify_own_bundle(getattr(args,'host_freeze_sha256',None))
    verify_tree(args.source,'FREEZE_SHA256.json',SOURCE_FREEZE)
    verify_tree(args.standing_source,'FREEZE_SHA256.json',STANDING_FREEZE)
    verify_tree(args.supervisor_source,'campaign_source_hashes.json',SUPERVISOR_MAP)
    if sha(args.supervisor_source/'tools/launch_reference_physics_spark.py')!=SUPERVISOR_CODE:raise ValueError('Ownership supervisor bytes changed')
    for key in ('bindings','asset','admission','standing_one','standing32'):
        p=getattr(args,key)
        if p.is_symlink()or (p.is_dir()and any(q.is_symlink()for q in p.rglob('*'))):raise ValueError('Symbolic admission/input:'+key)
    selector_path=Path(__file__).resolve().parent/'inputs/active_model.json'
    if sha(selector_path)!=SELECTOR_SHA256:raise ValueError('Canonical selection snapshot changed')
    selector=read(selector_path)
    for path,bound in [('robot.usda',selector['usd_sha256']),('source/model.json',selector['model']['sha256']),('source/source.urdf',selector['urdf']['sha256'])]:
        if sha(args.asset/path)!=bound:raise ValueError('Asset differs from canonical corrected-mass selection:'+path)
    CONTRACT=load_contract(args.source)
    identity=CONTRACT.verify_inputs(native_args(args))
    return validate_identity(identity,args)

validate_inputs=verify_inputs

def command(args,name,phase):
    if phase!=PHASE:raise ValueError('Only one fresh canonical PPO smoke phase')
    mounts=[(args.source,'/ppo'),(args.standing_source,'/standing'),(args.asset,'/asset'),(args.admission,'/admission'),
        (args.standing_one,'/standing_one'),(args.standing32,'/standing32'),(args.bindings,'/bindings.json')]
    return ['docker','compose','--env-file','docker/.env.base','-f','docker/docker-compose.yaml','--profile','base',
        'run','--rm','--no-deps','--name',name,'-w','/output','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
        '-e','PYTHONPATH=/ppo:/workspace/isaaclab/source/isaaclab','-v',str(args.output)+':/output:rw',
        *[arg for path,target in mounts for arg in ('-v',str(path)+':'+target+':ro')],
        '--entrypoint','/workspace/isaaclab/_isaac_sim/python.sh','isaac-lab-base','/ppo/run_native_smoke.py',
        '--asset','/asset','--admission','/admission','--standing-source','/standing','--standing-one','/standing_one',
        '--standing32','/standing32','--bindings','/bindings.json','--output','/output/'+PHASE,'--device','cuda:0','--headless']

def load_supervisor(args,identity):
    global PARENT
    sys.path.insert(0,str(args.supervisor_source/'tools'))
    PARENT=load_module('_canonical_ppo_ownership_source009',args.supervisor_source/'tools/launch_reference_physics_spark.py')
    PARENT.command=lambda source,output,name,phase:command(args,name,phase)
    def native_verified_source(source):
        if source!=args.source:raise ValueError('Unexpected native source path')
        if verify_inputs(args)!=identity:raise ValueError('Inputs changed before native process creation')
    PARENT.verified_source=native_verified_source
    PARENT.RUNTIME_TREE=identity['runtime_binding']['runtime_tree_sha256']
    original_save=PARENT.save
    def metadata_save(path,data):
        if Path(path).parent.name=='jobs'and Path(path).name==PHASE+'.json':
            data={**data,'no_policy_loaded':False,'no_checkpoint_loaded':True,'checkpoint_input':None,
                'policy_initialization':'fresh actor only after same-process neutral admission','training_integration_only':True,
                'quality_admitted':False,'physical_admission':False,'Stage2_complete':False,'ppo_scope':SCHEMA,'legacy_supervisor_reused_for_ownership_only':True}
        return original_save(path,data)
    PARENT.save=metadata_save
    return PARENT

def run_campaign(args,identity):
    report={'status':'preparing','started_unix':time.time(),'identity':identity,'host_freeze_sha256':args.host_freeze_sha256,
        'planned_phases':[PHASE],'training_integration_only':True,'quality_admitted':False,'physical_admission':False,'Stage2_complete':False,
        'automatic_continuation':False,'checkpoint_input':None,'source_freeze_sha256':SOURCE_FREEZE,
        'standing_source_freeze_sha256':STANDING_FREEZE,'scope':'Fresh zero-command two-update infrastructure screen; no walking or hardware admission'}
    save(args.output/'campaign.json',report);phase_payloads=None
    try:
        PARENT.run_owned(args,PHASE)
        if verify_inputs(args)!=identity:raise ValueError('Original inputs changed after container exit')
        report['post_exit_original_inputs_reverified']=True
        report['accepted_phase']=CONTRACT.validate_result(args.output/PHASE,identity)
        if report['accepted_phase'].get('phase')!=PHASE or report['accepted_phase'].get('updates_completed')!=2:raise ValueError('Unexpected accepted phase receipt')
        phase_payloads=tree(args.output/PHASE)
        save(args.output/(PHASE+'_immutable.sha256.json'),phase_payloads)
        report['phase_payload_manifest_sha256']=sha(args.output/(PHASE+'_immutable.sha256.json'))
        report['post_exit_all_phase_payloads_inventoried']=True
        report['status']='completed'
    except Exception as exc:
        report.update(status='stopped'if isinstance(exc,InterruptedError)else'failed',error=repr(exc));raise
    finally:
        try:
            if verify_inputs(args)!=identity:raise ValueError('Terminal input identity differs')
            report['terminal_inputs_unchanged']=True
            if phase_payloads is not None and tree(args.output/PHASE)!=phase_payloads:raise ValueError('Native payload changed after post-exit seal')
        except Exception as exc:report.update(status='failed',terminal_inputs_unchanged=False,integrity_error=repr(exc))
        report['finished_unix']=time.time();save(args.output/'campaign.json',report)
    if report['status']!='completed':raise RuntimeError('Native PPO terminal integrity failed')
    return report

def main():
    parser=argparse.ArgumentParser(allow_abbrev=False,description=__doc__)
    for name in (*INPUT_ARGS,'output'):parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    parser.add_argument('--isaaclab',type=Path,default=Path('/home/orionh/IsaacLab'))
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    # Check dangling output before resolve removes symlink identity.
    require_fresh_output(args)
    for name in (*INPUT_ARGS,'output','isaaclab'):setattr(args,name,getattr(args,name).resolve())
    require_fresh_output(args);args.host_freeze_sha256=verify_own_bundle();identity=verify_inputs(args)
    if args.preflight_only:print(json.dumps({'identity':identity,'host_freeze_sha256':args.host_freeze_sha256,'no_GPU':True},indent=2));return
    args.coordination_sha256=sha(COORDINATION)
    if args.coordination_sha256!=EXPECTED_COORDINATION:raise ValueError('Coordination changed; no allocation')
    load_supervisor(args,identity);args.output.mkdir(parents=True)
    for name in ('jobs','logs'):(args.output/name).mkdir()
    signal.signal(signal.SIGTERM,lambda *_:(args.output/'stop.request').touch());signal.signal(signal.SIGINT,lambda *_:(args.output/'stop.request').touch())
    run_campaign(args,identity)
if __name__=='__main__':main()
