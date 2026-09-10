"""Bounded provisional canonical standing; one explicit1 or32 allocation, no policy."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,importlib.util,json,signal,sys,time
sys.dont_write_bytecode=True
SOURCE_FREEZE='acb589708fcba8be2f6ef64171795889eca8b6583fdbb4bc2fc25d61717b0467'
SUPERVISOR_MAP='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
SUPERVISOR_CODE='9ebabaf254cc77fabad7d4759cc80fc26d7efd3912ea84bd55304594c88ed561'
SELECTOR_SHA256='d11edd466902e2122d6cec7cd1b0c4399b566a6ed1598d6bfe911317197353e4'
COORDINATION=Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md')
EXPECTED_COORDINATION='35af5100e39980f55a3f5e19654e4550b3c3fb6da114b64fbe0d143817242cf3'
ADMISSION_STATE='5a561006d0e63f2f69b7d4129836a33a55b5a9c5a2b576bdd91fb112fe845a8b'
CONTRACT=None;PARENT=None

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)
def tree(p):return {f.relative_to(p).as_posix():sha(f) for f in sorted(p.rglob('*')) if f.is_file()}
def valid_hash(h):return isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h)
def verify_tree(root,manifest,bound):
 if not valid_hash(bound):raise ValueError('Final native source binding pending')
 if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):raise ValueError('Symbolic immutable input')
 if sha(root/manifest)!=bound:raise ValueError('Wrong immutable manifest: '+str(root))
 actual=tree(root);actual.pop(manifest)
 if actual!=read(root/manifest):raise ValueError('Changed/unlisted immutable input: '+str(root))
def verify_own_bundle(expected=None):
 p=Path(__file__).resolve().parent;bound=sha(p/'FREEZE_SHA256.json')
 if expected is not None and expected!=bound:raise ValueError('Host changed during standing')
 verify_tree(p,'FREEZE_SHA256.json',bound);return bound

def require_fresh_output(args):
 if args.output.exists() or args.output.is_symlink():raise ValueError('Fresh standing output required')
 for p in (args.source,args.asset,args.admission,args.supervisor_source,Path(__file__).resolve().parent,*([args.standing_one] if args.standing_one else [])):
  if args.output==p or p in args.output.parents or args.output in p.parents:raise ValueError('Output overlaps immutable input')
def load_module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def verify_inputs(args):
 global CONTRACT
 if type(args.num_envs) is not int or args.num_envs not in (1,32):raise ValueError('Explicit standing1 or32 only')
 if (args.num_envs==32)!=(args.standing_one is not None):raise ValueError('Standing32 requires same-source standing1; standing1 consumes none')
 if not valid_hash(SOURCE_FREEZE):raise ValueError('Root native source binding pending; no allocation')
 verify_own_bundle(getattr(args,'host_freeze_sha256',None))
 verify_tree(args.source,'FREEZE_SHA256.json',SOURCE_FREEZE)
 verify_tree(args.supervisor_source,'campaign_source_hashes.json',SUPERVISOR_MAP)
 if sha(args.supervisor_source/'tools/launch_reference_physics_spark.py')!=SUPERVISOR_CODE:raise ValueError('Ownership supervisor bytes changed')
 selector_path=Path(__file__).resolve().parent/'inputs/active_model.json'
 if sha(selector_path)!=SELECTOR_SHA256:raise ValueError('Canonical selection snapshot changed')
 selector=read(selector_path)
 for path,bound in [('robot.usda',selector['usd_sha256']),('source/model.json',selector['model']['sha256']),('source/source.urdf',selector['urdf']['sha256'])]:
  if sha(args.asset/path)!=bound:raise ValueError('Asset differs from canonical corrected-mass selection: '+path)
 if CONTRACT is None:CONTRACT=load_module('_canonical_native_standing_contract',args.source/'standing_contract.py')
 if Path(CONTRACT.__file__).resolve()!=args.source/'standing_contract.py':raise ValueError('Wrong native contract import origin')
 identity=CONTRACT.verify_inputs(SimpleNamespace(source=args.source,asset=args.asset,admission=args.admission,output=args.output/'standing',num_envs=args.num_envs,standing_one=args.standing_one))
 if identity.get('training_allowed') is not False or identity.get('physical_admission') is not False:raise ValueError('Native standing cannot admit training or physics')
 if identity.get('schema')!='canonical_native_standing_v1' or identity.get('inspector_freeze_sha256')!=SOURCE_FREEZE or identity.get('runtime_binding')!={'runtime_tree_sha256':SOURCE_FREEZE,'scope':'canonical_provisional_native_standing_only'}:raise ValueError('Native standing runtime identity missing or incompatible')
 if identity.get('steps')!=8000 or identity.get('controls')!=1000 or identity.get('settle_controls')!=200 or identity.get('num_envs')!=args.num_envs or identity.get('dt')!=.0025 or identity.get('gravity')!=[0.,0.,-9.81]:raise ValueError('Unexpected standing allocation')
 if identity.get('actuation_state_sha256')!=ADMISSION_STATE:raise ValueError('Wrong completed native coordinate/effort admission')
 if identity.get('ground') is not True or identity.get('control_dt')!=.02 or identity.get('substeps_per_control')!=8:raise ValueError('Standing floor must be explicit')
 if args.num_envs==32 and (not valid_hash(identity.get('standing_one_state_sha256')) or not valid_hash(identity.get('standing_one_inventory_sha256'))):raise ValueError('Standing32 lacks exact previous1 evidence')
 return identity

validate_inputs=verify_inputs

def command(args,name,phase):
 if phase!='standing':raise ValueError('Only native standing is allocated')
 return ['docker','compose','--env-file','docker/.env.base','-f','docker/docker-compose.yaml','--profile','base',
  'run','--rm','--no-deps','--name',name,'-w','/output','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
  '-e','PYTHONPATH=/standing:/workspace/isaaclab/source/isaaclab',
  '-v',str(args.output)+':/output:rw','-v',str(args.source)+':/standing:ro','-v',str(args.asset)+':/asset:ro','-v',str(args.admission)+':/admission:ro',*(['-v',str(args.standing_one)+':/standing_one:ro'] if args.standing_one else []),
  '--entrypoint','/workspace/isaaclab/_isaac_sim/python.sh','isaac-lab-base','/standing/run_standing.py',
  '--asset','/asset','--admission','/admission','--output','/output/standing','--num-envs',str(args.num_envs),*(['--standing-one','/standing_one'] if args.standing_one else []),'--device','cuda:0','--headless']

def load_supervisor(args,identity):
 global PARENT
 sys.path.insert(0,str(args.supervisor_source/'tools'))
 PARENT=load_module('_canonical_ownership_source009',args.supervisor_source/'tools/launch_reference_physics_spark.py')
 PARENT.command=lambda source,output,name,phase:command(args,name,phase)
 def native_verified_source(source):
  if source!=args.source:raise ValueError('Unexpected native source path')
  verify_tree(source,'FREEZE_SHA256.json',SOURCE_FREEZE)
 PARENT.verified_source=native_verified_source
 PARENT.RUNTIME_TREE=identity['runtime_binding']['runtime_tree_sha256']
 original_save=PARENT.save
 def metadata_save(path,data):
  if Path(path).parent.name=='jobs' and Path(path).name=='standing.json':
   data={**data,'no_policy_loaded':True,'training_allowed':False,'physical_admission':False,
    'standing_scope':'canonical_native_standing_v1','legacy_supervisor_reused_for_ownership_only':True}
  return original_save(path,data)
 PARENT.save=metadata_save
 return PARENT

def run_campaign(args,identity):
 report={'status':'preparing','started_unix':time.time(),'identity':identity,'host_freeze_sha256':args.host_freeze_sha256,
  'planned_phases':['standing'],'training_allowed':False,'physical_admission':False,'stage2_complete':False,
  'automatic_continuation':False,'source_freeze_sha256':SOURCE_FREEZE,'scope':'Provisional native standing screen only; no hardware, walking or training admission'}
 save(args.output/'campaign.json',report);phase_payloads=None
 try:
  PARENT.run_owned(args,'standing')
  if verify_inputs(args)!=identity:raise ValueError('Original inputs changed after container exit')
  report['post_exit_original_inputs_reverified']=True
  report['standing']=CONTRACT.validate_result(args.output/'standing',identity)
  phase_payloads=tree(args.output/'standing')
  save(args.output/'standing_immutable.sha256.json',phase_payloads)
  report['standing_payload_manifest_sha256']=sha(args.output/'standing_immutable.sha256.json')
  report['post_exit_all_standing_payloads_inventoried']=True
  report['status']='completed'
 except Exception as exc:
  report.update(status='stopped' if isinstance(exc,InterruptedError) else 'failed',error=repr(exc));raise
 finally:
  try:
   if verify_inputs(args)!=identity:raise ValueError('Terminal input identity differs')
   report['terminal_inputs_unchanged']=True
   if phase_payloads is not None and tree(args.output/'standing')!=phase_payloads:raise ValueError('Query payload changed after post-exit seal')
  except Exception as exc:report.update(status='failed',terminal_inputs_unchanged=False,integrity_error=repr(exc))
  report['finished_unix']=time.time();save(args.output/'campaign.json',report)
 if report['status']!='completed':raise RuntimeError('Native standing terminal integrity failed')
 return report

def main():
 p=argparse.ArgumentParser(allow_abbrev=False,description=__doc__)
 for name in ('source','asset','admission','supervisor-source','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--num-envs',type=int,choices=[1,32],required=True);p.add_argument('--standing-one',type=Path)
 p.add_argument('--isaaclab',type=Path,default=Path('/home/orionh/IsaacLab'));p.add_argument('--preflight-only',action='store_true')
 args=p.parse_args()
 for name in ('source','asset','admission','supervisor_source','output','isaaclab'):setattr(args,name,getattr(args,name).resolve())
 if args.standing_one is not None:args.standing_one=args.standing_one.resolve()
 require_fresh_output(args);args.host_freeze_sha256=verify_own_bundle();identity=verify_inputs(args)
 if args.preflight_only:print(json.dumps({'identity':identity,'host_freeze_sha256':args.host_freeze_sha256,'no_GPU':True},indent=2));return
 args.coordination_sha256=sha(COORDINATION)
 if args.coordination_sha256!=EXPECTED_COORDINATION:raise ValueError('Coordination changed; no allocation')
 load_supervisor(args,identity);args.output.mkdir(parents=True)
 for name in ('jobs','logs'):(args.output/name).mkdir()
 signal.signal(signal.SIGTERM,lambda *_:(args.output/'stop.request').touch());signal.signal(signal.SIGINT,lambda *_:(args.output/'stop.request').touch())
 run_campaign(args,identity)
if __name__=='__main__':main()
