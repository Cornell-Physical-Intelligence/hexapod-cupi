"""Single native canonical-model inspection; no controller, training or physical admission."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,importlib.util,json,signal,sys,time
sys.dont_write_bytecode=True
SOURCE_FREEZE='a9a4dbe0e28c0ed85a5d5d6dd7788bdf8bc1b9154f676aa7262fc6eb4beac3f9'
SUPERVISOR_MAP='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
SUPERVISOR_CODE='9ebabaf254cc77fabad7d4759cc80fc26d7efd3912ea84bd55304594c88ed561'
SELECTOR_SHA256='d11edd466902e2122d6cec7cd1b0c4399b566a6ed1598d6bfe911317197353e4'
COORDINATION=Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md')
EXPECTED_COORDINATION='35af5100e39980f55a3f5e19654e4550b3c3fb6da114b64fbe0d143817242cf3'
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
 if expected is not None and expected!=bound:raise ValueError('Host changed during inspection')
 verify_tree(p,'FREEZE_SHA256.json',bound);return bound

def require_fresh_output(args):
 if args.output.exists() or args.output.is_symlink():raise ValueError('Fresh inspection output required')
 for p in (args.source,args.asset,args.supervisor_source,Path(__file__).resolve().parent):
  if args.output==p or p in args.output.parents or args.output in p.parents:raise ValueError('Output overlaps immutable input')
def load_module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def verify_inputs(args):
 global CONTRACT
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
 if CONTRACT is None:CONTRACT=load_module('_canonical_native_inspection_contract',args.source/'inspection_contract.py')
 if Path(CONTRACT.__file__).resolve()!=args.source/'inspection_contract.py':raise ValueError('Wrong native contract import origin')
 identity=CONTRACT.verify_inputs(SimpleNamespace(source=args.source,asset=args.asset,output=args.output/'inspection'))
 if identity.get('training_allowed') is not False or identity.get('physical_admission') is not False:raise ValueError('Native inspection cannot admit training or physics')
 if identity.get('schema')!='canonical_native_inspection_v1' or identity.get('inspector_freeze_sha256')!=SOURCE_FREEZE or identity.get('runtime_binding')!={'runtime_tree_sha256':SOURCE_FREEZE,'scope':'canonical_native_inspection_only'}:raise ValueError('Native inspection runtime identity missing or incompatible')
 if identity.get('steps')!=8 or identity.get('dt')!=.0025 or identity.get('gravity')!=[0.,0.,0.]:raise ValueError('Unexpected inspection allocation')
 return identity

validate_inputs=verify_inputs

def command(args,name,phase):
 if phase!='inspection':raise ValueError('Only native inspection is allocated')
 return ['docker','compose','--env-file','docker/.env.base','-f','docker/docker-compose.yaml','--profile','base',
  'run','--rm','--no-deps','--name',name,'-w','/output','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
  '-e','PYTHONPATH=/inspection:/workspace/isaaclab/source/isaaclab',
  '-v',str(args.output)+':/output:rw','-v',str(args.source)+':/inspection:ro','-v',str(args.asset)+':/asset:ro',
  '--entrypoint','/workspace/isaaclab/_isaac_sim/python.sh','isaac-lab-base','/inspection/run_inspection.py',
  '--asset','/asset','--output','/output/inspection','--device','cuda:0','--headless']

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
  if Path(path).parent.name=='jobs' and Path(path).name=='inspection.json':
   data={**data,'no_policy_loaded':True,'training_allowed':False,'physical_admission':False,
    'inspection_scope':'canonical_native_inspection_v1','legacy_supervisor_reused_for_ownership_only':True}
  return original_save(path,data)
 PARENT.save=metadata_save
 return PARENT

def run_campaign(args,identity):
 report={'status':'preparing','started_unix':time.time(),'identity':identity,'host_freeze_sha256':args.host_freeze_sha256,
  'planned_phases':['inspection'],'training_allowed':False,'physical_admission':False,'stage2_complete':False,
  'automatic_continuation':False,'source_freeze_sha256':SOURCE_FREEZE,'scope':'Native canonical asset inspection only; no controller/gain/target or physical admission'}
 save(args.output/'campaign.json',report);phase_payloads=None
 try:
  PARENT.run_owned(args,'inspection')
  if verify_inputs(args)!=identity:raise ValueError('Original inputs changed after container exit')
  report['post_exit_original_inputs_reverified']=True
  report['inspection']=CONTRACT.validate_result(args.output/'inspection',identity)
  phase_payloads=tree(args.output/'inspection')
  save(args.output/'inspection_immutable.sha256.json',phase_payloads)
  report['inspection_payload_manifest_sha256']=sha(args.output/'inspection_immutable.sha256.json')
  report['post_exit_all_inspection_payloads_inventoried']=True
  report['status']='completed'
 except Exception as exc:
  report.update(status='stopped' if isinstance(exc,InterruptedError) else 'failed',error=repr(exc));raise
 finally:
  try:
   if verify_inputs(args)!=identity:raise ValueError('Terminal input identity differs')
   report['terminal_inputs_unchanged']=True
   if phase_payloads is not None and tree(args.output/'inspection')!=phase_payloads:raise ValueError('Inspection payload changed after post-exit seal')
  except Exception as exc:report.update(status='failed',terminal_inputs_unchanged=False,integrity_error=repr(exc))
  report['finished_unix']=time.time();save(args.output/'campaign.json',report)
 if report['status']!='completed':raise RuntimeError('Native inspection terminal integrity failed')
 return report

def main():
 p=argparse.ArgumentParser(allow_abbrev=False,description=__doc__)
 for name in ('source','asset','supervisor-source','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--isaaclab',type=Path,default=Path('/home/orionh/IsaacLab'));p.add_argument('--preflight-only',action='store_true')
 args=p.parse_args()
 for name in ('source','asset','supervisor_source','output','isaaclab'):setattr(args,name,getattr(args,name).resolve())
 require_fresh_output(args);args.host_freeze_sha256=verify_own_bundle();identity=verify_inputs(args)
 if args.preflight_only:print(json.dumps({'identity':identity,'host_freeze_sha256':args.host_freeze_sha256,'no_GPU':True},indent=2));return
 args.coordination_sha256=sha(COORDINATION)
 if args.coordination_sha256!=EXPECTED_COORDINATION:raise ValueError('Coordination changed; no allocation')
 load_supervisor(args,identity);args.output.mkdir(parents=True)
 for name in ('jobs','logs'):(args.output/name).mkdir()
 signal.signal(signal.SIGTERM,lambda *_:(args.output/'stop.request').touch());signal.signal(signal.SIGINT,lambda *_:(args.output/'stop.request').touch())
 run_campaign(args,identity)
if __name__=='__main__':main()
