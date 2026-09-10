"""Cold preserved315 comparison only: fresh standing then complete 12-case diagnostic."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,importlib.util,json,signal,sys,time
sys.dont_write_bytecode=True
SOURCE_MAP='4671eab012aaf79ce49769cfdc1667142237a4b4a45966fd0399a9caa900c51b'
LEGACY_RUNTIME_TREE='abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280'
CONTRACT_FREEZE='eb87f1dd456771950f7c1bf50ed0a51c1fc6ccb4631217e29d99fa8a84717095'
SUPERVISOR_MAP='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
SUPERVISOR_CODE='9ebabaf254cc77fabad7d4759cc80fc26d7efd3912ea84bd55304594c88ed561'
COORDINATION=Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md')
PHASES=('standing','baseline')
CONTRACT=None
PARENT=None

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)
def tree_hashes(p):return {f.relative_to(p).as_posix():sha(f) for f in sorted(p.rglob('*')) if f.is_file()}
def verify_tree(root,manifest,bound):
 if len(bound)!=64 or sha(root/manifest)!=bound:raise ValueError('Wrong or unbound immutable manifest: '+str(root))
 if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):raise ValueError('Symbolic input substitution')
 actual=tree_hashes(root);actual.pop(manifest)
 if actual!=read(root/manifest):raise ValueError('Changed/unlisted frozen input: '+str(root))
def verify_own_bundle(expected=None):
 p=Path(__file__).resolve().parent;bound=sha(p/'FREEZE_SHA256.json')
 if expected is not None and expected!=bound:raise ValueError('Host changed during campaign')
 verify_tree(p,'FREEZE_SHA256.json',bound);return bound

def verify_legacy_runtime(source):
 expected=read(Path(__file__).resolve().parent/'LEGACY_RUNTIME_SHA256.json')
 normalized={k.removeprefix('isaaclab/hexapod_rl/'):v for k,v in expected.items()}
 if len(expected)!=16 or hashlib.sha256(json.dumps(normalized,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=LEGACY_RUNTIME_TREE:raise ValueError('Legacy runtime map identity mismatch')
 actual={p.relative_to(source).as_posix() for p in (source/'isaaclab/hexapod_rl').rglob('*.py')}
 if actual!=set(expected):raise ValueError('Legacy runtime inventory mismatch')
 for k,bound in expected.items():
  if (source/k).is_symlink() or sha(source/k)!=bound:raise ValueError('Legacy runtime changed: '+k)
 return expected

def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def verify_inputs(args):
 global CONTRACT
 verify_own_bundle(getattr(args,'host_freeze_sha256',None))
 verify_tree(args.source,'campaign_source_hashes.json',SOURCE_MAP)
 verify_legacy_runtime(args.source)
 verify_tree(args.contract,'FREEZE_SHA256.json',CONTRACT_FREEZE)
 verify_tree(args.supervisor_source,'campaign_source_hashes.json',SUPERVISOR_MAP)
 if sha(args.supervisor_source/'tools/launch_reference_physics_spark.py')!=SUPERVISOR_CODE:raise ValueError('Wrong source009 supervisor')
 if CONTRACT is None:CONTRACT=load_module('_frozen_direct315_cold_contract',args.contract/'cold_contract.py')
 if Path(CONTRACT.__file__).resolve()!=args.contract/'cold_contract.py':raise ValueError('Wrong imported contract')
 identity=CONTRACT.verify_inputs(args)
 if identity['source_manifest_sha256']!=SOURCE_MAP:raise ValueError('Cold source identity mismatch')
 return identity

validate_inputs=verify_inputs

def command(args,name,phase):
 if phase not in PHASES:raise ValueError('Only standing and baseline exist')
 cli=['docker','compose','--env-file','docker/.env.base','-f','docker/docker-compose.yaml','--profile','base',
      'run','--rm','--no-deps','--name',name,'-w','/output','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
      '-e','PYTHONPATH=/workspace/isaaclab/source/isaaclab:/source/isaaclab:/source/tools',
      '-v',str(Path(__file__).resolve().parent)+':/cold-host:ro','-v',str(args.source)+':/source:ro','-v',str(args.checkpoint)+':/checkpoint/original.pt:ro',
      '-v',str(args.output)+':/output:rw']
 if phase=='baseline':cli+=['-v',str(args.output/'standing')+':/admission:ro','-v',str(args.output/'standing')+':/output/standing:ro']
 return cli+['--entrypoint','/workspace/isaaclab/_isaac_sim/python.sh','isaac-lab-base','/cold-host/run_cold_entry.py','--source-root','/source','--']+CONTRACT.runtime_arguments(phase)

def load_supervisor(args):
 global PARENT
 sys.path.insert(0,str(args.supervisor_source/'tools'))
 PARENT=load_module('_frozen_cold_source009_supervisor',args.supervisor_source/'tools/launch_reference_physics_spark.py')
 PARENT.command=lambda source,output,name,phase:command(args,name,phase)
 def cold_verified_source(source):
  if source!=args.source:raise ValueError('Unexpected source path in inherited supervisor')
  verify_tree(source,'campaign_source_hashes.json',SOURCE_MAP)
  verify_legacy_runtime(source)
 PARENT.verified_source=cold_verified_source
 PARENT.RUNTIME_TREE=LEGACY_RUNTIME_TREE
 original_save=PARENT.save
 def metadata_save(path,data):
  if Path(path).parent.name=='jobs' and Path(path).name in ('standing.json','baseline.json'):
   data=dict(data)
   if 'no_policy_loaded' in data:data['legacy_supervisor_no_policy_loaded']=data['no_policy_loaded']
   data['policy_phase_requested']=data.get('phase')=='baseline'
   state_path=args.output/'baseline/state.json'
   baseline_state=read(state_path) if state_path.is_file() else {}
   loaded=baseline_state.get('status')=='completed' and baseline_state.get('checkpoint_sha256')=='1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8'
   data['checkpoint_load_verified_by_completed_state']=loaded
   data['no_policy_loaded']=True if data.get('phase')=='standing' else (False if loaded else None)
  return original_save(path,data)
 PARENT.save=metadata_save
 return PARENT

def run_campaign(args,identity):
 report=dict(status='preparing',started_unix=time.time(),identity=identity,host_freeze_sha256=args.host_freeze_sha256,
             standing_admitted=False,baseline_complete=False,policy_training_started=False,PPO_updates_completed=0,
             Stage2_complete=False,automatic_continuation=False,phase_scope={'standing':'32x1000','baseline':'48 replicas / 12 cases / 12 seconds'},
             accepted_phases={})
 save(args.output/'campaign.json',report)
 standing_map=None
 try:
  for phase in PHASES:
   if (args.output/'stop.request').exists():raise InterruptedError('Stop requested before next phase')
   if verify_inputs(args)!=identity:raise ValueError('Identity changed before phase')
   if standing_map is not None and tree_hashes(args.output/'standing')!=standing_map:raise ValueError('Completed standing changed')
   PARENT.run_owned(args,phase)
   if verify_inputs(args)!=identity:raise ValueError('Identity changed after phase')
   accepted=CONTRACT.validate_result(args.output/phase,phase,identity)
   if accepted['phase']!=phase:raise ValueError('Result phase mismatch')
   if phase=='standing':
    standing_map=tree_hashes(args.output/'standing');save(args.output/'standing_immutable.sha256.json',standing_map)
    report['standing_admitted']=True
   elif tree_hashes(args.output/'standing')!=standing_map:raise ValueError('Baseline altered standing evidence')
   report['accepted_phases'][phase]=accepted
   save(args.output/(phase+'_accepted.json'),accepted)
   report['last_completed_phase']=phase;save(args.output/'campaign.json',report)
  report.update(status='completed',baseline_complete=True)
 except Exception as e:
  report.update(status='stopped' if isinstance(e,InterruptedError) else 'failed',error=repr(e))
  raise
 finally:
  try:
   if verify_inputs(args)!=identity:raise ValueError('Terminal input identity differs')
   if standing_map is not None and tree_hashes(args.output/'standing')!=standing_map:raise ValueError('Terminal standing changed')
   report['terminal_inputs_unchanged']=True
  except Exception as e:report.update(status='failed',terminal_inputs_unchanged=False,terminal_integrity_error=repr(e))
  report['finished_unix']=time.time();save(args.output/'campaign.json',report)
 if report['status']!='completed':raise RuntimeError('Terminal integrity failed')
 return report

def main():
 p=argparse.ArgumentParser(allow_abbrev=False,description=__doc__)
 for k in ('source','checkpoint','contract','supervisor-source','output'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--isaaclab',type=Path,default=Path('/home/orionh/IsaacLab'));p.add_argument('--preflight-only',action='store_true')
 args=p.parse_args()
 for k in ('source','checkpoint','contract','supervisor_source','output','isaaclab'):setattr(args,k,getattr(args,k).resolve())
 for immutable in (args.source,args.contract,args.supervisor_source,args.checkpoint.parent):
  if args.output==immutable or immutable in args.output.parents:p.error('Output must be outside immutable inputs')
 if args.output.exists():p.error('Fresh output required; no implicit continuation')
 args.host_freeze_sha256=verify_own_bundle();identity=verify_inputs(args)
 if args.preflight_only:print(json.dumps({'identity':identity,'host_freeze_sha256':args.host_freeze_sha256,'no_GPU':True},indent=2));return
 load_supervisor(args)
 args.coordination_sha256=sha(COORDINATION)
 args.output.mkdir(parents=True)
 for name in ('logs','jobs'): (args.output/name).mkdir()
 signal.signal(signal.SIGTERM,lambda *_:(args.output/'stop.request').touch())
 signal.signal(signal.SIGINT,lambda *_:(args.output/'stop.request').touch())
 run_campaign(args,identity)
if __name__=='__main__':main()
