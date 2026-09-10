"""Explicit native005 smoke, pilot and extended campaigns; versioned extended deadline adapter."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,importlib.util,json,signal,sys,time
sys.dont_write_bytecode=True
SOURCE_MAP='aa70659d33243f1e28e9c71b1a7f9d33341642db770fd7356192573a26fe904e'
LEGACY_RUNTIME_TREE='abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280'
CONTRACT_FREEZE='0bdd2abf3811e68ac59e4fbc325eb5d2bb96afd2183eb84c4d37faeb86ff9e9f'
SUPERVISOR_MAP='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
SUPERVISOR_CODE='9ebabaf254cc77fabad7d4759cc80fc26d7efd3912ea84bd55304594c88ed561'
COORDINATION=Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md')
ORIGINAL_CHECKPOINT='1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8'
PHASES={'smoke':('standing','train','final_constant','final_stop'),
        'pilot':('standing','initial_constant','initial_stop','train','final_constant','final_stop'),
        'extended':('standing','initial_constant','initial_stop','train','final_constant','final_stop')}
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

def load_deadline_adapter():
 return load_module('_direct_host004_deadline_adapter',Path(__file__).resolve().parent/'deadline_adapter.py')

def selected_phases(args):
 if args.allocation not in PHASES or args.branch not in ('caps','curriculum','quiet_priority'):raise ValueError('Explicit bounded allocation/branch required')
 if args.allocation=='smoke' and args.branch!='quiet_priority':raise ValueError('Only quiet-priority smoke is allocated')
 if args.allocation in ('pilot','extended') and getattr(args,'smoke',None) is None:raise ValueError('Pilot/extended requires separate completed smoke proof')
 if args.allocation=='smoke' and getattr(args,'smoke',None) is not None:raise ValueError('Smoke cannot inherit another campaign')
 return PHASES[args.allocation]

def verify_inputs(args):
 global CONTRACT
 selected_phases(args)
 verify_own_bundle(getattr(args,'host_freeze_sha256',None))
 verify_tree(args.source,'campaign_source_hashes.json',SOURCE_MAP);verify_legacy_runtime(args.source)
 verify_tree(args.contract,'FREEZE_SHA256.json',CONTRACT_FREEZE)
 verify_tree(args.supervisor_source,'campaign_source_hashes.json',SUPERVISOR_MAP)
 if sha(args.supervisor_source/'tools/launch_reference_physics_spark.py')!=SUPERVISOR_CODE:raise ValueError('Wrong source009 supervisor')
 if args.allocation=='extended':load_deadline_adapter().inspect_supervisor(args.supervisor_source/'tools/launch_reference_physics_spark.py')
 if CONTRACT is None:
  sys.path.insert(0,str(args.contract))
  CONTRACT=load_module('_frozen_direct315_native_contract',args.contract/'direct_contract.py')
 if Path(CONTRACT.__file__).resolve()!=args.contract/'direct_contract.py':raise ValueError('Wrong imported native contract')
 if Path(sys.modules['direct_config'].__file__).resolve()!=args.contract/'direct_config.py':raise ValueError('Wrong imported native configuration')
 identity=CONTRACT.verify_inputs(args)
 if identity['source_manifest_sha256']!=SOURCE_MAP or identity['checkpoint_sha256']!=ORIGINAL_CHECKPOINT:raise ValueError('Native source/original checkpoint identity mismatch')
 return identity

validate_inputs=verify_inputs

def phase_checkpoint(args,phase):
 if phase not in selected_phases(args):raise ValueError('Unallocated phase')
 if phase=='standing':return None,None
 if phase in ('train','initial_constant','initial_stop'):return args.checkpoint,ORIGINAL_CHECKPOINT
 receipt=read(args.output/'train/training_receipt.json');path=args.output/'train/policy/final.pt'
 if receipt.get('complete') is not True:raise ValueError('No complete training checkpoint')
 bound=receipt['final_checkpoint_sha256']
 if len(bound)!=64 or sha(path)!=bound:raise ValueError('Final checkpoint differs from immutable training receipt')
 return path,bound

def command(args,name,phase):
 phases=selected_phases(args)
 if phase not in phases:raise ValueError('Unallocated phase')
 checkpoint,bound=phase_checkpoint(args,phase)
 cli=['docker','compose','--env-file','docker/.env.base','-f','docker/docker-compose.yaml','--profile','base',
      'run','--rm','--no-deps','--name',name,'-w','/output','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
      '-e','PYTHONPATH=/workspace/isaaclab/source/isaaclab:/source/isaaclab:/source/tools',
      '-v',str(Path(__file__).resolve().parent)+':/train-host:ro','-v',str(args.source)+':/source:ro',
      '-v',str(args.checkpoint)+':/checkpoint/original.pt:ro','-v',str(args.output)+':/output:rw']
 for completed in phases[:phases.index(phase)]:
  cli+=['-v',str(args.output/completed)+':/output/'+completed+':ro']
 if phase!='standing':cli+=['-v',str(args.output/'standing')+':/admission:ro']
 if phase not in ('standing','train'):cli+=['-v',str(checkpoint)+':/checkpoint/evaluated.pt:ro']
 if args.allocation in ('pilot','extended'):cli+=['-v',str(args.smoke)+':/smoke:ro']
 return cli+['--entrypoint','/workspace/isaaclab/_isaac_sim/python.sh','isaac-lab-base','/train-host/run_train_entry.py','--source-root','/source','--']+CONTRACT.runtime_arguments(phase,args.allocation,args.branch)

def load_supervisor(args):
 global PARENT
 sys.path.insert(0,str(args.supervisor_source/'tools'))
 PARENT=load_module('_frozen_direct_native_source009_supervisor',args.supervisor_source/'tools/launch_reference_physics_spark.py')
 if args.allocation=='extended':load_deadline_adapter().install(PARENT,args.supervisor_source/'tools/launch_reference_physics_spark.py')
 PARENT.command=lambda source,output,name,phase:command(args,name,phase)
 def native_verified_source(source):
  if source!=args.source:raise ValueError('Unexpected source path in inherited supervisor')
  verify_tree(source,'campaign_source_hashes.json',SOURCE_MAP);verify_legacy_runtime(source)
 PARENT.verified_source=native_verified_source;PARENT.RUNTIME_TREE=LEGACY_RUNTIME_TREE
 original_save=PARENT.save
 def metadata_save(path,data):
  if Path(path).parent.name=='jobs' and Path(path).name in [phase+'.json' for phase in selected_phases(args)]:
   data=dict(data);phase=data['phase']
   if 'no_policy_loaded' in data:data['legacy_supervisor_no_policy_loaded']=data['no_policy_loaded']
   data['policy_phase_requested']=phase!='standing'
   state_path=args.output/phase/'state.json';state={};initialization={}
   try:
    state=read(state_path) if state_path.is_file() else {}
    init_path=args.output/'train/repair_initialization.json'
    if phase=='train' and init_path.is_file():initialization=read(init_path)
   except Exception as exc:data['policy_metadata_read_error']=repr(exc)
   expected=getattr(args,'phase_expected_checkpoints',{}).get(phase)
   loaded=phase!='standing' and expected is not None and state.get('status')=='completed' and (initialization.get('checkpoint_sha256') if phase=='train' else state.get('checkpoint_sha256'))==expected
   data['checkpoint_load_verified_by_completed_state']=loaded
   data['no_policy_loaded']=True if phase=='standing' else (False if loaded else None)
   data['allocation']=args.allocation;data['branch']=args.branch
   if args.allocation=='extended':data['supervisor_runtime_adapter']=dict(PARENT.DIRECT_DEADLINE_ADAPTER)
  return original_save(path,data)
 PARENT.save=metadata_save
 return PARENT

def validated_completed_updates(args,identity,accepted):
 selected=identity.get('selection',{})
 updates=selected.get('updates')
 if selected.get('allocation')!=args.allocation or selected.get('branch')!=args.branch or type(updates) is not int or updates<=0:
  raise ValueError('Validated training selection is missing or mismatched')
 receipt=read(args.output/'train/training_receipt.json')
 if accepted.get('phase')!='train' or type(accepted.get('updates_completed')) is not int or accepted['updates_completed']!=updates or receipt.get('complete') is not True or type(receipt.get('updates_completed')) is not int or receipt['updates_completed']!=updates:
  raise ValueError('Accepted training update count differs from validated selection/receipt')
 return updates

def observed_training(args):
 path=args.output/'train/training_receipt.json'
 if not path.is_file():return {'training_receipt_exists':False,'observed_updates_completed':None}
 receipt=read(path)
 return {'training_receipt_exists':True,'training_receipt_sha256':sha(path),
         'observed_updates_completed':receipt.get('updates_completed'),
         'training_receipt_complete':receipt.get('complete',False),'training_error':receipt.get('error'),
         'receipt_scope':'Preserve producer-reported updates even if a later phase fails; missing means unavailable.'}

def run_campaign(args,identity):
 phases=selected_phases(args);completed={}
 smoke_map=tree_hashes(args.smoke) if args.allocation in ('pilot','extended') else None
 def unchanged():
  if verify_inputs(args)!=identity:raise ValueError('Input identity changed')
  if smoke_map is not None and tree_hashes(args.smoke)!=smoke_map:raise ValueError('Completed prior smoke changed')
  for phase,mapping in completed.items():
   if tree_hashes(args.output/phase)!=mapping:raise ValueError('Completed phase changed: '+phase)
 report={'status':'preparing','started_unix':time.time(),'identity':identity,'host_freeze_sha256':args.host_freeze_sha256,
         'allocation':args.allocation,'branch':args.branch,'planned_phases':list(phases),'accepted_phases':{},
         'PPO_updates_completed':0,'training_attempted':False,'Stage2_complete':False,
         'automatic_continuation':False,'qualification':'Training/evaluation completion is unqualified diagnostic evidence'}
 save(args.output/'campaign.json',report)
 try:
  for phase in phases:
   if (args.output/'stop.request').exists():raise InterruptedError('Stop requested before next phase')
   unchanged();checkpoint,bound=phase_checkpoint(args,phase)
   if not hasattr(args,'phase_expected_checkpoints'):args.phase_expected_checkpoints={}
   args.phase_expected_checkpoints[phase]=bound
   if phase=='train':report['training_attempted']=True;save(args.output/'campaign.json',report)
   PARENT.run_owned(args,phase);unchanged()
   if checkpoint is not None and sha(checkpoint)!=bound:raise ValueError('Input checkpoint changed during phase')
   accepted=CONTRACT.validate_result(args.output/phase,phase,identity,expected_checkpoint_sha256=bound)
   if accepted['phase']!=phase:raise ValueError('Result phase mismatch')
   completed_updates=validated_completed_updates(args,identity,accepted) if phase=='train' else None
   completed[phase]=tree_hashes(args.output/phase)
   save(args.output/(phase+'_immutable.sha256.json'),completed[phase])
   report['accepted_phases'][phase]=accepted;report['last_completed_phase']=phase
   if phase=='train':report['PPO_updates_completed']=completed_updates
   save(args.output/(phase+'_accepted.json'),accepted);save(args.output/'campaign.json',report)
  report.update(status='completed',bounded_campaign_complete=True)
 except Exception as exc:
  report.update(status='stopped' if isinstance(exc,InterruptedError) else 'failed',error=repr(exc));raise
 finally:
  try:unchanged();report['terminal_inputs_unchanged']=True
  except Exception as exc:report.update(status='failed',terminal_inputs_unchanged=False,terminal_integrity_error=repr(exc))
  try:report['observed_training']=observed_training(args)
  except Exception as exc:report.update(status='failed',training_export_error=repr(exc))
  report['finished_unix']=time.time();save(args.output/'campaign.json',report)
 if report['status']!='completed':raise RuntimeError('Terminal integrity failed')
 return report

def main():
 p=argparse.ArgumentParser(allow_abbrev=False,description=__doc__)
 for name in ('source','checkpoint','contract','supervisor-source','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--allocation',choices=tuple(PHASES),required=True);p.add_argument('--branch',choices=('curriculum','caps','quiet_priority'),required=True)
 p.add_argument('--smoke',type=Path);p.add_argument('--isaaclab',type=Path,default=Path('/home/orionh/IsaacLab'));p.add_argument('--preflight-only',action='store_true')
 args=p.parse_args()
 for name in ('source','checkpoint','contract','supervisor_source','output','isaaclab','smoke'):
  if getattr(args,name) is not None:setattr(args,name,getattr(args,name).resolve())
 for immutable in [args.source,args.contract,args.supervisor_source,args.checkpoint.parent]+([args.smoke] if args.smoke else []):
  if args.output==immutable or immutable in args.output.parents:p.error('Output must be outside immutable inputs')
 if args.output.exists():p.error('Fresh output required; no implicit continuation')
 args.host_freeze_sha256=verify_own_bundle();identity=verify_inputs(args)
 if args.preflight_only:print(json.dumps({'identity':identity,'host_freeze_sha256':args.host_freeze_sha256,'no_GPU':True},indent=2));return
 load_supervisor(args);args.coordination_sha256=sha(COORDINATION);args.output.mkdir(parents=True)
 for name in ('logs','jobs'):(args.output/name).mkdir()
 signal.signal(signal.SIGTERM,lambda *_:(args.output/'stop.request').touch())
 signal.signal(signal.SIGINT,lambda *_:(args.output/'stop.request').touch())
 run_campaign(args,identity)
if __name__=='__main__':main()
