"""One unqualified native005 CAPS500 direct-PPO progress recording; no training."""
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
ADAPTER_FREEZE='fb60da984d73c01725e453521ca861e02b061103dfc2fb266a56aa9dd1fa7a20'
ADAPTER=None
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

def require_fresh_output(args):
 if args.output.exists():raise ValueError('Fresh output required; no recording continuation')
 for immutable in (args.source,args.contract,args.supervisor_source,args.adapter,args.pilot,args.checkpoint.parent,Path(__file__).resolve().parent):
  if args.output==immutable or immutable in args.output.parents or args.output in immutable.parents:raise ValueError('Output overlaps immutable input')

def adapter_args(args):
 return SimpleNamespace(source_root=args.source,native_contract=args.contract,pilot=args.pilot,
  checkpoint=args.checkpoint,checkpoint_sha256=args.checkpoint_sha256,admission=args.pilot/'standing/admission.json',
  output=args.output/'recording',seed=7057)

def verify_inputs(args):
 global ADAPTER
 verify_own_bundle(getattr(args,'host_freeze_sha256',None))
 verify_tree(args.source,'campaign_source_hashes.json',SOURCE_MAP);verify_legacy_runtime(args.source)
 verify_tree(args.contract,'FREEZE_SHA256.json',CONTRACT_FREEZE)
 verify_tree(args.supervisor_source,'campaign_source_hashes.json',SUPERVISOR_MAP)
 verify_tree(args.adapter,'FREEZE_SHA256.json',ADAPTER_FREEZE)
 if sha(args.supervisor_source/'tools/launch_reference_physics_spark.py')!=SUPERVISOR_CODE:raise ValueError('Wrong source009 supervisor')
 if len(args.campaign_sha256)!=64 or sha(args.pilot/'campaign.json')!=args.campaign_sha256:raise ValueError('Selected campaign SHA mismatch or pending')
 if ADAPTER is None:ADAPTER=load_module('_bound_direct_preview_contract',args.adapter/'preview_contract.py')
 if Path(ADAPTER.__file__).resolve()!=args.adapter/'preview_contract.py':raise ValueError('Wrong preview contract origin')
 identity=ADAPTER.verify_inputs(adapter_args(args))
 if identity['source_manifest_sha256']!=SOURCE_MAP or identity['checkpoint_sha256']!=args.checkpoint_sha256:raise ValueError('Preview identity differs')
 return identity

validate_inputs=verify_inputs

def command(args,name,phase):
 if phase!='recording':raise ValueError('Only recording is allocated')
 return ['docker','compose','--env-file','docker/.env.base','-f','docker/docker-compose.yaml','--profile','base',
  'run','--rm','--no-deps','--name',name,'-w','/output','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
  '-e','PYTHONPATH=/workspace/isaaclab/source/isaaclab:/source/isaaclab:/source/tools',
  '-v',str(args.output)+':/output:rw','-v',str(args.source)+':/source:ro',
  '-v',str(args.contract)+':/contract:ro','-v',str(args.adapter)+':/recording:ro',
  '-v',str(args.pilot)+':/pilot:ro','-v',str(args.pilot/'standing')+':/admission:ro',
  '-v',str(args.checkpoint)+':/checkpoint/evaluated.pt:ro',
  '--entrypoint','/workspace/isaaclab/_isaac_sim/python.sh','isaac-lab-base','/recording/record_direct_preview.py',
  '--source-root','/source','--native-contract','/contract','--pilot','/pilot','--checkpoint','/checkpoint/evaluated.pt',
  '--checkpoint-sha256',args.checkpoint_sha256,'--admission','/admission/admission.json',
  '--output','/output/recording','--seed','7057','--device','cuda:0']

def load_supervisor(args):
 global PARENT
 sys.path.insert(0,str(args.supervisor_source/'tools'))
 PARENT=load_module('_frozen_preview_source009_supervisor',args.supervisor_source/'tools/launch_reference_physics_spark.py')
 PARENT.command=lambda source,output,name,phase:command(args,name,phase)
 def native_verified_source(source):
  if source!=args.source:raise ValueError('Unexpected source path in inherited supervisor')
  verify_tree(source,'campaign_source_hashes.json',SOURCE_MAP);verify_legacy_runtime(source)
 PARENT.verified_source=native_verified_source;PARENT.RUNTIME_TREE=LEGACY_RUNTIME_TREE
 original_save=PARENT.save
 def metadata_save(path,data):
  if Path(path).parent.name=='jobs' and Path(path).name=='recording.json':
   data=dict(data)
   if 'no_policy_loaded' in data:data['legacy_supervisor_no_policy_loaded']=data['no_policy_loaded']
   data['policy_phase_requested']=True;data['no_policy_loaded']=None
   state_path=args.output/'recording/state.json'
   try:state=read(state_path) if state_path.is_file() else {}
   except Exception as exc:state={};data['policy_metadata_read_error']=repr(exc)
   loaded=state.get('status')=='completed' and state.get('checkpoint_sha256')==args.checkpoint_sha256
   data['checkpoint_load_verified_by_completed_state']=loaded
   if loaded:data['no_policy_loaded']=False
   data['recording_only']=True
  return original_save(path,data)
 PARENT.save=metadata_save
 return PARENT

def validate_result(output,identity):
 state=read(output/'state.json');video=read(output/'video.json');final=read(output/'pre_shutdown_integrity.json')
 if state.get('status')!='completed' or state.get('checkpoint_sha256')!=identity['checkpoint_sha256'] or state.get('runtime_binding',{}).get('runtime_tree_sha256')!=LEGACY_RUNTIME_TREE:raise ValueError('Incomplete or wrong native recording state')
 expected={'complete':True,'frames':950,'fps':25,'recorded_control_steps':1900,'planned_control_steps':1900,
  'planned_frames':950,'physics_duration_s':38.,'playback_duration_s':38.,'playback_rate':1.,'dt_s':.02,
  'source_and_inputs_reverified_after_recording':True,'stage2_complete':False,'policy_training_started':False,
  'pose_forcing':False,'qualification_performed':False,'terminal_event':None,'error':None}
 for key,value in expected.items():
  if video.get(key)!=value:raise ValueError('Incomplete/mismatched preview: '+key)
 for key in ('source_manifest_sha256','plan_sha256','native_contract_freeze_sha256','adapter_freeze_sha256','checkpoint_sha256','pilot','rendering_overrides'):
  if video.get(key)!=identity[key]:raise ValueError('Preview provenance mismatch: '+key)
 if (final.get('schema')!='direct_preview_pre_shutdown_v1' or final.get('scope')!='before_native_app_close' or final.get('process_exit_verified') is not False or final.get('passed') is not True or final.get('provenance')!=state.get('recording_provenance')):raise ValueError('Pre-shutdown integrity receipt mismatch')
 shutdown=final.get('shutdown_runtime',{})
 if shutdown.get('source_sha256')!='6af5372bb0cdda6e665a30cd285ff724b3e5bd012dbdd4df458e0726173fc903' or shutdown.get('fast_shutdown') is not True or shutdown.get('close_return_required') is not False:raise ValueError('Unpinned installed shutdown contract')
 sealed=tree_hashes(output);sealed.pop('pre_shutdown_integrity.json')
 if sealed!=final.get('output_hashes'):raise ValueError('Sealed recording evidence changed or missing after process exit')
 for key,value in state['recording_provenance'].items():
  if video.get(key)!=value:raise ValueError('Video/native-state provenance differs: '+key)
 for when in ('before','after'):
  receipt=video.get('checkpoint_readback_'+when,{})
  if receipt.get('passed') is not True or receipt.get('actor_and_critic_including_normalizers_exact') is not True or receipt.get('checkpoint_sha256')!=identity['checkpoint_sha256']:raise ValueError('Strict checkpoint readback missing')
 for file,key in (('rollout.mp4','video_sha256'),('trace.npz','trace_sha256')):
  if not (output/file).is_file() or sha(output/file)!=video.get(key):raise ValueError('Recording payload differs: '+file)
 return {'complete':True,'video_sha256':video['video_sha256'],'trace_sha256':video['trace_sha256'],
  'checkpoint_sha256':identity['checkpoint_sha256'],'frames':950,'control_steps':1900,'qualification_performed':False}

def run_campaign(args,identity):
 report={'status':'preparing','started_unix':time.time(),'identity':identity,'host_freeze_sha256':args.host_freeze_sha256,
  'planned_phases':['recording'],'recording_only':True,'policy_training_started':False,'stage2_complete':False,
  'qualification':'Fresh progress video only; no quality acceptance or automatic continuation'}
 save(args.output/'campaign.json',report)
 try:
  PARENT.run_owned(args,'recording')
  if verify_inputs(args)!=identity:raise ValueError('Original inputs changed after container exit')
  report['post_exit_original_inputs_reverified']=True
  report['recording']=validate_result(args.output/'recording',identity);report['status']='completed'
 except Exception as exc:
  report.update(status='stopped' if isinstance(exc,InterruptedError) else 'failed',error=repr(exc));raise
 finally:
  try:
   if verify_inputs(args)!=identity:raise ValueError('Preview input identity changed')
   report['terminal_inputs_unchanged']=True
  except Exception as exc:report.update(status='failed',terminal_inputs_unchanged=False,integrity_error=repr(exc))
  report['finished_unix']=time.time();save(args.output/'campaign.json',report)
 if report['status']!='completed':raise RuntimeError('Terminal recording integrity failed')
 return report

def main():
 p=argparse.ArgumentParser(allow_abbrev=False,description=__doc__)
 for name in ('source','contract','supervisor-source','adapter','pilot','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--campaign-sha256',required=True)
 p.add_argument('--isaaclab',type=Path,default=Path('/home/orionh/IsaacLab'));p.add_argument('--preflight-only',action='store_true')
 args=p.parse_args()
 for name in ('source','contract','supervisor_source','adapter','pilot','checkpoint','output','isaaclab'):setattr(args,name,getattr(args,name).resolve())
 require_fresh_output(args);args.host_freeze_sha256=verify_own_bundle();identity=verify_inputs(args)
 if args.preflight_only:print(json.dumps({'identity':identity,'host_freeze_sha256':args.host_freeze_sha256,'no_GPU':True},indent=2));return
 load_supervisor(args);args.coordination_sha256=sha(COORDINATION);args.output.mkdir(parents=True)
 for name in ('logs','jobs'):(args.output/name).mkdir()
 signal.signal(signal.SIGTERM,lambda *_:(args.output/'stop.request').touch())
 signal.signal(signal.SIGINT,lambda *_:(args.output/'stop.request').touch())
 run_campaign(args,identity)
if __name__=='__main__':main()
