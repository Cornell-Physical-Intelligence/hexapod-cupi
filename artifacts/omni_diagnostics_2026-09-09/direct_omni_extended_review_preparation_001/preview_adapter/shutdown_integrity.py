"""Pre-shutdown evidence seal; process completion is verified by the host."""
from pathlib import Path
import hashlib,inspect,json
SCHEMA='direct_preview_pre_shutdown_v1'
SIMULATION_APP_SHA256='6af5372bb0cdda6e665a30cd285ff724b3e5bd012dbdd4df458e0726173fc903'
NAME='pre_shutdown_integrity.json'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def tree(root):return {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
def write(path,value):
 temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def shutdown_runtime(app):
 source=Path(inspect.getsourcefile(type(app))).resolve()
 if sha(source)!=SIMULATION_APP_SHA256 or app.config.get('fast_shutdown') is not True:raise ValueError('Installed shutdown source/default differs')
 return {'source_path':str(source),'source_sha256':SIMULATION_APP_SHA256,'fast_shutdown':True,'close_return_required':False}
def validate_shutdown_seal(output,provenance):
 record=read(output/NAME)
 if record.get('schema')!=SCHEMA or record.get('passed') is not True or record.get('provenance')!=provenance or record.get('scope')!='before_native_app_close' or record.get('process_exit_verified') is not False:raise ValueError('Missing/mismatched pre-shutdown seal')
 actual=tree(output);actual.pop(NAME)
 if actual!=record.get('output_hashes'):raise ValueError('Post-seal output changed or missing')
 if record.get('shutdown_runtime',{}).get('source_sha256')!=SIMULATION_APP_SHA256:raise ValueError('Unpinned shutdown API')
 return record

def seal_before_shutdown(args,provenance,app,verify_inputs):
 output=args.output
 if (output/NAME).exists():raise ValueError('Duplicate pre-shutdown seal')
 if any(p.is_symlink() for p in output.rglob('*')):raise ValueError('Symbolic recording output')
 if verify_inputs(args)!=provenance:raise ValueError('Inputs changed before native shutdown')
 video=read(output/'video.json');state=read(output/'state.json')
 if any((output/name).exists() for name in ('failure.json','recording_failure.json')) or state.get('status')!='completed' or video.get('complete') is not True or video.get('terminal_event') is not None or video.get('error') is not None:raise ValueError('Failed/incomplete recording cannot be sealed')
 if state.get('recording_provenance')!=provenance or any(video.get(k)!=v for k,v in provenance.items()):raise ValueError('Recording provenance differs')
 if (video.get('frames'),video.get('recorded_control_steps'),video.get('fps'))!=(950,1900,25):raise ValueError('Incomplete frame/control count')
 for name in ('before','after'):
  check=read(output/('checkpoint_readback_'+name+'.json'))
  if check!=video.get('checkpoint_readback_'+name) or check.get('passed') is not True or check.get('actor_and_critic_including_normalizers_exact') is not True or check.get('checkpoint_sha256')!=provenance['checkpoint_sha256']:raise ValueError('Original exact checkpoint readback missing')
 for name,key in (('rollout.mp4','video_sha256'),('trace.npz','trace_sha256')):
  if sha(output/name)!=video.get(key):raise ValueError('Recorded media/trace changed')
 for name in ('provenance.json','entry_seams.json','native_arguments.json','environment.yaml','matched_evaluation_environment.yaml'):
  if not (output/name).is_file():raise ValueError('Required entry/configuration evidence missing: '+name)
 runtime=shutdown_runtime(app)
 write(output/NAME,{'schema':SCHEMA,'passed':True,'scope':'before_native_app_close','process_exit_verified':False,
  'provenance':provenance,'shutdown_runtime':runtime,'output_hashes':tree(output),
  'host_obligation':'Verify actual exit0, cleanup, unchanged original inputs and all sealed output bytes independently.'})
 return validate_shutdown_seal(output,provenance)
