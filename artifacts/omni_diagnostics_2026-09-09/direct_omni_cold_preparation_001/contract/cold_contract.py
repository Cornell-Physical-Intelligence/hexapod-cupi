"""Standard-library-only cold baseline contract. No dispatch or allocation."""
import hashlib,json,math
from pathlib import Path
CHECKPOINT='1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8'
PARENT='f469abdad81cab7bf63719c72cf4468b8827b802499f85a66d6b1abe8e9bd5e3'
PLAN='robot/hexapod_mkii_length_study/training_plan.json'
URDF='e916b1ebbb2720c90f23974bbffa5da120b32d2e5409fe419fb1492b8556746c'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def verify_inputs(args):
 source=args.source.resolve();m=read(source/'campaign_source_hashes.json')
 for k,v in m.items():
  p=source/k
  if p.is_symlink() or not p.resolve().is_relative_to(source) or sha(p)!=v:raise ValueError('Source mismatch: '+k)
 files={str(p.relative_to(source)) for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
 if files!=set(m)|{'campaign_source_hashes.json'}:raise ValueError('Unexpected source inventory')
 origin=read(source/'source_origin.json');plan=read(source/PLAN);o=plan['omni']
 if origin['schema']!='direct315_cold_formal004_v1' or origin['parent_historical_manifest_sha256']!=PARENT:raise ValueError('Wrong baseline lineage')
 if sha(args.checkpoint)!=CHECKPOINT:raise ValueError('Wrong original315 checkpoint')
 if (plan['validation_num_envs'],plan['validation_control_steps'],plan['evaluation_num_envs'],plan['physics_dt_s'],plan['decimation'])!=(32,1000,48,.0025,8):raise ValueError('Unexpected baseline allocation/dt')
 if o['overrides']['target_slew_rad_per_20ms']!=.04 or o['overrides']['observation_noise_scale']!=1 or o['overrides']['target_filter_time_constant_s']!=0:raise ValueError('Wrong declared control profile')
 if o['diagnostics']!={'duration_s':12,'settle_s':2,'seed':7057,'trace_envs_per_scenario':1,'controller':'policy'}:raise ValueError('Wrong diagnostic cases/bounds')
 return {'schema':'direct315_cold_formal004_v1','source_manifest_sha256':sha(source/'campaign_source_hashes.json'),'plan_sha256':sha(source/PLAN),'checkpoint_sha256':CHECKPOINT,'actor_width':315,'critic_width':318,'training_allowed':False,'Stage2_complete':False,'overrides':o['overrides'],'diagnostic_options':o['diagnostics']}
def validate_result(directory,phase,identity):
 directory=Path(directory);s=read(directory/'state.json')
 if s.get('status')!='completed' or s.get('variant')!='f050_t060' or s.get('urdf_sha256')!=URDF or s.get('plan_sha256')!=identity['plan_sha256'] or s.get('stance_index')!=0:raise ValueError('Incomplete/wrong-source phase state')
 if phase=='standing':
  a=read(directory/'admission.json')
  if not a.get('gate',{}).get('passed') or any(a.get(k)!=s.get(k) for k in ['variant','urdf_sha256','plan_sha256','stance_index']):raise ValueError('Standing not admitted')
  return {'phase':phase,'passed':True,'state_sha256':sha(directory/'state.json'),'admission_sha256':sha(directory/'admission.json')}
 if phase!='baseline':raise ValueError('Cold host has no training or video phase')
 d=read(directory/'diagnostics.json')
 if not d.get('complete') or d.get('kind')!='diagnostic_not_qualification' or d.get('checkpoint_sha256')!=CHECKPOINT or len(d.get('scenarios',[]))!=12 or not (directory/'diagnostic_trace.npz').is_file():raise ValueError('Missing/wrong diagnostic evidence')
 if d.get('overrides')!=identity['overrides'] or d.get('options')!=identity['diagnostic_options']:raise ValueError('Diagnostic control profile/options mismatch')
 audit=d['observation_audit']
 if audit['actor_width']!=315 or audit['critic_width']!=318 or any(audit[k]!=0 for k in ['max_same_step_repeat_difference','max_command_slice_difference','max_history_shift_difference']):raise ValueError('Observation audit failed')
 for row in d['scenarios']:
  cap=row['windows']['all']['applied_torque_abs_max_nm']
  if not isinstance(cap,(int,float)) or not math.isfinite(cap) or cap>1.60001:raise ValueError('Applied torque diagnostic invalid')
 return {'phase':phase,'complete':True,'state_sha256':sha(directory/'state.json'),'diagnostics_sha256':sha(directory/'diagnostics.json'),'scope':'Measured baseline, not physical or training admission','Stage2_complete':False}
def runtime_arguments(phase):
 if phase not in ['standing','baseline']:raise ValueError('No training dispatch')
 argv=['/source/tools/train_length_study.py','--package','/source/robot/hexapod_mkii_length_study','--output','/output/'+phase,'--variant','f050_t060','--stance-index','0','--mode','validate' if phase=='standing' else 'evaluate','--headless','--device','cuda:0','--kit_args=--/exts/omni.kit.telemetry/skipDeferredStartup=true --/app/extensions/excluded/4=omni.kit.telemetry']
 if phase=='baseline':argv+=['--admission','/admission/admission.json','--checkpoint','/checkpoint/original.pt']
 return argv
