"""Stdlib source/admission/result contract for bounded canonical coordinate/effort evidence."""
from pathlib import Path
import math
import json
import importlib.util
_spec=importlib.util.spec_from_file_location('_canonical_phase_b_parent_contract',Path(__file__).resolve().with_name('inspection_contract.py'))
_parent=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_parent)
sha=_parent.sha;read=_parent.read;check_map=_parent.check_map
verify_asset=_parent.verify_inputs;validate_a=_parent.validate_result

SCHEMA='canonical_native_actuation_v1'
DT=.0025
STEPS=8 # unchanged parent import prefix
BASELINE_STEPS=40
PULSE_STEPS=8
COAST_STEPS=8
PULSE_NM=.005
BOUND_NM=.1
BOUND_Q=.05
BOUND_DQ=2.
EXPECTED_CASES=36
EXPERIMENT_STEPS=BASELINE_STEPS+EXPECTED_CASES*(BASELINE_STEPS+PULSE_STEPS+COAST_STEPS)
PARENT_FREEZE='1d636f6b6909171e366be7b08b3590c0a65a8a9206e57b76792d3f4bcd166cd8'

def vector(row,key,n=18):
 value=row[key]
 if not isinstance(value,list)or len(value)!=1 or not isinstance(value[0],list)or len(value[0])!=n:raise ValueError('Wrong raw vector shape:'+key)
 if any(type(x)not in (int,float)or not math.isfinite(x)for x in value[0]):raise ValueError('Nonfinite raw vector:'+key)
 return value[0]

def close_vector(a,b,tolerance=1e-8):
 return len(a)==len(b)and all(abs(x-y)<=tolerance for x,y in zip(a,b))

def validate_raw(directory):
 d=Path(directory);names=read(d/'native_readback.json')['joint_names']
 if len(names)!=18 or len(set(names))!=18:raise ValueError('Invalid native name order')
 prefix=read(d/'samples.json');counter=prefix[-1]['explicit_step_counter']
 plan=[('initial_baseline','baseline',None,0)]*BASELINE_STEPS
 for j,name in enumerate(names):
  for sign in [-1,1]:
   label=f'pulse:{name}:{sign}'
   plan.extend([(label,'baseline',j,sign)]*BASELINE_STEPS+[(label,'pulse',j,sign)]*PULSE_STEPS+[(label,'coast',j,sign)]*COAST_STEPS)
 baseline={};endpoints={};previous=None;previous_label=None;seen=0
 with(d/'experiment.jsonl').open()as f:
  for i,line in enumerate(f):
   if i>=len(plan):raise ValueError('Unexpected extra physics row')
   row=json.loads(line);label,phase,j,sign=plan[i]
   if row.get('sequence')!=i or row.get('explicit_step_counter')!=counter+i+1 or row.get('label')!=label or row.get('phase')!=phase or row.get('dt_s')!=DT:raise ValueError('Raw clock/order/phase mismatch')
   expected=[0.]*18
   if phase=='pulse':expected[j]=sign*PULSE_NM
   for k in ['requested_external_nm','software_applied_nm','native_input_pre']:
    if not close_vector(vector(row,k),expected):raise ValueError('Raw force command mismatch:'+k)
   q=vector(row,'joint_position');dq=vector(row,'joint_velocity_sdk');interval=vector(row,'interval_angle_rate_rad_s')
   vector(row,'native_input_post');vector(row,'projected_joint_reaction_nm');vector(row,'root_com_velocity',6)
   if max(map(abs,q))>BOUND_Q or max(map(abs,dq))>BOUND_DQ or max(map(abs,interval))>BOUND_DQ:raise ValueError('Raw diagnostic bound exceeded')
   if previous_label!=label:previous=[0.]*18
   if not close_vector(interval,[(a-b)/DT for a,b in zip(q,previous)],2e-6):raise ValueError('Interval-angle channel mismatch')
   previous=q;previous_label=label
   if phase=='baseline':baseline[label]=q
   elif phase=='coast':endpoints[label]=q
   seen+=1
 if seen!=EXPERIMENT_STEPS:raise ValueError('Missing raw physics rows')
 cases=read(d/'case_results.json')
 if len(cases)!=36:raise ValueError('Missing actual response cases')
 for k,c in enumerate(cases):
  j=k//2;sign=[-1,1][k%2];name=names[j];label=f'pulse:{name}:{sign}'
  if c.get('joint')!=name or c.get('sign')!=sign or c.get('passed')is not True:raise ValueError('Wrong/failed response case')
  displacement=[a-b for a,b in zip(endpoints[label],baseline[label])]
  if not close_vector(displacement,c['actual_all_joint_displacement'],1e-10):raise ValueError('Response receipt differs from raw position')
  p=c['predicted_all_joint_displacement'][j];ratio=c['actuated_joint_response_ratio']
  if not math.isfinite(p)or p*sign<=0 or not math.isfinite(ratio)or not .5<=ratio<=1.5 or abs(displacement[j]/p-ratio)>1e-8:raise ValueError('Invalid response discriminator')
 resets=[json.loads(x)for x in(d/'reset_readbacks.jsonl').read_text().splitlines()]
 if len(resets)!=74:raise ValueError('Missing reset history')
 expected_resets=[('initial_baseline',[0.]*18,counter)]
 for j,name in enumerate(names):
  for sign in [-1,1]:
   q=[0.]*18;q[j]=sign*.01;expected_resets.append((f'coordinate:{name}:{sign}',q,counter+BASELINE_STEPS))
 expected_resets.append(('mass_matrix_origin',[0.]*18,counter+BASELINE_STEPS))
 for j,name in enumerate(names):
  for sign in [-1,1]:
   k=2*j+(sign==1);expected_resets.append((f'pulse:{name}:{sign}',[0.]*18,counter+BASELINE_STEPS+k*(BASELINE_STEPS+PULSE_STEPS+COAST_STEPS)))
 for i,(row,(label,q,step))in enumerate(zip(resets,expected_resets)):
  if row.get('reset_index')!=i or row.get('label')!=label or row.get('explicit_step_counter')!=step:raise ValueError('Reset clock/order mismatch')
  for key,want in [('joint_position',q),('position_target',q),('joint_velocity_sdk',[0.]*18),('velocity_target',[0.]*18),('native_external_input',[0.]*18),('root_com_velocity',[0.]*6)]:
   if not close_vector(vector(row,key,len(want)),want,2e-6):raise ValueError('Reset state/target mismatch:'+key)
 return {'raw_steps':seen,'raw_resets':len(resets),'raw_response_cases':len(cases)}

def verify_admission(directory):
 own=Path(__file__).resolve().parent
 admission=Path(directory).resolve();expected=read(own/'PHASE_A_SHA256.json')
 check_map(admission,expected)
 actual={str(p.relative_to(admission))for p in admission.rglob('*')if p.is_file()}
 if actual!=set(expected):raise ValueError('Phase A actual inventory changed')
 a=read(admission/'state.json');validate_a(admission,a['identity'])
 if a['identity']['inspector_freeze_sha256']!=PARENT_FREEZE:raise ValueError('Unreviewed Phase A')
 return {'phase_a_manifest_sha256':sha(own/'PHASE_A_SHA256.json'),'phase_a_state_sha256':sha(admission/'state.json')}

def verify_inputs(args):
 own=Path(__file__).resolve().parent
 identity=verify_asset(args);admission=Path(args.admission).resolve();prior=verify_admission(admission)
 if Path(args.output).resolve().is_relative_to(admission):raise ValueError('Output overlaps Phase A')
 identity.update(schema=SCHEMA,inspector_freeze_sha256=sha(own/'FREEZE_SHA256.json'),
  runtime_binding={'runtime_tree_sha256':sha(own/'FREEZE_SHA256.json'),'scope':'canonical_native_coordinate_effort_only'},
  **prior,
  steps=STEPS+EXPERIMENT_STEPS,import_prefix_steps=STEPS,
  diagnostic={'baseline_steps':BASELINE_STEPS,'pulse_steps':PULSE_STEPS,'coast_steps':COAST_STEPS,'pulse_nm':PULSE_NM,
   'cases':EXPECTED_CASES,'experiment_steps':EXPERIMENT_STEPS,'coordinate_rad':.01,
   'bound_nm':BOUND_NM,'bound_joint_excursion_rad':BOUND_Q,'bound_joint_rate_rad_s':BOUND_DQ,
   'gravity':[0.,0.,0.],'ground':False,'position_drives':False,'external_forces_every_iteration':True,
   'native_maxforce_field':'retain zero implicit-drive maximum; explicit input is software limited independently'})
 return identity

def validate_result(directory,identity):
 d=Path(directory);s=read(d/'state.json')
 if s.get('schema')!=SCHEMA or s.get('identity')!=identity:raise ValueError('Wrong actuation identity')
 if s.get('status')!='completed' or s.get('inputs_unchanged')is not True:raise ValueError('Incomplete/failed experiment')
 if s.get('errors')!=[]or s.get('native_error_events')!=[]or read(d/'native_errors.json')!=[]:raise ValueError('Native/runtime errors')
 if any(s.get(k)is not False for k in ['physics_admitted','physical_admission','training_allowed']):raise ValueError('Diagnostic cannot admit controller or learning')
 if s.get('explicit_steps_completed')!=STEPS+EXPERIMENT_STEPS:raise ValueError('Incomplete physics clock')
 required={'usd_identity','native_identity','native_frames','native_scene','native_sdf_paths','no_drive_gains','finite_samples','sdk_source_bound','external_force_setting','coordinate_effort_cases'}
 if not required.issubset(s.get('checks',{}))or not all(v is True for v in s['checks'].values()):raise ValueError('Missing/rejected checks')
 outputs=s.get('outputs',{});check_map(d,outputs)
 if not {'experiment.jsonl','experiment_summary.json','reset_readbacks.jsonl','sdk_readback.json','sdf_readback.json','samples.json','native_readback.json','case_results.json'}.issubset(outputs):raise ValueError('Missing raw experiment evidence')
 r=read(d/'experiment_summary.json')
 if r.get('status')!='completed' or r.get('case_count')!=36 or r.get('coordinate_count')!=36 or r.get('step_count')!=EXPERIMENT_STEPS:raise ValueError('Incomplete named cases')
 if r.get('all_native_inputs_match')is not True or r.get('all_coordinate_checks')is not True:raise ValueError('Rejected actuator/reset evidence')
 if r.get('errors')!=[]:raise ValueError('Recorded experiment failure')
 raw=validate_raw(d)
 return {'phase':'actuation','status':'completed','scope':'coordinate_effort_diagnostic_only',
  'physical_admission':False,'training_allowed':False,'state_sha256':sha(d/'state.json'),
  'case_count':r['case_count'],'coordinate_count':r['coordinate_count'],'step_count':r['step_count'],**raw}
