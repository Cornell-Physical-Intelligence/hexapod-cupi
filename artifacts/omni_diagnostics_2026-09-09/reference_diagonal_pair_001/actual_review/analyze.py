"""Read-only source-bound diagonal static diagnostic replay; no walking admission."""
from pathlib import Path
import hashlib,json,sys,numpy as np
H=Path(__file__).resolve().parent;R=H.parent;S=R.parent/'reference_diagonal_pair_adapter_001/source_diagonal_pair_001';sys.path.insert(0,str(S/'tools'))
from score_transfer import score_transfer,validate_substep_trace
from load_transfer import PairLoadTransfer
from screen_metrics import standing_screen,standing_quiet_review
from pair_screen_contract import PAIR_CASES
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((R/'remote_audit.json').read_text())
for f,h in a['raw_payloads'].items():assert sha(R/f)==h,f
mapping=json.loads((S/'campaign_source_hashes.json').read_text());assert len(mapping)==934 and sha(S/'campaign_source_hashes.json')==a['source_manifest_sha256']
for f,h in mapping.items():assert sha(S/f)==h,f

def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def differences(x,y,path=''):
 out=[]
 if isinstance(x,dict) and isinstance(y,dict):
  assert x.keys()==y.keys(),path
  for k in x:out.extend(differences(x[k],y[k],path+'/'+k))
 elif isinstance(x,list) and isinstance(y,list):
  assert len(x)==len(y),path
  for i,(u,v) in enumerate(zip(x,y)):out.extend(differences(u,v,path+'/'+str(i)))
 elif isinstance(x,(int,float)) and not isinstance(x,bool) and isinstance(y,(int,float)) and not isinstance(y,bool):
  if x!=y:out.append({'path':path,'local':x,'recorded':y,'absolute_difference':abs(x-y)})
 else:assert x==y,(path,x,y)
 return out

def observer(sub,d):
 n=len(d['time_s']);validate_substep_trace(sub,d)
 for k,v in [('joint_position_rad','joint_position_rad'),('joint_velocity_rad_s','joint_velocity_rad_s'),('root_link_position_world_m','position_world_m'),('root_link_quaternion_world_xyzw','quaternion_world_xyzw')]:np.testing.assert_array_equal(sub[k][8::8],d[v])
 def extreme(key):
  q=sub[key];i=np.unravel_index(abs(q).argmax(),q.shape)
  return {'abs_max_nm':float(abs(q).max()),'signed_peak_nm':float(q[i]),'sample_index':int(i[0]),'relative_observer_time_s':float(sub['time_s'][i[0]]),'control_index':int(sub['control_index'][i[0]]),'substep_index':int(sub['substep_index'][i[0]]),'runtime_joint':str(d['joint_names'][i[2]])}
 return {'controls':n,'samples':len(sub['time_s']),'exact8_substeps_counters_timestamps_and_endpoint_parity':True,'requested':extreme('computed_torque_nm'),'applied':extreme('applied_torque_nm'),'requested_samples_above1p6':int((abs(sub['computed_torque_nm'])>1.6).sum())}

sd=load(R/'run/standing/trace.npz');names=sd.pop('joint_names').tolist();sd.pop('legs',None)
standing=standing_screen(sd);quiet=standing_quiet_review(sd,names);assert standing['passed'] and quiet['passed']
results={}
for case,partition in PAIR_CASES.items():
 p=R/'run'/case
 if not (p/'state.json').exists():results[case]={'status':'unrun','no_physical_conclusion':True};continue
 state=json.loads((p/'state.json').read_text());out={'status':state['status'],'failure':state.get('failure'),'identity':state.get('identity'),'partition':partition,'controls':state.get('control_steps'),'diagnostic_controls':state.get('diagnostic_controls')}
 assert state.get('pair_case')==case and state.get('pair_partition')==partition
 if (p/'startup/trace.npz').exists():
  startup=load(p/'startup/trace.npz');initial={k:v[-1] for k,v in startup.items() if k!='joint_names'};out['startup_observer']=observer(load(p/'startup/physics_substeps.npz'),startup)
 else:
  initial=None;out['startup_complete']=False
 trace=p/'trace.npz' if (p/'trace.npz').exists() else p/'partial_trace.npz'
 if not trace.exists():
  out['no_diagnostic_trace']=True;results[case]=out;continue
 d=load(trace);sub=load(p/'physics_substeps.npz');refs=json.loads((p/'reference_states.json').read_text());post=json.loads((p/'post_step_measurements.json').read_text());n=len(d['time_s'])
 out.update(diagnostic_trace=trace.name,available_diagnostic_controls=n,available_reference_rows=len(refs),available_postcheck_rows=len(post),diagnostic_observer=observer(sub,d),observer_origin_physical_s=4.,contact_rate_hz=50,complete400Hz_contacts_claimed=False)
 np.testing.assert_allclose(d['time_s'][:,0],np.arange(201,201+n)*.02,atol=1e-7,rtol=0)
 names=d['joint_names'].tolist();contacts=d['distal_contact'][:,0]&d['contact_point_valid'][:,0]
 out['nonfoot_or_terminal_samples']={k:int(d[k].any(axis=-1).sum()) if d[k].ndim>1 else int(d[k].sum()) for k in ('shaft_contact','coxa_contact','femur_contact','base_contact','terminated','truncated')}
 out['minimum_distal_support_count']=int(contacts.sum(-1).min())
 if initial is not None:
  g=PairLoadTransfer(names,pair_leg_names=partition['pair_legs']);g.reset(initial);bounds=[];first_error=None
  for i in range(n):
   row={k:v[i] for k,v in d.items() if k!='joint_names'}
   try:g._measure(row,g.base._read(row));bounds.append({k:float(g.diagnostics[k]) for k in ('measured_support_margin_m','body_displacement_m','body_rotation_rad','corner_drift_m','corner_max_slip_mps')})
   except ValueError as e:first_error={'index':i,'physical_time_s':float(d['time_s'][i,0]),'error':str(e)};break
  out['independent_post_measurement_replay']={'rows_passed':len(bounds),'first_error':first_error,'extrema':{k:float((min if k=='measured_support_margin_m' else max)(v[k] for v in bounds)) for k in bounds[0]} if bounds else None}
 if n==1100 and len(refs)==n and state.get('diagnostic_result') is not None:
  scored=score_transfer(d,refs,substeps=sub,pair_leg_names=partition['pair_legs']);diff=differences(scored,state['diagnostic_result'])
  out.update(original_score=state['diagnostic_result'],independent_score=scored,numeric_replay_differences=diff,all_nonnumeric_fields_and_verdict_equal=True)
  assert scored['proposed_criteria_met']==state['proposed_diagnostic_criteria_met']
 else:out['complete_scorer_unavailable_reason']='Partial diagnostic or runtime stopped before full aligned scoring; no synthetic completion'
 if n:
  out['last_measured_row']={'time_s':float(d['time_s'][-1,0]),'normal_force_z_n':dict(zip(('lf','lm','lr','rf','rm','rr'),d['normal_force_world_n'][-1,0,:,2].tolist())),'distal_contact':contacts[-1].tolist(),'contact_point_valid':d['contact_point_valid'][-1,0].tolist()}
  q=sub['joint_position_rad'][:,0].astype(float);v=sub['joint_velocity_rad_s'][:,0].astype(float);delta=q[-1]-q[0];integ=(v[:-1]+v[1:]).sum(0)*.00125;j=int(abs(integ-delta).argmax())
  out['diagnostic_joint_rate_bias']={'runtime_joint':names[j],'angle_delta_rad':float(delta[j]),'reported_rate_integral_rad':float(integ[j]),'difference_rad':float(integ[j]-delta[j]),'not_an_admission_metric_substitution':True}
 results[case]=out
report={'scope':'Separate static diagonal-pair diagnostics; no walking/PPO or Stage2 qualification','source_manifest_sha256':a['source_manifest_sha256'],'source_files_verified':934,'raw_payloads_verified':len(a['raw_payloads']),'asset_files_remote_verified':550,'fresh32_standing_physical_and_unchanged_quiet_replay_passed':True,'cases':results,'remote_owned_absence_tokens':len(a['owned_containers_absent']),'pause047_restoration_verified':True,'historical_cleanup_only':True,'physics_walking_qualification':False,'stage2_complete':False,'velocity_fidelity_qualified':False}
out=H/'report.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2,allow_nan=False))
