"""Independent actual terminal and bounded-prefix replay; no physics writes."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--audit-primitives',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
if args.out.exists():raise FileExistsError('New immutable review result required')
sys.path.insert(0,str(args.source/'tools'));sys.path.insert(0,str(args.audit_primitives))
from screen_metrics import standing_screen,standing_quiet_review,physical_metrics
from omni_quiet_review import quiet_metrics,QUIET_GATES
from audit import substeps,clocks,moving
R=args.raw;RUN=R/'run';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def read(p):return json.loads(p.read_text())
def serial(x):
 if isinstance(x,np.ndarray):return serial(x.tolist())
 if isinstance(x,np.generic):return serial(x.item())
 if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [serial(v) for v in x]
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
rawaudit=read(R/'remote_audit.json')
for f,h in rawaudit['raw_payloads'].items():assert sha(R/f)==h,f
report={'scope':'Actual moving001 infrastructure failure after a real reference-support outcome; no PPO updates or policy evaluation','raw_payloads_verified':len(rawaudit['raw_payloads']),'remote_audit_sha256':sha(R/'remote_audit.json'),'standing':{},'calibration':{},'profile':{},'PPO_updates_completed':0,'Stage2_complete':False,'physical_training_policy_rejection':False,'profile_gate_passed':False}
def compare(actual,stored,path=''):
 diffs=[]
 if isinstance(actual,dict):
  assert set(actual)==set(stored),(path,set(actual)^set(stored))
  for k in actual:diffs+=compare(actual[k],stored[k],path+'/'+k)
 elif isinstance(actual,(list,tuple)):
  assert len(actual)==len(stored),path
  for i,(a,b) in enumerate(zip(actual,stored)):diffs+=compare(a,b,path+'/'+str(i))
 elif actual!=stored:
  assert path.endswith('/max_heading_excursion_deg') and abs(actual-stored)<1e-5,(path,actual,stored)
  diffs.append({'path':path,'local':actual,'stored':stored,'difference_deg':actual-stored})
 return diffs
def rate_and_cap(data,raw):
 req=abs(raw['computed_torque_nm'][1601:]);shape=req.shape;i=np.unravel_index(req.argmax(),shape);q=raw['joint_position_rad'][1600:].astype(float);v=raw['joint_velocity_rad_s'][1600:].astype(float)
 integral=(.5*(v[1:]+v[:-1])*.0025).sum(0);diff=integral-(q[-1]-q[0]);j=np.unravel_index(abs(diff).argmax(),diff.shape)
 return {'post200_max_requested_Nm':float(req.max()),'post200_requested_elements_over1p6':int((req>1.6).sum()),'peak_substep':int(i[0]+1601),'peak_replica':int(i[1]),'peak_joint_index':int(i[2]),'SDK_rate_vs_angle_max_difference_rad':float(abs(diff).max()),'worst_rate_replica':int(j[0]),'worst_rate_joint_index':int(j[1]),'native_rate_fidelity_pass':False}
stand=load(RUN/'standing/trace.npz');names=stand.pop('joint_names').tolist();stand.pop('legs',None);sr=load(RUN/'standing/physics_substeps.npz');st=read(RUN/'standing/state.json');m=standing_screen(stand);q=standing_quiet_review(stand,names)
for k,v in m.items():assert st['gate'][k]==v,k
qd=compare(q,st['gate']['all_replica_quiet']);_,ss=substeps(stand,sr)
report['standing']={'passed_all32':q['passed'],'controls':len(stand['time_s']),'physical_metrics_exact':True,'quiet_verdicts_exact':True,'tiny_heading_numeric_differences':qd,'substeps':ss,'rate_and_cap':rate_and_cap(stand,sr)}
del stand,sr
cal=load(RUN/'calibration/trace.npz');cal.pop('joint_names',None);cal.pop('legs',None);cr=load(RUN/'calibration/physics_substeps.npz');cc=load(RUN/'calibration/sensor_clocks.npz');state=read(RUN/'calibration/state.json');saved=read(RUN/'calibration/calibration.json');names=state['layout']['joint_names_runtime'];trials={}
for label,start,end in [('zero_mean',200,1200),('sampled',1200,2200)]:
 data={k:v[:end] for k,v in cal.items()};assert not data['requested_command'][start:].any();metrics=physical_metrics(data,settle_steps=start);each=[quiet_metrics(data,i,start,names,.02) for i in range(32)];lag=float(abs(data['reference_to_executable_lag_rad'][start:]).max())
 actual={'passed':bool(metrics['original_basic_standing_physics_pass'] and metrics['post_settle_max_requested_torque_nm']<=1.6 and metrics['post_settle_min_distal_support_count']==6 and metrics['max_target_cast_error_rad']<=2e-7 and lag<=.02000001 and all(r['pass'] and r['window_duration_s']>=10 for r in each)),'scope':'finite_position_residual_quiet_trial','start_control_index':start,'end_control_index_exclusive':end,'original_quiet_bounds':QUIET_GATES,'per_environment':each,'physical':metrics,'maximum_residual_offset_rad':lag,'rawSDK_rate_gate_unchanged':True,'physical_rate_fidelity_qualified':False}
 diffs=compare(actual,saved['trials'][label]);trials[label]={'passed_all32':actual['passed'],'numeric_differences':diffs,'maximum_residual_offset_rad':lag,'post_settle_max_requested_torque_nm':metrics['post_settle_max_requested_torque_nm']}
_,cs=substeps(cal,cr);clk=clocks(cc,np.zeros((len(cal['time_s']),32),bool))
meta=read(RUN/'calibration/initial.pt.json');initial=sha(RUN/'calibration/initial.pt');assert meta['checkpoint_sha256']==initial and state['initial_checkpoint']==meta
report['calibration']={'passed':saved['passed'],'std':saved['initial_std'],'controls':len(cal['time_s']),'trials':trials,'substeps':cs,'clocks':clk,'rate_and_cap':rate_and_cap(cal,cr),'initial_checkpoint_sha256':initial,'checkpoint_receipt_exact':True,'PPO_updates_completed':state['PPO_updates_completed']}
del cal,cr,cc
p=RUN/'profile_32';d=load(p/'raw/trace.npz');raw=load(p/'raw/physics_substeps.npz');clock=load(p/'raw/sensor_clocks.npz');ref=load(p/'raw/reference_states.npz');events=read(p/'raw/episodes.json');ps=read(p/'state.json');assert ps['input_checkpoint_sha256']==initial;assert ps['PPO_updates_completed']==0 and not ps['policy_training_started']
assert len(d['time_s'])==224 and np.array_equal(np.argwhere(d['training_terminated']),[[223,6]]) and not d['training_truncated'].any();assert events[0]['ended_rows']==[6] and events[0]['reference_failure'][6]==4 and 'reset_kind' not in events[0]
assert not d['terminated'].any() and not d['truncated'].any();assert np.max(abs(d['reference_to_executable_lag_rad']))==0
sat,sub=substeps(d,raw);clock_review=clocks(clock,np.zeros((224,32),bool))
# The complete primitive intentionally rejects the missing reset receipt. Replay
# only the223 controls before that attempted, never-completed reset separately.
prefix=moving({k:v[:223] for k,v in d.items()},{k:(v[:1785] if v.ndim and len(v)==1793 else v) for k,v in raw.items()}, {k:v[:223] for k,v in clock.items()}, {k:v[:223] for k,v in ref.items()},[],[])
retained=d['distal_contact'][-1,6].copy();retained[ref['current_leg'][-1,6]]=False;assert retained.sum()==4
window=[{'control':int(i+1),'time_s':float(d['time_s'][i,6]),'normal_forces_N':d['normal_force_world_n'][i,6,:,2].tolist(),'distal_contact':d['distal_contact'][i,6].tolist(),'current_leg':int(ref['current_leg'][i,6]),'command':d['requested_command'][i,6].tolist()} for i in range(213,224)]
report['profile']={'status':ps['status'],'controls_completed':224,'requested_profile_controls_after200':512,'achieved_controls_after200':24,'error':ps['error'],'traceback':ps['traceback'],'ended_replica':6,'final_reference_failure_code':4,'final_active_leg':'LF','final_RR_force_N':float(d['normal_force_world_n'][-1,6,5,2]),'final_LF_force_N':float(d['normal_force_world_n'][-1,6,0,2]),'total_contacts':int(d['distal_contact'][-1,6].sum()),'retained_contacts_excluding_LF':int(retained.sum()),'window':window,'substeps':sub,'clocks_all224_before_any_physical_reset':clock_review,'rate_and_cap':rate_and_cap(d,raw),'complete_pre_event223_control_audit':prefix,'attempted_reset_receipt_incomplete':True,'full_audit_preserved_failure':'KeyError(reset_kind)','reset_idx_not_reached_by_recorded_traceback':True,'initial_checkpoint_sha256':initial,'zero_residual_lag':True,'untrained_reference_support_outcome':True,'no_event_profile_would_still_reject_after_API_only_fix':True,'actual_physics_reset_completed':False,'policy_training_started':False,'PPO_updates_completed':0,'failure_export_method_restored':read(p/'raw/physics_substep_review.json')['method_restored']}
report['source_paths']={'physical_source':str(args.source),'audit_primitives':str(args.audit_primitives)}
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(serial(report),indent=2,allow_nan=False)+'\n');print(json.dumps({k:report[k] for k in ('raw_payloads_verified','PPO_updates_completed','Stage2_complete','profile_gate_passed')},indent=2))
