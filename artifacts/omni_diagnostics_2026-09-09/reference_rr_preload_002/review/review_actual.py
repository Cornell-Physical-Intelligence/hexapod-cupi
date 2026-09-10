"""Independent original-gate replay and actual RR correction/contact review."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--original-trace',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
ROOT=args.raw.resolve();SOURCE=args.source.resolve();OUTPUT=args.output.resolve();OUTPUT.mkdir(parents=True,exist_ok=True)
if any((OUTPUT/f).exists() for f in ['review_final.json','correction_states.json']):raise FileExistsError('Fresh review output files required')
sys.path.insert(0,str(SOURCE/'tools'))
from directional_metrics import score_direction
from screen_metrics import standing_screen,standing_quiet_review
from physics_substeps import displacement_check
from rr_preload_diagnostic import RRFirstLandingPreload
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x

def load(phase,file):
 with np.load(ROOT/'run'/phase/file,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def subreview(data,sub,names):
 n=len(data['time_s']);assert len(sub['time_s'])==n*8+1
 np.testing.assert_array_equal(sub['control_index'],np.r_[-1,np.repeat(np.arange(n),8)])
 np.testing.assert_array_equal(sub['substep_index'],np.r_[0,np.tile(np.arange(1,9),n)])
 np.testing.assert_array_equal(sub['sim_step_counter']-sub['sim_step_counter'][0],np.arange(n*8+1))
 np.testing.assert_allclose(np.diff(sub['sdk_sim_timestamp_s']),.0025,atol=1e-7,rtol=0)
 for a,b in [('joint_position_rad','joint_position_rad'),('joint_velocity_rad_s','joint_velocity_rad_s'),('root_link_position_world_m','position_world_m'),('root_link_quaternion_world_xyzw','quaternion_world_xyzw'),('computed_torque_nm','computed_torque_nm'),('applied_torque_nm','applied_torque_nm')]:np.testing.assert_array_equal(sub[a][8::8],data[b])
 assert sub['joint_names'].tolist()==list(names)
 tor=sub['computed_torque_nm'][1601:];i=np.unravel_index(abs(tor).argmax(),tor.shape);absolute=i[0]+1601
 q=sub['joint_position_rad'][1600:].astype(float);v=sub['joint_velocity_rad_s'][1600:].astype(float);delta=q[-1]-q[0];integral=.5*(v[:-1]+v[1:]).sum(0)*.0025;err=integral-delta;ii=np.unravel_index(abs(err).argmax(),err.shape)
 return {'samples':len(sub['time_s']),'every8th_control_endpoint_and_full_counter_timestamp_parity':True,'max_postsettle_requested_torque_nm':float(abs(tor).max()),'peak_time_s':float(sub['time_s'][absolute]),'peak_replica':int(i[1]),'peak_joint':names[i[2]],'postsettle_torque_elements_over_1p6':int((abs(tor)>1.6).sum()),'reported_joint_rate_vs_angle':{'worst_replica':int(ii[0]),'joint':names[ii[1]],'actual_angle_delta_rad':float(delta[ii]),'reported_rate_integral_rad':float(integral[ii]),'difference_rad':float(err[ii])},'contact_forces_recorded_at400Hz':False,'velocity_fidelity_qualified':False}

audit=json.loads((ROOT/'remote_audit.json').read_text())
for f,h in audit['raw_payloads'].items():assert sha(ROOT/f)==h,f
m=json.loads((SOURCE/'campaign_source_hashes.json').read_text());assert len(m)==932 and sha(SOURCE/'campaign_source_hashes.json')==audit['source_manifest_sha256']
for f,h in m.items():assert sha(SOURCE/f)==h,f
campaign=json.loads((ROOT/'run/campaign.json').read_text());standing_state=json.loads((ROOT/'run/standing/state.json').read_text())
sd=load('standing','trace.npz');sn=sd.pop('joint_names').tolist();sd.pop('legs');sg=standing_screen(sd);sq=standing_quiet_review(sd,sn)
assert all(standing_state['gate'][k]==v for k,v in sg.items()) and sq['passed'] is True
quiet_numeric_differences=[]
def compare_quiet(a,b,path=''):
 if isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:compare_quiet(a[k],b[k],path+'/'+k)
 elif isinstance(a,list):
  assert len(a)==len(b)
  for i,(x,y) in enumerate(zip(a,b)):compare_quiet(x,y,path+'/'+str(i))
 elif a!=b:
  assert path.endswith('/max_heading_excursion_deg') and abs(a-b)<1e-5,(path,a,b)
  quiet_numeric_differences.append({'path':path,'stored':a,'local_replay':b,'difference_deg':b-a})
compare_quiet(standing_state['gate']['all_replica_quiet'],sq)
standing_sub=subreview(sd,load('standing','physics_substeps.npz'),sn)
report={'scope':'Actual bounded RR preload002 trial, original physical gates; no PPO/Stage2 or native velocity-fidelity admission','source_manifest_sha256':audit['source_manifest_sha256'],'source_files_verified':len(m),'raw_files_verified':len(audit['raw_payloads']),'terminal_campaign_status':campaign['status'],'fresh32standing':{'status':standing_state['status'],'physical_gate':sg,'physical_gate_all_fields_exact':True,'quiet':sq,'quiet_boolean_verdicts_exact':True,'tiny_heading_numeric_replay_differences_deg':quiet_numeric_differences,'no_gate_changed_to_reconcile':True,'substeps':standing_sub},'wave':None,'stage2_complete':False,'policy_training_started':False,'old_threshold_or_gate_changed':False}
if (ROOT/'run/left_strafe/state.json').exists():
 state=json.loads((ROOT/'run/left_strafe/state.json').read_text());d=load('left_strafe','trace.npz');names=d.pop('joint_names').tolist();legs=d.pop('legs').tolist();sub=load('left_strafe','physics_substeps.npz');refs=json.loads((ROOT/'run/left_strafe/reference_states.json').read_text());n=len(d['time_s'])
 gate=score_direction(d,refs,case='left_strafe',joint_names=names,failure=state.get('failure'));assert state['gate']==gate
 assert state['control_steps']==n;json.dumps(state,allow_nan=False)
 for k in ['raw_residual_action','residual_position_rad','residual_velocity_rad_s','residual_acceleration_rad_s2']:np.testing.assert_array_equal(d[k],0)
 traces=[];times=[];allpre=True
 for row in refs:
  if 'result' not in row:continue
  r=row['result'];h=r['diagnostics']['rr_preload_diagnostic'];idx=row['physical_step']
  if r['valid'][0]:
   assert 0<=idx<n;np.testing.assert_allclose(r['target_time_s'],d['time_s'][idx,0],atol=3e-12,rtol=0);np.testing.assert_array_equal(np.asarray(r['q_ref'])[0].astype(np.float32),d['joint_target_rad'][idx,0]);np.testing.assert_array_equal(r['q_ref'],d['reference_position_rad'][idx])
  if h['started']:
   times.append(h['start_s']);traces.append({'control_index':idx,'target_time_s':r['target_time_s'],'diagnostic':h,'valid':r['valid'][0]})
   curve=RRFirstLandingPreload();curve.arm(5,1,h['start_s'],.3,h['anchor_at_start_world_m'],h['measured_anchor_at_start_world_m'],h['original_swing_endpoint_world_m'])
   if h['status']!='rejected':
    offset,v,a=curve.sample(h['last_sample_time_s']);np.testing.assert_allclose(h['offset_applied_world_m'],offset,atol=1e-15,rtol=0);np.testing.assert_allclose(h['analytic_velocity_world_mps'],v,atol=1e-15,rtol=0);np.testing.assert_allclose(h['analytic_acceleration_world_mps2'],a,atol=1e-15,rtol=0)
 assert len(set(times))<=1
 original=np.load(args.original_trace,allow_pickle=False)
 F=d['normal_force_world_n'][:,0,:,2];support=d['distal_contact'][:,0];comparison=[]
 for idx in [419,420,435,436,453,454,495,503,504]:
  if idx>=n:continue
  comparison.append({'control_index':idx,'time_s':d['time_s'][idx,0],'original_RR_force_n':original['normal_force_world_n'][idx,0,5,2],'new_all_leg_force_n':F[idx],'new_distal_support':support[idx]})
 prefix=min(420,n);prefix_equal={k:np.array_equal(d[k][:prefix],original[k][:prefix]) for k in ['joint_position_rad','joint_target_rad','position_world_m','normal_force_world_n','distal_contact']}
 failidx=np.flatnonzero(support[200:].sum(1)<5)+200
 last=refs[-1]['result'];hg=last['diagnostics']['rr_preload_diagnostic']
 wave={'status':state['status'],'failure':state.get('failure'),'control_steps':n,'gate_reproduced_exactly':True,'gate':gate,'all_zero_residual_actions_and_state':True,'no_checkpoint_payloads':not any(p.suffix=='.pt' for p in (ROOT/'run').rglob('*')),'policy_or_reward_optimization':False,'step_rewards_may_be_computed_by_env_but_not_used_to_train':True,'substeps':subreview(d,sub,names),'correction':{'started':hg['started'],'completed':hg['completed'],'status':hg['status'],'one_start_time':len(set(times))==1,'first_recorded_changed_state':traces[0] if traces else None,'final_state':hg,'all_recorded_C2_offsets_and_derivatives_match_exact_curve':True,'all_reference_target_times_and_emitted_knots_verified':True},'actual_original505_window_comparison':comparison,'pre_intervention_first420_rows_exact_equal':prefix_equal,'subfive_control_indices':failidx.tolist(),'final_actual_all_leg_normal_n':F[-1],'final_actual_all_leg_contact':support[-1],'final_current_leg':last['state']['current_leg'],'final_generator_mode':last['state']['mode'],'stop_requested_time_s':last['state']['stop_requested_time_s'],'reference_quiet_time_s':last['state']['reference_quiet_time_s'],'continued_after_old505_control_boundary':n>505,'only_actual_new_run_is_scored':True}
 if n>200:wave['available_prefix400Hz_pose_velocity_diagnostic']=displacement_check(sub,200,n)
 report['wave']=wave
 (OUTPUT/'correction_states.json').write_text(json.dumps(clean(traces),indent=2,allow_nan=False)+'\n')
report['remote_provenance']={'source932_assets550_unchanged':audit['source_unchanged'] and audit['admitted_assets_unchanged'],'owned_names_and_recorded_IDs_absent':audit['owned_containers_absent'],'pause_restored':audit['pause_restoration'],'unit_user_scope':audit['unit']}
p=OUTPUT/'review_final.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n');print(json.dumps(clean({k:v for k,v in report.items() if k not in ['fresh32standing','remote_provenance']}),indent=2))
