"""Source-bound actual003 turning/arc review; no physics, metric or gate changes."""
from pathlib import Path
import hashlib,json,sys,numpy as np
HERE=Path(__file__).resolve().parent;RAW=HERE.parent;SOURCE=RAW.parent/'reference_directional_adapter_003/source_directional_003'
sys.path.insert(0,str(SOURCE/'tools'))
from directional_metrics import score_direction
from screen_metrics import standing_screen,standing_quiet_review
from physics_substeps import displacement_check
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((RAW/'remote_audit.json').read_text())
for f,h in a['raw_payloads'].items():assert sha(RAW/f)==h,f
m=json.loads((SOURCE/'campaign_source_hashes.json').read_text());assert sha(SOURCE/'campaign_source_hashes.json')==a['source_manifest_sha256']
for f,h in m.items():assert sha(SOURCE/f)==h,f
assert len(m)==930 and len(a['raw_payloads'])==44

def read_phase(name):
 p=RAW/'run'/name;s=json.loads((p/'state.json').read_text())
 with np.load(p/'trace.npz',allow_pickle=False) as z:names=z['joint_names'].tolist();d={k:z[k].copy() for k in z.files if k not in ('joint_names','legs')}
 with np.load(p/'physics_substeps.npz',allow_pickle=False) as z:sub={k:z[k].copy() for k in z.files}
 n=len(d['time_s']);assert len(sub['time_s'])==n*8+1
 np.testing.assert_array_equal(sub['control_index'],np.r_[-1,np.repeat(np.arange(n),8)])
 np.testing.assert_array_equal(sub['substep_index'],np.r_[0,np.tile(np.arange(1,9),n)])
 np.testing.assert_array_equal(sub['sim_step_counter']-sub['sim_step_counter'][0],np.arange(n*8+1))
 np.testing.assert_allclose(np.diff(sub['sdk_sim_timestamp_s']),.0025,atol=1e-7,rtol=0)
 for raw,control in [('joint_position_rad','joint_position_rad'),('joint_velocity_rad_s','joint_velocity_rad_s'),('root_link_position_world_m','position_world_m'),('root_link_quaternion_world_xyzw','quaternion_world_xyzw'),('computed_torque_nm','computed_torque_nm'),('applied_torque_nm','applied_torque_nm')]:np.testing.assert_array_equal(sub[raw][8::8],d[control])
 def torque(start):
  q=sub['computed_torque_nm'][start:];i=np.unravel_index(abs(q).argmax(),q.shape);idx=i[0]+start
  return {'max_requested_nm':float(abs(q).max()),'max_applied_nm':float(abs(sub['applied_torque_nm'][start:]).max()),'requested_samples_above1p6':int((abs(q)>1.6).sum()),'peak_time_s':float(sub['time_s'][idx]),'peak_control_index':int(sub['control_index'][idx]),'peak_substep_index':int(sub['substep_index'][idx]),'environment':int(i[1]),'runtime_joint':names[i[2]]}
 def joint_rate(begin,end):
  q=sub['joint_position_rad'][begin*8:end*8+1].astype(np.float64);v=sub['joint_velocity_rad_s'][begin*8:end*8+1].astype(np.float64)
  delta=q[-1]-q[0];integral=.5*(v[:-1]+v[1:]).sum(0)*.0025;err=integral-delta;ind=np.unravel_index(abs(err).argmax(),err.shape)
  return {'start_control_boundary':begin,'end_control_boundary':end,'duration_s':(end-begin)*.02,'environment':int(ind[0]),'runtime_joint':names[ind[1]],'angle_delta_rad':float(delta[ind]),'reported_rate_integral_rad':float(integral[ind]),'difference_rad':float(err[ind]),'pairs_above0p01_rad':int((abs(err)>.01).sum()),'position_velocity_fidelity_admitted':False}
 out={'controls':n,'actual400Hz_samples':len(sub['time_s']),'every8th_q_v_pose_torque_endpoint_and_counter_timestamp_parity':True,'all_recorded400Hz_torque_including_reset':torque(0),'postsettle400Hz_torque':torque(1601),'postsettle400Hz_joint_rate_discrepancy':joint_rate(200,n)}
 return s,names,d,sub,out

ss,sn,sd,sub,standing=read_phase('standing');sg=standing_screen(sd);sq=standing_quiet_review(sd,sn)
assert sg['passed'] and sq['passed'];standing.update({'physical_pass':True,'all32_unchanged_quiet_pass':True})
results={}
for case in ['left_turn','forward_right_arc']:
 s,names,d,sub,out=read_phase(case);refs=json.loads((RAW/'run'/case/'reference_states.json').read_text());g=score_direction(d,refs,case=case,joint_names=names,failure=s['failure'])
 differences=[]
 def compare(x,y,path=''):
  if isinstance(x,dict):
   assert x.keys()==y.keys()
   for k in x:compare(x[k],y[k],path+'/'+k)
  elif isinstance(x,list):
   assert len(x)==len(y)
   for i,(u,v) in enumerate(zip(x,y)):compare(u,v,path+'/'+str(i))
  elif x!=y:differences.append({'path':path,'local':x,'remote':y,'local_minus_remote':x-y})
 compare(g,s['gate'])
 for e in differences:
  assert e['path']=='/final_quiet_stop_window/max_heading_excursion_deg' and abs(e['local_minus_remote'])<7e-6,e
 assert g['passed']==s['gate']['passed'];json.dumps(s,allow_nan=False);json.dumps(g,allow_nan=False)
 out.update({'status':s['status'],'failure':s['failure'],'exact_saved_scalar_gate_replay_equal':not differences,'portable_original_verdict_and_all_nonnumeric_fields_equal':True,'numeric_differences':differences,'strict_terminal_JSON_pass':True,'original_gate':s['gate'],'portable_recomputed_gate':g})
 end=min(len(d['time_s']),1400);out['400Hz_motion_interval_diagnostic']=displacement_check(sub,200,end)
 if case=='left_turn':
  assert s['status']=='completed' and g['passed'] and out['controls']==2400
  out['scope']='This one declared left-turn reference screen passes; not full omni/PPO qualification'
 else:
  assert s['status']=='rejected' and not g['passed'] and out['controls']==508
  supports=d['distal_contact'].sum(-1)[:,0];bad=np.where(supports[200:]<5)[0]+200;assert bad.tolist()==[507]
  state=refs[-1]['result']['state'];last=d['normal_force_world_n'][-1,0,:,2];legs=('lf','lm','lr','rf','rm','rr')
  out['failure_evidence']={'intended_swing_leg':state['current_leg'],'first_subfive_index':507,'time_s':float(d['time_s'][-1,0]),'distal_support':d['distal_contact'][-1,0].tolist(),'named_normal_force_z_n':dict(zip(legs,last.tolist())),'named_contact_point_finite':dict(zip(legs,d['contact_point_valid'][-1,0].tolist())),'named_reaction_world_n':dict(zip(legs,d['reaction_force_world_n'][-1,0].tolist())),'prior_generator_confirmed_landings':g['generator_confirmed_touchdowns'],'contact_sample_rate_hz':50,'400Hz_contact_forces_available':False}
  out['final20_support_rows']=[{'index':i,'time_s':float(d['time_s'][i,0]),'distal_support':d['distal_contact'][i,0].tolist(),'normal_force_z_n':d['normal_force_world_n'][i,0,:,2].tolist()} for i in range(out['controls']-20,out['controls'])]
  out['full24s_motion_and_quiet_reached']=False
 results[case]=out
report={'scope':'Actual003 source-bound scalar and400Hz review; original gates unchanged','source_manifest_sha256':a['source_manifest_sha256'],'source_files_verified':930,'raw_payloads_verified':44,'admitted_asset_files_remote_verified':550,'fresh_standing':standing,'cases':results,'remote_owned3_ids_and_names_absent':len(a['owned_containers_absent'])==6,'pause046_restoration_verified':True,'historical_cleanup_only_no_later_lock_claim':True,'stage2_complete':False,'omni_policy_admitted':False,'rate_fidelity_admitted':False,'initial_reset_requested_torque_warning':'All-recorded reset transients are retained separately from the declared settling exclusion; standing admission is not hardware-startup qualification'}
p=HERE/'report.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'standing':standing,'turn_torque':results['left_turn']['postsettle400Hz_torque'],'turn400Hz':results['left_turn']['400Hz_motion_interval_diagnostic'],'arc_torque':results['forward_right_arc']['postsettle400Hz_torque'],'arc_failure':results['forward_right_arc']['failure_evidence']},indent=2))
