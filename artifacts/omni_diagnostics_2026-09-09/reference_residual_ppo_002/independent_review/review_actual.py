"""Independent terminal002 replay from immutable raw values; CPU only, no learning."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent;TMP=ROOT.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def normal(v):
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,dict):return {k:normal(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [normal(x) for x in v]
 return v

def frozen(path,manifest,bound):
 assert sha(path/manifest)==bound
 m=read(path/manifest);assert {str(p.relative_to(path)):sha(p) for p in path.rglob('*') if p.is_file() and p!=path/manifest}==m
 return len(m)

def main():
 audit=read(ROOT/'remote_audit.json')
 for name,h in audit['raw_payloads'].items():assert sha(ROOT/name)==h,name
 source=TMP/'reference_physics_adapter_009/source_009';consumer=TMP/'reference_residual_ppo_source_002';obs=TMP/'reference_policy_observation_005_001'
 frozen(source,'campaign_source_hashes.json','04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e')
 frozen(consumer,'FREEZE_SHA256.json','e9a1d6d687504fd1323d8addbcb962c3025745a258f71b9f076879778eedaa0b')
 frozen(obs,'FREEZE_SHA256.json','22402b0079b0b5ac0acdd4e1d7e819fe206b20ee1883077f7de6eaca55808a63')
 for p in (source/'tools',obs,consumer):sys.path.insert(0,str(p))
 from screen_metrics import standing_screen,standing_quiet_review
 from observation import ObservationBuilder  # installs exact frozen reference module path
 from physical_scores import quiet_trial,forward_stop_trial
 from physical_contract import validate_result
 run=ROOT/'run';campaign=read(run/'campaign.json');results={}
 keys=['time_s','computed_torque_nm','applied_torque_nm','coxa_contact','femur_contact','shaft_contact','base_contact','velocity_body_mps','distal_contact','terminated','truncated','position_world_m','reference_to_executable_lag_rad','position_target_cast_error_rad','joint_position_rad','joint_target_rad','joint_velocity_rad_s','quaternion_world_wxyz','rotation_world_from_body','velocity_world_mps','reference_point_world_m','requested_command']
 for phase in ('standing','smoke','evaluate_initial','evaluate_final'):
  p=run/phase
  if not (p/'state.json').exists():results[phase]={'status':'not_run'};continue
  state=read(p/'state.json');entry={'status':state['status']};results[phase]=entry
  with np.load(p/'trace.npz',allow_pickle=False) as z:
   names=z['joint_names'].tolist();d={k:z[k] for k in keys};last_twist=z['velocity_body_mps'][-1].copy()
  count,n=d['time_s'].shape;entry.update(controls=count,replicas=n)
  np.testing.assert_allclose(d['time_s'],np.broadcast_to((np.arange(count)+1)[:,None]*.02,(count,n)),rtol=0,atol=1e-10)
  rows=[{k:v[i] for k,v in d.items()} for i in range(count)]
  if phase=='standing':
   gate=standing_screen(d);quiet=standing_quiet_review(d,names);entry.update(physical_pass=bool(gate['passed']),quiet_pass=bool(quiet['passed']),quiet_replicas_passed=sum(r['pass'] for r in quiet['per_environment']))
   assert state['gate']['passed']==bool(gate['passed'] and quiet['passed'])
  else:
   if state['status']=='completed':
    accepted=validate_result(p,state['identity']);assert accepted==read(run/(phase+'_accepted.json'));assert accepted==campaign['accepted_phases'][phase]
    entry['strict_consumer_result_receipt_equal']=True
   with np.load(p/'reference_state.npz',allow_pickle=False) as z:ref={k:z[k] for k in z.files}
   if phase=='smoke':
    cal=read(p/'calibration.json');trials={}
    for label in ('zero_mean','sampled'):
     saved=cal['trials'][label];actual=quiet_trial(rows,names,saved['start_control_index'],saved['end_control_index_exclusive']);assert normal(actual)==saved
     trials[label]={'passed':actual['passed'],'replicas_passed':sum(r['pass'] for r in actual['per_environment']),'start':actual['start_control_index'],'end':actual['end_control_index_exclusive'],'max_residual_offset_rad':actual['maximum_residual_offset_rad']}
    saved=read(p/'post_update_quiet.json');actual=quiet_trial(rows,names,saved['start_control_index'],saved['end_control_index_exclusive']);assert normal(actual)==saved
    entry.update(calibration_exact_replay=trials,post_update_quiet_exact_replay=True,post_update_quiet_passed=actual['passed'],post_update_quiet_replicas_passed=sum(r['pass'] for r in actual['per_environment']),PPO_updates=state['PPO_updates_completed'])
    assert not d['requested_command'].any()
   else:
    actual=forward_stop_trial(SimpleNamespace(rows=rows,names=names,wave=SimpleNamespace(s=ref)));saved=read(p/'retention.json');assert normal(actual)==saved
    command=np.zeros((count,n,3));command[200:1400,:,0]=.005;np.testing.assert_array_equal(d['requested_command'],command)
    qualified=[e for e in actual['measured_events'] if e['confirmed_measured_touchdown'] and e['measured_reference_point_lift_m']>=.002]
    entry.update(retention_exact_replay=True,retention_passed=actual['passed'],completed_legs=actual['completed_legs'],qualified_landings=len(qualified),measured_forward_mps=actual['measured_forward_mps'],forward_displacement_m=actual['independent_progress']['measured_forward_displacement_m'],old50Hz_position_rate_gap_m=actual['independent_progress']['displacement_integral_difference_m'],reference_quiet_time_s=float(ref['quiet_time'][0]),stop_latency_s=float(ref['quiet_time'][0])-28.,quiet_duration_s=actual['final_quiet']['per_environment'][0]['window_duration_s'],quiet_passed=actual['final_quiet']['passed'],checkpoint_sha256=state['checkpoint_sha256'])
   with np.load(p/'sensor_clocks.npz',allow_pickle=False) as z:
    c={k:z[k] for k in z.files};current=c['sensor_timestamp_s'];assert current.shape==(count,n,14) and current.dtype==np.float32
    expected=current[0].copy()
    for i in range(1,count):
     for _ in range(8):expected=np.add(expected,np.float32(.0025),dtype=np.float32)
     np.testing.assert_array_equal(expected,current[i])
    for k in ('sensor_last_update_s','expected_timestamp_s'):np.testing.assert_array_equal(c[k],current)
    assert c['contact_valid'].all() and c['all_sensors_valid'].all() and not c['sensor_outdated'].any() and not c['sensor_age_s'].any() and not c['contact_age_s'].any()
    entry.update(all14_sensor_clocks_every_control_exact=True,observed_contact_age_max_s=float(c['contact_age_s'].max()),sensor_clock_final_s=float(current[-1,0,0]))
  with np.load(p/'physics_substeps.npz',allow_pickle=False) as z:
   expected_count=8*count+1;assert len(z['time_s'])==expected_count
   np.testing.assert_array_equal(z['control_index'],np.r_[-1,np.repeat(np.arange(count),8)])
   np.testing.assert_array_equal(z['substep_index'],np.r_[0,np.tile(np.arange(1,9),count)])
   counters=z['sim_step_counter'];np.testing.assert_array_equal(counters-counters[0],np.arange(expected_count));np.testing.assert_allclose(np.diff(z['sdk_sim_timestamp_s']),.0025,rtol=0,atol=1e-7)
   for key,ckey in [('joint_position_rad','joint_position_rad'),('joint_velocity_rad_s','joint_velocity_rad_s'),('root_link_position_world_m','position_world_m'),('computed_torque_nm','computed_torque_nm'),('applied_torque_nm','applied_torque_nm')]:
    values=z[key];np.testing.assert_array_equal(values[8::8],d[ckey])
    if key in ('computed_torque_nm','applied_torque_nm'):entry['postsettle400Hz_'+key+'_peak']=float(np.abs(values[1601:]).max())
   with np.load(p/'trace.npz',allow_pickle=False) as trace:np.testing.assert_array_equal(z['root_link_quaternion_world_xyzw'][8::8],trace['quaternion_world_xyzw'])
   q=z['joint_position_rad'];v=z['joint_velocity_rad_s'];start=1600;end=(1400*8 if phase.startswith('evaluate') else len(q)-1)
   delta=q[end].astype(float)-q[start].astype(float);vr=v[start:end+1].astype(float);integral=(.5*(vr[:-1]+vr[1:])).sum(0)*.0025;difference=integral-delta;worst=np.unravel_index(np.argmax(np.abs(difference)),difference.shape)
   entry.update(substep_count=expected_count,all_substep_times_and_control_endpoint_parity=True,rate_angle_discrepancy={'scope':'raw reported rate versus actual joint angle, diagnostic not admission replacement','duration_s':(end-start)*.0025,'replica':int(worst[0]),'joint':names[worst[1]],'angle_delta_rad':float(delta[worst]),'reported_rate_integral_rad':float(integral[worst]),'difference_rad':float(difference[worst])})
   if phase!='standing':
    with np.load(p/'final_observation.npz',allow_pickle=False) as o:
     assert o['policy'].shape==(n,846) and o['critic'].shape==(n,849);np.testing.assert_array_equal(o['policy'],o['critic'][:,:846]);np.testing.assert_array_equal(o['critic'][:,846:],(last_twist*5).astype(np.float32))
     np.testing.assert_array_equal(o['raw_sdk_joint_velocity_rad_s'],v[-1]);interval=(q[-1].astype(float)-q[-9].astype(float))/.02;np.testing.assert_array_equal(o['interval_joint_rate_rad_s'],interval);assert o['interval_rate_valid'].all()
     schema=read(p/'observation_schema.json');fields={f['name']:(f['start'],f['stop']) for f in schema['fields']};a,b=fields['joint_position_interval_average_rate'];np.testing.assert_array_equal(o['policy'][:,a:b],(interval*.05).astype(np.float32))
     history=o['policy'][:,:315].reshape(n,5,63)
     for i,k in enumerate(range(count-4,count+1)):np.testing.assert_array_equal(history[:,i,27:45],(v[k*8].astype(float)*.05).astype(np.float32))
     entry.update(final846849_packet_and_raw_vs_interval_channels_exact=True,runtime_schema_sha256=schema['schema_sha256'])
  print('REPLAY_PHASE_COMPLETE',phase,flush=True)
 report={'scope':'Independent terminal002 raw scoring/time/checkpoint-receipt replay; no optimization or GPU','campaign_status':campaign['status'],'raw_payloads_verified':len(audit['raw_payloads']),'phases':results,'PPO_updates_completed':campaign.get('PPO_updates_completed'),'native_velocity_fidelity_qualified':False,'Stage2_complete':False,'restored_unix':audit['pause_restoration']['restored_unix']}
 out=ROOT/'review.json'
 if out.exists():raise FileExistsError(out)
 out.write_text(json.dumps(normal(report),indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
