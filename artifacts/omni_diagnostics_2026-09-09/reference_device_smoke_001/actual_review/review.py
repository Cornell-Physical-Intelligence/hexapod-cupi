"""Independent actual001 bridge audit; never starts an application or GPU."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent;TMP=ROOT.parent;RUN=ROOT/'raw/reference_device_smoke_001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def load(p):
 with np.load(p,allow_pickle=False) as d:return {k:d[k] for k in d.files}
def main():
 audit=read(ROOT/'remote_audit.json')
 for path,value in audit['raw_payloads'].items():assert sha(ROOT/'raw'/path)==value,path
 hostpath=TMP/'reference_device_smoke_launch_002';assert sha(hostpath/'FREEZE_SHA256.json')=='adbb60b3d77b8d6b268dcf3060232fcf3c493361259a0997582e694d62a35658'
 for f,h in read(hostpath/'FREEZE_SHA256.json').items():assert sha(hostpath/f)==h
 sys.path.insert(0,str(hostpath));import launch_device_smoke_spark as host
 campaign=read(RUN/'campaign.json');assert campaign['status']=='completed' and campaign['both_bridge_phases_passed'] and campaign['source_and_inputs_unchanged']
 rows=[];schemas=[]
 for phase,n in [('replicas_1',1),('replicas_32',32)]:
  p=RUN/phase;result=host.validate_result(p,campaign['identities'][phase]);assert result==read(RUN/(phase+'_accepted.json'))
  s=read(p/'state.json');m=load(p/'last_device_sample.npz');o=load(p/'final_observation.npz');d=load(p/'physics_substeps.npz');c=load(p/'sensor_clocks.npz');ref=load(p/'last_reference_state.npz');schema=read(p/'observation_schema.json');schemas.append(schema)
  # Independent NumPy recurrence, separate from the standard-library host reader.
  current=c['sensor_timestamp_s'];expected=current[0].copy()
  for i in range(1,264):
   for _ in range(8):expected=np.add(expected,np.float32(.0025),dtype=np.float32)
   np.testing.assert_array_equal(expected,current[i])
  for k in ('sensor_last_update_s','expected_timestamp_s'):np.testing.assert_array_equal(current,c[k])
  assert not c['sensor_outdated'].any() and not c['sensor_age_s'].any() and not c['contact_age_s'].any()
  assert c['all_sensors_valid'].all() and c['contact_valid'].all()
  np.testing.assert_array_equal(o['policy'],o['critic'][:,:846])
  np.testing.assert_array_equal(o['critic'][:,846:],(m['velocity_body_mps']*5).astype(np.float32))
  np.testing.assert_array_equal(o['raw_sdk_joint_velocity_rad_s'],m['joint_velocity_rad_s'])
  np.testing.assert_array_equal(m['joint_velocity_rad_s'],d['joint_velocity_rad_s'][-1])
  np.testing.assert_array_equal(m['joint_position_rad'],d['joint_position_rad'][-1])
  np.testing.assert_array_equal(m['position_world_m'],d['root_link_position_world_m'][-1])
  np.testing.assert_array_equal(m['quaternion_world_xyzw'],d['root_link_quaternion_world_xyzw'][-1])
  np.testing.assert_array_equal(m['quaternion_world_wxyz'],m['quaternion_world_xyzw'][:,[3,0,1,2]])
  interval=(d['joint_position_rad'][-1].astype(float)-d['joint_position_rad'][-9].astype(float))/.02
  np.testing.assert_array_equal(interval,o['interval_joint_rate_rad_s']);assert o['interval_rate_valid'].all()
  fields={f['name']:(f['start'],f['stop']) for f in schema['fields']}
  a,b=fields['joint_position_interval_average_rate'];np.testing.assert_array_equal(o['policy'][:,a:b],(interval*.05).astype(np.float32))
  history=o['policy'][:,:315].reshape(n,5,63)
  for frame,step in enumerate(range(260,265)):
   np.testing.assert_array_equal(history[:,frame,27:45],(d['joint_velocity_rad_s'][step*8].astype(float)*.05).astype(np.float32))
  np.testing.assert_array_equal(o['policy'][:,315:320],np.ones((n,5),np.float32))
  assert m['measurement_valid'].all() and m['distal_contact'].all() and m['contact_valid'].all()
  for k in ('terminated','truncated','base_contact','coxa_contact','femur_contact','shaft_contact'):assert not m[k].any()
  assert ref['ready'].all() and not ref['failure'].any() and not ref['liftoffs'].any() and not ref['touchdowns'].any()
  assert not ref['command'].any() and not ref['requested'].any() and not ref['v'].any() and not ref['a'].any()
  np.testing.assert_array_equal(ref['q'].astype(np.float32),m['joint_target_rad'].astype(np.float32))
  dt=np.diff(d['time_s'][1600:]);q=d['joint_position_rad'][1600:].astype(float);v=d['joint_velocity_rad_s'][1600:].astype(float)
  integral=np.sum((v[:-1]+v[1:])*.5*dt[:,None,None],axis=0);delta=q[-1]-q[0];err=integral-delta;worst=np.unravel_index(np.argmax(abs(err)),err.shape)
  timing=read(p/'timings.json')['samples'];metrics={}
  for key in ('reference_and_target_set_ms','env_step_including_capture_ms','nested_device_capture_ms','observation_history_ms'):
   values=np.asarray([r[key] for r in timing]);metrics[key]={'median':float(np.median(values)),'p95':float(np.percentile(values,95)),'max':float(values.max())}
  rows.append({'phase':phase,'replicas':n,'controls':264,'new_hold_controls':64,'new_hold_duration_s':1.28,'host_raw_gate_passed':True,'actor_width':846,'critic_width':849,'schema_sha256':schema['schema_sha256'],'critic_actor_prefix_exact':True,'critic_reported_twist_tail_exact':True,'all_five_history_reported_rate_slots_exact':True,'final_interval_rate_matches_actual400Hz_angle_endpoints_exact':True,'all_actual_clock_rows_valid':True,'sensor_clock_first_s':float(current[0,0,0]),'sensor_clock_last_s':float(current[-1,0,0]),'source_device_reader_compared_channels':len(s['comparison_keys']),'final_distal_supports_min':int(m['distal_contact'].sum(-1).min()),'final_bad_contact_or_terminal':False,'reference_liftoffs':0,'postsettle400Hz_requested_peak_nm':s['postsettle_max_requested_nm'],'postsettle400Hz_applied_peak_nm':s['postsettle_max_applied_nm'],'final_raw_rate_abs_max_rad_s':float(abs(o['raw_sdk_joint_velocity_rad_s']).max()),'final_interval_rate_abs_max_rad_s':float(abs(interval).max()),'final_raw_minus_interval_abs_max_rad_s':float(abs(o['raw_sdk_joint_velocity_rad_s']-interval).max()),'hold_rate_angle_discrepancy':{'duration_s':float(dt.sum()),'worst_replica':int(worst[0]),'joint':str(d['joint_names'][worst[1]]),'angle_delta_rad':float(delta[worst]),'reported_trapezoid_integral_rad':float(integral[worst]),'integral_minus_angle_delta_rad':float(err[worst])},'timing_ms':metrics,'physics_substeps':len(d['time_s']),'packet_schema_equal_across_replica_counts':True,'PPO_or_full_quiet_admitted':False})
 assert schemas[0]==schemas[1]
 report={'scope':'Actual short CUDA device bridge at1/32; no policy/fullquiet/walking/velocity-fidelity admission','raw_payloads_verified':len(audit['raw_payloads']),'remote_audit_sha256':sha(ROOT/'remote_audit.json'),'campaign_sha256':sha(RUN/'campaign.json'),'source_and_assets_unchanged':True,'exact_owned_absence_verified':len(audit['owned_containers_absent']),'pause_restored_unix':audit['pause_restoration']['restored_unix'],'phases':rows,'velocity_fidelity_qualified':False,'PPO_admitted':False,'GPU_launches_by_this_review':0}
 (ROOT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
