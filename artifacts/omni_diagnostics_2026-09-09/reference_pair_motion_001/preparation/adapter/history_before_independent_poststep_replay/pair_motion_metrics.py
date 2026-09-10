"""Explicit new moving-pair score; preserve old physical/progress/quiet gates."""
from pathlib import Path
import sys
import numpy as np
from pair_motion_contract import PROTOCOL,STARTUP_CONTROLS,MOTION_START,MOTION_END,STEPS
from pair_motion_measurements import check_sensor_row,torque_window
from screen_metrics import physical_metrics,measured_progress,measured_flight_touchdowns
sys.path.insert(0,str(Path(__file__).resolve().parent/'paired_runtime'))


def expected_request(step):
 return np.array([.01,0.,0.]) if MOTION_START<=step<MOTION_END else np.zeros(3)


def numeric_tree_difference(actual,expected,path='root'):
 """Structure/counters are exact; numeric float differences are explicit evidence."""
 if isinstance(actual,dict):
  if not isinstance(expected,dict) or set(actual)!=set(expected):raise ValueError('Replay keys differ: '+path)
  return max([numeric_tree_difference(actual[k],expected[k],path+'.'+k) for k in actual]+[0.])
 if isinstance(actual,(list,tuple,np.ndarray)):
  actual=list(actual)
  if not isinstance(expected,(list,tuple,np.ndarray)) or len(actual)!=len(expected):raise ValueError('Replay shape differs: '+path)
  return max([numeric_tree_difference(a,b,path+f'[{i}]') for i,(a,b) in enumerate(zip(actual,expected))]+[0.])
 if isinstance(actual,(bool,np.bool_,int,np.integer,str)) or actual is None:
  if actual!=expected:raise ValueError('Replay exact field differs: '+path)
  return 0.
 if not np.isfinite(actual) or not np.isfinite(expected):raise ValueError('Replay nonfinite scalar: '+path)
 return float(abs(actual-expected))


def replay_pair_reference(data,references,joint_names):
 from pair_motion import PairContactReference
 g=PairContactReference(joint_names);max_difference=0.;count=0;reset_seen=False
 for row in references:
  step=row['physical_step']
  if step<STARTUP_CONTROLS or step-1>=len(data['time_s']):raise ValueError('Missing measured prior row forpairedreplay')
  measured={k:v[step-1] for k,v in data.items()}
  if 'reset' in row:
   if reset_seen or step!=STARTUP_CONTROLS:raise ValueError('Exactlyonepairedreset required')
   result=g.reset(measured);expected=row['reset'];reset_seen=True
  else:
   if not reset_seen or step!=STARTUP_CONTROLS+count:raise ValueError('Sequential pairedreference rows required')
   result=g.step(measured,expected_request(step));expected=row['result'];count+=1
  difference=numeric_tree_difference(result,expected)
  max_difference=max(max_difference,difference)
  if difference>PROTOCOL['numerical_replay_absolute_tolerance']:raise ValueError('Numerical paired replay differs beyond1e-8')
 return {'reset_seen':reset_seen,'reference_controls_replayed':count,'max_absolute_difference':max_difference,
         'absolute_tolerance':PROTOCOL['numerical_replay_absolute_tolerance'],'complete_state_including_each_foot_replayed':True,
         'passed':bool(reset_seen and count>=len(data['time_s'])-STARTUP_CONTROLS)}


def substep_evidence(data,physics):
 n=len(data['time_s']);expected=n*8+1
 if len(physics.get('time_s',[]))!=expected:raise ValueError('Complete initial+8substeps/control required')
 if not np.array_equal(physics['relative_physics_index'],np.arange(expected)):raise ValueError('Substep index coverage differs')
 if not np.allclose(physics['time_s'],np.arange(expected)*.0025,atol=1e-12,rtol=0):raise ValueError('Substep clock differs')
 if not np.all(np.diff(physics['sim_step_counter'])==1):raise ValueError('Substep simulation counter gap')
 controls=physics['control_index'][1:].reshape(n,8);indices=physics['substep_index'][1:].reshape(n,8)
 if not np.array_equal(controls,np.broadcast_to(np.arange(n)[:,None],(n,8))) or not np.array_equal(indices,np.broadcast_to(np.arange(1,9),(n,8))):raise ValueError('Control/substep pairing differs')
 for field,ordinary in [('root_link_position_world_m','position_world_m'),('root_link_quaternion_world_xyzw','quaternion_world_xyzw'),
    ('root_link_velocity_world_mps','velocity_world_mps'),('joint_position_rad','joint_position_rad'),('joint_velocity_rad_s','joint_velocity_rad_s'),
    ('computed_torque_nm','computed_torque_nm'),('applied_torque_nm','applied_torque_nm')]:
  if not np.array_equal(physics[field][8::8],data[ordinary]):raise ValueError('Complete endpoint mismatch: '+field)
 selected=slice(STARTUP_CONTROLS*8+1,None)
 requested=physics['computed_torque_nm'][selected];applied=physics['applied_torque_nm'][selected]
 return {'samples':expected,'all8updates_and_control_endpoints_verified':True,
         'all_recorded_requested_peak_nm':float(abs(physics['computed_torque_nm']).max()),
         'all_recorded_applied_peak_nm':float(abs(physics['applied_torque_nm']).max()),
         'poststartup_requested_peak_nm':float(abs(requested).max()),'poststartup_applied_peak_nm':float(abs(applied).max()),
         'poststartup_requested_excess_samples':int((abs(requested)>1.6).sum()),
         'passed':bool(np.isfinite(requested).all() and np.isfinite(applied).all() and abs(requested).max()<=1.6 and abs(applied).max()<=1.60001)}


def score_pair_motion(data,references,physics,support_checks,*,joint_names,failure=None,dt=.02):
 from omni_quiet_review import quiet_metrics,QUIET_GATES
 n=len(data['time_s'])
 if data['time_s'].shape!=(n,1) or not np.allclose(data['time_s'][:,0],(np.arange(n)+1)*dt,rtol=0,atol=1e-7):raise ValueError('Exact paired50Hz timestamps required')
 expected=np.stack([expected_request(step) for step in range(n)])[:,None]
 if not np.array_equal(data['requested_command'],expected) or not np.array_equal(data['reference_admitted_target_twist'],expected):raise ValueError('Paired requested/admittedtwist sequence differs')
 actual=np.stack((-data['velocity_body_mps'][...,1],data['velocity_body_mps'][...,0],data['gyro_body_rad_s'][...,2]),-1)
 if not np.array_equal(data['actual_executed_body_navigation_twist'],actual):raise ValueError('Actual body/navigation frame differs')
 for i in range(n):check_sensor_row({k:v[i] for k,v in data.items()})
 gate=physical_metrics(data);substeps=substep_evidence(data,physics)
 replay=replay_pair_reference(data,references,joint_names)
 results=[r['result'] for r in references if 'result' in r]
 final=results[-1] if results else {};state=final.get('state',{})
 complete=[e for e in measured_flight_touchdowns(data,start_step=STARTUP_CONTROLS) if e['confirmed_measured_touchdown'] and e['measured_reference_point_lift_m']>=.002]
 no_derating=all(float(r['command_derating_factor'])==1. for r in results if np.asarray(r['valid']).all())
 checks_valid=(len(support_checks)==n-STARTUP_CONTROLS and all('failure' not in r and r['physical_step']==i+STARTUP_CONTROLS for i,r in enumerate(support_checks)))
 supports=[r['support']['support_margin_m'] for r in support_checks if 'support' in r]
 quiet=None;quiet_time=state.get('reference_quiet_time_s');quiet_start=None if quiet_time is None else int(np.ceil(quiet_time/dt))+100
 if quiet_start is not None and n>=quiet_start+500:
  if data['requested_command'][quiet_start:].any():raise ValueError('Nonzero request in pairedquietwindow')
  quiet=quiet_metrics(data,0,quiet_start,list(joint_names),dt);quiet['bounds']=QUIET_GATES
 motion=None
 if n>=MOTION_END:
  motion=measured_progress(data,start_step=MOTION_START-1,end_step=MOTION_END-1,dt=dt)
  motion['mean_measured_forward_mps']=float(-data['velocity_body_mps'][MOTION_START:MOTION_END,0,1].mean())
  motion['passed']=bool(motion['measured_forward_displacement_m']>=.5*.01*24. and motion['mean_measured_forward_mps']>=.005
    and motion['displacement_integral_difference_m']<=.005 and not motion['interval_terminations'] and not motion['interval_truncations'])
 gate.update(kind='explicit_four_support_paired_motion_not_scalarwave_orPPO',protocol=PROTOCOL,
  measured_qualified_flight_events=complete,completed_measured_leg_indices=sorted({e['leg_index'] for e in complete}),
  completed_pairs=state.get('completed_pairs',0),generator_confirmed_touchdowns=state.get('confirmed_touchdowns',0),
  generator_final_mode=state.get('mode'),full_state_numerical_replay=replay,complete400Hz=substeps,
  poststep_checks_complete=bool(checks_valid),minimum_measured_required_support_margin_m=min(supports) if supports else None,
  independent_forward_motion=motion,final_quiet_stop_window=quiet,no_command_derating=bool(no_derating),
  original_scalar_five_support_gate_unchanged=True,old846_849_packet_admitted=False,stage2_complete=False,
  passed=bool(failure is None and n==STEPS and gate['original_basic_standing_physics_pass'] and gate['post_settle_max_requested_torque_nm']<=1.6
   and gate['post_settle_min_distal_support_count']>=4 and substeps['passed'] and checks_valid and replay['passed'] and no_derating
   and len({e['leg_index'] for e in complete})==6 and state.get('completed_pairs',0)>=3
   and state.get('mode')=='reference_quiet_hold' and state.get('active_pair') is None
   and motion is not None and motion['passed'] and quiet is not None and quiet['pass'] and quiet['window_duration_s']>=10.
   and gate['max_reference_to_executable_lag_rad']<=1e-12 and gate['max_target_cast_error_rad']<=2e-7))
 return gate
