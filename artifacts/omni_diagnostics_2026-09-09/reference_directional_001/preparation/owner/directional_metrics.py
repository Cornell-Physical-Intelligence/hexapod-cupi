"""Explicit limited directional checks; retain raw rate/pose evidence and old5mm bound."""
import numpy as np
from directional_contract import CASES,PROTOCOL
from screen_metrics import physical_metrics,measured_flight_touchdowns,measured_progress

def motion_evidence(data,requested,dt=.02):
 """Same complete4..28s interval for independent position/rate/heading evidence."""
 start,end=199,1399
 if len(data['position_world_m'])<=end:return None
 requested=np.asarray(requested,dtype=float);speed=float(np.linalg.norm(requested[:2]));duration=(end-start)*dt
 p=data['position_world_m'][start:end+1,0].astype(np.float64);r=data['rotation_world_from_body'][start:end+1,0].astype(np.float64)
 world_v=data['velocity_world_mps'][start:end+1,0].astype(np.float64)
 body=data['velocity_body_mps'][start+1:end+1,0].astype(np.float64);gyro=data['gyro_body_rad_s'][start+1:end+1,0].astype(np.float64)
 actual_nav=np.stack([-body[:,1],body[:,0],gyro[:,2]],axis=-1)
 progress=measured_progress(data,start_step=start,end_step=end,dt=dt)
 heading=np.unwrap(np.arctan2(-r[:,1,1],-r[:,0,1]));yaw_change=float(heading[-1]-heading[0])
 out={'duration_s':duration,'requested_forward_left_yaw':requested.tolist(),'mean_measured_body_navigation_twist':actual_nav.mean(0).tolist(),
  'old_world_displacement_integral_evidence':progress,'actual_heading_change_rad':yaw_change,
  'reported_body_yaw_rate_integral_rad':float(gyro[:,2].sum()*dt),
  'heading_minus_reported_body_yaw_integral_rad':float(yaw_change-gyro[:,2].sum()*dt),
  'max_planar_excursion_from_start_m':float(np.linalg.norm(p[:,:2]-p[0,:2],axis=1).max()),
  'translation_required':speed>0,'yaw_required':abs(requested[2])>0,'signed_translation_pass':None,'signed_yaw_pass':None}
 if speed:
  nav_axis=requested[:2]/speed
  direction=r@np.array([nav_axis[1],-nav_axis[0],0.]);direction[:,2]=0.
  length=np.linalg.norm(direction,axis=-1)
  if np.any(length<.5):raise ValueError('Requested direction projects poorly onto world ground')
  direction/=length[:,None];interval=.5*(direction[:-1]+direction[1:]);interval/=np.linalg.norm(interval,axis=-1)[:,None]
  along=float(np.sum(np.diff(p,axis=0)*interval));rate_along=float(np.sum(.5*(world_v[:-1]+world_v[1:])*interval)*dt)
  mean_along=float(actual_nav[:,:2].mean(0)@nav_axis)
  out.update(requested_axis_in_body_navigation=nav_axis.tolist(),measured_command_axis_displacement_m=along,
   integrated_world_velocity_command_axis_m=rate_along,mean_body_command_axis_velocity_mps=mean_along,
   commanded_axis_frame='At each measured pose rotate [left,-forward,0] into world, projectXY; use normalized adjacent-direction mean for each actual displacement',
   signed_translation_pass=bool(along>=.5*speed*duration and mean_along>=.5*speed))
 else:out['signed_translation_pass']=bool(out['max_planar_excursion_from_start_m']<=PROTOCOL['pure_turn_max_planar_excursion_m'])
 if requested[2]:
  sign=float(np.sign(requested[2]));out['signed_yaw_pass']=bool(sign*yaw_change>=.5*abs(requested[2])*duration and sign*actual_nav[:,2].mean()>=.5*abs(requested[2]))
 else:out['signed_yaw_pass']=True
 return out

def score_direction(data,references,*,case,joint_names,failure=None,dt=.02):
 from omni_quiet_review import quiet_metrics,QUIET_GATES
 requested=np.asarray(CASES[case]);n=len(data['time_s'])
 if data['time_s'].shape!=(n,1) or not np.allclose(data['time_s'][:,0],(np.arange(n)+1)*dt,atol=1e-7,rtol=0):raise ValueError('Actual physical command timestamps differ from exact50Hz case')
 expected=np.zeros((n,1,3));expected[200:min(n,1400),0]=requested
 if not np.array_equal(data['requested_command'],expected):raise ValueError('Actual requested command/time sequence differs from named case')
 if not np.array_equal(data['reference_admitted_target_twist'],expected):raise ValueError('Admitted target twist differs from requested named case')
 actual=np.stack((-data['velocity_body_mps'][...,1],data['velocity_body_mps'][...,0],data['gyro_body_rad_s'][...,2]),axis=-1)
 if not np.array_equal(data['actual_executed_body_navigation_twist'],actual):raise ValueError('Actual executed twist channel/frame differs from measured body data')
 gate=physical_metrics(data);events=measured_flight_touchdowns(data,start_step=200)
 complete=[e for e in events if e['confirmed_measured_touchdown'] and e['measured_reference_point_lift_m']>=.002]
 final=references[-1].get('result',{}) if references else {};state=final.get('state',{})
 results=[r['result'] for r in references if 'result' in r and np.asarray(r['result'].get('valid',[False])).all()]
 if any(float(r['command_derating_factor'])!=1. for r in results):raise ValueError('Named low-speed cases cannot silently derate command')
 motion=motion_evidence(data,requested,dt)
 quiet=None;quiet_time=state.get('reference_quiet_time_s');quiet_start=None if quiet_time is None else int(np.ceil(quiet_time/dt))+100
 if quiet_start is not None and n>=quiet_start+500:
  if np.any(data['requested_command'][quiet_start:]):raise ValueError('Nonzero requested command in final quiet window')
  quiet=quiet_metrics(data,0,quiet_start,list(joint_names),dt);quiet['bounds']=QUIET_GATES
 gate.update(kind='limited_directional_reference_discriminator_not_omni_or_PPO',case=case,requested_forward_left_yaw=requested.tolist(),
  measured_flight_events=events,completed_measured_leg_indices=sorted({e['leg_index'] for e in complete}),
  generator_confirmed_touchdowns=state.get('confirmed_touchdowns',0),generator_final_mode=state.get('mode'),
  command_channels='Raw rows retain requested, admitted target, filtered reference twist and actual measured body/navigation twist; references retain complete wave005 state',
  independent_directional_motion=motion,final_quiet_stop_window=quiet,all_omni_directions_qualified=False,stage2_complete=False,
  passed=bool(failure is None and n==2400 and gate['original_basic_standing_physics_pass']
   and gate['post_settle_max_requested_torque_nm']<=1.6 and gate['post_settle_min_distal_support_count']>=5
   and len({e['leg_index'] for e in complete})==6 and motion is not None and motion['signed_translation_pass'] and motion['signed_yaw_pass']
   and motion['old_world_displacement_integral_evidence']['displacement_integral_difference_m']<=.005
   and quiet is not None and quiet['pass'] and quiet['window_duration_s']>=10.
   and gate['max_reference_to_executable_lag_rad']<=1e-12 and gate['max_target_cast_error_rad']<=2e-7
   and state.get('mode')=='reference_quiet_hold'))
 return gate
