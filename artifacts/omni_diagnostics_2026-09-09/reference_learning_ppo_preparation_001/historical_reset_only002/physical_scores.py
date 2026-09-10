"""New residual-scope scores; original physical/quiet bounds remain unchanged."""
import numpy as np
from omni_quiet_review import quiet_metrics,QUIET_GATES
from batch_wave import MODES
from screen_metrics import physical_metrics,measured_progress,measured_flight_touchdowns

def data_from_rows(rows):
    if not rows:raise ValueError('No physical samples')
    return {key:np.stack([r[key] for r in rows]) for key in rows[0] if key!='contact_point_world_m_raw'}

def quiet_trial(rows,names,start,end):
    if not 0<=start<end<=len(rows) or end-start<500:raise ValueError('Contiguous>=10s quiet interval required')
    data=data_from_rows(rows[:end]);window=data['requested_command'][start:]
    if np.any(window):raise ValueError('Quiet interval has a nonzero requested twist')
    metrics=physical_metrics(data,settle_steps=start)
    each=[quiet_metrics(data,i,start,list(names),.02) for i in range(data['joint_position_rad'].shape[1])]
    lag=float(np.abs(data['reference_to_executable_lag_rad'][start:]).max())
    passed=bool(metrics['original_basic_standing_physics_pass'] and metrics['post_settle_max_requested_torque_nm']<=1.6
        and metrics['post_settle_min_distal_support_count']==6 and metrics['max_target_cast_error_rad']<=2e-7
        and lag<=.02000001 and all(r['pass'] and r['window_duration_s']>=10. for r in each))
    return dict(passed=passed,scope='finite_position_residual_quiet_trial',start_control_index=start,end_control_index_exclusive=end,
        original_quiet_bounds=QUIET_GATES,per_environment=each,physical=metrics,
        maximum_residual_offset_rad=lag,rawSDK_rate_gate_unchanged=True,physical_rate_fidelity_qualified=False)

def forward_stop_trial(session):
    data=data_from_rows(session.rows)
    if len(session.rows)!=2400 or data['joint_position_rad'].shape[1]!=1:raise ValueError('Complete1x2400 forward-stop trial required')
    physical=physical_metrics(data);progress=measured_progress(data,start_step=199,end_step=1399,dt=.02)
    events=measured_flight_touchdowns(data,start_step=200)
    complete=sorted({e['leg_index'] for e in events if e['confirmed_measured_touchdown'] and e['measured_reference_point_lift_m']>=.002})
    s=session.wave.s
    quiet_start=int(np.ceil(float(s['quiet_time'][0])/.02))+100 if bool(s['quiet_valid'][0]) else 2400
    quiet=quiet_trial(session.rows,session.names,quiet_start,2400) if quiet_start+500<=2400 else None
    forward=float(-data['velocity_body_mps'][200:1400,0,1].mean())
    ref_lag=float(np.abs(data['reference_to_executable_lag_rad'][200:]).max())
    passed=bool(physical['original_basic_standing_physics_pass'] and physical['post_settle_max_requested_torque_nm']<=1.6
        and physical['post_settle_min_distal_support_count']>=5 and complete==list(range(6))
        and forward>=.0025 and progress['measured_forward_displacement_m']>=.0025*progress['duration_s']
        and progress['displacement_integral_difference_m']<=.005 and quiet and quiet['passed']
        and MODES[int(s['mode'][0])]=='reference_quiet_hold' and ref_lag<=.02000001)
    return dict(passed=passed,scope='deterministic_finite_residual_forward_stop_retention_not_omni',physical=physical,
        measured_events=events,completed_legs=complete,measured_forward_mps=forward,independent_progress=progress,
        final_quiet=quiet,reference_confirmed_touchdowns=int(s['touchdowns'][0]),reference_final_mode=int(s['mode'][0]),
        residual_lag_bound_rad=.02,zero_residual_lag_gate_not_applicable_to_explicit_new_residual_lineage=True)
