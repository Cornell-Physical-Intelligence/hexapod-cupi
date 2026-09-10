from pathlib import Path
import json,time,hashlib
import numpy as np
import torch
from time_governor import *
OUT=Path(__file__).resolve().parent
BASE=OUT.parent
torch.set_num_threads(1)
g=SerialGeometry()
commands=json.loads((BASE/'report.json').read_text())['command_screen']['rows']
phase=torch.arange(512,dtype=DTYPE)/512
segments=[('stand',1,[0,0,0]),('forward',2,[.1,0,0]),('reverse',2,[-.1,0,0]),
          ('left',2,[0,.1,0]),('left_arc',2,[0,.1,.2]),('right_arc',2,[.1,0,-.2]),
          ('yaw',2,[0,0,.3]),('stop',8,[0,0,0])]
reports=[];start=time.monotonic()
for label,cfg in [('beta065_60mm_10mm',Config(stance_travel_m=.06,lift_m=.01)),
                  ('beta050_60mm_5mm',Config(duty=.50,stance_travel_m=.06,lift_m=.005,min_frequency_hz=.25))]:
    r=TwistReference(g,cfg);max_rate=0.;invalid=0;minmargin=1.;maxfreq=0.;max_error=0.
    for row in commands:
        c=tensor([row['command']]).expand(len(phase),3)
        state=r.state(phase,c,torch.zeros_like(c))
        invalid+=int((~state['valid']).sum());max_rate=max(max_rate,float(state['q_velocity'].abs().max()))
        minmargin=min(minmargin,float(state['minimum_joint_margin_rad'].min()))
        maxfreq=max(maxfreq,float(state['frequency'].max()))
        max_error=max(max_error,float(state['error_m'].max()))
    # A separate dense virtual transition screen supplements the constant cycles.
    f=FilterState(r);transition_rate=0.;transition_invalid=0
    for _,duration,target in segments:
        for _ in range(round(duration/.01)):
            state=f.step([target],.01)
            transition_rate=max(transition_rate,float(state['q_velocity'].abs().max()))
            transition_invalid+=int((~state['valid']).sum())
    bound=max(max_rate,transition_rate)
    scale=time_scale_from_rate_bound(bound)
    result=dict(configuration=cfg.__dict__,
                constant_cycles=dict(cases=len(commands),phases_per_case=len(phase),invalid_leg_phases=invalid,
                                     max_virtual_joint_speed_rad_s=max_rate,min_soft_limit_margin_rad=minmargin,
                                     max_fk_error_m=max_error,max_virtual_frequency_hz=maxfreq),
                virtual_transition=dict(max_joint_speed_rad_s=transition_rate,invalid_leg_steps=transition_invalid),
                global_time_scale=scale,feedback_rate_reserve_rad_s=.25,
                sampled_proposed_command_envelope=dict(max_translation_mps=.2*scale,max_yaw_rad_s=.4*scale,
                    bearings_degrees=22.5,command_targets_are_derated=True,
                    note='All16 bearings and sampled interior speeds/yaws checked; not a formal continuum or dynamics guarantee.'),
                ideal_steady_stop_bounds=dict(translation_distance_m=.1,yaw_angle_rad=.2,
                    derivation='2*v/omega_filter and2*yaw/omega_filter for v=.2,yaw=.4,omega_filter4; coherent time scaling preserves these virtual path integrals.'))
    if invalid or transition_invalid:
        result['status']='rejected_before_physical_time_test';reports.append(result);continue
    # Independent real20ms stepping of the whole sequence. No q projection/slew.
    f=FilterState(r);trace=[];lastq=None;clock=0.;max_delta=0.;max_qrate=0.;accel=0.;yawaccel=0.
    stop_started=None;stopped=None;max_slip=0.;total_error=0.;max_derating=0.
    for segment,duration,target in segments:
        remaining=duration/scale
        if segment=='stop':stop_started=clock
        while remaining>1e-9:
            dt=min(.02,remaining)
            state=advance_governed(f,[target],dt,scale)
            q=state['q_checked'][0];delta=0. if lastq is None else float((q-lastq).abs().max())
            # Account for the one shorter segment-boundary sample explicitly.
            max_delta=max(max_delta,delta*min(1.,.02/dt))
            max_qrate=max(max_qrate,float(state['physical_q_velocity'].abs().max()))
            accel=max(accel,float(state['admitted_acceleration'][0,:2].norm()))
            yawaccel=max(yawaccel,float(state['admitted_acceleration'][0,2].abs()))
            xy=state['feet'][...,:2];c=state['admitted_command']
            body=nav_to_body(c)[:,None,:]+c[:,None,2,None]*torch.stack((-xy[...,1],xy[...,0]),-1)
            mask=torch.remainder(f.phase[:,None]+r.offsets,1)<cfg.duty
            slip=float((state['physical_foot_velocity'][...,:2]+body).norm(dim=-1)[mask].max())
            max_slip=max(max_slip,slip)
            max_derating=max(max_derating,float((tensor(target)-state['admitted_command_target'][0]).abs().max()))
            if segment=='stop' and stopped is None:
                all_near=(state['feet'][0]-g.feet0).norm(dim=-1).max()<.0002
                if all_near and state['physical_q_velocity'].abs().max()<.01 and state['admitted_command'].abs().max()<1e-4:stopped=clock+dt
            trace.append([clock,*state['admitted_command'][0].tolist(),*state['admitted_acceleration'][0].tolist(),delta,slip])
            lastq=q;clock+=dt;remaining-=dt
    np.savez_compressed(OUT/(label+'_trace.npz'),trace=np.array(trace),columns=np.array(['time_s','v_forward','v_left','yaw','accel_forward','accel_left','yaw_accel','max_joint_step_rad','stance_reference_slip_mps']))
    result.update(status='sampled_kinematic_limits_only_not_dynamic_admission',
                  physical_transition=dict(duration_s=clock,max_joint_step_rad=max_delta,max_q_velocity_rad_s=max_qrate,
                    limiter03_pass=max_delta<=.03,reference_budget_pass=max_qrate<=REFERENCE_RATE_BUDGET,
                    max_planar_command_acceleration_mps2=accel,max_yaw_command_acceleration_rad_s2=yawaccel,
                    stop_settle_s=None if stopped is None else stopped-stop_started,
                    max_stance_reference_world_slip_mps=max_slip,
                    max_abs_target_command_derating=max_derating,
                    note='Admitted vs requested commands explicitly differ; small q step is not original command tracking.'))
    reports.append(result)
report=dict(status='separate_unselected_CPU_global_clock_continuation',reports=reports,cpu_seconds=time.monotonic()-start,
            source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'study.py',OUT/'time_governor.py',BASE/'reference.py',BASE/'FREEZE_SHA256.json']},
            limits=['No torque/contact/support/pad collision qualification.',
                    'Uniform time scaling is held fixed over each declared sequence; online scale changes require additional continuity analysis.',
                    'First prototype results unchanged. Samples with invalid IK are rejected; individual joints are not clipped to make this screen pass.',
                    'Derivative extrema are sampled with10% margin, not proven global between-sample bounds.',
                    'Feedback reserve is only a rate budget; a real combined residual target still needs an enforced shared limiter and verification.'])
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
