from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from test_landing import ActualPrefix,LandedFixture,TRACE
p=ActualPrefix();out=p.replay(touch_command=[0,0,0]);d=p.d
s=p.r.swing;l=p.r.landing;t0=l.t0
pts=np.array([l.sample(t)[0] for t in np.linspace(t0,l.end_time,501)])
original_target=s.end.copy();trigger=l.sample(t0)
fixture=LandedFixture(p);v=[];a=[]
for k in range(600):
    out=fixture.step()
    if not out['valid'][0]:raise RuntimeError(out['failure_reason'])
    v.append(float(abs(out['v_ref']).max()));a.append(float(abs(out['a_ref']).max()))
report=dict(scope='Frozen actualreference002 prefix + synthetic controllercontinuation only; no newphysics',actual_trace_sha256=hashlib.sha256(TRACE.read_bytes()).hexdigest(),actual=dict(standing_passed_replicas=32,flight_first_step=223,flight_last_step=272,flight_samples=50,return_step=273,return_time_s=float(d['time_s'][273,0]),return_normal_force_n=float(d['normal_force_world_n'][273,0,0,2]),lift_from_immediately_before_flight_m=float(d['reference_point_world_m'][250,0,0,2]-d['reference_point_world_m'][222,0,0,2]),lift_from_standing_sample199_m=float(d['reference_point_world_m'][250,0,0,2]-d['reference_point_world_m'][199,0,0,2]),body_displacement_m=(d['position_world_m'][273,0]-d['position_world_m'][199,0]).tolist(),measured_touchdown_confirmed=False),new_landing_reference=dict(start_s=t0,end_s=l.end_time,original_endpoint_world_m=original_target.tolist(),endpoint_world_m=l.end.tolist(),initial_position_world_m=trigger[0].tolist(),initial_velocity_world_mps=trigger[1].tolist(),initial_acceleration_world_mps2=trigger[2].tolist(),max_planar_overshoot_then_return_m=float(np.linalg.norm(pts[:,:2]-pts[0,:2],axis=1).max()),max_total_target_excursion_m=float(np.linalg.norm(pts-pts[0],axis=1).max()),original_endpoint_error_m=p.r.landing_original_contact_error_m),synthetic_stop_continuation=dict(max_reference_velocity_rad_s=max(v),max_reference_acceleration_rad_s2=max(a),liftoffs=p.r.liftoffs,confirmed_synthetic_touchdowns=p.r.touchdowns,final_mode=out['state']['mode'],finite_reference_stop_latency_s=p.r.reference_quiet_time-p.r.stop_requested_time,physical_result=False),independent_review='PPOagent7newtests and focusedsource review clear;16totaltests pass; no physicaladmission')
outdir=Path(__file__).parent;(outdir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');time=d['time_s'][:,0];ix=time>=4
ax.plot(time[ix]-4,d['reference_point_world_m'][ix,0,0,2]*1000,label='Actual LF toe (frozen failed screen)')
times=np.linspace(4,6,201);ax.plot(times-4,np.array([s.sample(t)[0][2] for t in times])*1000,'--',label='Original virtual swing target')
times2=np.linspace(t0,l.end_time,101);ax.plot(times2-4,np.array([l.sample(t)[0][2] for t in times2])*1000,':',linewidth=2,label='New landing target (CPU only)')
ax.axvline(1.48,color='gray',alpha=.6);ax.axhline(0,color='gray',alpha=.3);ax.set(xlabel='Seconds after motion request',ylabel='World toe/target height (mm)',title='Actual early contact after real clearance; new bounded landing awaits physics')
ax.legend(fontsize=8);fig.savefig(outdir/'landing_diagnosis.png',dpi=160)
print(json.dumps(report,indent=2))
