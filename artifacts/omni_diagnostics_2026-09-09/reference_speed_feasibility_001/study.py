"""Versioned CPU-only faster-wave matrix using unchanged frozen scalar code."""
from dataclasses import replace,asdict
from pathlib import Path
import hashlib,json,math,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'oracle'))
from wave_reference import WaveConfig,WaveContactReference
from test_wave_reference import Fixture
DT=.02
torch.set_num_threads(1)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify():
    plan=json.loads((ROOT/'PLAN.json').read_text())
    for section in ('source_files','inputs'):
        for path,value in plan[section].items():
            if sha(ROOT/path)!=value:raise ValueError('Bound immutable study input changed: '+path)
    return plan

def configuration(duration,speed):
    if duration not in (.5,.75,1.,1.5,2.) or speed not in (.005,.01,.02,.04):raise ValueError('Only predeclared duration/speed matrix')
    return replace(WaveConfig(),swing_s=duration,max_translation_mps=speed)

class IdealFixture(Fixture):
    """Perfect body/joint following with synthetic plane contacts; not physics.

    Uses actual009 named soft limits. It has zero standing PD preload, zero
    forbidden-contact flags and no forces/dynamics; all are explicit idealities.
    """
    def __init__(self):
        super().__init__();record=json.loads((ROOT/'inputs/actual009_soft_limits.json').read_text())
        values=np.asarray(record['soft_joint_pos_limits_rad'])
        self.actual_soft_limits=values[[record['names_runtime'].index(n) for n in self.names]]
    def snapshot(self):
        result=super().snapshot();result['soft_joint_pos_limits_rad']=self.actual_soft_limits[None].copy();return result

class AuditReference(WaveContactReference):
    """Read-only candidate-bound observation; parent checks remain authoritative."""
    def __init__(self,*args,**kwargs):
        self.bound_attempts=[];super().__init__(*args,**kwargs)
    def _bounds(self,q,v,a):
        attempt=dict(time_s=self.time,maximum_velocity_rad_s=float(np.abs(v).max()),maximum_acceleration_rad_s2=float(np.abs(a).max()),minimum_joint_margin_rad=float(np.minimum(q-self.lower,self.upper-q).min()),q_candidate=self._runtime(q).tolist(),v_candidate=self._runtime(v).tolist(),a_candidate=self._runtime(a).tolist())
        self.bound_attempts.append(attempt)
        return super()._bounds(q,v,a)

def case_id(duration,speed,bearing,yaw=0.):
    return f'swing{duration:g}_speed{speed:g}_bearing{bearing:g}_yaw{yaw:+g}'.replace('.','p').replace('+','plus').replace('-','minus')

def run_case(duration,speed,bearing=0.,yaw=0.,*,baseline=False,save_trace=True,move_seconds=24.,stop_seconds=12.):
    cfg=configuration(duration,speed);f=IdealFixture();r=AuditReference(f.names,cfg);initial=f.snapshot();r.reset(initial)
    command=np.array([speed*math.cos(math.radians(bearing)),speed*math.sin(math.radians(bearing)),yaw]);start=100;stop=start+round(move_seconds/DT);count=stop+round(stop_seconds/DT)
    rows=[];failure=None;liftoffs_at_stop=None;events=[];last_touch=0;last_lift=0;minsupport=math.inf;minmargin=math.inf;maxv=maxa=0.;minfactor=1.
    for k in range(count):
        requested=command if start<=k<stop else np.zeros(3)
        if k==stop:liftoffs_at_stop=r.liftoffs
        measured=f.snapshot();out=r.step(measured,requested)
        if not out['valid'][0]:
            candidate=r.bound_attempts[-1] if r.bound_attempts and abs(r.bound_attempts[-1]['time_s']-r.time)<1e-10 else None
            failure=dict(control_index=k,measured_time_s=float(measured['time_s'][0]),reference_time_s=r.time,reason=out['failure_reason'],current_leg=out['state']['current_leg'],mode=r.mode,bound_candidate=candidate)
            break
        diag=out['diagnostics'];maxv=max(maxv,float(np.abs(out['v_ref']).max()));maxa=max(maxa,float(np.abs(out['a_ref']).max()));minmargin=min(minmargin,diag['minimum_joint_margin_rad']);minsupport=min(minsupport,diag['measured_projected_COM_support_margin_m']);minfactor=min(minfactor,float(out['command_derating_factor']))
        if r.liftoffs!=last_lift:
            events.append(dict(kind='synthetic_liftoff',leg=out['state']['current_leg'],time_s=r.swing.t0,endpoint_world_m=r.swing.end.tolist(),stride_m=float(np.linalg.norm(r.swing.end-r.swing.coeff[0]))));last_lift=r.liftoffs
        if r.touchdowns!=last_touch:
            events.append(dict(kind='synthetic_confirmed_touchdown',time_s=r.time,confirmed_count=r.touchdowns));last_touch=r.touchdowns
        rows.append(dict(time=r.time,q=out['q_ref'][0],v=out['v_ref'][0],a=out['a_ref'][0],position=r.position.copy(),requested=requested.copy(),admitted=r.command.copy(),mode=r.mode,support=diag['measured_projected_COM_support_margin_m'],joint_margin=diag['minimum_joint_margin_rad'],distal_count=int(measured['distal_contact'].sum())))
        f.advance(out)
    stopped=r.reference_quiet_time is not None and r.current_leg is None and np.array_equal(r.command,np.zeros(3))
    latency=None if r.reference_quiet_time is None or r.stop_requested_time is None else r.reference_quiet_time-r.stop_requested_time
    complete=failure is None and len(rows)==count and stopped and liftoffs_at_stop==r.liftoffs
    ident=case_id(duration,speed,bearing,yaw)
    row=dict(id=ident,baseline=baseline,swing_s=duration,requested_speed_mps=speed,bearing_deg=bearing,requested_yaw_rad_s=yaw,requested_twist=command.tolist(),configuration=asdict(cfg),completed_ideal_sequence=complete,failure=failure,controls_completed=len(rows),planned_controls=count,confirmed_synthetic_touchdowns=r.touchdowns,planned_liftoffs=r.liftoffs,liftoffs_after_stop=None if liftoffs_at_stop is None else r.liftoffs-liftoffs_at_stop,minimum_admission_factor=minfactor,maximum_valid_target_velocity_rad_s=maxv,maximum_valid_target_acceleration_rad_s2=maxa,minimum_valid_joint_margin_rad=None if not math.isfinite(minmargin) else minmargin,minimum_synthetic_projected_COM_margin_m=None if not math.isfinite(minsupport) else minsupport,maximum_attempted_velocity_rad_s=max(x['maximum_velocity_rad_s'] for x in r.bound_attempts),maximum_attempted_acceleration_rad_s2=max(x['maximum_acceleration_rad_s2'] for x in r.bound_attempts),finite_reference_stop_latency_s=latency,final_reference_mode=r.mode,events=events,physics_or_torque_admitted=False,forbidden_contact_geometry_checked=False,actual_measured_motion_retimed=False)
    if save_trace:
        (ROOT/'traces').mkdir(exist_ok=True)
        np.savez_compressed(ROOT/'traces'/f'{ident}.npz',**{key:np.array([x[key] for x in rows]) for key in rows[0]} if rows else {},joint_names=np.array(f.names))
        (ROOT/'traces'/f'{ident}.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
    return row

def main():
    plan=verify();started=time.monotonic();rows=[]
    configs=[(2.,.005,True)]+[(d,s,False) for d in plan['stage_one']['swing_durations_s'] for s in plan['stage_one']['requested_translation_mps']]
    for d,s,baseline in configs:
        row=run_case(d,s,baseline=baseline);rows.append(row)
        print(row['id'], 'PASS' if row['completed_ideal_sequence'] else {k:row['failure'][k] for k in ['control_index','reason','current_leg']}, 'steps',row['confirmed_synthetic_touchdowns'],'v/a',round(row['maximum_valid_target_velocity_rad_s'],3),round(row['maximum_valid_target_acceleration_rad_s2'],3),flush=True)
    eligible=[r for r in rows if not r['baseline'] and r['completed_ideal_sequence'] and r['confirmed_synthetic_touchdowns']>=12 and r['minimum_admission_factor']>=1.-1e-12]
    selected=sorted(eligible,key=lambda r:(-r['requested_speed_mps'],r['maximum_valid_target_acceleration_rad_s2']))[:3]
    follow=[]
    for candidate in selected:
        for bearing,yaw in [(180.,0.),(90.,0.),(270.,0.),(45.,0.),(0.,.01),(0.,-.01)]:
            row=run_case(candidate['swing_s'],candidate['requested_speed_mps'],bearing,yaw);follow.append(row)
            print(row['id'],'PASS' if row['completed_ideal_sequence'] else {k:row['failure'][k] for k in ['control_index','reason','current_leg']},flush=True)
    report=dict(scope='Synthetic ideal body/joint following and plane contacts; no forces or physical retiming',plan=plan,cpu_seconds=time.monotonic()-started,stage_one=rows,selected_for_directional_followup=[r['id'] for r in selected],directional_followup=follow,physics_admitted=False,GPU_launches=0,formal_reference_budget={'velocity_rad_s':1.75,'acceleration_rad_s2':6.,'joint_margin_rad':.02},formal_total_budget={'velocity_rad_s':2.,'acceleration_rad_s2':8.},comparison_note='Formal .04rad/20ms is an executable target budget, not a measured motor speed specification; 1.6Nm torque feasibility is untested here')
    (ROOT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');verify()
if __name__=='__main__':main()
