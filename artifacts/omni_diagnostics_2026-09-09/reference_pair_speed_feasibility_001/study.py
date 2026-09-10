"""Bounded target-only speed/cadence study using the byte-identical parent oracle."""
from pathlib import Path
import copy,hashlib,json,sys,time
import numpy as np
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H/'oracle'))
from planned_target_oracle import evaluate
from wave_reference import WaveContactReference,LEGS,tensor

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def inputs():
 frozen=json.loads((H/'PARENT_INPUTS_SHA256.json').read_text())
 for f,h in frozen['copied_files_sha256'].items():
  if sha(H/f)!=h:raise ValueError('Parent copied input changed: '+f)
 with np.load(H/'inputs/actual_pair001_startup_trace.npz',allow_pickle=False) as z:
  names=tuple(z['joint_names'].tolist());snapshot={k:z[k][-1].copy() for k in z.files if k!='joint_names'}
 return json.loads((H/'inputs/planned_target_base_plan.json').read_text()),names,snapshot

def proposal(base,speed,swing,study):
 if not np.isfinite([speed,swing]).all() or speed<=0 or swing<=0:raise ValueError('Finite positive speed/duration')
 p=copy.deepcopy(base);p['candidate_id']=f'forward_{round(speed*1000):03d}mmps_swing_{round(swing*1000):04d}ms'
 p.update(requested_forward_left_yaw=[speed,0.,0.],swing_s=swing,cycle_s=3*(swing+study['handoff_hold_s']),
          placement_horizon_s=swing+.5*(3*(swing+study['handoff_hold_s'])-swing),
          lift_m=study['lift_m'],handoff_hold_s=study['handoff_hold_s'],horizontal_fraction=study['horizontal_fraction'])
 p['scope']='Target-only planned endpoints; timings are new proposed parameters and are not the contact-aware controller or a physical recording'
 return p

def classify(p,r,data,names,snapshot):
 g=WaveContactReference(names);g.reset(snapshot);lower=g._runtime(g.lower);upper=g._runtime(g.upper)
 jnames=[n for nameset in g.g.names for n in nameset];legof={n:LEGS[i//3] for i,n in enumerate(jnames)}
 events=[e for e in r['events'] if e['kind']=='planned_pair_swing_not_measured_liftoff']
 peakstrides=[length for e in events for length in e['planned_stride_m']]
 stop=p['initial_hold_s']+p['motion_s'];times=data['time_s'];command=data['command'][:,0]
 after=times>=stop;before=times<=stop
 integral=float((.5*(command[after][1:]+command[after][:-1])*np.diff(times[after])).sum()) if after.any() else None
 if r['failure']:
  t=r['failure']['time_s'];q,v,a=data['first_rejected_qva'];active=r['failure']['planned_pair'];category=[]
  if r['failure']['error'].startswith('Unreachable'):category.append('IK_geometry_or_soft_limit')
  if float(np.minimum(q-lower,upper-q).min())<.02:category.append('joint_residual_margin')
  if abs(v).max()>1.75+1e-5:category.append('reference_velocity')
  if abs(a).max()>6.+1e-5:category.append('reference_acceleration')
  if r['failure']['planned_support_margin_m']<.05:category.append('planned_support_margin')
  event=next((e for e in reversed(events) if e['time_s']<=t),None)
  phase='scheduled_stance_or_hold'
  if active is not None:
   age=t-event['time_s'];phase='swing_XY_and_Z' if age<p['swing_s']*p['horizontal_fraction'] else 'swing_Z_after_XY_end'
  if t>=stop:phase='stop_'+phase
  joints={kind:{'joint':names[int(abs(arr).argmax())],'leg':legof[names[int(abs(arr).argmax())]],'value':float(arr[int(abs(arr).argmax())])} for kind,arr in [('v',v),('a',a)]}
  joints['minimum_margin']={'joint':names[int(np.minimum(q-lower,upper-q).argmin())],'rad':float(np.minimum(q-lower,upper-q).min())}
  _,J,_=g.g.fk(tensor(g._leg(q)));sigma=np.linalg.svd(J.numpy(),compute_uv=False)[...,-1]
  crossing={'categories':category,'time_s':t,'phase':phase,'active_pair':active,'joint_peaks':joints,'min_jacobian_singular_value_m':float(sigma.min()),'planned_stride_of_active_pair_m':None if active is None or event is None else event['planned_stride_m'],'planned_stride_of_last_started_pair_m':None if event is None else event['planned_stride_m'],'first_rejected_q_is_diagnostic_only':True,'not_emitted':True}
 else:crossing=None
 # No angle or joint-rate evidence from actual physics is replayed/time-scaled.
 qleg=np.stack([g._leg(q) for q in data['q']]);_,J,_=g.g.fk(tensor(qleg));sigma=np.linalg.svd(J.numpy(),compute_uv=False)[...,-1]
 return {'first_crossing':crossing,'max_planned_stride_m':max(peakstrides,default=0.),'nominal_steady_stride_speed_times_cycle_m':p['requested_forward_left_yaw'][0]*p['cycle_s'],
         'minimum_accepted_Jacobian_singular_value_m':float(sigma.min()),'reference_speed_fraction':r['target_velocity_peak_rad_s']/1.75,'reference_acceleration_fraction':r['target_acceleration_peak_rad_s2']/6.,
         'joint_margin_beyond_reserved_0p02_rad':r['minimum_executed_joint_margin_rad']-.02,'requested_stop_time_s':stop,'planned_stop_command_integral_m':integral,'stop_integral_complete':r['reference_quiet_time_s'] is not None,
         'actual_stop_or_torque_measured':False,'qualified_contact_simulator_used':False}

def main():
 output=H/'results'
 if output.exists():raise FileExistsError('Use a new result directory/version')
 output.mkdir();matrix=json.loads((H/'PLAN.json').read_text());base,names,snapshot=inputs()
 cases=[(s,2.) for s in matrix['baselines_mps']]+[(s,t) for s in matrix['speeds_mps'] for t in matrix['swing_s']]
 result=[]
 for speed,swing in cases:
  p=proposal(base,speed,swing,matrix);r,data=evaluate(p,names,snapshot);diagnostic=classify(p,r,data,names,snapshot)
  cid=p['candidate_id'];np.savez_compressed(output/(cid+'.npz'),**data,joint_names=np.asarray(names))
  row={'parameters':p,'parameters_sha256':hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest(),'result':r,'diagnostic':diagnostic,'target_npz_sha256':sha(output/(cid+'.npz'))}
  (output/(cid+'.json')).write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');result.append(row)
  print(json.dumps({'id':cid,'passed':r['completed_target_sequence'],'v':r['target_velocity_peak_rad_s'],'a':r['target_acceleration_peak_rad_s2'],'margin':r['minimum_executed_joint_margin_rad'],'crossing':diagnostic['first_crossing']},allow_nan=False),flush=True)
 summary={'scope':matrix['scope'],'matrix_plan_sha256':sha(H/'PLAN.json'),'parent_inputs_sha256':sha(H/'PARENT_INPUTS_SHA256.json'),'driver_sha256':sha(Path(__file__)),'cases':result,'GPU_dispatches':0,'contact_gates_modified':False,'measured_physics_admission':False,'actor_adoption':False}
 (H/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
