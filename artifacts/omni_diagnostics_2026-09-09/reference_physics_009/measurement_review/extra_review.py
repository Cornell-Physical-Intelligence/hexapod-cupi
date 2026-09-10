"""Actual009 joint-angle/rate, quiet and torque replay. No scoring substitution."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from frozen_screen_metrics import measured_progress,standing_quiet_review
from omni_quiet_review import quiet_metrics,QUIET_GATES
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def joint_review(d,t,start,end):
 assert np.array_equal(d['joint_names'],t['joint_names'])
 for key in ('joint_position_rad','joint_velocity_rad_s'):assert np.array_equal(d[key][8::8],t[key])
 q=d['joint_position_rad'][start*8:end*8+1].astype(float);v=d['joint_velocity_rad_s'][start*8:end*8+1].astype(float)
 dt=np.diff(d['time_s'][start*8:end*8+1]);assert np.allclose(dt,.0025,rtol=0,atol=1e-12)
 delta=q[-1]-q[0];fd=np.diff(q,axis=0)/dt[:,None,None]
 integrations={name:np.einsum('t,tej->ej',dt,x) for name,x in [('left',v[:-1]),('right',v[1:]),('trapezoid',.5*(v[:-1]+v[1:]))]}
 rows=[]
 for e in range(q.shape[1]):
  for j,name in enumerate(d['joint_names'].tolist()):
   rows.append(dict(environment=e,joint=name,actual_angle_delta_rad=float(delta[e,j]),actual400Hz_angle_range_rad=float(np.ptp(q[:,e,j])),world_root_position_at_start_m=d['root_link_position_world_m'][start*8,e].astype(float).tolist(),integrated_reported_rate_rad={k:float(x[e,j]) for k,x in integrations.items()},integral_minus_angle_delta_rad={k:float(x[e,j]-delta[e,j]) for k,x in integrations.items()},reported_rate_rms_rad_s=float(np.sqrt(np.mean(v[1:,e,j]**2))),angle_difference_interval_rate_rms_rad_s=float(np.sqrt(np.mean(fd[:,e,j]**2))),rate_disagreement_rms_rad_s=float(np.sqrt(np.mean((.5*(v[1:,e,j]+v[:-1,e,j])-fd[:,e,j])**2)))))
 return dict(control_boundaries=[start,end],duration_s=(end-start)*.02,all_control_q_and_v_endpoints_exact=True,rate_from_angles_semantics='interval-averaged actual angle differences; not independently sensed instantaneous velocity',pairs=len(rows),pairs_abs_integral_difference_gt_001rad=sum(abs(r['integral_minus_angle_delta_rad']['trapezoid'])>.01 for r in rows),worst_pair=max(rows,key=lambda r:abs(r['integral_minus_angle_delta_rad']['trapezoid'])),per_environment_joint=rows,velocity_measurement_qualified=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 result={}
 for phase in ['standing','wave']:
  root=a.run/phase;t=load(root/'trace.npz');d=load(root/'physics_substeps.npz');state=json.loads((root/'state.json').read_text());n=len(t['time_s'])
  assert len(d['time_s'])==n*8+1
  rows=[]
  for name,lo,hi in [('initial_before_first_observed_update',0,1),('actual_updates_before_scoring',1,1601),('all_scored_updates',1601,len(d['time_s']))]:
   v=d['computed_torque_nm'][lo:hi];k,e,j=np.unravel_index(np.abs(v).argmax(),v.shape);k+=lo
   rows.append(dict(interval=name,sample_index=int(k),environment=int(e),joint=str(d['joint_names'][j]),time_s=float(d['time_s'][k]),requested_signed_nm=float(d['computed_torque_nm'][k,e,j]),applied_signed_nm=float(d['applied_torque_nm'][k,e,j]),max_abs_requested_nm=float(np.abs(v).max()),samples_above1p6=int((np.abs(v)>1.6).sum()),max_abs_applied_nm=float(np.abs(d['applied_torque_nm'][lo:hi]).max())))
  r=dict(controls=n,trace_sha256=sha(root/'trace.npz'),substeps_sha256=sha(root/'physics_substeps.npz'),torque_intervals=rows,hardware_startup_qualified=False)
  r['post_settle_joints']=joint_review(d,t,200,n)
  if phase=='standing':r['quiet']=standing_quiet_review(t,t['joint_names'].tolist())
  else:
   refs=json.loads((root/'reference_states.json').read_text());final=refs[-1]['result']['state'];quiet_start=int(np.ceil(final['reference_quiet_time_s']/.02))+100
   r['original50Hz_progress_full24s']=measured_progress(t,start_step=199,end_step=1399)
   r['moving_joints']=joint_review(d,t,200,1400);r['quiet_joints']=joint_review(d,t,quiet_start,n)
   r['quiet_start_step']=quiet_start;r['quiet']=quiet_metrics(t,0,quiet_start,t['joint_names'].tolist(),.02)
   r['original_gate_passed']=state['gate']['passed'];r['original_quiet_bounds']=QUIET_GATES
   r['reported_final_mode']=final['mode'];r['generator_confirmed_touchdowns']=final['confirmed_touchdowns']
  result[phase]=r
 a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 for phase,r in result.items():print(json.dumps({'phase':phase,'torque':r['torque_intervals'],'worst_q_rate':r['post_settle_joints']['worst_pair'],'moving_progress':r.get('original50Hz_progress_full24s'),'quiet_start':r.get('quiet_start_step'),'quiet_pass':r['quiet'].get('passed',r['quiet'].get('pass'))},indent=2))
if __name__=='__main__':main()
