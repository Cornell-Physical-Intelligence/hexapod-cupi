"""Counterfactual target-only prefix: not a controller, physics replay or admission."""
from diagnose import *

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);out=ap.parse_args().output;out.mkdir(parents=True,exist_ok=False)
 z=np.load(RAW/'run/left_strafe/trace.npz');rows=json.loads((RAW/'run/left_strafe/reference_states.json').read_text());refs={e['physical_step']:e['result'] for e in rows if 'result'in e}
 g=SerialGeometry();order=[list(z['joint_names']).index(x) for ns in g.names for x in ns];inv=np.argsort(order)
 start=420;t0=float(z['time_s'][start,0]);duration=.30;dz=-.0005;dt=.02
 qs=[];base=[];times=[];ikvalid=[];jm=[];sigma=[];dzs=[]
 for k in range(200,505):
  r=refs[k];state=r['state'];q=np.array(r['q_ref'])[0][order].reshape(6,3);qc=q.copy();u=np.clip((r['target_time_s']-t0)/duration,0,1);blend=10*u**3-15*u**4+6*u**5;offset=dz*blend
  if offset!=0:
   # Preserve every other joint exactly; adjust only RR world Z in the same desired-body frame.
   R=np.array(state['desired_rotation_world_from_body']);p=np.array(state['desired_position_world_m']);feet=g.fk(tensor(q))[0].numpy();world=feet@R.T+p;world[5,2]+=offset;target=(world-p)@R;result=g.ik(tensor(target));qc[5]=result['q_checked'].numpy()[5];ikvalid.append(bool(result['valid'][5]));jm.append(float(result['minimum_joint_margin_rad'][5]));sigma.append(float(result['min_jacobian_singular_value_m'][5]))
  qs.append(qc.reshape(-1)[inv]);base.append(np.array(r['q_ref'])[0]);times.append(r['target_time_s']);dzs.append(offset)
 qs=np.array(qs);base=np.array(base);times=np.array(times);v=np.diff(qs,axis=0)/dt;a=np.diff(v,axis=0)/dt
 limits=z['soft_joint_pos_limits_rad'][200:505,0];margin=np.minimum(qs-limits[...,0],limits[...,1]-qs);delta=qs-base
 # Same measured-state instantaneous proportional term only, not a future torque prediction.
 pd_delta=30*delta
 after=np.arange(start-200,len(qs));velocity_peak=np.unravel_index(np.abs(v).argmax(),v.shape);accel_peak=np.unravel_index(np.abs(a).argmax(),a.shape)
 report=dict(scope='Single-change CPU target-prefix sensitivity only; no changed actual verdict, contact history or physical rollout',source_manifest_sha256=json.loads((RAW/'remote_audit.json').read_text())['source_manifest_sha256'],raw_trace_sha256=sha(RAW/'run/left_strafe/trace.npz'),requested_body_twist=[0,.005,0],only_changed_reference='RR stance world anchor Z lowered 0.5 mm with quintic C2 interpolation after first confirmed RR landing; frozen original desired-body trajectory retained',parameters=dict(leg='rr',offset_z_m=dz,begin_control_index=start,begin_time_s=t0,duration_s=duration,end_time_s=t0+duration),verified_prefix_controls=len(qs),end_at_actual_rejection_s=times[-1],original_real_rejection_preserved=True,original_gate_change=False,all_counterfactual_IK_valid=all(ikvalid),minimum_named_soft_joint_margin_rad=margin.min(),required_joint_margin_rad=.02,max_reference_velocity_rad_s=np.abs(v).max(),reference_velocity_budget_rad_s=1.75,max_reference_acceleration_rad_s2=np.abs(a).max(),reference_acceleration_budget_rad_s2=6.,max_target_change_rad=np.abs(delta).max(),max_instantaneous_fixed_measurement_proportional_torque_delta_nm=np.abs(pd_delta).max(),minimum_RR_jacobian_singular_value_m=min(sigma),velocity_peak=dict(time_s=times[velocity_peak[0]+1],joint=str(z['joint_names'][velocity_peak[1]])),acceleration_peak=dict(time_s=times[accel_peak[0]+2],joint=str(z['joint_names'][accel_peak[1]])),hypothesis='Restore roughly the 0.49 mm lost downward target-versus-toe vertical offset seen between RR landing and failure, without relabeling contact or lowering the 1 N threshold',limitations=['Original measured pose/contact data are not a counterfactual physics prediction.','The 0.5 mm value tests the observed preload change; it is not an identified force-controller gain.','Full swing/contact/stop and torque gates require a fresh physical run.','Recorded torque headroom is only 0.0824 N m on another leg; global redistribution cannot be certified by the local PD delta.','The fixed-foot reference point lies above the true contact patch during tilted-pad landing; no measured penetration is available.'])
 np.testing.assert_array_equal(qs[:start-200+1],base[:start-200+1])
 unchanged=[i for i in range(18) if i not in order[15:18]]
 np.testing.assert_array_equal(qs[:,unchanged],base[:,unchanged])
 assert all(ikvalid) and margin.min()>=.02 and np.abs(v).max()<=1.75 and np.abs(a).max()<=6
 (out/'candidate_prefix.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n');np.savez_compressed(out/'candidate_prefix_arrays.npz',time_s=times,target_rad=qs,original_target_rad=base,anchor_offset_z_m=dzs)
 print(json.dumps(clean(report),indent=2))
if __name__=='__main__':main()
