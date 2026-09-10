"""Read-only, narrow diagnosis of the actual train001 recovery rejection."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def analyze(root):
 phase=root/'run/train_10';raw=phase/'raw';state=json.loads((phase/'state.json').read_text())
 z=np.load(raw/'trace.npz');keys=['time_s','learning_active','episode_id','distal_contact','normal_force_world_n','computed_torque_nm','applied_torque_nm','base_contact','shaft_contact','coxa_contact','femur_contact','terminated','truncated','target_position_rad','reference_position_rad','residual_position_rad','residual_velocity_rad_s','raw_residual_action','joint_position_rad','joint_velocity_rad_s','position_world_m','quaternion_world_xyzw','training_truncated']
 t={k:z[k] for k in keys};s=np.load(raw/'physics_substeps.npz');tau=s['computed_torque_nm'];applied=s['applied_torque_nm'];events=json.loads((raw/'episodes.json').read_text())
 assert len(t['time_s'])==2600 and tau.shape==(20801,32,18)
 assert np.array_equal(s['joint_names'],state['layout']['joint_names_runtime'])
 assert np.array_equal(t['computed_torque_nm'],tau[8::8].astype(np.float64))
 assert np.array_equal(t['joint_position_rad'],s['joint_position_rad'][8::8].astype(np.float64))
 assert np.array_equal(s['relative_physics_index'],np.arange(20801))
 assert np.allclose(np.diff(s['time_s']),.0025,rtol=0,atol=1e-12)
 ended=np.flatnonzero(t['training_truncated'][2399]);ev=next(e for e in events if e['control']==2400)
 assert ev['ended_rows']==ended.tolist() and len(ended)==26
 final_tau=np.abs(tau[-8:]).max((0,2));nonfoot=t['base_contact'][-1]|t['shaft_contact'][-1].any(-1)|t['coxa_contact'][-1].any(-1)|t['femur_contact'][-1].any(-1)
 failed=ended[(final_tau[ended]>1.6)|(t['distal_contact'][-1,ended].sum(-1)!=6)|nonfoot[ended]|t['terminated'][-1,ended]]
 assert failed.tolist()==[8]
 r=8;j=int(np.argmax(np.abs(tau[-8:,r]).max(0)));names=state['layout']['joint_names_runtime']
 target=t['target_position_rad'][-1,r,j];q=t['joint_position_rad'][-1,r,j];qd=t['joint_velocity_rad_s'][-1,r,j]
 sublast=tau[-800:,r,j];endpointlast=t['computed_torque_nm'][-100:,r,j]
 reset_q=np.array(ev['reset_reference_joint_target_rad'])[ev['ended_rows'].index(r)]
 # First post-reset target is the unchanged RecoveryTargets quintic at age1.
 nominal=t['reference_position_rad'][-1,r];u=.01;first=reset_q+(nominal-reset_q)*u**3*(10+u*(-15+6*u))
 assert np.allclose(t['reference_position_rad'][2400,r],first,rtol=0,atol=1e-14)
 assert np.array_equal(t['target_position_rad'][-100:,r],np.broadcast_to(nominal,(100,18)))
 assert np.array_equal(t['reference_position_rad'][-1,r],t['reference_position_rad'][199,r])
 assert not np.any(t['raw_residual_action'][2400:,r]) and not np.any(t['residual_position_rad'][2400:,r])
 data={
 'schema':'actual_train001_recovery_failure_review_v1','qualifying':False,'input_root':str(root),
 'input_sha256':{str(p.relative_to(root)):sha(p) for p in [phase/'state.json',raw/'trace.npz',raw/'physics_substeps.npz',raw/'episodes.json']},
 'PPO_updates_completed':state['PPO_updates_completed'],'controls':2600,'decision_checkpoint_produced':False,
 'timeout_control':2400,'timeout_rows':ended.tolist(),'recovery_controls':200,'failed_rows':failed.tolist(),
 'six_contacts_at_failure_all_rows':bool((t['distal_contact'][-1].sum(-1)==6).all()),
 'recovered_rows_nonfoot':np.flatnonzero(nonfoot & np.isin(np.arange(32),ended)).tolist(),
 'recovered_rows_native_termination':np.flatnonzero(t['terminated'][-1]&np.isin(np.arange(32),ended)).tolist(),
 'final_eight_substep_requested_peak_by_row_nm':final_tau.tolist(),
 'all_applied_peak_nm_float32':float(np.abs(applied).max()),
 'failed_joint':{'row':r,'runtime_index':j,'runtime_name':names[j],'named_leg':'RM','coordinate':'tibia',
 'last_requested_nm':float(t['computed_torque_nm'][-1,r,j]),'last_eight_substep_peak_nm':float(np.abs(tau[-8:,r]).max()),
 'last_two_seconds_400Hz_peak_nm':float(np.abs(sublast).max()),'last_two_seconds_400Hz_min_abs_nm':float(np.abs(sublast).min()),
 'last_two_seconds_50Hz_min_abs_nm':float(np.abs(endpointlast).min()),'last_two_seconds_50Hz_max_abs_nm':float(np.abs(endpointlast).max()),
 'target_rad':float(target),'measured_position_rad':float(q),'reported_velocity_rad_s':float(qd),
 'endpoint_position_error_Kp30_nm':float(30*(target-q)),'endpoint_reported_rate_Kd06_nm':float(-.6*qd),
 'PD_component_caveat':'Endpoint component decomposition is diagnostic; actuator torque is computed before the endpoint integration, so these are not asserted bit-exact equal.',
 'last_two_seconds_interval_angle_rate_mean_rad_s':float((t['joint_position_rad'][-1,r,j]-t['joint_position_rad'][-101,r,j])/2.),
 'final_distal_force_z_N':t['normal_force_world_n'][-1,r,:,2].tolist()},
 'reset_contract_checks':{'reset_target_rad':reset_q.tolist(),'first_reference_matches_existing_quintic':True,
 'first_postreset_root_position_delta_from_initial_m':(t['position_world_m'][2400,r]-t['position_world_m'][0,r]).tolist(),
 'first_postreset_XYZW_equals_initial':bool(np.array_equal(t['quaternion_world_xyzw'][2400,r],t['quaternion_world_xyzw'][0,r])),
 'final_target_exact_same_canonical_as_initial_recovery':True,'final_two_seconds_target_constant':True,
 'recovery_raw_action_and_residual_exact_zero':True},
 'interpretation':'A persistent requested-torque failure after randomized timeout reset; all contacts remain. Nominal target and reset transform evidence do not support a stale action or frame mismatch. Reset-dependent contact/load distribution is a hypothesis, not an identified native solver cause.',
 'recommendation':'No unchanged retry. First compare the exact failing reset pose with an explicitly new canonical-joint reset distribution in a zero-residual repeated-recovery screen; unchanged physical targets/cap/gates, no PPO until all selected recovery criteria pass. Preserve randomized baseline failure.'}
 return data
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--input',type=Path,required=True);a.add_argument('--output',type=Path,required=True);x=a.parse_args()
 if x.output.exists():raise FileExistsError(x.output)
 x.output.write_text(json.dumps(analyze(x.input),indent=2)+'\n')
