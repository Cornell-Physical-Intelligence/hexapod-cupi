"""Actual directional002 strafe rejection; exact frozen gate and full available telemetry."""
from pathlib import Path
import hashlib,json,sys,numpy as np
HERE=Path(__file__).resolve().parent;RAW=HERE.parent;SOURCE=RAW.parent/'reference_directional_adapter_002/source_directional_002'
sys.path.insert(0,str(SOURCE/'tools'))
from directional_metrics import score_direction
from screen_metrics import standing_screen,standing_quiet_review
from physics_substeps import displacement_check
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((RAW/'remote_audit.json').read_text())
for f,h in a['raw_payloads'].items():assert sha(RAW/f)==h,f
mapping=json.loads((SOURCE/'campaign_source_hashes.json').read_text());assert sha(SOURCE/'campaign_source_hashes.json')==a['source_manifest_sha256']
for f,h in mapping.items():assert sha(SOURCE/f)==h,f
phase=RAW/'run/left_strafe';state=json.loads((phase/'state.json').read_text())
with np.load(phase/'trace.npz',allow_pickle=False) as z:names=z['joint_names'].tolist();d={k:z[k].copy() for k in z.files if k not in ('joint_names','legs')}
with np.load(phase/'physics_substeps.npz',allow_pickle=False) as z:sub={k:z[k].copy() for k in z.files}
refs=json.loads((phase/'reference_states.json').read_text());gate=score_direction(d,refs,case='left_strafe',joint_names=names,failure=state['failure'])
assert state['status']=='rejected' and state['control_steps']==505 and state['gate']==gate and gate['passed'] is False
json.dumps(state,allow_nan=False);json.dumps(gate,allow_nan=False)
n=len(d['time_s']);assert n==505 and len(sub['time_s'])==n*8+1
np.testing.assert_array_equal(sub['control_index'],np.r_[-1,np.repeat(np.arange(n),8)])
np.testing.assert_array_equal(sub['substep_index'],np.r_[0,np.tile(np.arange(1,9),n)])
np.testing.assert_array_equal(sub['sim_step_counter']-sub['sim_step_counter'][0],np.arange(n*8+1))
np.testing.assert_allclose(np.diff(sub['sdk_sim_timestamp_s']),.0025,atol=1e-7,rtol=0)
for raw,control in [('joint_position_rad','joint_position_rad'),('joint_velocity_rad_s','joint_velocity_rad_s'),('root_link_position_world_m','position_world_m'),('root_link_quaternion_world_xyzw','quaternion_world_xyzw'),('computed_torque_nm','computed_torque_nm'),('applied_torque_nm','applied_torque_nm')]:np.testing.assert_array_equal(sub[raw][8::8],d[control])
settled=sub['computed_torque_nm'][1601:];peak=np.unravel_index(abs(settled).argmax(),settled.shape);absolute_index=peak[0]+1601
prefix=displacement_check(sub,200,n)
q=sub['joint_position_rad'][1600:,0].astype(np.float64);v=sub['joint_velocity_rad_s'][1600:,0].astype(np.float64)
delta=q[-1]-q[0];integral=.5*(v[:-1]+v[1:]).sum(0)*.0025;j=int(abs(integral-delta).argmax())
with np.load(RAW/'run/standing/trace.npz',allow_pickle=False) as z:sn=z['joint_names'].tolist();sd={k:z[k] for k in z.files if k not in ('joint_names','legs')}
standing=standing_screen(sd);quiet=standing_quiet_review(sd,sn);assert standing['passed'] and quiet['passed']
legs=('lf','lm','lr','rf','rm','rr');last=refs[-1]['result']['state'];supports=d['distal_contact'].sum(-1)[:,0]
assert np.where(supports[200:]<5)[0].tolist()==[304]
lastrows=[]
for i in range(n-10,n):lastrows.append({'control_index':i,'time_s':float(d['time_s'][i,0]),'distal_support':d['distal_contact'][i,0].tolist(),'normal_force_z_n':d['normal_force_world_n'][i,0,:,2].tolist()})
report={'scope':'Exact directional002 strafe rejection; remaining turn/arc unmeasured, no source/gate modification',
'source_manifest_sha256':a['source_manifest_sha256'],'source_files_verified':len(mapping),'raw_payloads_verified':len(a['raw_payloads']),
'fresh32_standing_physical_and_quiet_recomputed_pass':True,'actual_terminal_state':'rejected','exact_full_scalar_gate_replay_equal':True,'strict_terminal_JSON_serialization_passed':True,
'controls':n,'actual400Hz_samples':len(sub['time_s']),'every8th_control_endpoint_and_counter_timestamp_parity':True,'original_gate':gate,
'failure_classification':{'intended_swing_leg':last['current_leg'],'first_subfive_support_index':504,'time_s':10.1,'other_support_below_threshold':'rr',
 'RR_normal_force_n':float(np.linalg.norm(d['normal_force_world_n'][-1,0,5])),'unchanged_distal_support_threshold_n':1.,
 'RR_contact_point_finite':bool(d['contact_point_valid'][-1,0,5]),'RR_reaction_force_world_n':d['reaction_force_world_n'][-1,0,5].tolist(),
 'meaning':'RR carries nonzero force but is below admitted1N support while LM is airborne; no detached-foot or fallen-body claim',
 'preceding_confirmed_landings':['lf','rr'],'nonfoot_samples':gate['post_settle_nonfoot_env_steps'],'last10_control_rows':lastrows},
'postsettle400Hz_torque':{'max_requested_nm':float(abs(settled).max()),'max_applied_nm':float(abs(sub['applied_torque_nm'][1601:]).max()),
 'peak_time_s':float(sub['time_s'][absolute_index]),'runtime_joint':names[peak[2]],'samples_above1p6':int((abs(settled)>1.6).sum())},
'available_prefix400Hz_motion_4_to10p1s_not_full24s_gate':prefix,
'available_prefix400Hz_joint_rate_discrepancy':{'runtime_joint':names[j],'angle_delta_rad':float(delta[j]),'reported_rate_integral_rad':float(integral[j]),'difference_rad':float(integral[j]-delta[j])},
'no_full24s_motion_or_final_quiet_window':True,'unmeasured_cases':['left_turn','forward_right_arc'],
'remote_source930_assets550_owned_absence_and_pause044_restoration_verified':True,'new_live_lock_snapshot_claimed':False,
'next_declared_scope':'Fresh standing and only two unmeasured cases, same wave005/physics/gates; no identical strafe repeat',
'stage2_complete':False,'left_strafe_admitted':False,'velocity_fidelity_qualified':False}
p=HERE/'report.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'passed':gate['passed'],'trigger':report['failure_classification']['RR_normal_force_n'],'torque400Hz':report['postsettle400Hz_torque'],'prefix_q_rate':report['available_prefix400Hz_joint_rate_discrepancy']},indent=2))
