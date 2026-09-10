"""Exact recorded input prefix only; no counterfactual physics/force claim."""
from pathlib import Path
import hashlib,json,sys
import numpy as np,torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from test_preload import TRACE,NAMES,snapshot,parent,WaveContactReference,SOURCE,PARENT
from reference_residual import ReferenceResidualTarget,ResidualConfig
c=WaveContactReference(NAMES);old=parent.WaveContactReference(NAMES);seed=snapshot(199);r=c.reset(seed);old.reset(seed)
lim=seed['soft_joint_pos_limits_rad'][0];core=ReferenceResidualTarget(NAMES,dict(zip(NAMES,lim[:,0])),dict(zip(NAMES,lim[:,1])),1,ResidualConfig('formal_004',.02,.25,2.,8.),dtype=torch.float64);core.reset(r['q_ref'],r['q_ref'])
results=[];changed=[];vmax=amax=0.;minimum_margin=1e9;maxdelta=0.;maximum_correction_speed=maximum_correction_accel=0.
for k in range(200,505):
 r=c.step(snapshot(k-1),[0,.005,0]);o=old.step(snapshot(k-1),[0,.005,0]);assert r['valid'][0] and o['valid'][0]
 out=core.step(r['q_ref'],np.zeros((1,18)),reference_valid=r['valid'],analytic_reference_velocity=r['v_ref'],analytic_reference_acceleration=r['a_ref']);np.testing.assert_array_equal(out['target_position_rad'].numpy(),r['q_ref'])
 delta=r['q_ref']-o['q_ref'];peak=float(np.max(np.abs(delta)));maxdelta=max(maxdelta,peak)
 if peak!=0:changed.append(k)
 v=float(out['target_velocity_rad_s'].abs().max());a=float(out['target_acceleration_rad_s2'].abs().max());vmax=max(vmax,v);amax=max(amax,a);minimum_margin=min(minimum_margin,r['diagnostics']['minimum_joint_margin_rad'])
 h=r['diagnostics']['rr_preload_diagnostic'];maximum_correction_speed=max(maximum_correction_speed,np.linalg.norm(h['analytic_velocity_world_mps']));maximum_correction_accel=max(maximum_correction_accel,np.linalg.norm(h['analytic_acceleration_world_mps2']))
 if k in [419,420,421,433,434,435,436,504]:results.append({'control_index':k,'target_time_s':r['target_time_s'],'max_target_difference_rad':peak,'rr_diagnostic':h,'base_mode':r['state']['mode'],'next_wave_order_index':r['state']['next_wave_order_index'],'confirmed_touchdowns':r['state']['confirmed_touchdowns']})
failed=c.step(snapshot(504),[0,.005,0]);assert not failed['valid'][0] and failed['q_ref'] is None

def serialize(x):
 if isinstance(x,dict):return {k:serialize(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [serialize(v) for v in x]
 if isinstance(x,np.ndarray):return serialize(x.tolist())
 if isinstance(x,np.generic):return x.item()
 return x
report={'source_manifest_sha256':hashlib.sha256((SOURCE/'campaign_source_hashes.json').read_bytes()).hexdigest(),'parent_manifest_sha256':hashlib.sha256((PARENT/'campaign_source_hashes.json').read_bytes()).hexdigest(),'raw_trace_sha256':hashlib.sha256((HERE.parent/'reference_directional_results_002/run/left_strafe/trace.npz').read_bytes()).hexdigest(),'scope':'Actual recorded inputs through old rejection; target-only changed controller and exact residual core; no changed contact/force or physical continuation claim','controls':305,'first_changed_control_index':changed[0],'first_changed_target_time_s':float(TRACE['time_s'][changed[0],0]),'exact_old_targets_before_first_change':True,'other_five_legs_targets_unchanged':True,'original_measured_contact_anchors_and_touchdown_counts_unchanged':True,'original_liftoff_cadence_unchanged':True,'max_joint_target_difference_rad':maxdelta,'maximum_instantaneous_fixed_measurement_P_term_difference_nm':30*maxdelta,'max_discrete_reference_velocity_rad_s':vmax,'max_discrete_reference_acceleration_rad_s2':amax,'minimum_named_soft_joint_margin_rad':minimum_margin,'maximum_sampled_cartesian_correction_speed_mps':maximum_correction_speed,'maximum_sampled_cartesian_correction_acceleration_mps2':maximum_correction_accel,'zero_residual_emits_reference_exactly':True,'rejected_when_fed_original_subfive_support_row':failed['failure_reason'],'no_torque_prediction':True,'not_a_new_physical_admission':True,'selected_transition_rows':results}
p=HERE/'ACTUAL_PREFIX_REPLAY.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(serialize(report),indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='selected_transition_rows'},indent=2))
