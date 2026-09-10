"""Actual reverse001 posthoc replay; preserves failed run and original consistency gate."""
from pathlib import Path
import sys,json,hashlib,numpy as np
HERE=Path(__file__).resolve().parent;RAW=HERE.parent;SOURCE=RAW.parent/'reference_directional_adapter_001/source_directional_001'
sys.path.insert(0,str(SOURCE/'tools'))
from directional_metrics import score_direction
from screen_metrics import standing_screen,standing_quiet_review
from physics_substeps import displacement_check
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def normal(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,dict):return {k:normal(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [normal(v) for v in x]
 return x
a=json.loads((RAW/'remote_audit.json').read_text())
for f,h in a['raw_payloads'].items():assert sha(RAW/f)==h,f
source_map=json.loads((SOURCE/'campaign_source_hashes.json').read_text())
for f,h in source_map.items():assert sha(SOURCE/f)==h,f
with np.load(RAW/'run/reverse/trace.npz',allow_pickle=False) as z:names=z['joint_names'].tolist();d={k:z[k] for k in z.files if k not in ('joint_names','legs')}
with np.load(RAW/'run/reverse/physics_substeps.npz',allow_pickle=False) as z:sub={k:z[k] for k in z.files}
with np.load(RAW/'run/reverse/partial_trace.npz',allow_pickle=False) as z:
 for k in z.files:np.testing.assert_array_equal(z[k],d[k])
refs=json.loads((RAW/'run/reverse/reference_states.json').read_text());g=score_direction(d,refs,case='reverse',joint_names=names)
try:json.dumps(g,allow_nan=False)
except TypeError as error:serialization_error=str(error)
else:raise AssertionError('Original gate unexpectedly serialized')
assert isinstance(g['independent_directional_motion']['yaw_required'],np.bool_)
assert not g['passed'] and g['independent_directional_motion']['old_world_displacement_integral_evidence']['displacement_integral_difference_m']>.005
assert len(d['time_s'])==2400 and len(sub['time_s'])==19201
assert np.array_equal(sub['control_index'],np.r_[-1,np.repeat(np.arange(2400),8)])
assert np.array_equal(sub['substep_index'],np.r_[0,np.tile(np.arange(1,9),2400)])
assert np.array_equal(sub['sim_step_counter']-sub['sim_step_counter'][0],np.arange(19201))
assert np.allclose(np.diff(sub['sdk_sim_timestamp_s']),.0025,atol=1e-7,rtol=0)
for raw,control in [('joint_position_rad','joint_position_rad'),('joint_velocity_rad_s','joint_velocity_rad_s'),('root_link_position_world_m','position_world_m'),('root_link_quaternion_world_xyzw','quaternion_world_xyzw'),('computed_torque_nm','computed_torque_nm'),('applied_torque_nm','applied_torque_nm')]:np.testing.assert_array_equal(sub[raw][8::8],d[control])
review400=displacement_check(sub,200,1400)
q=sub['joint_position_rad'][1600:11201,0].astype(np.float64);v=sub['joint_velocity_rad_s'][1600:11201,0].astype(np.float64)
qdelta=q[-1]-q[0];integral=.5*(v[:-1]+v[1:]).sum(0)*.0025;worst=int(np.abs(integral-qdelta).argmax())
with np.load(RAW/'run/standing/trace.npz',allow_pickle=False) as z:sn=z['joint_names'].tolist();sd={k:z[k] for k in z.files if k not in ('joint_names','legs')}
standing=standing_screen(sd);quiet=standing_quiet_review(sd,sn);assert standing['passed'] and quiet['passed']
report={'scope':'Posthoc replay of unchanged reverse001 scalar gates; run remains failed and reverse is not admitted',
 'source_files_verified':len(source_map),'source_manifest_sha256':a['source_manifest_sha256'],'raw_payloads_verified':len(a['raw_payloads']),
 'standing32_physical_and_quiet_recomputed_pass':True,'original_raw_state_status':json.loads((RAW/'run/reverse/state.json').read_text())['status'],
 'full_controls':2400,'full_substeps':19201,'partial_trace_and_fulltrace_same_arrays':True,'exact_indices_timestamps_and_control_endpoint_parity':True,
 'separate_finalization_failure':{'reproduced_error':serialization_error,'field':'gate.independent_directional_motion.yaw_required','type':str(type(g['independent_directional_motion']['yaw_required'])),
 'explanation':'Numpy bool blocks strict final-state JSON. Exception handler retains the same gate, so its second JSON save also fails; previous running2400 state remains. No physical state is rewritten.'},
 'original_gate_posthoc_replay':normal(g),'replay_serialization_conversion_scope':'Only converts numpy scalars for this separate review artifact, not original source or gate values',
 'independent400Hz_motion':review400,
 'postsettle400Hz_requested_peak_nm':float(np.abs(sub['computed_torque_nm'][1601:]).max()),
 'postsettle400Hz_applied_peak_nm':float(np.abs(sub['applied_torque_nm'][1601:]).max()),
 'moving400Hz_joint_rate_discrepancy':{'joint':names[worst],'actual_angle_delta_rad':float(qdelta[worst]),'reported_rate_integral_rad':float(integral[worst]),'difference_rad':float(integral[worst]-qdelta[worst])},
 'untested_cases':['left_strafe','left_turn','forward_right_arc'],'next_explicit_scope':'Fresh002 for these three unmeasured cases; strict JSON scalar/failure finalization fix only, same physics/gates/wave005. Do not repeat identical reverse or waive5mm.',
 'root_remote_cleanup_receipt_verified':True,'new_live_lock_check_performed':False,'stage2_complete':False,'reverse_admitted':False,'velocity_fidelity_qualified':False}
p=HERE/'report.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'passed':g['passed'],'original50Hz_gap_m':g['independent_directional_motion']['old_world_displacement_integral_evidence']['displacement_integral_difference_m'],
 'independent400Hz_link_gap_m':review400['link']['integrals']['trapezoid']['position_difference_norm_m'][0],'postsettle400Hz_requested_peak_nm':report['postsettle400Hz_requested_peak_nm'],'rate_discrepancy':report['moving400Hz_joint_rate_discrepancy']},indent=2))
