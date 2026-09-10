"""Read-only replay using the exact external directional001 source; no Isaac/GPU."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def normal(v):
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,dict):return {k:normal(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [normal(x) for x in v]
 return v

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);args=parser.parse_args();source=args.source.resolve()
 expected=json.loads((ROOT/'preparation/source_overlay/campaign_source_hashes.json').read_text())
 assert sha(source/'campaign_source_hashes.json')=='373c9ea08406f6f3593279fa9fda24badf56ae86bd938e140dba7d734b5ee919'
 actual={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file() and p.name!='campaign_source_hashes.json'}
 assert actual==expected
 sys.path.insert(0,str(source/'tools'))
 from directional_metrics import score_direction
 from screen_metrics import standing_screen,standing_quiet_review
 from physics_substeps import displacement_check
 audit=json.loads((ROOT/'remote_audit.json').read_text())
 for name,h in audit['raw_payloads'].items():assert sha(ROOT/'raw'/name)==h
 def load(phase):
  with np.load(ROOT/'raw/run'/phase/'trace.npz',allow_pickle=False) as z:return z['joint_names'].tolist(),{k:z[k] for k in z.files if k not in ('joint_names','legs')}
 names,d=load('reverse');sn,sd=load('standing')
 assert standing_screen(sd)['passed'] and standing_quiet_review(sd,sn)['passed']
 refs=json.loads((ROOT/'raw/run/reverse/reference_states.json').read_text());gate=score_direction(d,refs,case='reverse',joint_names=names)
 root=json.loads((ROOT/'root_actual_review/report.json').read_text())
 assert normal(gate)==root['original_gate_posthoc_replay']
 assert isinstance(gate['independent_directional_motion']['yaw_required'],np.bool_)
 try:json.dumps(gate,allow_nan=False)
 except TypeError as error:serialization_error=str(error)
 else:raise AssertionError('Original JSON failure no longer reproduces')
 with np.load(ROOT/'raw/run/reverse/partial_trace.npz',allow_pickle=False) as z:
  for k in z.files:np.testing.assert_array_equal(z[k],d[k])
 with np.load(ROOT/'raw/run/reverse/physics_substeps.npz',allow_pickle=False) as z:sub={k:z[k] for k in z.files}
 assert len(d['time_s'])==2400 and len(sub['time_s'])==19201
 np.testing.assert_array_equal(sub['control_index'],np.r_[-1,np.repeat(np.arange(2400),8)])
 np.testing.assert_array_equal(sub['substep_index'],np.r_[0,np.tile(np.arange(1,9),2400)])
 np.testing.assert_array_equal(sub['sim_step_counter']-sub['sim_step_counter'][0],np.arange(19201))
 np.testing.assert_allclose(np.diff(sub['sdk_sim_timestamp_s']),.0025,atol=1e-7,rtol=0)
 for field,control in [('joint_position_rad','joint_position_rad'),('joint_velocity_rad_s','joint_velocity_rad_s'),('root_link_position_world_m','position_world_m'),('root_link_quaternion_world_xyzw','quaternion_world_xyzw'),('computed_torque_nm','computed_torque_nm'),('applied_torque_nm','applied_torque_nm')]:np.testing.assert_array_equal(sub[field][8::8],d[control])
 independent400=displacement_check(sub,200,1400);assert normal(independent400)==root['independent400Hz_motion']
 complete=[e for e in gate['measured_flight_events'] if e['confirmed_measured_touchdown'] and e['measured_reference_point_lift_m']>=.002]
 final=refs[-1]['result']['state'];quiet=gate['final_quiet_stop_window']
 assert len(complete)==11 and gate['completed_measured_leg_indices']==list(range(6))
 assert not gate['passed'] and quiet['pass'] and quiet['window_duration_s']>=10
 print(json.dumps({'scope':'Independent read-only raw replay; original failed state preserved','standing32_quiet_and_physics_pass':True,
  'source_payloads_verified':len(expected),'raw_payloads_verified':len(audit['raw_payloads']),
  'full_controls':len(d['time_s']),'full_substeps':len(sub['time_s']),
  'original_gate_exactly_matches_root_replay':True,'original_gate_passed':False,
  'original50Hz_gap_m':gate['independent_directional_motion']['old_world_displacement_integral_evidence']['displacement_integral_difference_m'],
  'independent400Hz_link_trapezoid_gap_m':independent400['link']['integrals']['trapezoid']['position_difference_norm_m'][0],
  'independent400Hz_is_diagnostic_not_substitute_gate':True,'measured_qualified_landings':len(complete),'all_six_legs':True,
  'quiet_duration_s':quiet['window_duration_s'],'reference_quiet_time_s':final['reference_quiet_time_s'],
  'serialization_failure_reproduced':serialization_error,'original_raw_state_status':json.loads((ROOT/'raw/run/reverse/state.json').read_text())['status'],
  'later_cases_unrun':['left_strafe','left_turn','forward_right_arc'],'velocity_fidelity_qualified':False,'Stage2_complete':False},indent=2))
if __name__=='__main__':main()
