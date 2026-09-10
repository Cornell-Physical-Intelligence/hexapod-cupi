"""Compare all declared matched-origin outcomes without selecting favorable cases."""
import json,math
from origin_contract import CASES

def compare_cases(output,states):
 if list(states)!=list(CASES):raise ValueError('All five cases in fixed order are required')
 baseline=states['origin_a']['initial_readback']['actual'];names=None;rows=[]
 for case,xy in CASES.items():
  s=states[case];actual=s['initial_readback']['actual']
  for key in ('q','v','target','root_com_velocity'):
   if actual[key]!=baseline[key]:raise ValueError('Initial '+key+' differs across matched cases')
  if actual['root_pose'][0][2:]!=baseline['root_pose'][0][2:]:raise ValueError('Initial root Z/quaternion differs')
  if actual['root_pose'][0][:2]!=list(xy):raise ValueError('Initial translation differs from declaration')
  d=json.loads((output/case/'rate_discrepancy.json').read_text())
  if names is None:names=d['joint_names_runtime']
  if names!=d['joint_names_runtime']:raise ValueError('Runtime name order differs across origin cases')
  rows.append({'case':case,'translation_xy_m':list(xy),'unchanged_physical_gate':s['unchanged_physical_gate'],
   'unchanged_quiet_gate':s['unchanged_quiet_gate'],'all_existing_bounds_met':s['all_existing_bounds_met'],
   'initial_readback':s['initial_readback'],'ground_readback':s['ground_readback'],'rate_discrepancy':d})
 first=rows[0]['rate_discrepancy'];repeat=rows[-1]['rate_discrepancy']
 repeats=[{'joint':a['joint'],'reported_integral_bias_repeat_minus_first_rad':b['reported_integral_minus_delta_rad']['trapezoid']-a['reported_integral_minus_delta_rad']['trapezoid'],
  'actual_angle_delta_repeat_minus_first_rad':b['angle_delta_rad']-a['angle_delta_rad']} for a,b in zip(first['joints'],repeat['joints'])]
 return {'scope':'Five matched cold-state translated-world measurements; no production/PPO/velocity admission',
  'all_declared_cases_recorded':True,'matched_initial_nonXY_generalized_state_and_target_inputs_exact':True,'derived19body_poses_are_retained_for_quantization_review_not_claimed_bit_exact':True,'joint_names_runtime':names,'cases':rows,
  'origin_repeat_difference':repeats,'native_cause_established':False,'new_rate_pass_threshold':None,
  'interpretation':'Use all paired effects and origin-repeat variation. Translation sensitivity is evidence only; no native mechanism follows automatically.'}
