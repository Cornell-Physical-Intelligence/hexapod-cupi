from pathlib import Path
import hashlib,json,os,unittest
import numpy as np
from study import proposal,inputs
H=Path(__file__).resolve().parent
PARENT=Path(os.environ.get('PAIR_MOTION_OWNER',str(H.parent/'reference_pair_motion_001')))
class StudyTests(unittest.TestCase):
 def test_parent_oracle_and_inputs_are_unchanged(self):
  expected=json.loads((H/'PARENT_INPUTS_SHA256.json').read_text())['copied_files_sha256']
  for f,h in expected.items():self.assertEqual(hashlib.sha256((H/f).read_bytes()).hexdigest(),h)
 def test_two_baseline_speeds_exact_original_target_oracle(self):
  for speed in (.015,.02):
   new=H/'results'/f'forward_{round(speed*1000):03d}mmps_swing_2000ms.npz'
   with np.load(new,allow_pickle=False) as a,np.load(PARENT/f'target_only_{speed}.npz',allow_pickle=False) as b:
    self.assertEqual(set(a.files),set(b.files))
    for k in a.files:np.testing.assert_array_equal(a[k],b[k],err_msg=k)
 def test_retiming_recomputes_cycle_and_horizon_preserving_limits(self):
  base,_,_=inputs();study=json.loads((H/'PLAN.json').read_text());p=proposal(base,.03,1.5,study)
  self.assertAlmostEqual(p['cycle_s'],5.4);self.assertAlmostEqual(p['placement_horizon_s'],3.45)
  self.assertEqual(p['fixed_target_limits'],base['fixed_target_limits']);self.assertEqual(p['lift_m'],.007);self.assertEqual(p['handoff_hold_s'],.3)
  for v in (float('nan'),float('inf'),0.,-1.):
   with self.assertRaises(ValueError):proposal(base,v,1.5,study)
 def test_first_rejection_never_emitted_and_failure_stage_preserved(self):
  summary=json.loads((H/'SUMMARY.json').read_text());self.assertEqual(len(summary['cases']),22)
  for case in summary['cases']:
   r=case['result'];d=case['diagnostic'];p=case['parameters']
   with np.load(H/'results'/(p['candidate_id']+'.npz'),allow_pickle=False) as z:
    self.assertEqual(len(z['time_s']),r['accepted_controls']+1)
    if r['failure']:
     self.assertTrue(d['first_crossing']['not_emitted']);self.assertGreater(r['failure']['time_s'],z['time_s'][-1]);self.assertEqual(z['first_rejected_qva'].shape,(3,18));self.assertTrue(d['first_crossing']['categories'])
    else:
     self.assertEqual(r['accepted_controls'],2200);self.assertEqual(r['max_zero_residual_executable_lag_rad'],0);self.assertEqual(r['planned_liftoffs_after_stop_request'],0);self.assertIsNotNone(r['reference_quiet_time_s']);self.assertLessEqual(r['target_velocity_peak_rad_s'],1.75);self.assertLessEqual(r['target_acceleration_peak_rad_s2'],6.);self.assertGreaterEqual(r['minimum_executed_joint_margin_rad'],.02)
 def test_stopping_integral_is_known_planned_motion_only(self):
  s=json.loads((H/'SUMMARY.json').read_text())
  for c in s['cases']:
   r=c['result'];d=c['diagnostic'];self.assertFalse(d['actual_stop_or_torque_measured']);self.assertFalse(d['qualified_contact_simulator_used'])
   if r['completed_target_sequence']:
    v=c['parameters']['requested_forward_left_yaw'][0]
    self.assertAlmostEqual(d['planned_stop_command_integral_m'],v,delta=2e-5)
if __name__=='__main__':unittest.main()
