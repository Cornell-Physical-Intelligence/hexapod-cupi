import ast,sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'reference_physics_adapter_009/source_009/tools'));sys.path.insert(0,str(HERE))
from pair_rollout import emit_control,check_measured_pair_state

class PairRolloutTests(unittest.TestCase):
 def build(self,*,terminal=False,nonfinite=False,endpoint_failure=False):
  captured=[];dest=[];calls=[]
  term=torch.tensor([terminal]);trunc=torch.tensor([False]);row={'terminated':np.array([terminal]),'truncated':np.array([False]),'measured':np.array([7.])}
  def step(action):
   calls.append('step');captured.append(row)
   return {'policy':torch.tensor([float('nan') if nonfinite else 0.])},torch.zeros(1),term,trunc,{}
  def end(value):
   self.assertIs(value,row);self.assertIs(dest[-1],row)
   if endpoint_failure:raise RuntimeError('endpoint fault')
  recorder=SimpleNamespace(begin_control=lambda i:calls.append(('begin',i)),end_control=end)
  def targets(q,v,a,valid):
   np.testing.assert_array_equal(q.numpy(),np.ones((1,18)));self.assertTrue(valid.all());calls.append('targets')
  env=SimpleNamespace(device='cpu',set_reference_targets=targets,set_evaluation_targets=lambda v:self.assertFalse(v.any()),step=step)
  ref={'q_ref':np.ones((1,18)),'v_ref':np.zeros((1,18)),'a_ref':np.zeros((1,18)),'valid':np.ones(1,bool)}
  return env,ref,captured,recorder,dest,calls,row
 def test_one_original_step_and_real_terminal_flags_preserved(self):
  e,r,c,o,d,calls,row=self.build(terminal=True)
  got,term,trunc=emit_control(e,r,c,o,0,d,torch.zeros(1,18));self.assertIs(got,row);self.assertTrue(term.item());self.assertEqual(calls.count('step'),1);self.assertEqual(len(d),1)
 def test_failing_finite_check_keeps_pre_reset_row(self):
  e,r,c,o,d,calls,row=self.build(nonfinite=True)
  with self.assertRaises(ValueError):emit_control(e,r,c,o,0,d,torch.zeros(1,18))
  self.assertIs(d[-1],row)
 def test_failing_observer_endpoint_keeps_pre_reset_row(self):
  e,r,c,o,d,calls,row=self.build(endpoint_failure=True)
  with self.assertRaises(RuntimeError):emit_control(e,r,c,o,0,d,torch.zeros(1,18))
  self.assertIs(d[-1],row)
 def test_no_direct_pose_or_extra_physics_steps(self):
  text=ast.unparse(ast.parse((HERE/'run_pair_physics.py').read_text()))
  self.assertNotIn('write_root',text);self.assertNotIn('write_joint',text);self.assertNotIn('sim.step',text)
  self.assertNotIn('startup.next',text);self.assertIn('startup.sample(step + 1)',text)
  self.assertIn('snapshot = startup_rows[-1]',text);self.assertIn('check_substep_batch',text)
class FinalMeasurementTests(unittest.TestCase):
 def test_final_row_checks_existing_measurement_bounds_and_exact_target(self):
  seen=[]
  g=SimpleNamespace(base=SimpleNamespace(_read=lambda r:r),_measure=lambda r,m:seen.append(r),diagnostics={'measured_support_margin_m':.06})
  row={'time_s':np.array([26.]),'reference_to_executable_lag_rad':np.zeros((1,18)),'position_target_cast_error_rad':np.zeros((1,18))}
  self.assertEqual(check_measured_pair_state(g,row)['time_s'],26.);self.assertIs(seen[-1],row)
  row['reference_to_executable_lag_rad'][0,0]=.001
  with self.assertRaises(ValueError):check_measured_pair_state(g,row)

if __name__=='__main__':unittest.main()
