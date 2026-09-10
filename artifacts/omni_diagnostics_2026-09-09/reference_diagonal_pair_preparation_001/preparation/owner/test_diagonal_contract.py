import ast,copy,importlib.util,json,sys,unittest
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;P=H.parent/'reference_pair_physics_adapter_001/source_pair_001/tools';S=H/'source_diagonal_pair_001/tools'
sys.path.insert(0,str(H.parent/'reference_load_transfer_001'));sys.path.insert(0,str(S))
from load_transfer import PairLoadTransfer,TransferConfig,partition_for_names
from score_transfer import score_transfer
from synthetic_fixture import Fixture

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
class ContractTests(unittest.TestCase):
 def test_exact_allowed_named_partitions(self):
  self.assertEqual(partition_for_names(('lf','rr')),((0,5),(1,2,3,4)))
  self.assertEqual(partition_for_names(('lr','rf')),((2,3),(0,1,4,5)))
  for p in [('lf','lf'),('rr','lf'),('lf','rm'),('lf',),('bogus','rr')]:
   with self.assertRaises(ValueError):partition_for_names(p)
 def test_required_retained_support_loss_rejects_before_target(self):
  for pair in [('lf','rr'),('lr','rf')]:
   f=Fixture();g=PairLoadTransfer(f.names,pair_leg_names=pair);row=f.snapshot();g.reset(row)
   bad=copy.deepcopy(row);bad['distal_contact'][0,g.supports[0]]=False;bad['contact_point_valid'][0,g.supports[0]]=False
   out=g.step(bad);self.assertFalse(out['valid'][0]);self.assertIsNone(out['q_ref']);self.assertIn('support',out['failure_reason'])
 def test_old_middle_target_and_configuration_parity(self):
  old=load('old_transfer_for_parity',P/'load_transfer.py');self.assertEqual(vars(TransferConfig()),vars(old.TransferConfig()))
  f=Fixture();g=PairLoadTransfer(f.names);o=old.PairLoadTransfer(f.names);row=f.snapshot();g.reset(row);o.reset(row)
  for _ in range(1100):
   row=f.snapshot();a=g.step(row);b=o.step(row)
   for k in ('q_ref','v_ref','a_ref','analytic_velocity_rad_s','analytic_acceleration_rad_s2'):np.testing.assert_array_equal(a[k],b[k])
   self.assertEqual(a['failure_reason'],b['failure_reason']);self.assertTrue(a['valid'][0]);f.advance(a)
 def test_actual_middle_scorer_result_exactly_preserved(self):
  p=H.parent/'reference_pair_results_001/run/pair';old=load('old_pair_scorer_for_parity',P/'score_transfer.py')
  with np.load(p/'trace.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
  with np.load(p/'physics_substeps.npz',allow_pickle=False) as z:sub={k:z[k] for k in z.files}
  refs=json.loads((p/'reference_states.json').read_text());new=score_transfer(d,refs,substeps=sub);original=old.score_transfer(d,refs,substeps=sub)
  self.assertEqual(new,original);self.assertTrue(new['proposed_criteria_met'])
 def test_existing_physics_steps_and_observers_unchanged(self):
  for f in ['pair_rollout.py','physics_substeps.py','physics_telemetry.py','reference_physics_env.py','canonical_stance_startup.py','reference_residual.py','reference_residual_env.py','wave_reference.py']:
   self.assertEqual((P/f).read_bytes(),(S/f).read_bytes())
  a=ast.parse((P/'run_pair_physics.py').read_text());b=ast.parse((S/'run_pair_physics.py').read_text())
  loops=lambda t:[n for n in ast.walk(t) if isinstance(n,ast.For) and isinstance(n.iter,ast.Call) and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='range']
  self.assertEqual([ast.dump(n) for n in loops(a)],[ast.dump(n) for n in loops(b)])
if __name__=='__main__':unittest.main()
