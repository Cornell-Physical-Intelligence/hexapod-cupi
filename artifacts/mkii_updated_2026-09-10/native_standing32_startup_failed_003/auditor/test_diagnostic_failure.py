import importlib.util,unittest,ast
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('a6',HERE/'audit_remote.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
class DiagnosticFailureTests(unittest.TestCase):
 def test_early_getter_failure_remains_authentic_failure_without_complete_rows(self):
  c={'status':'failed','planned_phases':['standing'],'training_allowed':False,'physical_admission':False,'error':"RuntimeError('legacy getter unavailable')"}
  state={'status':'failed','explicit_steps_completed':0,'errors':["RuntimeError('legacy getter unavailable')"],'checks':{'solver_recipe_readback':False}}
  self.assertEqual(a.classify(c,{'status':'failed','exit_code':1},state,{'ExecMainStatus':'1'},{'attempted':False,'passed':False}),'authentic_terminal_failure')
  self.assertEqual(state['explicit_steps_completed'],0)
 def test_exact_source_owned_diagnostic_validator_and_readbacks_retained(self):
  t=(HERE/'audit_remote.py').read_text();self.assertIn("h.CONTRACT.validate_result(RUN/'standing',identity)",t);self.assertIn("'standing/legacy_friction_readback.json'",t);self.assertIn("attempt('session'",t)
  self.assertEqual([x[4]for x in a.PINS],[109,63,59,926]);self.assertEqual(a.INV,'6eac4972e2e64f9991955a1b7bb73afc')
  self.assertIn("'velocity_iterations':0",t);self.assertNotIn("'velocity_iterations':4",t)
if __name__=='__main__':unittest.main()
