"""Exercise actual entrypoint finalization AST with failing local resources.

No application, GPU, filesystem mutation outside a temporary test directory,
checkpoint, or simulator is invoked. No source is rewritten.
"""
import ast,copy,time,unittest
from pathlib import Path
from types import SimpleNamespace
OWNER=Path(__file__).resolve().parents[1]/'reference_residual_ppo_source_001'

def block():
 main=next(n for n in ast.parse((OWNER/'run_residual_ppo.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
 final=next(n.finalbody for n in main.body if isinstance(n,ast.Try) and n.finalbody)
 return compile(ast.Module(body=final,type_ignores=[]),str(OWNER/'run_residual_ppo.py')+'[actual finally]','exec')

class FinalizationTests(unittest.TestCase):
 def run_case(self,failed=()):
  calls=[];saved=[];state={'status':'completed'};identity={'bound':True}
  def perform(name,result=None):
   calls.append(name)
   if name in failed:raise RuntimeError('injected '+name)
   return result
  env=SimpleNamespace(close=lambda:perform('env_close'));app=SimpleNamespace(close=lambda:perform('app_close'))
  runner=SimpleNamespace(logger=SimpleNamespace(writer=SimpleNamespace(flush=lambda:perform('logger_flush'),close=lambda:perform('logger_close'))))
  namespace=dict(state=state,identity=identity,args=SimpleNamespace(output=Path('/synthetic/output')),session=SimpleNamespace(export=lambda _:perform('export',{'complete':True})),runner=runner,env=env,app=app,time=time,
   verify_inputs=lambda _:perform('verify_inputs',identity),save=lambda path,value:(calls.append('state_save'),saved.append(copy.deepcopy(value))))
  exec(block(),namespace);return state,calls,saved
 def test_normal_export_verify_and_both_closes(self):
  state,calls,saved=self.run_case();self.assertEqual(state['status'],'completed');self.assertTrue(state['source_inputs_unchanged']);self.assertIn('env_close',calls);self.assertIn('app_close',calls);self.assertTrue(saved)
 def test_logger_failure_preserves_failure_and_continues_integrity_and_close(self):
  for failure in ('logger_flush','logger_close'):
   with self.subTest(failure=failure):
    state,calls,saved=self.run_case((failure,));self.assertEqual(state['status'],'failed');self.assertIn('logger_finalize_error',saved[-1]);self.assertIn('verify_inputs',calls);self.assertIn('env_close',calls);self.assertIn('app_close',calls)
 def test_export_failure_does_not_skip_final_receipt_or_cleanup(self):
  state,calls,saved=self.run_case(('export',));self.assertEqual(state['status'],'failed');self.assertIn('export_error',saved[-1]);self.assertIn('verify_inputs',calls);self.assertIn('app_close',calls)
 def test_env_and_application_close_failures_are_recorded_and_app_attempted(self):
  state,calls,saved=self.run_case(('env_close','app_close'));self.assertEqual(state['status'],'failed');self.assertIn('app_close',calls);self.assertIn('environment_close_error',saved[-1]);self.assertIn('application_close_error',saved[-1])
 def test_integrity_failure_is_not_completed(self):
  state,calls,saved=self.run_case(('verify_inputs',));self.assertEqual(state['status'],'failed');self.assertFalse(saved[-1]['source_inputs_unchanged']);self.assertIn('app_close',calls)
if __name__=='__main__':unittest.main()
