"""Independent actual-tail control-flow counterexamples; no native allocation."""
import copy,sys,textwrap,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=next(p for p in Path(__file__).resolve().parents if(p/'robot/active_model.json').is_file())
sys.path.insert(0,str(ROOT/'tmp/canonical_ppo_integration_003'))
from canonical_direct_ppo import native_entry_adapter as entry

class TailFailure(unittest.TestCase):
 def fixture(self,final_failure=False,callback_failure=False):
  events=[];state={'checks':{}};session=SimpleNamespace(count=8000)
  def close():events.append(('close',session.count))
  session.close=close
  def verify(*args):
   events.append(('verify',session.count))
   if final_failure and session.count==8384:raise ValueError('final solver read changed')
   return {'CPU_ONLY':True}
  def callback(*args):
   events.append(('callback',session.count));session.count=8384
   if callback_failure:raise ValueError('actual learner failure retained')
  env={'solver_record':{},'verify_solver':verify,'stage':None,'roots':None,'PhysxSchema':None,
   'save':lambda path,value:events.append(('save',session.count)),'out':Path('/CPU_ONLY'),
   'state':state,'learner_callback':callback,'session':session,'model':None,'errors':[],
   'a':None,'time':SimpleNamespace(monotonic=lambda:0.),'started':0.,'print':lambda *a,**kw:None}
  return env,events
 def test_after_close_solver_failure_cannot_create_completed_state(self):
  env,events=self.fixture(final_failure=True)
  with self.assertRaisesRegex(ValueError,'final solver'):exec(textwrap.dedent(entry.LEARNER_TAIL),env)
  self.assertEqual(events,[('verify',8000),('save',8000),('callback',8000),('close',8384),('verify',8384)])
  self.assertIsNone(env['session']);self.assertEqual(env['state']['explicit_steps_completed'],8384)
  self.assertNotIn('status',env['state']);self.assertNotIn('solver_recipe_readback',env['state']['checks'])
 def test_callback_failure_retains_live_session_for_unchanged_outer_finalizer(self):
  env,events=self.fixture(callback_failure=True)
  with self.assertRaisesRegex(ValueError,'learner failure'):exec(textwrap.dedent(entry.LEARNER_TAIL),env)
  self.assertEqual(events,[('verify',8000),('save',8000),('callback',8000)])
  self.assertIsNotNone(env['session']);self.assertEqual(env['session'].count,8384)
  self.assertNotIn('status',env['state']);self.assertNotIn('solver_recipe_readback',env['state']['checks'])

if __name__=='__main__':unittest.main()
