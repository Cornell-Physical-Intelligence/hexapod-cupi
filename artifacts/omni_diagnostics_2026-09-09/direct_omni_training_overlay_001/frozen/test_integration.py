import ast,copy,json,tempfile,unittest
from pathlib import Path
import yaml
from make_overlay import patch_entry,plan
from training_config import configure,options
from checkpoint_load import load_repair_checkpoint
HERE=Path(__file__).parent
BASE=HERE.parent/'baseline/inputs'
class Tests(unittest.TestCase):
 def test_two_plans_only_caps_diff(self):
  old=json.loads((BASE/'old_plan.json').read_text());old['omni']['overrides']['target_slew_rad_per_20ms']=.04
  a=plan(old,'curriculum');b=plan(old,'caps')
  self.assertEqual(a['training_num_envs'],128);self.assertEqual(a['training_iterations'],10)
  a['omni']['direct_recovery_training']=b['omni']['direct_recovery_training'];self.assertEqual(a,b)
  self.assertEqual(plan(old,'caps')['omni']['overrides'],old['omni']['overrides'])
 def test_main_physics_assignments_unchanged(self):
  old=(BASE/'old_entry.py').read_text();new=patch_entry(old)
  def physics(text):
   t=ast.parse(text);m=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='main')
   out=[]
   for n in ast.walk(m):
    if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id in ['cfg','env_module'] for x in ast.walk(n.targets[0])):out.append(ast.dump(n))
   return out
  self.assertEqual(physics(old),physics(new));self.assertIn('wrapped = CapsPairWrapper(wrapped)',new)
 def test_allocation_and_observation_guard(self):
  old=yaml.safe_load((HERE/'inputs/agent.yaml').read_text());new=configure(old,options('caps'),2);self.assertEqual(new['num_steps_per_env'],256);self.assertEqual(new['save_interval'],1);self.assertEqual(old['num_steps_per_env'],24)
  for n in [1,3,25,True]:
   with self.assertRaises(ValueError):configure(old,options('caps'),n)
  old['obs_groups']['actor'].append('caps_previous_policy')
  with self.assertRaises(ValueError):configure(old,options('caps'),10)
 def test_wrong_checkpoint_rejected_before_runner_load(self):
  class Runner:
   def load(self,*args,**kwargs):raise AssertionError('Wrong checkpoint touched runner')
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'wrong.pt';p.write_bytes(b'wrong')
   o={'checkpoint_sha256':'1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8','exploration_std':.1,'entropy_coef':0.,'optimizer':'reset','learning_rate':5e-5}
   with self.assertRaises(ValueError):load_repair_checkpoint(Runner(),p,o)
if __name__=='__main__':unittest.main()
