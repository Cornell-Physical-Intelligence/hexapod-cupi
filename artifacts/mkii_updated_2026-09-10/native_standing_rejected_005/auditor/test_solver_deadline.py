import copy,importlib.util,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('new_audit',HERE/'audit_remote.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
class NewChecks(unittest.TestCase):
 def test_missing_actual_invocation_refuses_before_any_external_read(self):
  with patch.object(a,'INV','PENDING'),patch.object(a,'call')as call:
   r=a.audit();self.assertFalse(r['audit_verified']);self.assertEqual(r['errors'][0]['check'],'invocation_binding');call.assert_not_called()
 def test_complete_solver_evidence_strict_all_phases_roots_and_flags(self):
  identity={'num_envs':1,'solver_recipe':{'position_iterations':32,'velocity_iterations':4}};state={'checks':{'solver_recipe_readback':True}}
  r={p:{'/Robot':{'position_iterations':32,'velocity_iterations':1 if p=='before' else 4,'authored':[p!='before']*2}}for p in ('before','after_authoring','after_reset','after_controlled_steps')}
  self.assertTrue(a.solver_check(r,identity,state)['passed'])
  for phase in r:
   bad=copy.deepcopy(r);del bad[phase]
   with self.assertRaises(ValueError):a.solver_check(bad,identity,state)
   bad=copy.deepcopy(r);bad[phase]['/Robot']['velocity_iterations']=16
   with self.assertRaises(ValueError):a.solver_check(bad,identity,state)
   bad=copy.deepcopy(r);bad[phase]['/Robot_001']=bad[phase]['/Robot']
   with self.assertRaises(ValueError):a.solver_check(bad,identity,state)
  with self.assertRaises(ValueError):a.solver_check(r,identity,{'checks':{}})
 def test_actual_frozen_deadline_metadata_and_wrong_budget_reject(self):
  path=HERE.parent/'canonical_native_standing_host_006/deadline_adapter.py';spec=importlib.util.spec_from_file_location('deadline',path);d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
  proof=d.inspect_supervisor(HERE.parent/'reference_physics_adapter_009/source_009/tools/launch_reference_physics_spark.py');meta={k:v for k,v in proof.items()if k not in ('original_function_source','adapted_function_source','_original_module_source')};j={'deadline_seconds':1200,'app_ready_deadline_seconds':90,'supervisor_runtime_adapter':meta}
  self.assertTrue(a.deadline_check(j,proof)['passed'])
  for patch in [{'deadline_seconds':600},{'app_ready_deadline_seconds':120},{'supervisor_runtime_adapter':{}}]:
   with self.assertRaises(ValueError):a.deadline_check({**j,**patch},proof)
  self.assertIn('--property=RuntimeMaxSec=1320',a.expected_launch_command())
if __name__=='__main__':unittest.main()
