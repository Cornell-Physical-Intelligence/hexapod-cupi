import ast,copy,hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from build_source import new_plan,new_entry
from cold_contract import runtime_arguments,validate_result
HERE=Path(__file__).parent
OLD=HERE/'inputs'
class Tests(unittest.TestCase):
 def test_only_slew_plan_delta(self):
  old=json.loads((OLD/'old_plan.json').read_text());got=new_plan(old)
  self.assertEqual(old['omni']['overrides']['target_slew_rad_per_20ms'],.03)
  self.assertEqual(got['omni']['overrides']['target_slew_rad_per_20ms'],.04)
  got['omni']['overrides']['target_slew_rad_per_20ms']=.03;self.assertEqual(got,old)
 def test_only_readonly_import_ast_delta(self):
  old=(OLD/'old_entry.py').read_text();new=new_entry(old)
  a=ast.parse(old);b=ast.parse(new)
  for i,(x,y) in enumerate(zip(a.body,b.body)):
   if isinstance(x,ast.ImportFrom) and x.module=='repair_length_study_inertias':
    self.assertEqual(y.module,'candidate_asset_audit');self.assertEqual(y.names[0].asname,'repair_and_verify');b.body[i]=x
  self.assertEqual(ast.dump(a),ast.dump(b))
 def test_exact_phase_commands_no_train(self):
  for p in ['standing','baseline']:
   args=runtime_arguments(p);self.assertEqual(args[args.index('--mode')+1],{'standing':'validate','baseline':'evaluate'}[p]);self.assertNotIn('--iterations',args)
  with self.assertRaises(ValueError):runtime_arguments('train')
 def test_standing_binding_and_rejection(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);i={'plan_sha256':'p'};s={'status':'completed','variant':'f050_t060','urdf_sha256':'e916b1ebbb2720c90f23974bbffa5da120b32d2e5409fe419fb1492b8556746c','plan_sha256':'p','stance_index':0}
   (p/'state.json').write_text(json.dumps(s));(p/'admission.json').write_text(json.dumps(dict(s,gate={'passed':True})))
   self.assertTrue(validate_result(p,'standing',i)['passed'])
   (p/'admission.json').write_text(json.dumps(dict(s,gate={'passed':False})))
   with self.assertRaises(ValueError):validate_result(p,'standing',i)
 def test_actual_report_and_nan_applied_rejection(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);s={'status':'completed','variant':'f050_t060','urdf_sha256':'e916b1ebbb2720c90f23974bbffa5da120b32d2e5409fe419fb1492b8556746c','plan_sha256':'p','stance_index':0}
   (p/'state.json').write_text(json.dumps(s));report=json.loads((OLD/'old_diagnostics.json').read_text());(p/'diagnostic_trace.npz').write_bytes(b'fixture')
   identity={'plan_sha256':'p','overrides':copy.deepcopy(report['overrides']),'diagnostic_options':report['options']};identity['overrides']['target_slew_rad_per_20ms']=.04
   (p/'diagnostics.json').write_text(json.dumps(report))
   with self.assertRaises(ValueError):validate_result(p,'baseline',identity)
   report['overrides']['target_slew_rad_per_20ms']=.04;(p/'diagnostics.json').write_text(json.dumps(report));self.assertTrue(validate_result(p,'baseline',identity)['complete'])
   report['scenarios'][0]['windows']['all']['applied_torque_abs_max_nm']=float('nan');(p/'diagnostics.json').write_text(json.dumps(report))
   with self.assertRaises(ValueError):validate_result(p,'baseline',identity)
if __name__=='__main__':unittest.main()
