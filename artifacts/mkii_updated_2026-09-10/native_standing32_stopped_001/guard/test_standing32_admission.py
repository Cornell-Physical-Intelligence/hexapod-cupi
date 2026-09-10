import ast,copy,importlib.util,json,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('standing32_guard_admission',HERE/'launch_guarded_remote.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
class Standing32Admission(unittest.TestCase):
 def host(self):
  identity=json.loads((HERE/'bound_inputs_preflight.json').read_text())['identity']
  return SimpleNamespace(SOURCE_FREEZE=g.SOURCE_SHA256,SUPERVISOR_MAP=g.SUPERVISOR_SHA256,require_fresh_output=Mock(),verify_inputs=Mock(return_value=identity))
 def test_explicit_32_and_actual_standing_one_host_arguments(self):
  h=self.host()
  with patch.object(g,'verify_manifest'),patch.object(g,'sha',return_value=g.HOST_SHA256):r=g.validate_standing_inputs(h)
  args=h.verify_inputs.call_args.args[0];self.assertEqual(args.num_envs,32);self.assertEqual(args.standing_one,g.STANDING_ONE);self.assertEqual(args.admission,g.ADMISSION)
  self.assertEqual(r['standing_one_state_sha256'],g.STANDING_ONE_STATE_SHA256)
  self.assertEqual(r['standing_one_inventory_sha256'],g.STANDING_ONE_INVENTORY_SHA256)
 def test_wrong_1_source_scope_or_admission_hash_rejected(self):
  for key,value in [('num_envs',1),('standing_one_state_sha256','f'*64),('standing_one_inventory_sha256','f'*64),('training_allowed',True)]:
   h=self.host();h.verify_inputs.return_value[key]=value
   with patch.object(g,'verify_manifest'),patch.object(g,'sha',return_value=g.HOST_SHA256),self.assertRaises(RuntimeError):g.validate_standing_inputs(h)
 def test_full_prior_map_binds_every_external_large_payload(self):
  inventory=json.loads((HERE/'inputs/PRIOR_RAW_INVENTORY.json').read_text());audit=json.loads((HERE/'inputs/previous_terminal_audit.json').read_text())
  self.assertEqual(inventory,audit['raw_inventory']);self.assertEqual(len(inventory),38)
  expected={k.replace('run/','native_standing_002/',1) if k.startswith('run/') else k.replace('forecast_pause/','forecast_pause_007/',1):v['sha256'] for k,v in inventory.items()}
  self.assertEqual(g.PRIOR_PINS,expected)
  self.assertEqual(inventory['run/standing/contacts.jsonl']['size_bytes'],169991706)
  self.assertFalse((HERE/'inputs/previous_owner/run/standing/contacts.jsonl').exists())
 def test_host_actual_readonly_standing_one_mount_seam_is_frozen(self):
  mapping=json.loads((HERE/'inputs/HOST_FREEZE_SHA256.json').read_text());self.assertEqual(mapping['launch_standing_spark.py'],g.HOST_SHA256)
  tree=ast.parse((HERE/'launch_guarded_remote.py').read_text());verify=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify_previous_owner')
  text=ast.unparse(verify)
  for value in ["'authentic_completed_standing'","'standing_pass'","'explicit_steps_completed'","'captured_steps'","'reset_count'","'post_exit_all_standing_payloads_inventoried'"]:self.assertIn(value,text)
if __name__=='__main__':unittest.main()
