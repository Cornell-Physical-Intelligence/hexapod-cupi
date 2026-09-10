"""Narrow source004 and completed CAPS500 lineage checks; no actual future checkpoint."""
import ast,hashlib,json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import preview_contract as c
import test_preview as fixture_module
HERE=Path(__file__).resolve().parent

class ExtendedTests(unittest.TestCase):
 def setUp(self):
  p=patch.object(c,'HOST','a'*64);p.start();self.addCleanup(p.stop)
 def fixture(self,root):return fixture_module.ReceiptTests().fixture(root)
 def test_only_completed_caps500_with_all_six_phases(self):
  with tempfile.TemporaryDirectory() as t:
   args,native,original=self.fixture(Path(t));c.verify_pilot(args,native)
   variants=[{'allocation':'smoke'},{'allocation':'pilot'},{'branch':'curriculum'},{'branch':'quiet_priority'},{'status':'failed'},{'PPO_updates_completed':499},{'bounded_campaign_complete':False},{'last_completed_phase':'train'}]
   for changes in variants:
    candidate={**original,**changes};c.save(args.pilot/'campaign.json',candidate)
    with self.subTest(changes=changes),self.assertRaises(ValueError):c.verify_pilot(args,native)
   c.save(args.pilot/'campaign.json',original);del original['accepted_phases']['final_stop'];c.save(args.pilot/'campaign.json',original)
   with self.assertRaisesRegex(ValueError,'Incomplete'):c.verify_pilot(args,native)
 def test_wrong_source_schema_host_or_selected_checkpoint_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   args,native,original=self.fixture(Path(t))
   for key,value in [('source_manifest_sha256','0'*64),('schema','direct315_quiet_priority_native_v3')]:
    candidate={**original,'identity':{**original['identity'],key:value}};c.save(args.pilot/'campaign.json',candidate)
    with self.subTest(key=key),self.assertRaises(ValueError):c.verify_pilot(args,native)
   candidate={**original,'host_freeze_sha256':'b'*64};c.save(args.pilot/'campaign.json',candidate)
   with self.assertRaises(ValueError):c.verify_pilot(args,native)
   c.save(args.pilot/'campaign.json',original);args.checkpoint_sha256='0'*64
   with self.assertRaisesRegex(ValueError,'checkpoint'):c.verify_pilot(args,native)
 def test_pending_training_host_rejects_without_reading_campaign(self):
  with patch.object(c,'HOST','PENDING_HOST005_FREEZE'),patch.object(c,'read') as read:
   with self.assertRaisesRegex(ValueError,'pending'):c.verify_pilot(types.SimpleNamespace(),None)
   read.assert_not_called()
 def test_wrong_native_manifest_rejected_before_loading_contract(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);(root/'FREEZE_SHA256.json').write_text('{}');a=types.SimpleNamespace(seed=7057,source_root=root,native_contract=root)
   with patch.object(c,'verify_source',return_value={}),patch.object(c,'load_native') as load:
    with self.assertRaisesRegex(ValueError,'manifest'):c.verify_inputs(a,check_self=False)
    load.assert_not_called()
 def test_actual_source599_entry_reconstructed_exactly(self):
  mapping=c.read(HERE/'inputs/native_source_manifest.json')
  self.assertEqual(len(mapping),599);self.assertEqual(c.sha(HERE/'inputs/native_source_manifest.json'),c.SOURCE)
  scope={}
  for filename,names in [('cold_build_source.py',('new_entry',)),('native005_build_source.py',('one','patch_entry'))]:
   tree=ast.parse((HERE/'inputs'/filename).read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
   exec(compile(ast.Module(body=nodes,type_ignores=[]),filename,'exec'),scope)
  text=scope['patch_entry'](scope['new_entry']((HERE/'inputs/cold_old_entry.py').read_text()))
  self.assertEqual(text,(HERE/'inputs/native_entry.py').read_text());self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),mapping['tools/train_length_study.py'])
  self.assertEqual(c.sha(HERE/'inputs/native_omni_flat_env.py'),mapping['tools/omni_flat_env.py'])
 def test_actual_evaluation_bypasses_training_caps_and_curriculum_wrappers(self):
  tree=ast.parse((HERE/'inputs/native_entry.py').read_text())
  # Every assignment that changes the environment/runner via training wrappers remains under the exact mode guard.
  found=[]
  for node in ast.walk(tree):
   if isinstance(node,ast.If) and ast.dump(node.test)==ast.dump(ast.parse('args.mode == "train"',mode='eval').body):
    found.append(ast.unparse(node))
  joined='\n'.join(found)
  self.assertIn('audited_environment(make_environment(OmniFlatEnv))',joined)
  self.assertIn('CapsPairWrapper(wrapped)',joined)
  self.assertIn('configure(run_cfg, direct_selection)',joined)
  self.assertIn('runner.load(str(args.checkpoint))',(HERE/'inputs/native_entry.py').read_text())

if __name__=='__main__':unittest.main()
