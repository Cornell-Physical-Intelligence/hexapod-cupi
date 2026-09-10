import ast,hashlib,importlib.util,json,sys,unittest
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

g=load('guard_linkage',HERE/'launch_guarded_remote.py')
class LinkageTests(unittest.TestCase):
 def test_native005_actual_selection_equals_guard500(self):
  native=load('bound_native_config',HERE/'inputs/direct_config.py')
  self.assertEqual(native.selection('train','extended','caps',None,None),g.PREVIOUS_SELECTION)
  self.assertEqual(native.selection('train','extended','caps',None,None)['updates'],500)
 def test_all_frozen_dependency_maps_and_source004_entry(self):
  ins=HERE/'inputs'
  for name,bound in [('adapter004',g.ADAPTER_SHA256),('host003',g.HOST_FREEZE_SHA256),('training_host005',g.TRAIN_HOST_SHA256),('native005',g.CONTRACT_SHA256)]:self.assertEqual(g.sha(ins/(name+'_FREEZE_SHA256.json')),bound)
  native=json.loads((ins/'native005_FREEZE_SHA256.json').read_text());self.assertEqual(g.sha(ins/'direct_config.py'),native['direct_config.py'])
  host=json.loads((ins/'host003_FREEZE_SHA256.json').read_text());self.assertEqual(g.sha(ins/'preview_host003.py'),host['launch_preview_spark.py']);self.assertEqual(host['launch_preview_spark.py'],g.HOST_SHA256)
  adapter=json.loads((ins/'adapter004_FREEZE_SHA256.json').read_text());self.assertEqual(g.sha(ins/'preview_contract004.py'),adapter['preview_contract.py'])
  manifest=json.loads((ins/'source004_manifest.json').read_text());self.assertEqual(len(manifest),599);self.assertEqual(g.sha(ins/'source004_manifest.json'),g.SOURCE_SHA256)
  self.assertEqual(g.sha(ins/'source004_entry.py'),manifest['tools/train_length_study.py'])
  contract=load('bound_preview004',ins/'preview_contract004.py');self.assertEqual(contract.HOST,g.TRAIN_HOST_SHA256);self.assertEqual(contract.SOURCE,g.SOURCE_SHA256);self.assertEqual(contract.NATIVE,g.CONTRACT_SHA256)
  self.assertEqual((len(contract.schedule()),contract.FRAMES,contract.FPS),(1900,950,25))
 def test_actual_host_command_only_recording_with_readonly_mounts(self):
  h=load('bound_preview_host003',HERE/'inputs/preview_host003.py')
  args=SimpleNamespace(output=Path('/output-new'),source=g.SOURCE,contract=g.CONTRACT,adapter=g.ADAPTER,pilot=g.PREVIOUS_ROOT,checkpoint=g.PREVIOUS_ROOT/'train/policy/final.pt',checkpoint_sha256='a'*64)
  cmd=h.command(args,'exact-test-owner','recording')
  mounts=[cmd[i+1] for i,v in enumerate(cmd) if v=='-v']
  self.assertEqual([x for x in mounts if x.endswith(':rw')],['/output-new:/output:rw'])
  for suffix in (':/source:ro',':/contract:ro',':/recording:ro',':/pilot:ro',':/admission:ro',':/checkpoint/evaluated.pt:ro'):self.assertEqual(sum(x.endswith(suffix) for x in mounts),1)
  self.assertIn('/recording/record_direct_preview.py',cmd);self.assertIn('--checkpoint-sha256',cmd)
  for phase in ('train','standing','final_stop'):
   with self.assertRaises(ValueError):h.command(args,'exact-test-owner',phase)
  self.assertEqual((h.SOURCE_MAP,h.CONTRACT_FREEZE,h.ADAPTER_FREEZE),(g.SOURCE_SHA256,g.CONTRACT_SHA256,g.ADAPTER_SHA256))
 def test_prior_invocation_has_actual_historical_receipt_not_terminal_claim(self):
  d=json.loads((HERE/'inputs/retry002_active_snapshot.json').read_text())
  self.assertIn('InvocationID='+g.PREVIOUS_INVOCATION,d['unit_state']);self.assertEqual(d['unit'],g.PREVIOUS_UNIT)
  self.assertEqual(d['campaign']['status'],'preparing');self.assertFalse(d['campaign'].get('bounded_campaign_complete',False))
 def test_guard_restoration_twice_checks_source_and_previous_owner(self):
  tree=ast.parse((HERE/'launch_guarded_remote.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
  for name in ('validate_preview_inputs','verify_previous_owner'):
   calls=[n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==name];self.assertEqual(len(calls),2)
  text=(HERE/'inputs/preview_host003.py').read_text();self.assertIn("if verify_inputs(args)!=identity",text);self.assertIn("report['post_exit_original_inputs_reverified']=True",text);self.assertIn("pre_shutdown_integrity.json",text)
if __name__=='__main__':unittest.main()

class StdlibTests(unittest.TestCase):
 def test_stdlib_only_import_no_gpu_or_runtime_start(self):
  import subprocess
  code="import importlib.util,sys; s=importlib.util.spec_from_file_location('frozen_candidate',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m); assert m.PREVIOUS_SELECTION['updates']==500; assert not any(k in sys.modules for k in ('torch','numpy','isaaclab'))"
  subprocess.run([sys.executable,'-B','-S','-c',code,str(HERE/'launch_guarded_remote.py')],check=True,capture_output=True,text=True,timeout=20)
