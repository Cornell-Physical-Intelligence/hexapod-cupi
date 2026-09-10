"""Bind the actual frozen native/host deadline behavior without a simulator."""
from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,importlib.util,json,sys,unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
g=module(HERE/'launch_guarded_remote.py','extended_contract_guard')
class ExtendedContractTests(unittest.TestCase):
 def test_actual_native500_caps_and_host_phase_deadlines(self):
  native=ROOT/'tmp/direct_omni_recovery_001/native_005'
  cfg=module(native/'direct_config.py','extended_cfg_guard')
  expected=cfg.selection('train','extended','caps',None,None)
  self.assertEqual(expected['schema'],'direct315_extended_native_v4');self.assertEqual((expected['replicas'],expected['controls_per_update'],expected['updates']),(1024,24,500));self.assertEqual(expected['caps']['quiet_temporal_weight'],.1)
  h=module(ROOT/'tmp/direct_omni_train_host_004/launch_train_spark.py','guard_actual_host004')
  self.assertEqual(h.verify_own_bundle(),g.HOST_FREEZE_SHA256)
  self.assertEqual(h.SOURCE_MAP,g.SOURCE_SHA256);self.assertEqual(h.CONTRACT_FREEZE,g.CONTRACT_SHA256)
  args=SimpleNamespace(allocation='extended',branch='caps',smoke=g.SMOKE)
  phases=h.selected_phases(args);proof=h.load_deadline_adapter().inspect_supervisor(ROOT/'tmp/reference_physics_adapter_009/source_009/tools/launch_reference_physics_spark.py')
  self.assertEqual(proof['phase_deadline_seconds'],{p:1800 if p=='train' else 600 for p in phases});self.assertEqual(proof['app_ready_deadline_seconds'],90);self.assertTrue(proof['cleanup_AST_unchanged']);self.assertEqual(sum(proof['phase_deadline_seconds'].values()),4800)
 def test_outer_and_fallback_allow_declared_phases_and_cleanup(self):
  tree=ast.parse((HERE/'launch_guarded_remote.py').read_text());strings={n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
  self.assertIn('--property=RuntimeMaxSec=5400',strings);self.assertIn('--property=TimeoutStopSec=180',strings);self.assertIn('--on-active=95m',strings);self.assertGreater(95*60,5400+180);self.assertGreaterEqual(5400,1800+5*600)
 def test_runtime_pending_prior_never_claims_ready(self):
  if all(len(v)==64 for v in g.PRIOR_PINS.values()) and len(g.PREVIOUS_INVOCATION)==32:return
  with patch.object(g,'PAUSE',Path('/uncreated-test-pause062')),patch.object(g,'OUTPUT',Path('/uncreated-test-extended500')),patch.object(g.subprocess,'run') as run,patch.object(g.subprocess,'check_output') as read:
   with self.assertRaisesRegex(RuntimeError,'pending'):g.main()
   run.assert_not_called();read.assert_not_called()
 def test_unknown_or_wrong_completed_update_allocation_rejects(self):
  # Actual typed native selector keeps500 bounded; no iterations CLI escape.
  cfg=module(ROOT/'tmp/direct_omni_recovery_001/native_005/direct_config.py','bounded_cfg_guard')
  for iterations in (50,501,True):
   with self.assertRaises(ValueError):cfg.selection('train','extended','caps',None,iterations)
  self.assertFalse(cfg.extended_protocol()['automatic_continuation'])
if __name__=='__main__':unittest.main()
