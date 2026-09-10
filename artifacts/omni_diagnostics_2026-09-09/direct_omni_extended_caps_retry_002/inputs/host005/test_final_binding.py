"""Exact final source/native binding checks, without remote or GPU access."""
from pathlib import Path
import hashlib,json,unittest
import launch_train_spark as h
HERE=Path(__file__).resolve().parent
NATIVE=HERE.parents[1]/'tmp/direct_omni_recovery_001/native_005'
class FinalBindingTests(unittest.TestCase):
 def test_actual_root_build_and_launcher_identity(self):
  raw=(HERE/'source_build/remote_build_and_audit_001.log').read_text();first,rest=raw.split('\n',1);build=json.loads(first);audit=json.loads(rest);pins=json.loads(build['build_stdout']);final=json.loads((HERE/'FINAL_BINDINGS.json').read_text())
  self.assertEqual(build['build_returncode'],0);self.assertTrue(audit['verified']);self.assertEqual(pins,audit['pins']);self.assertEqual(pins['source_manifest_sha256'],h.SOURCE_MAP);self.assertEqual(pins['files'],599);self.assertEqual(audit['parent_files'],589);self.assertEqual(len(audit['changed']),13);self.assertEqual(audit['removed'],[])
  for key,value in pins.items():self.assertEqual(final[key],value)
  self.assertEqual(final['host_launcher_sha256'],h.sha(HERE/'launch_train_spark.py'));self.assertEqual(final['deadline_adapter_sha256'],h.sha(HERE/'deadline_adapter.py'));self.assertEqual(final['source_build_log_sha256'],h.sha(HERE/'source_build/remote_build_and_audit_001.log'))
 def test_native_exact28_and_original_supervisor_remain_bound(self):
  self.assertEqual(h.CONTRACT_FREEZE,'0bdd2abf3811e68ac59e4fbc325eb5d2bb96afd2183eb84c4d37faeb86ff9e9f');self.assertEqual(h.sha(NATIVE/'FREEZE_SHA256.json'),h.CONTRACT_FREEZE)
  expected=json.loads((NATIVE/'FREEZE_SHA256.json').read_text());self.assertEqual(len(expected),28);self.assertEqual({p.relative_to(NATIVE).as_posix():h.sha(p) for p in NATIVE.rglob('*') if p.is_file() and p!=NATIVE/'FREEZE_SHA256.json'},expected)
  self.assertEqual(h.SUPERVISOR_CODE,h.load_deadline_adapter().SUPERVISOR_SHA256);self.assertNotIn('PENDING',h.SOURCE_MAP);self.assertEqual(len(h.SOURCE_MAP),64)
if __name__=='__main__':unittest.main()
