"""Exact failed owner remains separate from successful smoke admission."""
import contextlib,importlib.util,json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('retryguard',H/'launch_guarded_remote.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)

class FailedPredecessor(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name)
  self.output=self.base/'direct_omni_train_extended_caps_001';pins={}
  for n in g.FAILED_PINS:
   p=self.base/n;p.parent.mkdir(parents=True,exist_ok=True);data=json.loads((H/'failed_owner'/n).read_text())
   if n.endswith('/pause.json'):data.update(output=str(self.output),source=str(g.SOURCE))
   if n.endswith('/launch.json'):data['command'][data['command'].index('--output')+1]=str(self.output)
   p.write_text(json.dumps(data,indent=2)+'\n');pins[n]=g.sha(p)
  self.stack=contextlib.ExitStack();self.addCleanup(self.stack.close)
  for key,value in [('BASE',self.base),('FAILED_OUTPUT',self.output),('FAILED_PINS',pins)]:self.stack.enter_context(patch.object(g,key,value))
  self.owner={'ActiveState':'failed','SubState':'failed','InvocationID':g.FAILED_INVOCATION,'MainPID':'0','ExecMainStatus':'1','Result':'exit-code'}
 def text(self):return '\n'.join(k+'='+v for k,v in self.owner.items())
 def test_exact_failed_no_output_no_container_calls_passes(self):
  with patch.object(g,'call',return_value=self.text()),patch.object(g.subprocess,'run') as proc:g.verify_failed_previous_owner();proc.assert_not_called()
 def test_unknown_active_changed_status_or_invocation_reject(self):
  original=self.owner.copy()
  for key,value in [('InvocationID',''),('InvocationID','1'*32),('ActiveState','inactive'),('ActiveState','active'),('SubState','dead'),('MainPID','123'),('ExecMainStatus','0'),('Result','success')]:
   self.owner={**original,key:value}
   with patch.object(g,'call',return_value=self.text()),self.assertRaisesRegex(RuntimeError,'owner/status'):g.verify_failed_previous_owner()
  with patch.object(g,'call',side_effect=subprocess.CalledProcessError(1,['systemctl'])),self.assertRaises(subprocess.CalledProcessError):g.verify_failed_previous_owner()
 def test_any_new_output_file_directory_or_dangling_link_rejects(self):
  with patch.object(g,'call',return_value=self.text()):
   self.output.mkdir()
   with self.assertRaisesRegex(RuntimeError,'output appeared'):g.verify_failed_previous_owner()
   self.output.rmdir();self.output.write_text('new evidence')
   with self.assertRaisesRegex(RuntimeError,'output appeared'):g.verify_failed_previous_owner()
   self.output.unlink();self.output.symlink_to(self.base/'missing')
   with self.assertRaisesRegex(RuntimeError,'output appeared'):g.verify_failed_previous_owner()
 def test_unknown_lstat_is_not_absence(self):
  with patch.object(g,'call',return_value=self.text()),patch.object(Path,'lstat',side_effect=PermissionError('unknown')),self.assertRaises(PermissionError):g.verify_failed_previous_owner()
 def test_each_failure_pin_changed_or_missing_rejects(self):
  with patch.object(g,'call',return_value=self.text()):
   for name in g.FAILED_PINS:
    p=self.base/name;raw=p.read_bytes();p.write_bytes(raw+b' ')
    with self.assertRaisesRegex(RuntimeError,'receipt changed'):g.verify_failed_previous_owner()
    p.unlink()
    with self.assertRaises(FileNotFoundError):g.verify_failed_previous_owner()
    p.write_bytes(raw)
 def test_rebound_semantically_wrong_restore_or_launch_rejects(self):
  for name,change in [('restored.json',lambda x:x.update(timers=[])),('restored.json',lambda x:x.update(owned_cleanup_checked=['unexpected-container'])),('pause.json',lambda x:x.update(unit='different.service')),('launch.json',lambda x:x.update(command=['echo','not a launch']))]:
   p=self.base/'forecast_pause_062'/name;raw=p.read_bytes();obj=json.loads(raw);change(obj);p.write_text(json.dumps(obj));pins={**g.FAILED_PINS,'forecast_pause_062/'+name:g.sha(p)}
   with patch.object(g,'FAILED_PINS',pins),patch.object(g,'call',return_value=self.text()),self.assertRaises(RuntimeError):g.verify_failed_previous_owner()
   p.write_bytes(raw)
 def test_actual_failure_receipt_reconstruction_matches_captured_bytes(self):
  audit=json.loads((H/'FAILED_EXTENDED001_AUDIT.json').read_text())
  self.assertTrue(audit['verified_pre_simulator_failure']);self.assertFalse(audit['output_exists'])
  for name,row in audit['pause_files'].items():
   if name in ('pause.json','launch.json','restored.json'):
    p=H/'failed_owner/forecast_pause_062'/name;self.assertEqual(g.sha(p),row['sha256']);self.assertEqual(p.stat().st_size,row['bytes'])
 def test_old_successful_smoke_host_pin_is_not_replaced_by_retry_host(self):
  self.assertEqual(g.SMOKE_HOST_FREEZE_SHA256,'b55282c968d64dca218baf10278339ff2013c23e28dc7cbe9a0d0b8dfd05d51f')
  self.assertNotEqual(g.HOST_FREEZE_SHA256,g.SMOKE_HOST_FREEZE_SHA256)

if __name__=='__main__':unittest.main()
