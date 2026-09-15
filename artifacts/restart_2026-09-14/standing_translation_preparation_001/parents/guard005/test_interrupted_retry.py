import contextlib,importlib.util,json,subprocess,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('interruption_guard',HERE/'launch_guarded_remote.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
class InterruptedRetryTests(unittest.TestCase):
 @contextlib.contextmanager
 def actual(self,unit=None,docker=None,state_patch=None):
  base=HERE/'inputs/interrupted_owner';orig_read=Path.read_text;orig_glob=Path.glob;sha=g.sha;inventory=json.loads((HERE/'inputs/INTERRUPTED_RAW_INVENTORY.json').read_text())
  def local(p):
   if p.is_relative_to(g.RETRY_ROOT):return base/'run'/p.relative_to(g.RETRY_ROOT)
   if p.is_relative_to(g.RETRY_PAUSE):return base/'forecast_pause'/p.relative_to(g.RETRY_PAUSE)
   return p
  def rd(p,*a,**kw):
   text=orig_read(local(p),*a,**kw)
   if state_patch is not None and p==g.RETRY_ROOT/'standing/state.json':return json.dumps({**json.loads(text),**state_patch})
   return text
  def glob(p,*a,**kw):return [g.RETRY_ROOT/'jobs'/x.name for x in orig_glob(local(p),*a,**kw)]if p==g.RETRY_ROOT/'jobs'else orig_glob(p,*a,**kw)
  def localsha(p):
   target=local(p)
   if target.is_file():return sha(target)
   return inventory[target.relative_to(base).as_posix()]['sha256']
  with patch.object(Path,'read_text',rd),patch.object(Path,'glob',glob),patch.object(g,'sha',side_effect=localsha),patch.object(g,'call',return_value=unit or 'ActiveState=failed\nMainPID=0\nExecMainStatus=1\nInvocationID='+g.RETRY_INVOCATION),patch.object(g.subprocess,'run',return_value=docker or subprocess.CompletedProcess([],1,'','No such object'))as run:yield run
 def test_exact_actual_interruption_and_two_absences(self):
  with self.actual()as run:g.verify_retry_cleanup();self.assertEqual([c.args[0][-1]for c in run.call_args_list],[g.RETRY_NAME,g.RETRY_ID])
  self.assertEqual(len(g.RETRY_RAW_PINS),34);self.assertEqual(sum('substeps_' in k for k in g.RETRY_RAW_PINS),9)
  self.assertFalse((HERE/'inputs/interrupted_owner/run/standing/contacts.jsonl').exists())
  audit=json.loads((HERE/'inputs/interruption_audit.json').read_text());self.assertEqual(g.RETRY_COMPETITORS,audit['job']['competitors']);self.assertEqual(audit['raw_total_bytes'],4666042464)
 def test_running_or_changed_owner_rejects(self):
  for unit in ['ActiveState=active\nMainPID=1\nExecMainStatus=0','ActiveState=failed\nMainPID=0\nExecMainStatus=1\nInvocationID='+'f'*32]:
   with self.actual(unit=unit)as run,self.assertRaises(RuntimeError):g.verify_retry_cleanup()
   run.assert_not_called()
 def test_invented_physics_or_native_finalization_rejects(self):
  for d in [{'explicit_steps_completed':1},{'status':'completed'},{'checks':{'scene':True}}]:
   with self.actual(state_patch=d)as run,self.assertRaisesRegex(RuntimeError,'unfinalized'):g.verify_retry_cleanup()
   run.assert_not_called()
 def test_unknown_owned_absence_and_changed_raw_reject(self):
  with self.actual(docker=subprocess.CompletedProcess([],1,'','daemon unreachable')),self.assertRaisesRegex(RuntimeError,'unknown'):g.verify_retry_cleanup()
  with self.actual()as run,patch.object(g,'RETRY_RAW_PINS',{next(iter(g.RETRY_RAW_PINS)):'f'*64}),self.assertRaisesRegex(RuntimeError,'raw changed'):g.verify_retry_cleanup()
  run.assert_not_called()
 def test_pending_retry_audit_refuses_before_external_calls(self):
  with patch.object(g,'RETRY_AUDIT_SHA256',None),patch.object(g,'call')as call,patch.object(g.subprocess,'run')as run,self.assertRaisesRegex(RuntimeError,'pending'):g.require_final_bindings()
  call.assert_not_called();run.assert_not_called()
if __name__=='__main__':unittest.main()
