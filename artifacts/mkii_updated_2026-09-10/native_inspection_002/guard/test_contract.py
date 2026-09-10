import ast,contextlib,hashlib,importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('canonical_guard002',HERE/'launch_guarded_remote.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
class GuardTests(unittest.TestCase):
 @contextlib.contextmanager
 def actual_prior(self,unit=None,docker=None):
  base=HERE/'inputs/previous_owner';orig_read=Path.read_text;orig_glob=Path.glob
  def local(p):
   if p.is_relative_to(g.PREVIOUS_ROOT):return base/'run'/p.relative_to(g.PREVIOUS_ROOT)
   if p.is_relative_to(g.PREVIOUS_PAUSE):return base/'forecast_pause'/p.relative_to(g.PREVIOUS_PAUSE)
   return p
  def read(p,*a,**kw):return orig_read(local(p),*a,**kw)
  def glob(p,*a,**kw):return [g.PREVIOUS_ROOT/'jobs'/x.name for x in orig_glob(local(p),*a,**kw)] if p==g.PREVIOUS_ROOT/'jobs' else orig_glob(p,*a,**kw)
  with patch.object(Path,'read_text',read),patch.object(Path,'glob',glob),patch.object(g,'sha',side_effect=lambda p:hashlib.sha256(local(p).read_bytes()).hexdigest()),patch.object(g,'call',return_value=unit or 'ActiveState=failed\nMainPID=0\nExecMainStatus=1\nInvocationID='+g.PREVIOUS_INVOCATION),patch.object(g.subprocess,'run',return_value=docker or subprocess.CompletedProcess([],1,'','No such object')) as run:
   yield run
 def test_exact_thirteen_previous_payloads_and_audit(self):
  d=json.loads((HERE/'inputs/previous_terminal_audit.json').read_text());self.assertTrue(d['audit_verified']);self.assertFalse(d['inspection_completed']);self.assertEqual(len(g.PRIOR_PINS),13)
  with self.actual_prior():
   for name,value in g.PRIOR_PINS.items():self.assertEqual(g.sha(g.PRIOR_BASE/name),value)
 def test_authentic_failed_prior_has_exact_two_absences(self):
  with self.actual_prior() as run:g.verify_previous_owner();self.assertEqual(run.call_count,2);self.assertEqual([c.args[0][-1] for c in run.call_args_list],[g.PREVIOUS_NAME,g.PREVIOUS_ID])
  with self.actual_prior('ActiveState=inactive\nMainPID=0\nExecMainStatus=1\nInvocationID='):g.verify_previous_owner()
 def test_wrong_live_owner_or_false_success_rejects(self):
  for fields in ('ActiveState=active\nMainPID=1\nExecMainStatus=1','ActiveState=inactive\nMainPID=0\nExecMainStatus=0','ActiveState=failed\nMainPID=0\nExecMainStatus=1\nInvocationID='+'f'*32):
   with self.actual_prior(fields) as run,self.assertRaises(RuntimeError):g.verify_previous_owner()
   run.assert_not_called()
 def test_changed_prior_payload_rejects_before_docker(self):
  with self.actual_prior() as run,patch.object(g,'PRIOR_PINS',{next(iter(g.PRIOR_PINS)):'f'*64}),self.assertRaisesRegex(RuntimeError,'changed'):g.verify_previous_owner()
  run.assert_not_called()
 def test_unknown_absence_or_existing_container_rejects(self):
  for result in (subprocess.CompletedProcess([],1,'','daemon unavailable'),subprocess.CompletedProcess([],0,'id name false','')):
   with self.actual_prior(docker=result),self.assertRaisesRegex(RuntimeError,'unknown'):g.verify_previous_owner()
 def test_pending_binding_no_external_calls(self):
  with patch.object(g,'SOURCE_SHA256','PENDING_TEST'),patch.object(g.subprocess,'run') as run,patch.object(g,'call') as call,self.assertRaisesRegex(RuntimeError,'pending'):g.require_final_bindings()
  run.assert_not_called();call.assert_not_called()
 def test_reused_output_fails_before_any_process(self):
  with tempfile.TemporaryDirectory() as t,patch.object(g,'OUTPUT',Path(t)),patch.object(g,'PAUSE',Path(t)/'pause'),patch.object(g.subprocess,'run') as run,patch.object(g,'call') as call,self.assertRaisesRegex(RuntimeError,'Fresh'):g.main()
  run.assert_not_called();call.assert_not_called()
 def test_restorer_gpu_and_main_unchanged_except_unique_names(self):
  old=ast.parse((HERE/'inputs/guard_parent.py').read_text());new=ast.parse((HERE/'launch_guarded_remote.py').read_text())
  def embed(t):return next(n.args[0].value for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='restorer' and n.func.attr=='write_text')
  self.assertEqual(embed(old),embed(new))
  for name in ('main','validate_inspection_inputs'):
   a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name);b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
   for n in ast.walk(a):
    if isinstance(n,ast.Constant) and n.value=='--unit=hexapod-canonical-forecast-restore-001':n.value='--unit=hexapod-canonical-forecast-restore-002'
   self.assertEqual(ast.dump(a),ast.dump(b))
 def test_unique_namespace_and_fixed_bounds(self):
  self.assertEqual(g.OUTPUT.name,'native_inspection_002');self.assertEqual(g.PAUSE.name,'forecast_pause_002');self.assertEqual(g.ASSET.name,'asset_001')
  t=(HERE/'launch_guarded_remote.py').read_text()
  for x in ('hexapod-canonical-forecast-restore-002','RuntimeMaxSec=720','TimeoutStopSec=180','--on-active=15m','/opt/wx/gpu.lock','/tmp/hexapod-isaac-gpu.lock'):self.assertIn(x,t)
  self.assertNotIn("'--checkpoint'",t)
 def test_stdlib_import(self):
  subprocess.run([sys.executable,'-B','-S','-c','import launch_guarded_remote,sys;assert not any(x in sys.modules for x in ("numpy","torch","isaaclab"))'],cwd=HERE,check=True,capture_output=True)
if __name__=='__main__':unittest.main()
