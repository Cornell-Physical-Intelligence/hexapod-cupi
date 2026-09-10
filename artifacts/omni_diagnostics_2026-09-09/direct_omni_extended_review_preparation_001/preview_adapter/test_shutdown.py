import ast,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import entry_adapter,shutdown_integrity as s
HERE=Path(__file__).resolve().parent

def fixture(path):
 p=Path(path);p.mkdir(exist_ok=True);provenance={'checkpoint_sha256':'a'*64};v={**provenance,'complete':True,'frames':950,'recorded_control_steps':1900,'fps':25,'terminal_event':None,'error':None}
 for when in ('before','after'):
  check={'passed':True,'actor_and_critic_including_normalizers_exact':True,'checkpoint_sha256':'a'*64};s.write(p/('checkpoint_readback_'+when+'.json'),check);v['checkpoint_readback_'+when]=check
 for name,key in (('rollout.mp4','video_sha256'),('trace.npz','trace_sha256')):(p/name).write_bytes(name.encode());v[key]=s.sha(p/name)
 for name in ('provenance.json','entry_seams.json','native_arguments.json','environment.yaml','matched_evaluation_environment.yaml'):(p/name).write_text('{}')
 s.write(p/'video.json',v);s.write(p/'state.json',{'status':'completed','recording_provenance':provenance})
 return NS(output=p),provenance

class Tests(unittest.TestCase):
 def test_before_close_seal_survives_process_exit_but_after_close_does_not(self):
  with tempfile.TemporaryDirectory() as t:
   code="""from test_shutdown import *
a,p=fixture(sys.argv[1])
s.shutdown_runtime=lambda app:{'source_sha256':s.SIMULATION_APP_SHA256,'fast_shutdown':True,'close_return_required':False}
try:s.seal_before_shutdown(a,p,None,lambda _:p)
finally:os._exit(0)
(a.output/'unreachable').write_text('bad')
"""
   r=subprocess.run([sys.executable,'-B','-c',code,t],cwd=HERE);self.assertEqual(r.returncode,0)
   s.validate_shutdown_seal(Path(t),{'checkpoint_sha256':'a'*64});self.assertFalse((Path(t)/'unreachable').exists())
 def test_source_failure_even_if_close_exits_zero_cannot_seal(self):
  with tempfile.TemporaryDirectory() as t:
   code="""from test_shutdown import *
from record_direct_preview import preserve_failure
a,p=fixture(sys.argv[1])
try:
 try:s.seal_before_shutdown(a,p,None,lambda _:{'wrong':'input'})
 except BaseException as e:preserve_failure(a.output,e);raise
finally:os._exit(0)
"""
   r=subprocess.run([sys.executable,'-B','-c',code,t],cwd=HERE);self.assertEqual(r.returncode,0)
   self.assertFalse((Path(t)/s.NAME).exists());self.assertEqual(s.read(Path(t)/'state.json')['status'],'failed')
 def test_missing_and_changed_output_or_failure_reject(self):
  with tempfile.TemporaryDirectory() as t:
   args,p=fixture(t)
   with patch.object(s,'shutdown_runtime',return_value={'source_sha256':s.SIMULATION_APP_SHA256}):s.seal_before_shutdown(args,p,None,lambda _:p)
   (args.output/'trace.npz').write_bytes(b'changed')
   with self.assertRaisesRegex(ValueError,'Post-seal'):s.validate_shutdown_seal(args.output,p)
  with tempfile.TemporaryDirectory() as t:
   args,p=fixture(t);s.write(args.output/'failure.json',{'failure':'preserved'})
   with self.assertRaisesRegex(ValueError,'Failed'):s.seal_before_shutdown(args,p,None,lambda _:p)
 def test_pending_state_or_wrong_readback_reject(self):
  with tempfile.TemporaryDirectory() as t:
   args,p=fixture(t);s.write(args.output/'checkpoint_readback_after.json',{'passed':False})
   with self.assertRaisesRegex(ValueError,'readback'):s.seal_before_shutdown(args,p,None,lambda _:p)
  with tempfile.TemporaryDirectory() as t:
   args,p=fixture(t);(args.output/'native_arguments.json').unlink()
   with self.assertRaisesRegex(ValueError,'entry/configuration'):s.seal_before_shutdown(args,p,None,lambda _:p)
 def test_finally_keeps_actual_close_after_seal_error(self):
  tree,counts=entry_adapter.instrument((HERE/'inputs/native_entry.py').read_text());self.assertEqual(counts['shutdown'],1)
  node=next(n for n in ast.walk(tree) if isinstance(n,ast.Try) and len(n.body)==1 and entry_adapter.same(n.body[0],'_preview_prepare_shutdown()'))
  calls=[]
  def reject():calls.append('seal');raise ValueError('missing')
  with self.assertRaises(ValueError):exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_shutdown_seam','exec'),{'_preview_prepare_shutdown':reject,'app':NS(close=lambda:calls.append('close'))})
  self.assertEqual(calls,['seal','close'])
 def test_render_runtime_bytes_remain_unchanged(self):
  expected={'preview_recorder.py':'60c1c9b58a7132f0506dcfef060e86f5912093694979c61f57bad6669914b94c','render_helpers.py':'b245d473abce3e1cdd795a6c1ff5f628000d5a4c235002b2a6d9eafc265037a5'}
  for name,bound in expected.items():self.assertEqual(s.sha(HERE/name),bound)
if __name__=='__main__':unittest.main()
