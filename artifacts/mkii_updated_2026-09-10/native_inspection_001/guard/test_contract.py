import ast,importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('canonical_guard',HERE/'launch_guarded_remote.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
class GuardTests(unittest.TestCase):
 def local_prior(self):
  base=HERE/'inputs/previous_owner';return patch.multiple(g,PRIOR_BASE=base,PREVIOUS_ROOT=base/'run',PREVIOUS_PAUSE=base/'forecast_pause',PRIOR_PINS={('run/'+k.split('/',1)[1] if k.startswith('direct_omni') else 'forecast_pause/'+k.split('/',1)[1]):v for k,v in g.PRIOR_PINS.items()})
 def call(self):return patch.object(g,'call',return_value='ActiveState=inactive\nMainPID=0\nExecMainStatus=0\nInvocationID=')
 def test_pending_native_host_stops_before_any_external_calls(self):
  with patch.object(g,'SOURCE_SHA256','PENDING_TEST_BINDING'),patch.object(g.subprocess,'run') as run,patch.object(g,'call') as call:
   with self.assertRaisesRegex(RuntimeError,'pending'):g.require_final_bindings()
  run.assert_not_called();call.assert_not_called()
 def test_exact_real_previous_payload_hashes(self):
  a=json.loads((HERE/'inputs/previous_terminal_audit.json').read_text())
  self.assertEqual(a['expected_invocation'],g.PREVIOUS_INVOCATION);self.assertTrue(a['recording_completion_replay_passed']);self.assertEqual(len(a['owned_names_IDs_absent']),2)
  with self.local_prior():
   for name,value in g.PRIOR_PINS.items():self.assertEqual(g.sha(g.PRIOR_BASE/name),value)
 def test_actual_prior_two_absences_and_restore_receipt(self):
  # Paths in actual prior receipts retain their true remote identity; no fabricated prior fixture.
  base=HERE/'inputs/previous_owner';real_read=Path.read_text
  mapping={g.PREVIOUS_ROOT/'campaign.json':base/'run/campaign.json',g.PREVIOUS_ROOT/'jobs/recording.json':base/'run/jobs/recording.json',g.PREVIOUS_PAUSE/'pause.json':base/'forecast_pause/pause.json',g.PREVIOUS_PAUSE/'restored.json':base/'forecast_pause/restored.json'}
  def read(p,*a,**kw):return real_read(mapping.get(p,p),*a,**kw)
  def digest(p):
   k=p.relative_to(g.PRIOR_BASE).as_posix();rel=('run/'+k.split('/',1)[1] if k.startswith('direct_omni') else 'forecast_pause/'+k.split('/',1)[1]);return __import__('hashlib').sha256((base/rel).read_bytes()).hexdigest()
  with patch.object(Path,'read_text',read),patch.object(g,'sha',side_effect=digest),patch.object(Path,'glob',return_value=[g.PREVIOUS_ROOT/'jobs/recording.json']),self.call(),patch.object(g.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','No such object')) as run:
   g.verify_previous_owner();self.assertEqual(run.call_count,2)
   run.return_value=subprocess.CompletedProcess([],1,'','daemon unavailable')
   with self.assertRaisesRegex(RuntimeError,'unknown'):g.verify_previous_owner()
 def test_active_wrong_invocation_or_failed_previous_blocks(self):
  for value in ('ActiveState=active\nMainPID=1\nExecMainStatus=0','ActiveState=inactive\nMainPID=0\nExecMainStatus=0\nInvocationID='+'f'*32,'ActiveState=failed\nMainPID=0\nExecMainStatus=1'):
   with patch.object(g,'call',return_value=value),self.assertRaises(RuntimeError):g.verify_previous_owner()
 def test_reused_output_blocks_before_process(self):
  with tempfile.TemporaryDirectory() as t:
   out=Path(t);pause=out/'pause'
   with patch.object(g,'OUTPUT',out),patch.object(g,'PAUSE',pause),patch.object(g.subprocess,'run') as run,patch.object(g,'call') as call,self.assertRaisesRegex(RuntimeError,'Fresh'):g.main()
   run.assert_not_called();call.assert_not_called()
 def test_restorer_and_gpu_ancestry_unchanged(self):
  old=ast.parse((HERE/'inputs/preview_guard_parent.py').read_text());new=ast.parse((HERE/'launch_guarded_remote.py').read_text())
  def embed(t):return next(n.args[0].value for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='restorer' and n.func.attr=='write_text')
  self.assertEqual(embed(old),embed(new))
  def gpu(t):return next(n for n in ast.walk(t) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='line' and any(isinstance(x,ast.Constant) and isinstance(x.value,str) and x.value.startswith('Unrelated CUDA') for x in ast.walk(n)))
  self.assertEqual(ast.dump(gpu(old)),ast.dump(gpu(new)))
 def test_unique_canonical_namespace_and_phase_bound(self):
  text=(HERE/'launch_guarded_remote.py').read_text()
  for x in ('canonical_direct_20260910','hexapod-canonical-forecast-restore-001','RuntimeMaxSec=720','TimeoutStopSec=180','--on-active=15m','/opt/wx/gpu.lock','/tmp/hexapod-isaac-gpu.lock','35af5100e39980f55a3f5e19654e4550b3c3fb6da114b64fbe0d143817242cf3'):self.assertIn(x,text)
  self.assertNotIn("'--checkpoint'",text)
 def test_stdlib_import(self):
  subprocess.run([sys.executable,'-B','-S','-c',"import launch_guarded_remote,sys;assert not any(k in sys.modules for k in ('torch','numpy','isaaclab'))"],cwd=HERE,check=True,capture_output=True)
if __name__=='__main__':unittest.main()
