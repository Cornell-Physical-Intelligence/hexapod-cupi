import ast,hashlib,importlib.util,json,shutil,subprocess,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('coldguard',HERE/'launch_guarded_remote.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)

class GuardTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name)
  shutil.copytree(HERE/'previous_owner',self.base,dirs_exist_ok=True)
  p=patch.object(g,'BASE',self.base);p.start();self.addCleanup(p.stop)
  self.unit='ActiveState=failed\nSubState=failed\nInvocationID='+g.PREVIOUS_INVOCATION
 def absent(self,*a,**kw):return subprocess.CompletedProcess(a,1,'','No such object')
 def test_unbound_no_pause_or_process(self):
  with patch.object(g,'PAUSE',self.base/'pause'),patch.object(g,'OUTPUT',self.base/'out'),patch.object(g,'HOST_SHA256','PENDING'),patch.object(g.subprocess,'run') as run,patch.object(g.subprocess,'check_output') as call:
   with self.assertRaisesRegex(RuntimeError,'pending'):g.main()
  run.assert_not_called();call.assert_not_called();self.assertFalse((self.base/'pause').exists())
 def test_exact_failed_previous_two_absences(self):
  with patch.object(g,'call',return_value=self.unit),patch.object(g.subprocess,'run',side_effect=self.absent) as run:g.verify_previous_owner()
  self.assertEqual(run.call_count,2)
 def test_collected_unit_with_exact_receipts_allowed(self):
  with patch.object(g,'call',return_value='ActiveState=inactive\nInvocationID='),patch.object(g.subprocess,'run',side_effect=self.absent):g.verify_previous_owner()
 def test_active_or_conflicting_invocation_fails(self):
  for value in [self.unit.replace('ActiveState=failed','ActiveState=active'),self.unit.replace(g.PREVIOUS_INVOCATION,'a'*32)]:
   with patch.object(g,'call',return_value=value),patch.object(g.subprocess,'run') as run:
    with self.assertRaises(RuntimeError):g.verify_previous_owner()
    run.assert_not_called()
 def test_changed_restoration_fails(self):
  (self.base/'forecast_pause_052/restored.json').write_text('{}')
  with patch.object(g,'call',return_value=self.unit),patch.object(g.subprocess,'run') as run:
   with self.assertRaisesRegex(RuntimeError,'receipt changed'):g.verify_previous_owner()
   run.assert_not_called()
 def test_container_live_or_unknown_not_absence(self):
  for r in [subprocess.CompletedProcess([],0,'x /owner false',''),subprocess.CompletedProcess([],1,'','daemon unavailable')]:
   with patch.object(g,'call',return_value=self.unit),patch.object(g.subprocess,'run',return_value=r):
    with self.assertRaisesRegex(RuntimeError,'not proven absent'):g.verify_previous_owner()
 def test_inputs_reject_changed_source_before_host(self):
  host=SimpleNamespace(verify_inputs=Mock())
  def digest(p):
   if p==g.SUPERVISOR_SOURCE/'campaign_source_hashes.json':return g.SUPERVISOR_SHA256
   if p==g.SOURCE/'campaign_source_hashes.json':return '0'*64
   raise AssertionError('No later input should be read')
  with patch.object(g,'sha',side_effect=digest):
   with self.assertRaisesRegex(RuntimeError,'Changed cold source'):g.validate_cold_inputs(host)
  host.verify_inputs.assert_not_called()
 def test_full_preflight_identity_and_no_learning(self):
  identity=dict(source_manifest_sha256=g.SOURCE_SHA256,checkpoint_sha256=g.CHECKPOINT_SHA256,training_allowed=False)
  host=SimpleNamespace(verify_inputs=Mock(return_value=identity))
  d={g.SUPERVISOR_SOURCE/'campaign_source_hashes.json':g.SUPERVISOR_SHA256,g.SOURCE/'campaign_source_hashes.json':g.SOURCE_SHA256,g.CHECKPOINT:g.CHECKPOINT_SHA256,g.HOST:'a'*64}
  with patch.object(g,'sha',side_effect=lambda p:d[p]),patch.object(g,'HOST_SHA256','a'*64),patch.object(g,'verify_frozen') as verify:
   self.assertEqual(g.validate_cold_inputs(host),identity);self.assertEqual(verify.call_count,2)
   identity['training_allowed']=True
   with self.assertRaisesRegex(RuntimeError,'permits learning'):g.validate_cold_inputs(host)
  args=host.verify_inputs.call_args.args[0];self.assertEqual(args.supervisor_source,g.SUPERVISOR_SOURCE)
 def test_frozen_payload_mutation_and_addition_reject(self):
  b=self.base/'bundle';b.mkdir();p=b/'runtime.py';p.write_text('exact');m=b/'FREEZE_SHA256.json';m.write_text(json.dumps({'runtime.py':g.sha(p)}));bound=g.sha(m)
  g.verify_frozen(b,bound);p.write_text('changed')
  with self.assertRaises(RuntimeError):g.verify_frozen(b,bound)
  p.write_text('exact');(b/'extra').write_text('extra')
  with self.assertRaises(RuntimeError):g.verify_frozen(b,bound)
 def test_two_phase_cli_no_training_and_bounds(self):
  t=ast.parse((HERE/'launch_guarded_remote.py').read_text())
  c=[n.value for n in ast.walk(t) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='cmd' for x in n.targets)]
  self.assertEqual(len(c),1)
  names={k:getattr(g,k) for k in ['UNIT','HOST','SOURCE','CHECKPOINT','CONTRACT','SUPERVISOR_SOURCE','OUTPUT']};names['restorer']=Path('/owned/resume.py')
  cmd=eval(compile(ast.Expression(c[0]),'<command>','eval'),{'str':str},names)
  self.assertIn('--property=RuntimeMaxSec=1800',cmd);self.assertIn('--property=TimeoutStopSec=180',cmd)
  self.assertEqual(cmd[cmd.index('--checkpoint')+1],str(g.CHECKPOINT))
  self.assertFalse(set(cmd)&{'train','learn','--decision-receipt','--phase-group'})
  self.assertIn('--on-active=35m',[n.value for n in ast.walk(t) if isinstance(n,ast.Constant)])
 def test_preflight_before_pause_and_again_before_dispatch(self):
  t=ast.parse((HERE/'launch_guarded_remote.py').read_text());main=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='main')
  calls=[n for n in ast.walk(main) if isinstance(n,ast.Call)]
  valid=sorted(n.lineno for n in calls if isinstance(n.func,ast.Name) and n.func.id=='validate_cold_inputs')
  pause=next(n.lineno for n in calls if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='PAUSE' and n.func.attr=='mkdir')
  self.assertEqual(len(valid),2);self.assertLess(valid[0],pause);self.assertGreater(valid[1],pause)
  safe=next(n for n in ast.walk(main) if isinstance(n,ast.Try) and any(isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Name) and x.value.func.id=='validate_cold_inputs' for x in n.body))
  text=ast.unparse(safe.handlers[0]);self.assertIn('str(restorer)',text);self.assertNotIn('--stop-owner',text)
 def test_embedded_restorer_identical_reviewed_parent(self):
  def embedded(path):
   t=ast.parse(path.read_text());return next(n.args[0].value for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='restorer' and n.func.attr=='write_text')
  self.assertEqual(embedded(HERE/'launch_guarded_remote.py'),embedded(HERE.parents[0]/'reference_learning_ppo_guard_001/launch_guarded_remote.py'))

if __name__=='__main__':unittest.main()
