import ast,hashlib,importlib.util,json,shutil,subprocess,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch,Mock
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('nativeguard',HERE/'launch_guarded_remote.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)

class GuardTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name)
  # Synthetic future receipts are test-only and never copied into runtime bindings.
  fixtures={
   'direct_omni_train_smoke_004/campaign.json':dict(status='completed',terminal_inputs_unchanged=True,bounded_campaign_complete=True,allocation='smoke',branch='quiet_priority',PPO_updates_completed=2,planned_phases=['standing','train','final_constant','final_stop'],accepted_phases={k:{} for k in ['standing','train','final_constant','final_stop']},host_freeze_sha256=g.HOST_FREEZE_SHA256,identity={'source_manifest_sha256':g.SOURCE_SHA256}),
   'forecast_pause_061/pause.json':dict(unit=g.PREVIOUS_UNIT,units={'weather.timer':'ActiveState=active','other.timer':'ActiveState=inactive'}),
   'forecast_pause_061/restored.json':dict(timers=['weather.timer'],restored_unix=123)}
  for i,phase in enumerate(('standing','train','final_constant','final_stop')):
   fixtures['direct_omni_train_smoke_004/jobs/'+phase+'.json']=dict(status='completed',phase=phase,cleanup_checked=True,container_name='test-only-'+phase,container_id=str(i+1)*64)
  pins={}
  for name,data in fixtures.items():
   file=self.base/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_text(json.dumps(data));pins[name]=g.sha(file)
  for name,value in [('BASE',self.base),('SMOKE',self.base/'direct_omni_train_smoke_004'),('PRIOR_PINS',pins),('PREVIOUS_INVOCATION','b'*32)]:
   p=patch.object(g,name,value);p.start();self.addCleanup(p.stop)
  self.unit='ActiveState=failed\nSubState=failed\nInvocationID='+g.PREVIOUS_INVOCATION
 def absent(self,*a,**kw):return subprocess.CompletedProcess(a,1,'','No such object')
 def test_unbound_no_pause_or_process(self):
  with patch.object(g,'PAUSE',self.base/'pause'),patch.object(g,'OUTPUT',self.base/'out'),patch.object(g,'HOST_SHA256','PENDING'),patch.object(g.subprocess,'run') as run,patch.object(g.subprocess,'check_output') as call:
   with self.assertRaisesRegex(RuntimeError,'pending'):g.main()
  run.assert_not_called();call.assert_not_called();self.assertFalse((self.base/'pause').exists())
 def test_exact_completed_previous_eight_absences(self):
  with patch.object(g,'call',return_value=self.unit),patch.object(g.subprocess,'run',side_effect=self.absent) as run:g.verify_previous_owner()
  self.assertEqual(run.call_count,8)
 def test_collected_unit_with_exact_receipts_allowed(self):
  with patch.object(g,'call',return_value='ActiveState=inactive\nInvocationID='),patch.object(g.subprocess,'run',side_effect=self.absent):g.verify_previous_owner()
 def test_active_or_conflicting_invocation_fails(self):
  for value in [self.unit.replace('ActiveState=failed','ActiveState=active'),self.unit.replace(g.PREVIOUS_INVOCATION,'a'*32)]:
   with patch.object(g,'call',return_value=value),patch.object(g.subprocess,'run') as run:
    with self.assertRaises(RuntimeError):g.verify_previous_owner()
    run.assert_not_called()
 def test_changed_restoration_fails(self):
  (self.base/'forecast_pause_061/restored.json').write_text('{}')
  with patch.object(g,'call',return_value=self.unit),patch.object(g.subprocess,'run') as run:
   with self.assertRaisesRegex(RuntimeError,'receipt changed'):g.verify_previous_owner()
   run.assert_not_called()
 def test_semantically_incomplete_prior_campaign_rejects_even_with_matching_hash(self):
  name='direct_omni_train_smoke_004/campaign.json';path=self.base/name;d=json.loads(path.read_text());d['terminal_inputs_unchanged']=False;path.write_text(json.dumps(d));pins={**g.PRIOR_PINS,name:g.sha(path)}
  with patch.object(g,'PRIOR_PINS',pins),patch.object(g,'call',return_value=self.unit),patch.object(g.subprocess,'run') as run:
   with self.assertRaisesRegex(RuntimeError,'not complete'):g.verify_previous_owner()
   run.assert_not_called()
 def test_exact_timer_restore_required_even_when_receipt_hash_matches(self):
  name='forecast_pause_061/restored.json';path=self.base/name;d=json.loads(path.read_text());d['timers']=[];path.write_text(json.dumps(d));pins={**g.PRIOR_PINS,name:g.sha(path)}
  with patch.object(g,'PRIOR_PINS',pins),patch.object(g,'call',return_value=self.unit),patch.object(g.subprocess,'run') as run:
   with self.assertRaisesRegex(RuntimeError,'timer restoration'):g.verify_previous_owner()
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
   with self.assertRaisesRegex(RuntimeError,'Changed native source'):g.validate_train_inputs(host)
  host.verify_inputs.assert_not_called()
 def test_full_preflight_identity_and_exact_pilot_selection(self):
  identity=dict(source_manifest_sha256=g.SOURCE_SHA256,checkpoint_sha256=g.CHECKPOINT_SHA256,actor_width=315,critic_width=318,smoke_campaign_sha256=g.PRIOR_PINS['direct_omni_train_smoke_004/campaign.json'],selection={'schema':'direct315_extended_native_v4','allocation':'extended','branch':'caps','replicas':1024,'controls_per_update':24,'updates':500,'caps':{'temporal_weight':.1,'spatial_weight':.1,'noise_scale':1.,'noise_seed':1157,'quiet_temporal_weight':.1}})
  host=SimpleNamespace(verify_inputs=Mock(return_value=identity),selected_phases=Mock(return_value=('standing','initial_constant','initial_stop','train','final_constant','final_stop')))
  d={g.SUPERVISOR_SOURCE/'campaign_source_hashes.json':g.SUPERVISOR_SHA256,g.SOURCE/'campaign_source_hashes.json':g.SOURCE_SHA256,g.CHECKPOINT:g.CHECKPOINT_SHA256,g.HOST:'a'*64}
  with patch.object(g,'sha',side_effect=lambda p:d[p]),patch.object(g,'HOST_SHA256','a'*64),patch.object(g,'verify_frozen') as verify:
   self.assertEqual(g.validate_train_inputs(host),identity);self.assertEqual(verify.call_count,2)
   old_smoke=identity['smoke_campaign_sha256'];identity['smoke_campaign_sha256']='0'*64
   with self.assertRaisesRegex(RuntimeError,'different prior smoke'):g.validate_train_inputs(host)
   identity['smoke_campaign_sha256']=old_smoke
   identity['selection']['updates']=2
   with self.assertRaisesRegex(RuntimeError,'500-update'):g.validate_train_inputs(host)
   identity['selection']['updates']=500;host.selected_phases.return_value=('standing','train')
   with self.assertRaisesRegex(RuntimeError,'phase order'):g.validate_train_inputs(host)
  args=host.verify_inputs.call_args.args[0];self.assertEqual(args.supervisor_source,g.SUPERVISOR_SOURCE);self.assertEqual((args.allocation,args.branch,args.smoke),('extended','caps',g.SMOKE))
 def test_pending_actual_prior_hash_rejects_even_with_other_bindings_complete(self):
  with patch.object(g,'PRIOR_PINS',{**g.PRIOR_PINS,'direct_omni_train_smoke_004/jobs/train.json':'PENDING_TERMINAL_SMOKE'}):
   with self.assertRaisesRegex(RuntimeError,'pending'):g.require_final_bindings()
 def test_exact_prior_seven_receipts_required(self):
  with patch.object(g,'PRIOR_PINS',{k:v for k,v in g.PRIOR_PINS.items() if k!='direct_omni_train_smoke_004/jobs/final_stop.json'}):
   with self.assertRaisesRegex(RuntimeError,'receipts required'):g.require_final_bindings()
 def test_failed_train_or_missing_final_diagnostics_reject(self):
  path=self.base/'direct_omni_train_smoke_004/campaign.json';original=json.loads(path.read_text())
  for change in [{'PPO_updates_completed':1},{'allocation':'pilot'},{'branch':'caps'},{'planned_phases':['standing','train']},{'accepted_phases':{'standing':{}}},{'identity':{'source_manifest_sha256':'a'*64}}]:
   path.write_text(json.dumps({**original,**change}));pins={**g.PRIOR_PINS,'direct_omni_train_smoke_004/campaign.json':g.sha(path)}
   with self.subTest(change=change),patch.object(g,'PRIOR_PINS',pins),patch.object(g,'call',return_value=self.unit),patch.object(g.subprocess,'run') as run:
    with self.assertRaisesRegex(RuntimeError,'allocation/source/phase'):g.verify_previous_owner()
    run.assert_not_called()
 def test_missing_prior_job_cleanup_rejects(self):
  name='direct_omni_train_smoke_004/jobs/final_stop.json';path=self.base/name;d=json.loads(path.read_text());d['cleanup_checked']=False;path.write_text(json.dumps(d));pins={**g.PRIOR_PINS,name:g.sha(path)}
  with patch.object(g,'PRIOR_PINS',pins),patch.object(g,'call',return_value=self.unit),patch.object(g.subprocess,'run',side_effect=self.absent):
   with self.assertRaisesRegex(RuntimeError,'cleanup proof'):g.verify_previous_owner()
 def test_frozen_payload_mutation_and_addition_reject(self):
  b=self.base/'bundle';b.mkdir();p=b/'runtime.py';p.write_text('exact');m=b/'FREEZE_SHA256.json';m.write_text(json.dumps({'runtime.py':g.sha(p)}));bound=g.sha(m)
  g.verify_frozen(b,bound);p.write_text('changed')
  with self.assertRaises(RuntimeError):g.verify_frozen(b,bound)
  p.write_text('exact');(b/'extra').write_text('extra')
  with self.assertRaises(RuntimeError):g.verify_frozen(b,bound)
 def test_six_phase_pilot_cli_and_bounds(self):
  t=ast.parse((HERE/'launch_guarded_remote.py').read_text())
  c=[n.value for n in ast.walk(t) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='cmd' for x in n.targets)]
  self.assertEqual(len(c),1)
  names={k:getattr(g,k) for k in ['UNIT','HOST','SOURCE','CHECKPOINT','CONTRACT','SUPERVISOR_SOURCE','OUTPUT','SMOKE']};names['restorer']=Path('/owned/resume.py')
  cmd=eval(compile(ast.Expression(c[0]),'<command>','eval'),{'str':str},names)
  self.assertIn('--property=RuntimeMaxSec=5400',cmd);self.assertIn('--property=TimeoutStopSec=180',cmd)
  self.assertEqual(cmd[cmd.index('--checkpoint')+1],str(g.CHECKPOINT))
  self.assertFalse(set(cmd)&{'learn','--decision-receipt','--phase-group'});self.assertEqual(cmd[cmd.index('--allocation')+1],'extended');self.assertEqual(cmd[cmd.index('--branch')+1],'caps');self.assertEqual(cmd[cmd.index('--smoke')+1],str(g.SMOKE))
  self.assertIn('--on-active=95m',[n.value for n in ast.walk(t) if isinstance(n,ast.Constant)])
 def test_preflight_before_pause_and_again_before_dispatch(self):
  t=ast.parse((HERE/'launch_guarded_remote.py').read_text());main=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='main')
  calls=[n for n in ast.walk(main) if isinstance(n,ast.Call)]
  valid=sorted(n.lineno for n in calls if isinstance(n.func,ast.Name) and n.func.id=='validate_train_inputs')
  pause=next(n.lineno for n in calls if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='PAUSE' and n.func.attr=='mkdir')
  self.assertEqual(len(valid),2);self.assertLess(valid[0],pause);self.assertGreater(valid[1],pause)
  safe=next(n for n in ast.walk(main) if isinstance(n,ast.Try) and any(isinstance(x,ast.Expr) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Name) and x.value.func.id=='validate_train_inputs' for x in n.body))
  text=ast.unparse(safe.handlers[0]);self.assertIn('str(restorer)',text);self.assertNotIn('--stop-owner',text)
 def test_embedded_restorer_identical_reviewed_parent(self):
  def embedded(path):
   t=ast.parse(path.read_text());return next(n.args[0].value for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='restorer' and n.func.attr=='write_text')
  self.assertEqual(embedded(HERE/'launch_guarded_remote.py'),embedded(HERE.parents[0]/'direct_omni_train_guard_002/launch_guarded_remote.py'))

class RuntimePendingTests(unittest.TestCase):
 def test_real_unbound_draft_fails_before_any_external_effect(self):
  if all(len(v)==64 for v in g.PRIOR_PINS.values()):
   g.require_final_bindings()
   for name,bound in g.PRIOR_PINS.items():self.assertEqual(g.sha(HERE/'previous_owner'/name),bound)
   return
  with patch.object(g.subprocess,'run') as run,patch.object(g.subprocess,'check_output') as output:
   with self.assertRaisesRegex(RuntimeError,'pending'):g.require_final_bindings()
  run.assert_not_called();output.assert_not_called()
 def test_actual_frozen_host_selects_exact_six_phases(self):
  spec=importlib.util.spec_from_file_location('actual_pilot_host_review',HERE.parent/'direct_omni_train_host_004/launch_train_spark.py')
  host=importlib.util.module_from_spec(spec);spec.loader.exec_module(host)
  self.assertEqual(host.selected_phases(SimpleNamespace(allocation='extended',branch='caps',smoke=g.SMOKE)),('standing','initial_constant','initial_stop','train','final_constant','final_stop'))
  with self.assertRaisesRegex(ValueError,'completed smoke'):host.selected_phases(SimpleNamespace(allocation='extended',branch='caps',smoke=None))

if __name__=='__main__':unittest.main()
