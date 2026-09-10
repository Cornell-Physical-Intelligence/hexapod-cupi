import ast,copy,hashlib,importlib.util,json,subprocess,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('epg',HERE/'launch_guarded_remote.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
class Tests(unittest.TestCase):
 def fixture(self,base):
  root=base/'direct_omni_train_extended_caps_002';(root/'jobs').mkdir(parents=True);pause=base/'forecast_pause_063';pause.mkdir()
  checkpoint=root/'train/policy/final.pt';checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'synthetic final checkpoint for CPU tests only');cp=g.sha(checkpoint)
  identity={'schema':'direct315_extended_native_v4','source_manifest_sha256':g.SOURCE_SHA256,'plan_sha256':g.PLAN_SHA256,'selection':g.PREVIOUS_SELECTION,'checkpoint_sha256':g.ORIGINAL_CHECKPOINT_SHA256,'actor_width':315,'critic_width':318}
  accepted={phase:{'phase':phase,'source_manifest_sha256':g.SOURCE_SHA256,'Stage2_complete':False,'passed' if phase=='standing' else 'complete':True} for phase in g.PHASES};accepted['train'].update(updates_completed=500,checkpoint_sha256=cp)
  campaign={'status':'completed','terminal_inputs_unchanged':True,'bounded_campaign_complete':True,'last_completed_phase':'final_stop','allocation':'extended','branch':'caps','planned_phases':list(g.PHASES),'accepted_phases':accepted,'PPO_updates_completed':500,'Stage2_complete':False,'automatic_continuation':False,'host_freeze_sha256':g.TRAIN_HOST_SHA256,'identity':identity}
  (root/'campaign.json').write_text(json.dumps(campaign))
  for i,phase in enumerate(g.PHASES):(root/'jobs'/(phase+'.json')).write_text(json.dumps({'phase':phase,'status':'completed','exit_code':0,'cleanup_checked':True,'container_name':'own'+str(i),'container_id':str(i)*64,'allocation':'extended','branch':'caps'}))
  (pause/'pause.json').write_text(json.dumps({'unit':g.PREVIOUS_UNIT,'output':str(root),'source':str(g.SOURCE),'units':{'active.timer':'ActiveState=active','inactive.timer':'ActiveState=inactive'}}))
  (pause/'restored.json').write_text(json.dumps({'restored_unix':123,'timers':['active.timer']}))
  pins={p.relative_to(base).as_posix():g.sha(p) for p in base.rglob('*.json')}
  data={'schema':'direct_extended_preview_selection_v1','pilot_name':root.name,'checkpoint_sha256':cp,'pilot_campaign_sha256':g.sha(root/'campaign.json'),'previous_owner':{'invocation':g.PREVIOUS_INVOCATION,'attempted_phases':list(g.PHASES),'pins':pins}}
  selection=base/'selection.json';selection.write_text(json.dumps(data));return selection,data,root
 def bind(self,base,path,root):
  self.addCleanup(patch.stopall);patch.object(g,'BASE',base).start();patch.object(g,'PREVIOUS_ROOT',root).start()
  g.bind_selection(['--selection',str(path),'--selection-sha256',g.sha(path)])
 def verify(self):
  with patch.object(g,'call',return_value='ActiveState=inactive\nMainPID=0\nExecMainStatus=0\nInvocationID='),patch.object(g.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','No such object')) as run:g.verify_previous_owner();return run.call_count
 def mutate_campaign(self,root,change):
  p=root/'campaign.json';d=json.loads(p.read_text());change(d);p.write_text(json.dumps(d));g.SELECTION['previous_owner']['pins'][p.relative_to(g.BASE).as_posix()]=g.sha(p)
 def test_pending_example_stops_before_process_or_pause(self):
  p=HERE/'SELECTION.example.json'
  with patch.object(g.subprocess,'run') as run,patch.object(g,'call') as call:
   with self.assertRaisesRegex(RuntimeError,'pending'):g.main(['--selection',str(p),'--selection-sha256',g.sha(p)])
  run.assert_not_called();call.assert_not_called()
 def test_exact_owner_twelve_absences_with_collected_invocation(self):
  with tempfile.TemporaryDirectory() as t:
   base=Path(t);path,data,root=self.fixture(base);self.bind(base,path,root);self.assertEqual(self.verify(),12)
 def test_partial_smoke_pilot_wrong_invocation_or_hash_reject_before_process(self):
  with tempfile.TemporaryDirectory() as t:
   path,data,root=self.fixture(Path(t))
   for change in (lambda d:d.update(pilot_name='direct_omni_train_smoke_004'),lambda d:d.update(pilot_name='direct_omni_train_pilot_caps_001'),lambda d:d.update(checkpoint_sha256='PENDING'),lambda d:d.update(pilot_campaign_sha256='a'*64),lambda d:d['previous_owner'].update(attempted_phases=list(g.PHASES[:-1])),lambda d:d['previous_owner'].update(invocation='b'*32)):
    d=copy.deepcopy(data);change(d);path.write_text(json.dumps(d))
    with patch.object(g.subprocess,'run') as run,patch.object(g,'call') as call,self.assertRaises(RuntimeError):g.bind_selection(['--selection',str(path),'--selection-sha256',g.sha(path)])
    run.assert_not_called();call.assert_not_called()
 def test_partial_failed_or_499_update_campaign_reject(self):
  for change in (lambda d:d.update(status='failed'),lambda d:d.update(PPO_updates_completed=499),lambda d:d['accepted_phases'].pop('final_stop'),lambda d:d['accepted_phases']['train'].update(updates_completed=499),lambda d:d.update(last_completed_phase='train')):
   with tempfile.TemporaryDirectory() as t:
    base=Path(t);path,data,root=self.fixture(base);self.bind(base,path,root);self.mutate_campaign(root,change)
    with self.assertRaises(RuntimeError):self.verify()
 def test_wrong_source_native_schema_host_original_or_branch_reject(self):
  for change in (lambda d:d['identity'].update(source_manifest_sha256='a'*64),lambda d:d['identity'].update(schema='direct315_native_v2'),lambda d:d.update(host_freeze_sha256='a'*64),lambda d:d['identity'].update(checkpoint_sha256='a'*64),lambda d:d['identity']['selection'].update(updates=50),lambda d:d.update(branch='quiet_priority')):
   with tempfile.TemporaryDirectory() as t:
    base=Path(t);path,data,root=self.fixture(base);self.bind(base,path,root);self.mutate_campaign(root,change)
    with self.assertRaises(RuntimeError):self.verify()
 def test_changed_checkpoint_rejects_even_when_receipts_match(self):
  with tempfile.TemporaryDirectory() as t:
   base=Path(t);path,data,root=self.fixture(base);self.bind(base,path,root);g.CHECKPOINT.write_bytes(b'changed')
   with self.assertRaisesRegex(RuntimeError,'checkpoint'):self.verify()
 def test_live_failed_or_wrong_invocation_reject(self):
  with tempfile.TemporaryDirectory() as t:
   base=Path(t);path,data,root=self.fixture(base);self.bind(base,path,root)
   for state in ('ActiveState=active\nMainPID=1\nExecMainStatus=0','ActiveState=failed\nMainPID=0\nExecMainStatus=1','ActiveState=inactive\nMainPID=0\nExecMainStatus=0\nInvocationID='+'d'*32):
    with patch.object(g,'call',return_value=state),self.assertRaises(RuntimeError):g.verify_previous_owner()
 def test_unknown_docker_and_missing_timer_reject(self):
  with tempfile.TemporaryDirectory() as t:
   base=Path(t);path,data,root=self.fixture(base);self.bind(base,path,root)
   with patch.object(g,'call',return_value='ActiveState=inactive\nMainPID=0\nExecMainStatus=0'),patch.object(g.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','daemon unavailable')),self.assertRaisesRegex(RuntimeError,'absence'):g.verify_previous_owner()
   p=base/'forecast_pause_063/restored.json';p.write_text(json.dumps({'restored_unix':123,'timers':[]}));g.SELECTION['previous_owner']['pins'][p.relative_to(base).as_posix()]=g.sha(p)
   with self.assertRaisesRegex(RuntimeError,'restoration'):self.verify()
 def test_extra_job_or_duplicate_owned_id_reject(self):
  for extra in (True,False):
   with tempfile.TemporaryDirectory() as t:
    base=Path(t);path,data,root=self.fixture(base);self.bind(base,path,root)
    if extra:(root/'jobs/extra.json').write_text('{}')
    else:
     p=root/'jobs/train.json';d=json.loads(p.read_text());d['container_id']='0'*64;p.write_text(json.dumps(d));g.SELECTION['previous_owner']['pins'][p.relative_to(base).as_posix()]=g.sha(p)
    with self.assertRaises(RuntimeError):self.verify()
 def test_restorer_byte_equal_original_deadlines_and_locks(self):
  def embedded(p):return next(n.args[0].value for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='restorer' and n.func.attr=='write_text')
  self.assertEqual(embedded(HERE/'launch_guarded_remote.py'),embedded(HERE/'inputs/preview_guard001_parent.py'))
  text=(HERE/'launch_guarded_remote.py').read_text()
  for value in ('RuntimeMaxSec=720','TimeoutStopSec=180','--on-active=15m','/opt/wx/gpu.lock','/tmp/hexapod-isaac-gpu.lock','35af5100e39980f55a3f5e19654e4550b3c3fb6da114b64fbe0d143817242cf3'):self.assertIn(value,text)
 def test_reused_output_no_external_calls(self):
  with tempfile.TemporaryDirectory() as t:
   out=Path(t)/'out';out.mkdir()
   with patch.object(g,'bind_selection'),patch.object(g,'OUTPUT',out),patch.object(g,'PAUSE',Path(t)/'pause'),patch.object(g.subprocess,'run') as run,patch.object(g,'call') as call:
    with self.assertRaisesRegex(RuntimeError,'Fresh'):g.main()
   run.assert_not_called();call.assert_not_called()
 def test_host_admission_500_and_exact_checkpoint(self):
  expected={'source_manifest_sha256':g.SOURCE_SHA256,'checkpoint_sha256':'a'*64,'pilot':{'campaign_sha256':'b'*64,'allocation':'extended','branch':'caps','completed_updates':500}}
  with patch.object(g,'require_final_bindings'),patch.object(g,'sha',side_effect=lambda p:g.SUPERVISOR_SHA256 if p==g.SUPERVISOR_SOURCE/'campaign_source_hashes.json' else g.SOURCE_SHA256 if p==g.SOURCE/'campaign_source_hashes.json' else g.HOST_SHA256),patch.object(g,'verify_frozen'),patch.object(g,'CHECKPOINT_SHA256','a'*64),patch.object(g,'CAMPAIGN_SHA256','b'*64):
   host=SimpleNamespace(require_fresh_output=lambda args:None,verify_inputs=lambda args:expected)
   self.assertEqual(g.validate_preview_inputs(host),expected)
   for field,value in [('completed_updates',50),('allocation','pilot'),('branch','quiet_priority')]:
    bad=copy.deepcopy(expected);bad['pilot'][field]=value
    with self.assertRaisesRegex(RuntimeError,'admitted'):g.validate_preview_inputs(SimpleNamespace(require_fresh_output=lambda args:None,verify_inputs=lambda args:bad))
if __name__=='__main__':unittest.main()
