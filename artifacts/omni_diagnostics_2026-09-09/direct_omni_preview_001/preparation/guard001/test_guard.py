import ast,importlib.util,json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('pg',HERE/'launch_guarded_remote.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
class Tests(unittest.TestCase):
 def fixture(self,base):
  root=base/'direct_omni_train_pilot_caps_001';(root/'jobs').mkdir(parents=True);pause=base/'forecast_pause_057';pause.mkdir()
  campaign={'status':'completed','terminal_inputs_unchanged':True,'allocation':'pilot','branch':'caps','planned_phases':list(g.PHASES),'host_freeze_sha256':g.TRAIN_HOST_SHA256,'identity':{'source_manifest_sha256':g.SOURCE_SHA256}}
  (root/'campaign.json').write_text(json.dumps(campaign))
  for i,phase in enumerate(g.PHASES):(root/'jobs'/ (phase+'.json')).write_text(json.dumps({'phase':phase,'status':'completed','cleanup_checked':True,'container_name':'own'+str(i),'container_id':str(i)*64}))
  (pause/'pause.json').write_text(json.dumps({'unit':g.PREVIOUS_UNIT,'output':str(root),'units':{'active.timer':'ActiveState=active','inactive.timer':'ActiveState=inactive'}}))
  (pause/'restored.json').write_text(json.dumps({'restored_unix':123,'timers':['active.timer']}))
  pins={p.relative_to(base).as_posix():g.sha(p) for p in base.rglob('*.json')}
  data={'schema':'direct_preview_selection_v1','pilot_name':'direct_omni_train_pilot_curriculum_001','checkpoint_sha256':'a'*64,'pilot_campaign_sha256':'b'*64,'previous_owner':{'invocation':'c'*32,'attempted_phases':list(g.PHASES),'pins':pins}}
  selection=base/'selection.json';selection.write_text(json.dumps(data));return selection,data,root
 def test_unbound_selection_stops_before_any_process_or_pause(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'selection.json';p.write_text('{}')
   with patch.object(g.subprocess,'run') as run,patch.object(g,'call') as call:
    with self.assertRaises(RuntimeError):g.main(['--selection',str(p),'--selection-sha256','PENDING'])
   run.assert_not_called();call.assert_not_called()
 def test_exact_owner_all_twelve_absences_and_collected_invocation(self):
  with tempfile.TemporaryDirectory() as t:
   base=Path(t);path,data,root=self.fixture(base)
   with patch.object(g,'BASE',base),patch.object(g,'PREVIOUS_ROOT',root),patch.object(g,'call',return_value='ActiveState=inactive\nInvocationID='),patch.object(g.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','No such object')) as run:
    g.bind_selection(['--selection',str(path),'--selection-sha256',g.sha(path)]);g.verify_previous_owner()
   self.assertEqual(run.call_count,12)
 def test_smoke_or_changed_checkpoint_receipt_reject(self):
  with tempfile.TemporaryDirectory() as t:
   path,data,_=self.fixture(Path(t));data['pilot_name']='direct_omni_train_smoke_002';path.write_text(json.dumps(data))
   with self.assertRaisesRegex(RuntimeError,'pilot selection'):g.bind_selection(['--selection',str(path),'--selection-sha256',g.sha(path)])
   data['pilot_name']='direct_omni_train_pilot_caps_001';data['checkpoint_sha256']='PENDING';path.write_text(json.dumps(data))
   with self.assertRaisesRegex(RuntimeError,'pending'):g.bind_selection(['--selection',str(path),'--selection-sha256',g.sha(path)])
 def test_unknown_docker_absence_and_missing_timer_do_not_admit(self):
  with tempfile.TemporaryDirectory() as t:
   base=Path(t);path,data,root=self.fixture(base)
   with patch.object(g,'BASE',base),patch.object(g,'PREVIOUS_ROOT',root),patch.object(g,'call',return_value='ActiveState=inactive\nInvocationID='):
    g.bind_selection(['--selection',str(path),'--selection-sha256',g.sha(path)])
    with patch.object(g.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','daemon unavailable')):
     with self.assertRaisesRegex(RuntimeError,'absence'):g.verify_previous_owner()
    restore=base/'forecast_pause_057/restored.json';restore.write_text(json.dumps({'restored_unix':123,'timers':[]}));g.SELECTION['previous_owner']['pins']['forecast_pause_057/restored.json']=g.sha(restore)
    with self.assertRaisesRegex(RuntimeError,'restoration'):g.verify_previous_owner()
 def test_prior_active_or_wrong_invocation_reject(self):
  with tempfile.TemporaryDirectory() as t:
   base=Path(t);path,data,root=self.fixture(base)
   with patch.object(g,'BASE',base),patch.object(g,'PREVIOUS_ROOT',root):
    g.bind_selection(['--selection',str(path),'--selection-sha256',g.sha(path)])
    for state in ('ActiveState=active\nInvocationID=','ActiveState=inactive\nInvocationID='+'d'*32):
     with patch.object(g,'call',return_value=state),self.assertRaises(RuntimeError):g.verify_previous_owner()
 def test_restorer_byte_equal_and_original_lock_deadlines(self):
  def embedded(p):
   return next(n.args[0].value for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='restorer' and n.func.attr=='write_text')
  self.assertEqual(embedded(HERE/'launch_guarded_remote.py'),embedded(HERE/'inputs/guard003_parent.py'))
  text=(HERE/'launch_guarded_remote.py').read_text()
  for value in ('RuntimeMaxSec=720','TimeoutStopSec=180','--on-active=15m','/opt/wx/gpu.lock','/tmp/hexapod-isaac-gpu.lock','22c615d67c6c4dc14ab370b291f8eddd72f8475ad34118b91a4237a34ee05fab'):self.assertIn(value,text)
 def test_reused_output_no_external_calls(self):
  with tempfile.TemporaryDirectory() as t:
   out=Path(t)/'out';out.mkdir()
   with patch.object(g,'bind_selection'),patch.object(g,'OUTPUT',out),patch.object(g,'PAUSE',Path(t)/'pause'),patch.object(g.subprocess,'run') as run,patch.object(g,'call') as call:
    with self.assertRaisesRegex(RuntimeError,'Fresh'):g.main()
   run.assert_not_called();call.assert_not_called()
if __name__=='__main__':unittest.main()
