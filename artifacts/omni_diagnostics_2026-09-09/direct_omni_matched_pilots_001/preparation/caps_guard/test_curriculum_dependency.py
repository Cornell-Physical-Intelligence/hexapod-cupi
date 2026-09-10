"""Curriculum is an ownership dependency; a poor/failed result is not a CAPS veto."""
import importlib.util,json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('caps_owner',HERE/'launch_guarded_remote.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
class CurriculumDependency(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name);self.root=self.base/'direct_omni_train_pilot_curriculum_001'
  self.attempted=('standing','initial_constant','initial_stop','train')
  self.docs={'direct_omni_train_pilot_curriculum_001/campaign.json':{'status':'failed','terminal_inputs_unchanged':True,'allocation':'pilot','branch':'curriculum','planned_phases':list(g.PREVIOUS_PHASES),'host_freeze_sha256':g.HOST_FREEZE_SHA256,'identity':{'source_manifest_sha256':g.SOURCE_SHA256},'PPO_updates_completed':0,'Stage2_complete':False,'accepted_phases':{'standing':{}},'quality_passed':False},'forecast_pause_056/pause.json':{'unit':g.PREVIOUS_UNIT,'output':str(self.root),'units':{'weather.timer':'ActiveState=active'}},'forecast_pause_056/restored.json':{'timers':['weather.timer'],'restored_unix':123}}
  for i,phase in enumerate(self.attempted):self.docs['direct_omni_train_pilot_curriculum_001/jobs/'+phase+'.json']={'phase':phase,'status':'failed' if phase=='train' else 'completed','cleanup_checked':True,'container_name':'curriculum-'+phase,'container_id':str(i+1)*64}
  self.pins={}
  for name,data in self.docs.items():
   p=self.base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data));self.pins[name]=g.sha(p)
  for key,value in [('BASE',self.base),('PREVIOUS_ROOT',self.root),('PRIOR_PINS',self.pins),('PREVIOUS_ATTEMPTED_PHASES',self.attempted)]:
   p=patch.object(g,key,value);p.start();self.addCleanup(p.stop)
 def invoke(self,state=None,inspect=None):
  state=state or 'ActiveState=failed\nSubState=failed\nInvocationID='+g.PREVIOUS_INVOCATION
  inspect=inspect or subprocess.CompletedProcess([],1,'','No such object')
  with patch.object(g,'verify_smoke_owner') as smoke,patch.object(g,'call',return_value=state),patch.object(g.subprocess,'run',return_value=inspect) as run:
   g.verify_previous_owner();smoke.assert_called_once();self.assertEqual(run.call_count,2*len(self.attempted))
 def edit(self,name,change):
  p=self.base/name;d=json.loads(p.read_text());d.update(change);p.write_text(json.dumps(d));self.pins[name]=g.sha(p)
 def test_failed_unqualified_curriculum_with_zero_updates_is_not_a_quality_veto(self):self.invoke()
 def test_completed_curriculum_with_failed_quality_is_not_a_quality_veto(self):
  self.edit('direct_omni_train_pilot_curriculum_001/campaign.json',{'status':'completed','PPO_updates_completed':50,'quality_passed':False})
  self.invoke('ActiveState=inactive\nInvocationID=')
 def test_changed_input_integrity_or_wrong_branch_rejects(self):
  name='direct_omni_train_pilot_curriculum_001/campaign.json'
  self.edit(name,{'terminal_inputs_unchanged':False})
  with self.assertRaisesRegex(RuntimeError,'unchanged'):self.invoke()
  self.edit(name,{'terminal_inputs_unchanged':True,'branch':'caps'})
  with self.assertRaisesRegex(RuntimeError,'lineage'):self.invoke()
 def test_unpinned_extra_job_and_missing_cleanup_reject(self):
  path=self.root/'jobs/final_constant.json';path.write_text('{}')
  with self.assertRaisesRegex(RuntimeError,'inventory'):self.invoke()
  path.unlink();self.edit('direct_omni_train_pilot_curriculum_001/jobs/train.json',{'cleanup_checked':False})
  with self.assertRaisesRegex(RuntimeError,'cleanup'):self.invoke()
 def test_busy_owner_wrong_invocation_and_unknown_container_reject(self):
  for state in ['ActiveState=active\nInvocationID='+g.PREVIOUS_INVOCATION,'ActiveState=inactive\nInvocationID='+'e'*32]:
   with self.assertRaises(RuntimeError):self.invoke(state)
  with self.assertRaisesRegex(RuntimeError,'not proven absent'):self.invoke(inspect=subprocess.CompletedProcess([],1,'','Docker daemon unavailable'))
 def test_wrong_timer_restore_rejects(self):
  self.edit('forecast_pause_056/restored.json',{'timers':[]})
  with self.assertRaisesRegex(RuntimeError,'timer restoration'):self.invoke()
if __name__=='__main__':unittest.main()
