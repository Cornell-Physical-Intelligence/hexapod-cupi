"""Run the actual inherited supervisor through normal/error/cleanup paths; no real processes."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import importlib.util,json,subprocess,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[2]
HOST=ROOT/'tmp/direct_omni_cold_host_001'
sys.path.insert(0,str(HOST))
import launch_cold_spark as h
SUPERVISOR=ROOT/'tmp/reference_physics_adapter_009/source_009'
CONTRACT=ROOT/'tmp/direct_omni_recovery_001/baseline'
spec=importlib.util.spec_from_file_location('independent_cold_contract',CONTRACT/'cold_contract.py');contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)

class ActualRunOwnedTests(unittest.TestCase):
 def exercise(self,phase='standing',*,ready=True,wrong_runtime=False,contact_warning=False,instant_exit=False):
  temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);out=Path(temp.name)
  for k in ['jobs','logs',phase]:(out/k).mkdir()
  coord=out/'coordination';coord.write_text('fixed')
  a=SimpleNamespace(source=Path('/exact/coldsource'),checkpoint=Path('/exact/original.pt'),contract=CONTRACT,supervisor_source=SUPERVISOR,output=out,isaaclab=Path('/IsaacLab'),coordination_sha256=h.sha(coord))
  h.CONTRACT=contract;p=h.load_supervisor(a);calls=[];state={'name':None,'running':True};clock=[0]
  class Process:
   returncode=0
   def __init__(self):self.polls=0
   def poll(self):
    self.polls+=1
    if self.polls==1 and not instant_exit:return None
    return self.returncode
   def terminate(self):calls.append('terminate-client')
   def wait(self,timeout):return 0
   def kill(self):calls.append('kill-client')
  process=Process()
  def popen(cmd,**kwargs):
   calls.append(('command',cmd));state['name']=cmd[cmd.index('--name')+1]
   assert '/exact/coldsource:/source:ro' in cmd
   assert '/exact/original.pt:/checkpoint/original.pt:ro' in cmd
   assert cmd[cmd.index('--')+1:]==contract.runtime_arguments(phase)
   if phase=='baseline':assert str(out/'standing')+':/output/standing:ro' in cmd
   kwargs['stdout'].write(('REFERENCE_SCREEN_APP_READY\n' if ready else 'still in AppLauncher\n')+('Contact sensor: buffer overflow\n' if contact_warning else ''))
   kwargs['stdout'].flush()
   raw={'status':'completed','runtime_binding':{'runtime_tree_sha256':'wrong' if wrong_runtime else h.LEGACY_RUNTIME_TREE}}
   if phase=='baseline':raw['checkpoint_sha256']=contract.CHECKPOINT
   (out/phase/'state.json').write_text(json.dumps(raw))
   return process
  def run(cmd,**kwargs):
   calls.append(tuple(cmd[:2]))
   if cmd[:2]==['docker','inspect']:
    return subprocess.CompletedProcess(cmd,0,'a'*64+' /'+state['name']+' '+str(state['running']).lower(),'')
   if cmd[:2]==['docker','top']:return subprocess.CompletedProcess(cmd,0,'PID\n123\n','')
   if cmd[:2]==['docker','stop']:state['running']=False;return subprocess.CompletedProcess(cmd,0,'','')
   raise AssertionError('Unexpected subprocess '+str(cmd))
  def monotonic():clock[0]+=1;return 0 if clock[0]<=2 else 100
  error=None;result=None
  with patch.object(p,'COORDINATION',coord),patch.object(p,'preflight',return_value={'no_competitors':True}),patch.object(h,'verify_tree') as vt,patch.object(h,'verify_legacy_runtime') as vr,patch.object(p.os,'open',return_value=7),patch.object(p.os,'close') as close,patch.object(p.fcntl,'flock'),patch.object(p.subprocess,'Popen',side_effect=popen),patch.object(p.subprocess,'run',side_effect=run),patch.object(p,'resources',return_value=('',32*1024**3)),patch.object(p.time,'monotonic',side_effect=monotonic),patch.object(p.time,'sleep'):
   try:result=p.run_owned(a,phase)
   except Exception as e:error=e
   self.assertEqual(vt.call_args.args,(a.source,'campaign_source_hashes.json',h.SOURCE_MAP));vr.assert_called_once_with(a.source)
   self.assertEqual(close.call_count,2)
  job=json.loads((out/'jobs'/f'{phase}.json').read_text())
  self.assertTrue(job['cleanup_checked']);self.assertFalse(state['running'])
  return result,error,job,calls
 def test_complete_standing_real_supervisor_loop(self):
  result,error,job,calls=self.exercise();self.assertIsNone(error);self.assertEqual(job['status'],'completed');self.assertTrue(job['no_policy_loaded']);self.assertIn(('docker','stop'),calls)
 def test_complete_baseline_exact_checkpoint_metadata(self):
  result,error,job,calls=self.exercise('baseline');self.assertIsNone(error);self.assertFalse(job['no_policy_loaded']);self.assertTrue(job['checkpoint_load_verified_by_completed_state'])
 def test_no_AppReady_crossing90_rejects_and_cleans(self):
  _,error,job,_=self.exercise(ready=False);self.assertIsInstance(error,TimeoutError);self.assertEqual(job['startup_failure_kind'],'no_reference_AppReady_by90s')
 def test_wrong_runtime_in_completed_state_rejects_and_cleans(self):
  _,error,job,_=self.exercise(wrong_runtime=True);self.assertIn('Runtime identity',str(error));self.assertEqual(job['status'],'failed')
 def test_exited_client_recovers_and_stops_owned_container(self):
  _,error,job,calls=self.exercise(instant_exit=True);self.assertIsNone(error);self.assertEqual(job['container_id'],'a'*64);self.assertIn(('docker','stop'),calls)

if __name__=='__main__':unittest.main()
