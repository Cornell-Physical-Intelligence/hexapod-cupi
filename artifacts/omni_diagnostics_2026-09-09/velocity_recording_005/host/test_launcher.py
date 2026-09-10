from pathlib import Path
import importlib.util,json,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch,Mock
spec=importlib.util.spec_from_file_location('recordhost',Path(__file__).with_name('launch_recording_spark.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class TestHost(unittest.TestCase):
 def video(self,p):
  (p/'rollout.mp4').write_bytes(b'fixture-not-real-video')
  d={'checkpoint_sha256':m.CHECKPOINT_SHA,'strict_tensor_readback_completed':True,'stage2_complete':False,'qualification_performed':False,'runtime_binding':{'runtime_tree_sha256':m.RUNTIME_TREE},'frames':5,'complete':True,'terminal_event':None,'video_sha256':m.digest(p/'rollout.mp4'),'source_and_checkpoint_reverified_after_recording':True,'recorded_control_steps':1700,'planned_control_steps':1700}
  (p/'video.json').write_text(json.dumps(d));return d
 def test_accepts_video_report_without_train_state(self):
  with tempfile.TemporaryDirectory() as x:
   p=Path(x);self.video(p);self.assertTrue(m.validate_recording(p,0)['complete'])
   (p/'failure.json').write_text('{}')
   with self.assertRaises(RuntimeError):m.validate_recording(p,0)
 def test_short_clip_requires_terminal_evidence_and_strict_identity(self):
  with tempfile.TemporaryDirectory() as x:
   p=Path(x);d=self.video(p);d['complete']=False;(p/'video.json').write_text(json.dumps(d))
   with self.assertRaises(ValueError):m.validate_recording(p,0)
   d['terminal_event']={'termination':True};(p/'video.json').write_text(json.dumps(d));self.assertFalse(m.validate_recording(p,0)['complete'])
   d['strict_tensor_readback_completed']=False;(p/'video.json').write_text(json.dumps(d))
   with self.assertRaises(ValueError):m.validate_recording(p,0)
 def test_readonly_inputs_full_admitted_package_and_rgb(self):
  a=SimpleNamespace(source=Path('/source'),output=Path('/output'),probe=Path('/probe'),pilot=Path('/pilot'),adapter=Path('/adapter'))
  cmd=m.command(a,'owner')
  for v in ('/source:/workspace/hexapod:ro','/probe:/probe:ro','/pilot:/pilot:ro','/adapter:/recording:ro','/output:/outputs:rw'):self.assertIn(v,cmd)
  self.assertEqual(cmd[cmd.index('--package')+1],'/pilot/inputs/study');self.assertIn('--enable_cameras',cmd);self.assertNotIn('--iterations',cmd)
 def test_stopped_before_lock_never_launches(self):
  with tempfile.TemporaryDirectory() as x:
   p=Path(x);(p/'jobs').mkdir();(p/'stop.request').touch()
   with patch.object(m,'save',m.write,create=True),patch.object(m.subprocess,'Popen') as popen:
    with self.assertRaises(InterruptedError):m.run_owned(SimpleNamespace(output=p),'recording')
    popen.assert_not_called()
 def test_wrong_container_identity_never_signaled(self):
  with patch.object(m.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='wrong /owner true')):
   with self.assertRaises(RuntimeError):m.owned_container('owner','expected')
if __name__=='__main__':unittest.main()
