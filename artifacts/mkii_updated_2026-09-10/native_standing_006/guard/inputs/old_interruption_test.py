import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('interrupt_guard',HERE/'launch_guarded_remote.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
class InterruptionEvidence(unittest.TestCase):
 def test_actual_cause_is_shared_gpu_interruption_without_physical_rejection(self):
  p=HERE/'inputs/previous_owner';c=json.loads((p/'run/campaign.json').read_text());j=json.loads((p/'run/jobs/standing.json').read_text());state=json.loads((p/'run/standing/state.json').read_text())
  self.assertEqual((c['status'],j['status'],state['status']),('failed','failed','running'));self.assertEqual(state['explicit_steps_completed'],0);self.assertEqual(state['checks'],{})
  self.assertEqual(c['error'],g.PREVIOUS_ERROR);self.assertEqual(j['competitors'],g.PREVIOUS_COMPETITORS);self.assertIn('ithaca-reconstruction',j['competitors'][0]['process']);self.assertNotIn('exit_code',j)
 def test_all31_full_raw_pins_retained_without_bulk_copy(self):
  m=json.loads((HERE/'inputs/PRIOR_RAW_INVENTORY.json').read_text());self.assertEqual(len(m),31);self.assertEqual(sum(v['size_bytes'] for v in m.values()),144818179)
  expected={k.replace('run/','native_standing_003/',1) if k.startswith('run/') else k.replace('forecast_pause/','forecast_pause_009/',1):v['sha256'] for k,v in m.items()};self.assertEqual(g.PRIOR_PINS,expected)
  self.assertFalse(any(HERE.rglob('*.npz')));self.assertFalse((HERE/'inputs/previous_owner/run/standing/contacts.jsonl').exists())
 def test_same_source_host_and_single_environment_no_admission_reuse(self):
  self.assertEqual(g.SOURCE_SHA256,g.PREVIOUS_SOURCE);self.assertNotEqual(g.HOST_FREEZE_SHA256,g.PREVIOUS_HOST);self.assertNotIn("'--standing-one'",(HERE/'launch_guarded_remote.py').read_text())
  self.assertEqual(g.PAUSE.name,'forecast_pause_011');self.assertEqual(g.OUTPUT.name,'native_standing_004')
 def test_streamed_hash_exact(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'x';p.write_bytes(b'x'*9000000);self.assertEqual(g.sha(p),hashlib.sha256(p.read_bytes()).hexdigest())
if __name__=='__main__':unittest.main()
