import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('interrupt_guard',HERE/'launch_guarded_remote.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
class InterruptionEvidence(unittest.TestCase):
 def test_actual_quality_failure_and_independent_host_timeout_preserved(self):
  p=HERE/'inputs/previous_owner';c=json.loads((p/'run/campaign.json').read_text());j=json.loads((p/'run/jobs/standing.json').read_text());state=json.loads((p/'run/standing/state.json').read_text())
  self.assertEqual((c['status'],j['status'],state['status']),('failed','failed','completed'));self.assertEqual(state['explicit_steps_completed'],8000);self.assertFalse(state['standing_pass']);self.assertTrue(all(state['checks'].values()))
  self.assertEqual(c['error'],g.PREVIOUS_ERROR);self.assertEqual(j['error'],g.PREVIOUS_ERROR);self.assertIn('ten-minute bound',j['error']);self.assertNotIn('exit_code',j)
 def test_all36_full_raw_pins_retained_without_bulk_copy(self):
  m=json.loads((HERE/'inputs/PRIOR_RAW_INVENTORY.json').read_text());self.assertEqual(len(m),36);self.assertEqual(sum(v['size_bytes'] for v in m.values()),4969155341)
  expected={k.replace('run/','native_standing32_002/',1) if k.startswith('run/') else k.replace('forecast_pause/','forecast_pause_010/',1):v['sha256'] for k,v in m.items()};self.assertEqual(g.PRIOR_PINS,expected)
  self.assertFalse(any(HERE.rglob('*.npz')));self.assertFalse((HERE/'inputs/previous_owner/run/standing/contacts.jsonl').exists())
 def test_new_source_and_single_environment_no_admission_reuse(self):
  self.assertNotEqual(g.SOURCE_SHA256,g.PREVIOUS_SOURCE);self.assertNotEqual(g.HOST_FREEZE_SHA256,g.PREVIOUS_HOST);self.assertNotIn("'--standing-one'",(HERE/'launch_guarded_remote.py').read_text())
  self.assertEqual(g.PAUSE.name,'forecast_pause_013');self.assertEqual(g.OUTPUT.name,'native_standing_005')
 def test_streamed_hash_exact(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'x';p.write_bytes(b'x'*9000000);self.assertEqual(g.sha(p),hashlib.sha256(p.read_bytes()).hexdigest())
if __name__=='__main__':unittest.main()
