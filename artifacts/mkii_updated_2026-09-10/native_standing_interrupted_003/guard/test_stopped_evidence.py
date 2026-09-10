import ast,hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('stopped_guard',HERE/'launch_guarded_remote.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
class StoppedEvidence(unittest.TestCase):
 def test_actual_stopped32_has_unfinalized_state_and_no_fabricated_counter(self):
  p=HERE/'inputs/previous_owner';c=json.loads((p/'run/campaign.json').read_text());j=json.loads((p/'run/jobs/standing.json').read_text());state=json.loads((p/'run/standing/state.json').read_text())
  self.assertEqual((c['status'],j['status'],state['status']),('stopped','stopped','running'));self.assertEqual(state['explicit_steps_completed'],0);self.assertEqual(state['checks'],{});self.assertEqual(j['error'],"InterruptedError('Stop requested')")
  self.assertNotIn('exit_code',j);self.assertEqual(c['identity']['num_envs'],32)
  text=(HERE/'launch_guarded_remote.py').read_text();self.assertNotIn("standing/session.json",text);self.assertIn("native.get('status')!='running'",text)
 def test_all27_external_raw_hashes_retained_no_large_duplication(self):
  m=json.loads((HERE/'inputs/PRIOR_RAW_INVENTORY.json').read_text());self.assertEqual(len(m),27);self.assertEqual(sum(v['size_bytes'] for v in m.values()),1836888673)
  expected={k.replace('run/','native_standing32_001/',1) if k.startswith('run/') else k.replace('forecast_pause/','forecast_pause_008/',1):v['sha256'] for k,v in m.items()};self.assertEqual(g.PRIOR_PINS,expected)
  self.assertFalse((HERE/'inputs/previous_owner/run/standing/contacts.jsonl').exists());self.assertFalse(any(HERE.rglob('*.npz')))
 def test_streaming_hash_is_exact(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t)/'data';p.write_bytes(b'abc\x00'*2500000);self.assertEqual(g.sha(p),hashlib.sha256(p.read_bytes()).hexdigest())
 def test_no_standing002_admission_reuse_for_new_source(self):
  self.assertEqual(g.SOURCE.name,'standing_source_003');self.assertEqual(g.HOST.parent.name,'standing_host_003');self.assertNotEqual(g.SOURCE_SHA256,g.PREVIOUS_SOURCE)
  tree=ast.parse((HERE/'launch_guarded_remote.py').read_text());self.assertFalse(any(isinstance(n,ast.Constant) and n.value=='--standing-one' for n in ast.walk(tree)))
if __name__=='__main__':unittest.main()
