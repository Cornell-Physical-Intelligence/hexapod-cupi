from pathlib import Path
import importlib.util,json,tempfile,unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('guardreservation',HERE/'launch_guarded_remote.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
class Reservation(unittest.TestCase):
 def test_actual13_readback_matches_bound_marker_and_all10_dropins(self):
  d=json.loads((HERE/'inputs/reservation_readback.json').read_text())['files'];self.assertEqual(len(d),13);self.assertEqual(g.RESERVATION_PINS,{k:v['sha256']for k,v in d.items()})
  for i,(name,v) in enumerate(d.items()):self.assertEqual(g.sha(HERE/'inputs/reservation_files'/str(i)),v['sha256'])
  self.assertTrue(json.loads(d[str(g.RESERVATION/'ACTIVE')]['text'])['exclusive']);self.assertEqual(sum(k.endswith('.conf')for k in d),10)
 def test_marker_or_dropin_changed_fails_without_mutation(self):
  with tempfile.TemporaryDirectory()as t:
   base=Path(t);(base/'ACTIVE').write_text('{"exclusive":true}');drop=base/'drop';drop.write_text('exact');pins={str(p):g.sha(p) for p in [base/'ACTIVE',drop]}
   with patch.multiple(g,RESERVATION=base,RESERVATION_PINS=pins),patch.object(g.subprocess,'run')as run:
    self.assertTrue(g.verify_reservation()['reservation_active_at_preflight']);drop.write_text('changed')
    with self.assertRaises(RuntimeError):g.verify_reservation()
    run.assert_not_called()
 def test_loaded_dropin_or_active_scheduler_mismatch_rejects(self):
  with tempfile.TemporaryDirectory()as t:
   base=Path(t);(base/'ACTIVE').write_text('{"exclusive":true}');d=base/'covered.timer.d';d.mkdir();drop=d/'owned.conf';drop.write_text('exact');pins={str(p):g.sha(p)for p in [base/'ACTIVE',drop]}
   with patch.multiple(g,RESERVATION=base,RESERVATION_PINS=pins),patch.object(g,'call')as call:
    call.return_value='DropInPaths='+str(drop)+'\nNeedDaemonReload=no\nActiveState=inactive'
    self.assertTrue(g.verify_reservation()['reservation_active_at_preflight'])
    for bad in ['DropInPaths='+str(drop)+'\nNeedDaemonReload=yes\nActiveState=inactive','DropInPaths=other\nNeedDaemonReload=no\nActiveState=inactive','DropInPaths='+str(drop)+'\nNeedDaemonReload=no\nActiveState=active']:
     call.return_value=bad
     with self.assertRaises(RuntimeError):g.verify_reservation()
 def test_restorer_metadata_does_not_claim_persistent_release(self):
  text=(HERE/'launch_guarded_remote.py').read_text();self.assertIn("'scope':'per_job_snapshot_only'",text);self.assertIn("'persistent_reservation_release_attempted':False",text)
  self.assertNotIn("unlink",text);self.assertNotIn("daemon-reload",text)
if __name__=='__main__':unittest.main()
