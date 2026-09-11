import hashlib,json,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
class InstalledReceipt(unittest.TestCase):
 def test_actual_installed_readback_binds_current_runtime_and_policy(self):
  r=json.loads((HERE/'inputs/reservation_probe002.json').read_text())
  self.assertTrue(r['verified']);self.assertEqual(r['source_sha256'],hashlib.sha256((HERE/'launch_guarded_remote.py').read_bytes()).hexdigest());self.assertEqual(r['policy_sha256'],hashlib.sha256((HERE/'inputs/reservation_policy.json').read_bytes()).hexdigest())
  self.assertEqual(r['native_calls'],0);self.assertFalse(r['output_created']);self.assertFalse(r['pause_created']);self.assertEqual(r['reservation']['masked_user_units'],31);self.assertTrue(r['reservation']['queue_lock_held']);self.assertEqual(r['reservation']['queue_lock_owner_pid'],3814169)
  self.assertEqual((HERE/'inputs/reservation_probe002.stderr').read_bytes(),b'')
  mapping=json.loads((HERE/'inputs/ACTUAL_RESERVATION_PROBE_FREEZE_SHA256.json').read_text());self.assertEqual(mapping['launch_guarded_remote.py'],r['source_sha256']);self.assertEqual(mapping['inputs/reservation_policy.json'],r['policy_sha256'])
if __name__=='__main__':unittest.main()
