import hashlib,importlib.util,json,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('bound_auditor',HERE/'audit_remote.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
class MaskBinding(unittest.TestCase):
 def test_frozen_guard_manifest_and_policy_are_the_actual_reviewed_successor(self):
  root=HERE.parent/'canonical_native_standing32_guard_005';freeze=root/'FREEZE_SHA256.json';m=json.loads(freeze.read_text());pin=next(p for p in a.PINS if p[0]=='guard')
  self.assertEqual(hashlib.sha256(freeze.read_bytes()).hexdigest(),pin[3]);self.assertEqual(len(m),pin[4]);self.assertEqual(m['launch_guarded_remote.py'],'e80490c0f46aed03610e16d5140fc80a2376536409ced4715a1191b3267b5d10');self.assertEqual(m['inputs/reservation_policy.json'],'dcb623201a9ba63d50865ce1bcd527eb7844c0b65b629683582c747516e7d607')
  self.assertEqual(a.GUARD.name,root.name.removeprefix('canonical_native_'));self.assertEqual(a.INV,'c5397bd5a8dd4640899ad3dec2c30f47')
if __name__=='__main__':unittest.main()
