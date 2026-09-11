import importlib.util,json,hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('a32',HERE/'audit_remote.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
class StandingOneTests(unittest.TestCase):
 def test_exact_actual_guard_map_and_malformed_or_missing_admission(self):
  guard=HERE.parent/'canonical_native_standing32_guard_003';raw=json.loads((guard/'inputs/PRIOR_RAW_INVENTORY.json').read_text());mapping={k.removeprefix('run/standing/'):v for k,v in raw.items()if k.startswith('run/standing/')}
  with patch.object(a,'GUARD',guard),patch.object(a,'inventory',return_value=mapping):self.assertEqual(a.standing_one_check(),mapping)
  for change in ('missing','wrong'):
   bad=dict(mapping)
   if change=='missing':bad.pop('solver_readback.json')
   else:bad['state.json']={'sha256':'f'*64,'size_bytes':1}
   with patch.object(a,'GUARD',guard),patch.object(a,'inventory',return_value=bad),self.assertRaises(ValueError):a.standing_one_check()
 def test_same_source_standing_mount_in_launch(self):
  c=a.expected_launch_command();self.assertEqual(c[-4:],['--num-envs','32','--standing-one',str(a.STANDING_ONE)]);self.assertEqual(a.STANDING_ONE.name,'standing')
if __name__=='__main__':unittest.main()
