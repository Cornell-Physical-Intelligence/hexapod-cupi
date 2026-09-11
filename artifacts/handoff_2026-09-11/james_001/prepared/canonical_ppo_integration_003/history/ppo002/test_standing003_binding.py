"""Only source identity changes; no behavior or fabricated native admission."""
from pathlib import Path
import ast,hashlib,json,unittest
from test_support import ROOT,STANDING
from canonical_direct_ppo import native_entry_adapter as adapter
from canonical_direct_ppo.binding_contract import verify_admissions

PARENT=ROOT/'tmp/canonical_ppo_integration_001'
OLD=ROOT/'tmp/updated_native_standing_002'
HERE=Path(__file__).resolve().parent
NEW='e922ff1312b5106f957b384874ca5eb6ed4a9ed6566c0ef9a0602247a49670f0'
OLD_HASH='acb589708fcba8be2f6ef64171795889eca8b6583fdbb4bc2fc25d61717b0467'


class SourceBindingChecks(unittest.TestCase):
 def test_runtime_delta_is_only_exact_source_constant(self):
  before=(PARENT/'canonical_direct_ppo/native_entry_adapter.py').read_text()
  after=(HERE/'canonical_direct_ppo/native_entry_adapter.py').read_text()
  self.assertEqual(before.count(OLD_HASH),1)
  self.assertEqual(before.replace(OLD_HASH,NEW),after)
  for source in (PARENT/'canonical_direct_ppo').glob('*.py'):
   if source.name!='native_entry_adapter.py':self.assertEqual(source.read_bytes(),(HERE/'canonical_direct_ppo'/source.name).read_bytes(),source.name)
  self.assertEqual((PARENT/'run_native_smoke.py').read_bytes(),(HERE/'run_native_smoke.py').read_bytes())
 def test_native_composition_and_physics_inputs_same(self):
  self.assertEqual(adapter.STANDING_FREEZE,NEW)
  for name in ['run_standing.py','standing_session.py','standing_score.py','quiet_metrics.py','standing_contract.py','servo_candidate.json','geometry/geometry.json']:
   self.assertEqual((OLD/name).read_bytes(),(STANDING/name).read_bytes(),name)
  actual,receipt=adapter.adapted_source(STANDING)
  previous=(OLD/'run_standing.py').read_text().replace(adapter.ORIGINAL_TAIL,adapter.LEARNER_TAIL,1)
  self.assertEqual(actual,previous);self.assertTrue(receipt['neutral_physics_statements_unchanged']);self.assertTrue(receipt['exception_finally_statements_unchanged'])
  self.assertEqual(receipt['adapted_entry_sha256'],hashlib.sha256(previous.encode()).hexdigest())
 def test_old_source_refused_and_actual_admission_still_required(self):
  with self.assertRaisesRegex(ValueError,'Wrong frozen source'):adapter.adapted_source(OLD)
  binding=json.loads((HERE/'BINDINGS.json').read_text())
  self.assertEqual(binding['standing_source_freeze_sha256'],NEW)
  self.assertIsNone(binding['standing1_state_sha256']);self.assertIsNone(binding['standing32_state_sha256'])
  self.assertFalse(binding['ready_for_native_dispatch'])
  with self.assertRaisesRegex(ValueError,'Pending/invalid required binding'):verify_admissions(STANDING,Path('/nonexistent1'),Path('/nonexistent32'),binding)

if __name__=='__main__':unittest.main()
