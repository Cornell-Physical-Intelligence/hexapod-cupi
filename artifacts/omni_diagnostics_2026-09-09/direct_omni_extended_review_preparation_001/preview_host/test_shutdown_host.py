import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
from test_host import h
class ShutdownHostTests(unittest.TestCase):
 def args(self,p):return NS(output=p,host_freeze_sha256='host')
 def test_external_input_mutation_after_zero_exit_rejects(self):
  with tempfile.TemporaryDirectory() as t:
   a=self.args(Path(t));identity={'checkpoint':'original'}
   with patch.object(h,'PARENT',NS(run_owned=Mock(return_value={}))),patch.object(h,'verify_inputs',return_value={'changed':True}),patch.object(h,'validate_result') as validate:
    with self.assertRaisesRegex(ValueError,'Original inputs changed'):h.run_campaign(a,identity)
   validate.assert_not_called();report=h.read(a.output/'campaign.json');self.assertEqual(report['status'],'failed');self.assertFalse(report['terminal_inputs_unchanged'])
 def test_zero_exit_requires_host_seal_check_before_completed(self):
  with tempfile.TemporaryDirectory() as t:
   a=self.args(Path(t));identity={'checkpoint':'original'};events=[]
   with patch.object(h,'PARENT',NS(run_owned=lambda *_:events.append('exit_and_cleanup'))),patch.object(h,'verify_inputs',side_effect=lambda _:events.append('inputs') or identity),patch.object(h,'validate_result',side_effect=lambda *_:events.append('sealed_evidence') or {'complete':True}):
    result=h.run_campaign(a,identity)
   self.assertEqual(events,['exit_and_cleanup','inputs','sealed_evidence','inputs']);self.assertTrue(result['post_exit_original_inputs_reverified']);self.assertEqual(result['status'],'completed')
 def test_missing_seal_never_promoted_on_native_completed_state(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);h.save(p/'state.json',{'status':'completed'});h.save(p/'video.json',{'complete':True})
   with self.assertRaises(FileNotFoundError):h.validate_result(p,{})
if __name__=='__main__':unittest.main()
