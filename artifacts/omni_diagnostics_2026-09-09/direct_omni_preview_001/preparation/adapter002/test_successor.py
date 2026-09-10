import ast, json, tempfile, types, unittest
from pathlib import Path
import preview_contract as c
import test_preview

HERE=Path(__file__).resolve().parent

class SuccessorTests(unittest.TestCase):
    def test_rendering_and_inference_inputs_are_byte_unchanged(self):
        prior=c.read(HERE/'inputs/adapter001_sha256.json')
        for path in ('record_direct_preview.py','preview_recorder.py','entry_adapter.py','render_helpers.py',
                     'inputs/native_entry.py','inputs/native_omni_flat_env.py','inputs/legacy_runtime.json'):
            self.assertEqual(c.sha(HERE/path),prior[path],path)

    def test_native003_source_delta_is_only_finalizer_and_origin(self):
        a=c.read(HERE/'inputs/source001_manifest.json');b=c.read(HERE/'inputs/native_source_manifest.json')
        self.assertEqual(len(b),598);self.assertEqual(set(a),set(b))
        self.assertEqual({k for k in a if a[k]!=b[k]}, {'tools/direct_training.py','source_origin.json'})
        self.assertEqual(c.sha(HERE/'inputs/native_source_manifest.json'),c.SOURCE)
        self.assertEqual(c.sha(HERE/'inputs/native_entry.py'),b['tools/train_length_study.py'])
        self.assertEqual(c.sha(HERE/'inputs/native_omni_flat_env.py'),b['tools/omni_flat_env.py'])
        self.assertEqual(c.SOURCE,'64144b2c2466f17c809554ef53f82b736b4199136cc821451d7ac2788752bd2e')
        self.assertEqual(c.NATIVE,'20fbcd40a869b39d8e3143a6c9715ec62ada5cc6211eaf473f60f8b7167678cb')
        self.assertEqual(c.HOST,'19545653a00e635cc14ed9415f9799978da34674e84f6cad76eb42bd2a961ac4')

    def test_only_contract_changes_are_pins_and_pilot_restriction(self):
        text=(HERE/'preview_contract.py').read_text()
        for new,old in ((c.SOURCE,'37b4b6d5d75f6d697a122fc2a1903f8f625f6a67d1727e1257e1533300e44da6'),
                        (c.NATIVE,'9e591b93ae99ca3a76b7f7500ea20cc778da76c7f6b9b2c86fd22d31aceab3d7'),
                        (c.HOST,'332fad29e0a0f88ba07012700f2a21c3a4f0453f3d71224da84fcc477e48394f')):
            text=text.replace(new,old)
        newer="    if allocation!='pilot' or branch not in ('curriculum','caps'):raise ValueError('Preview requires a completed curriculum or CAPS pilot; smoke is not eligible')\n    selected=contract.selection('train',allocation,branch,None,None)\n    if identity.get('selection')!=selected or selected.get('updates')!=50:raise ValueError('Wrong campaign training selection')\n    phases=('standing','initial_constant','initial_stop','train','final_constant','final_stop')"
        older="    selected=contract.selection('train',allocation,branch,None,None)\n    if identity.get('selection')!=selected or (allocation=='smoke' and branch!='caps'):raise ValueError('Wrong campaign training selection')\n    phases=('standing','train','final_constant','final_stop') if allocation=='smoke' else ('standing','initial_constant','initial_stop','train','final_constant','final_stop')"
        self.assertEqual(text.count(newer),1)
        import hashlib
        restored=text.replace(newer,older)
        self.assertEqual(hashlib.sha256(restored.encode()).hexdigest(),c.read(HERE/'inputs/adapter001_sha256.json')['preview_contract.py'])

    def test_both_pilot_branches_accept_but_smoke_and_unknown_reject(self):
        fixture=test_preview.ReceiptTests()
        with tempfile.TemporaryDirectory() as t:
            args,native,campaign=fixture.fixture(Path(t))
            for branch in ('curriculum','caps'):
                campaign['branch']=branch;c.save(args.pilot/'campaign.json',campaign)
                result=c.verify_pilot(args,native)
                self.assertEqual(result['branch'],branch);self.assertEqual(result['completed_updates'],50)
            for allocation,branch in (('smoke','caps'),('pilot','unknown')):
                campaign['allocation']=allocation;campaign['branch']=branch;c.save(args.pilot/'campaign.json',campaign)
                with self.assertRaisesRegex(ValueError,'completed curriculum or CAPS pilot'):c.verify_pilot(args,native)

    def test_old_host_and_uncompleted_finals_reject(self):
        fixture=test_preview.ReceiptTests()
        with tempfile.TemporaryDirectory() as t:
            args,native,campaign=fixture.fixture(Path(t))
            campaign['host_freeze_sha256']='332fad29e0a0f88ba07012700f2a21c3a4f0453f3d71224da84fcc477e48394f'
            c.save(args.pilot/'campaign.json',campaign)
            with self.assertRaisesRegex(ValueError,'Unmatched'):c.verify_pilot(args,native)
            campaign['host_freeze_sha256']=c.HOST;campaign['accepted_phases'].pop('final_stop')
            c.save(args.pilot/'campaign.json',campaign)
            with self.assertRaisesRegex(ValueError,'Incomplete'):c.verify_pilot(args,native)

if __name__=='__main__':unittest.main()
