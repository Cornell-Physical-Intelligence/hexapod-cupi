"""Keep checkpoint provenance intact across a certified package migration."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import locomotion.evaluation.checkpoint as compat


class CheckpointCompatibilityTests(unittest.TestCase):
    def bundle(self, directory):
        from locomotion.tests.test_native_ranges import NativeRangeTests
        summary = NativeRangeTests().suite(1.)
        summary['allocation'].update(checkpoint_sha256=compat.CHECKPOINT_SHA, evaluator_sha256='a'*64)
        summary['loads'] = {str(index): {'status': 'available'} for index in range(13)}
        ranges = compat.freeze([deepcopy(summary) for _ in range(3)])
        ranges['baseline_provenance'] = [{'fixture': index} for index in range(3)]
        for name, value in (('ranges.json', ranges), ('summary.json', summary)):
            (directory/name).write_text(json.dumps(value))
        receipt = {'schema': compat.SCHEMA, 'mode': 'evaluate', 'eval_scope': 'probes',
                   'checkpoint_sha256': compat.CHECKPOINT_SHA, 'declaration_sha256': compat.DECLARATION_SHA,
                   'candidate_runtime_sources': {'files': {'locomotion/evaluation/scoring.py': 'a'*64}},
                   'files': {name: compat.sha(directory/name) for name in ('ranges.json', 'summary.json')}}
        path = directory/'receipt.json'
        path.write_text(json.dumps(receipt))
        return path

    def test_receipt_pins_evidence_and_rejects_failed_or_incomplete_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self.bundle(directory)
            digest = compat.sha(path)
            compat.read_receipt(path, digest)
            with self.assertRaisesRegex(ValueError, 'receipt hash'):
                compat.read_receipt(path, '0'*64)
            summary_path = directory/'summary.json'
            summary = json.loads(summary_path.read_text())
            summary['results'][0]['metrics']['drift'] = 2.
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'evidence hash'):
                compat.read_receipt(path, digest)
            receipt = json.loads(path.read_text())
            receipt['files']['summary.json'] = compat.sha(summary_path)
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'native parity'):
                compat.read_receipt(path, compat.sha(path))
            path = self.bundle(directory)
            summary = json.loads(summary_path.read_text())
            summary['loads'].pop('0')
            summary_path.write_text(json.dumps(summary))
            receipt = json.loads(path.read_text())
            receipt['files']['summary.json'] = compat.sha(summary_path)
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'load evidence'):
                compat.read_receipt(path, compat.sha(path))

    def test_candidate_dependency_changes_and_original_identity_changes_reject(self):
        from contracts.release import COMPATIBILITY_KEYS
        recorded = dict.fromkeys(COMPATIBILITY_KEYS, 'original')
        candidate = deepcopy(recorded)
        candidate.update(schema='hexapod_locomotion_ppo_v2', adapter_sha256='new', entry_sha256='new',
                         runtime_sources={'files': {'locomotion/omni/control.py': 'a'*64}})
        record = {'identity': recorded, 'checkpoint_sha256': compat.CHECKPOINT_SHA}
        receipt = {'original_identity': deepcopy(recorded),
                   'candidate_runtime_sources': deepcopy(candidate['runtime_sources'])}
        compat.require_identity(receipt, record, candidate)
        for key in ('model_sha256', 'physics_config', 'seed'):
            changed = deepcopy(candidate)
            changed[key] = 'different'
            with self.subTest(key=key), self.assertRaises(ValueError):
                compat.require_identity(receipt, record, changed)
        candidate['runtime_sources']['files']['locomotion/omni/control.py'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'candidate runtime'):
            compat.require_identity(receipt, record, candidate)
        record['identity']['entry_sha256'] = 'altered declaration'
        with self.assertRaisesRegex(ValueError, 'original identity'):
            compat.require_identity(receipt, record, candidate)

    def test_prepare_binds_receipt_and_refuses_training_or_uncertified_scope(self):
        from locomotion.spark.prepare import prepare
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = '/srv/cupi/hexapod/runs/james/checkpoint.pt'
            options = dict(mode='evaluate', checkpoint=checkpoint, checkpoint_sha=compat.CHECKPOINT_SHA,
                           checkpoint_declaration_sha=compat.DECLARATION_SHA,
                           checkpoint_compatibility='/srv/cupi/hexapod/runs/james/certified/receipt.json',
                           checkpoint_compatibility_sha256='a'*64, eval_scope='probes')
            binding = prepare(root/'pack', '/srv/cupi/hexapod/runs/james/fresh', **options)
            self.assertIn(['/srv/cupi/hexapod/runs/james/certified', '/compatibility'], binding['extra_mounts'])
            self.assertIn('/compatibility/receipt.json', binding['command_args'])
            for changed in ({'mode': 'train'}, {'eval_scope': 'full'}, {'checkpoint_compatibility_sha256': None}):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    prepare(root/'rejected', '/srv/cupi/hexapod/runs/james/fresh', **(options | changed))

    def test_translate_only_known_class_fields_without_mutation(self):
        original = {'class_name': 'locomotion.rate_schedule:CappedRatePPO',
                    'label': 'locomotion.rate_schedule:CappedRatePPO',
                    'nested': [{'class_name': 'locomotion.action_distribution:BoundedMeanGaussian'}]}
        snapshot = deepcopy(original)
        translated = compat.configuration(original)
        self.assertEqual(original, snapshot)
        self.assertEqual(translated['class_name'], 'locomotion.omni.ppo:CappedRatePPO')
        self.assertEqual(translated['label'], original['label'])
        self.assertEqual(translated['nested'][0]['class_name'],
                         'locomotion.omni.ppo:BoundedMeanGaussian')

    def test_translation_keeps_configuration_changes_visible(self):
        old = {'class_name': 'locomotion.rate_schedule:CappedRatePPO', 'gamma': .99}
        candidate = {'class_name': 'locomotion.omni.ppo:CappedRatePPO', 'gamma': .98}
        self.assertNotEqual(compat.configuration(old), candidate)

    def test_unknown_class_name_stays_unknown(self):
        self.assertEqual(compat.configuration({'class_name': 'unreviewed:PPO'}),
                         {'class_name': 'unreviewed:PPO'})


if __name__ == '__main__':
    unittest.main()
