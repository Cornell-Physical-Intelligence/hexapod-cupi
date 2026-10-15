"""Check candidate selection and frozen source identity without Isaac."""
import json
from pathlib import Path
import tempfile
import unittest

from locomotion.runs.prepare import prepare
from locomotion.runs.sources import physics_files, source_files, verify_runtime

ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/srv/cupi/hexapod/runs/jcc463/direct_binding_fixture'


class DirectBindingTests(unittest.TestCase):
    def test_prepare_preserves_legacy_default_and_binds_direct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = prepare(root/'legacy', REMOTE, mode='diagnostic')
            direct = prepare(root/'direct', REMOTE, mode='diagnostic', backend='direct')
            self.assertNotIn('--backend', legacy['command_args'])
            self.assertTrue(legacy['root_review_complete'])
            self.assertFalse(direct['root_review_complete'])
            args = direct['command_args']
            self.assertEqual(args[args.index('--backend')+1], 'direct')
            self.assertEqual(json.loads((root/'direct/PACK.json').read_text())['backend'], 'direct')
            verify_runtime(root/'direct/source')
            nested = root/'direct/source/locomotion/core/motor.py'
            self.assertTrue(nested.is_file())
            nested.write_text(nested.read_text()+'\n# tamper\n')
            with self.assertRaisesRegex(ValueError, 'manifest differs'):
                verify_runtime(root/'direct/source')

    def test_direct_rejects_modes_with_other_entrypoints_before_creating_pack(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ('throughput', 'tripod', 'replay'):
                output = Path(directory)/mode
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'Direct backend'):
                    prepare(output, REMOTE, mode=mode, backend='direct')
                self.assertFalse(output.exists())

    def test_direct_physics_closure_requires_candidate_sources(self):
        files = source_files(ROOT/'locomotion')
        files.pop('backends/isaac/direct_env.py', None)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            physics_files(files, backend='direct')
        with self.assertRaisesRegex(ValueError, 'Unknown native backend'):
            physics_files(files, backend='unknown')

    def test_bc_pack_binds_warmstart_and_keeps_review_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bc'
            binding = prepare(path, REMOTE, backend='direct', bc_warmstart=True, gait_clock=40)
            self.assertIn('--bc-warmstart', binding['command_args'])
            self.assertFalse(binding['root_review_complete'])
            self.assertTrue(json.loads((path/'PACK.json').read_text())['bc_warmstart'])

    def test_bc_rejects_unsupported_combinations_before_pack_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            for overrides in ({'backend': 'legacy'}, {'mode': 'diagnostic'}, {'learner': 'amp'},
                              {'gait_clock': 0}, {'reward_version': '4'}):
                options = dict(backend='direct', bc_warmstart=True, gait_clock=40)
                options.update(overrides)
                output = Path(directory)/'rejected'
                with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, 'BC warmstart'):
                    prepare(output, REMOTE, **options)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
