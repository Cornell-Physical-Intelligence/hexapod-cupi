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
    def test_pack_uses_one_backend_and_keeps_review_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = prepare(root/'pack', REMOTE, mode='diagnostic')
            self.assertNotIn('--backend', binding['command_args'])
            self.assertFalse(binding['root_review_complete'])
            self.assertEqual(binding['module'], 'locomotion.runs.train')
            verify_runtime(root/'pack/source')
            nested = root/'pack/source/locomotion/core/motor.py'
            nested.write_text(nested.read_text()+'\n# tamper\n')
            with self.assertRaisesRegex(ValueError, 'manifest differs'):
                verify_runtime(root/'pack/source')

    def test_retired_modes_fail_before_creating_pack(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ('tripod', 'replay', 'throughput'):
                output = Path(directory)/mode
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'Invalid native allocation'):
                    prepare(output, REMOTE, mode=mode)
                self.assertFalse(output.exists())

    def test_bc_pack_binds_warmstart_and_keeps_review_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bc'
            binding = prepare(path, REMOTE, bc_warmstart=True, gait_clock=40)
            self.assertIn('--bc-warmstart', binding['command_args'])
            self.assertFalse(binding['root_review_complete'])
            self.assertTrue(json.loads((path/'PACK.json').read_text())['bc_warmstart'])

    def test_bc_rejects_unsupported_combinations_before_pack_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            for overrides in ({'mode': 'diagnostic'},
                              {'gait_clock': 0}, {'reward_version': '4'}):
                options = dict(bc_warmstart=True, gait_clock=40)
                options.update(overrides)
                output = Path(directory)/'rejected'
                with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, 'BC warmstart'):
                    prepare(output, REMOTE, **options)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
