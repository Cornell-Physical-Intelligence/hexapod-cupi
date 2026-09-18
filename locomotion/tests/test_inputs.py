"""Exercise portable inputs with the frozen launcher's CPU preflight."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from locomotion import inputs, prepare


class PortableInputsTests(unittest.TestCase):
    def test_portable_pack_and_frozen_diagnostic_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packed = root / 'inputs'
            remote_inputs = '/home/orionh/HEXAPOD_runs/restart_20260914/foundation_inputs_test'
            inputs.pack(packed, remote_inputs)
            binding = prepare.prepare(root/'launch',
                '/home/orionh/HEXAPOD_runs/restart_20260914/foundation_test',
                mode='diagnostic', inputs=packed/'inputs.json')
            source = root/'launch/source'
            self.assertIn(prepare.REMOTE_ROOT, Path(binding['prior']).parents)
            result = subprocess.run([sys.executable, '-m', 'locomotion.train',
                '--mode', 'diagnostic', '--asset', str(packed/'asset'),
                '--model', str(packed/'asset/source/model.json'),
                '--geometry', str(packed/'geometry_source/geometry/geometry.json'),
                '--geometry-extrema', str(packed/'geometry_source/geometry/geometry_extrema.npz'),
                '--stance', str(packed/'prior/stance.json'), '--output', str(root/'unused'),
                '--num-envs', '1', '--source-freeze-sha256', binding['source_freeze_sha256'],
                '--headless', '--device', 'cuda:0', '--preflight-only'],
                cwd=source, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            identity = json.loads(result.stdout)
            self.assertFalse(identity['motion_prior'])
            self.assertFalse((root/'unused').exists())
            with self.assertRaises(FileExistsError):
                inputs.pack(packed, remote_inputs)

    def test_remote_input_root_must_be_absolute(self):
        for value in ('relative/path', '/absolute/../escape', '/outside/reservation'):
            with self.assertRaises(ValueError):
                inputs.declaration(value)

    def test_default_binding_matches_the_portable_input_pack(self):
        default = json.loads((inputs.ROOT/'configs/locomotion_spark.json').read_text())
        remote = Path(default['asset']).parent
        declared, _ = inputs.declaration(remote)
        self.assertEqual(default, declared)
        self.assertIn(prepare.REMOTE_ROOT, Path(default['stance']).parents)

    def test_input_tampering_fails_before_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / inputs.MODEL
            model.mkdir(parents=True)
            (model/'inputs.json').write_text(json.dumps({'files': {'wrong.bin': '0'*64}}))
            (root/'wrong.bin').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Canonical input bytes differ'):
                inputs.pack(root/'pack', '/absolute/inputs', root)
            self.assertFalse((root/'pack').exists())
