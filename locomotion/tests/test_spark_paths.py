"""Check CUPI packaging and preserve the historical launch boundary."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from locomotion import inputs, launch, prepare, reservation
from locomotion.spark_paths import CUPI_ROOT, LEGACY_ROOT, INPUT_ROOTS, RUN_ROOTS, within_roots


class SparkWorkspaceTests(unittest.TestCase):
    def test_cupi_input_pack_and_frozen_cpu_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packed = root / 'inputs'
            inputs.pack(packed, CUPI_ROOT / 'inputs/fixture')
            binding = prepare.prepare(root / 'attempt', CUPI_ROOT / 'runs/james/fixture',
                                      mode='diagnostic', inputs=packed / 'inputs.json')
            args = list(binding['command_args'])
            replacements = {'--asset': packed / 'asset', '--model': packed / 'asset/source/model.json',
                            '--geometry': packed / 'geometry_source/geometry/geometry.json',
                            '--geometry-extrema': packed / 'geometry_source/geometry/geometry_extrema.npz',
                            '--stance': packed / 'prior/stance.json'}
            for option, value in replacements.items():
                args[args.index(option) + 1] = str(value)
            result = subprocess.run([sys.executable, '-B', '-m', 'locomotion.train',
                                     '--output', str(root / 'unused'), '--preflight-only', *args],
                                    cwd=root / 'attempt/source', capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / 'unused').exists())
            reservation.verify_tree(root / 'attempt/source', binding['source_freeze_sha256'])

    def test_path_roles_and_legacy_compatibility(self):
        for prefix in RUN_ROOTS:
            self.assertTrue(within_roots(prefix / 'attempt', RUN_ROOTS))
        for prefix in INPUT_ROOTS:
            self.assertTrue(within_roots(prefix / 'bundle', INPUT_ROOTS))
        for path in (CUPI_ROOT, CUPI_ROOT / 'evidence/attempt', CUPI_ROOT / 'inputs/attempt',
                     '/srv/cupi/hexapod-other/runs/attempt', 'relative',
                     '/srv/cupi/hexapod/runs/../evidence/attempt'):
            with self.subTest(path=path):
                self.assertFalse(within_roots(path, RUN_ROOTS))
        self.assertFalse(within_roots(CUPI_ROOT / 'runs/james/input', INPUT_ROOTS))
        self.assertEqual(prepare.REMOTE_ROOT, LEGACY_ROOT)

    def test_launcher_rejects_output_in_inputs_and_prior_in_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            declared, _ = inputs.declaration(CUPI_ROOT / 'inputs/fixture')
            (root / 'inputs.json').write_text(json.dumps(declared))
            binding = prepare.prepare(root / 'attempt', CUPI_ROOT / 'runs/julian/fixture',
                                      mode='diagnostic', inputs=root / 'inputs.json')
            with ExitStack() as stack:
                stack.enter_context(patch.object(reservation, 'canonical_path', side_effect=Path))
                stack.enter_context(patch.object(reservation, 'verify_tree'))
                stack.enter_context(patch.object(reservation, 'pinned_file'))
                launch.verify(binding, Path(binding['source']))
                for key, value in (('output', str(CUPI_ROOT / 'inputs/overwrite')),
                                   ('prior', str(CUPI_ROOT / 'runs/julian/fixture/prior'))):
                    changed = copy.deepcopy(binding)
                    changed[key] = value
                    with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Guarded Spark'):
                        launch.verify(changed, Path(changed['source']))

    def test_host_still_rejects_symbolic_ancestry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'real').mkdir()
            (root / 'alias').symlink_to(root / 'real', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'Symbolic'):
                reservation.canonical_path(str(root / 'alias/output'))

    def test_cupi_checkpoint_binding_and_fresh_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'attempt'
            kwargs = dict(mode='evaluate', checkpoint=CUPI_ROOT / 'runs/shaurya/train/model_2000.pt',
                          checkpoint_sha='a' * 64, checkpoint_declaration_sha='b' * 64)
            binding = prepare.prepare(output, CUPI_ROOT / 'runs/shaurya/evaluation', **kwargs)
            self.assertIn(str(kwargs['checkpoint']), binding['input_files'])
            with self.assertRaises(FileExistsError):
                prepare.prepare(output, CUPI_ROOT / 'runs/shaurya/evaluation', **kwargs)
