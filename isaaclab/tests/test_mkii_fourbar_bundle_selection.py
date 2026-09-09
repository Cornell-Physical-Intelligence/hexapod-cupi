"""Select the executed physical formulation without changing its CAD coordinates."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'packages/hexapod_core'))
from hexapod_core import fourbar_v1 as contract


class BundleSelectionTests(unittest.TestCase):
    def bundle(self, version, root=ROOT):
        descriptor = contract.ASSET_BUNDLES[f'mkii_fourbar_v{version}']
        return contract.resolve_asset_bundle(descriptor['usd_path_relative'], repo_root=root)

    def test_default_and_immutable_registry_preserve_v3(self):
        self.assertEqual(contract.DEFAULT_ASSET_MODEL_ID, 'mkii_fourbar_v3')
        self.assertEqual(contract.MODEL_ID, 'mkii_fourbar_v3')
        self.assertEqual(contract.select_asset_bundle(repo_root=ROOT, environ={}), self.bundle(3))
        with self.assertRaises(TypeError):
            contract.ASSET_BUNDLES['new'] = {}
        with self.assertRaises(TypeError):
            contract.ASSET_BUNDLES['mkii_fourbar_v3']['model_id'] = 'changed'

    def test_generic_and_existing_environment_names_select_known_actual_bundle(self):
        path = contract.ASSET_BUNDLES['mkii_fourbar_v4']['usd_path_relative']
        for variable in ('HEXAPOD_USD_PATH', 'HEXAPOD_MKII_FOURBAR_USD_PATH'):
            self.assertEqual(contract.select_asset_bundle(repo_root=ROOT, environ={variable: path}), self.bundle(4))
        both = {'HEXAPOD_USD_PATH': str(ROOT/path), 'HEXAPOD_MKII_FOURBAR_USD_PATH': path}
        self.assertEqual(contract.select_asset_bundle(repo_root=ROOT, environ=both), self.bundle(4))
        both['HEXAPOD_USD_PATH'] = contract.ASSET_BUNDLES['mkii_fourbar_v3']['usd_path_relative']
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            contract.select_asset_bundle(repo_root=ROOT, environ=both)
        for value in ('', ' ', '/tmp/unregistered_robot.usda'):
            with self.assertRaises(ValueError):
                contract.select_asset_bundle(repo_root=ROOT, environ={'HEXAPOD_USD_PATH': value})

    def test_actual_config_selection_uses_bundle_not_old_kinematics_usd_path(self):
        path = ROOT/'packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1/config.py'
        tree = ast.parse(path.read_text())
        names = {'ASSET_BUNDLE', 'USD_PATH'}
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)]
        self.assertEqual(len(assignments), 2)
        selected = contract.ASSET_BUNDLES['mkii_fourbar_v4']['usd_path_relative']
        namespace = {'ROOT': ROOT, 'contract': contract, 'str': str,
                     'os': SimpleNamespace(environ={'HEXAPOD_USD_PATH': selected}),
                     'KINEMATICS': {'usd_path_relative': 'old-path-must-not-select-the-bundle'}}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), 'exec'), namespace)
        self.assertEqual(namespace['ASSET_BUNDLE'], self.bundle(4))
        self.assertEqual(namespace['USD_PATH'], str(ROOT/selected))

    def test_runtime_identity_tracks_selected_bundle_with_unchanged_coordinate_contract(self):
        kinematics = contract.load_kinematics(ROOT/contract.KINEMATICS_PATH)
        manifests = []
        for version in (3, 4):
            bundle = self.bundle(version)
            manifest = contract.runtime_manifest(kinematics, {'test_motor': True},
                kinematics_sha256=bundle['kinematics_sha256'], usd_sha256=bundle['usd_root_sha256'], asset_bundle=bundle)
            self.assertEqual(manifest['model_id'], f'mkii_fourbar_v{version}')
            self.assertEqual(manifest['asset_bundle'], bundle)
            self.assertEqual(manifest['closure_constraint_variant'], bundle['closure_constraint_variant'])
            self.assertEqual(manifest['usd_path_relative'], bundle['usd_path_relative'])
            for relative, expected in bundle['bundle_files_sha256'].items():
                self.assertEqual(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest(), expected)
            manifests.append(manifest)
        changed = {key for key in manifests[0] if manifests[0][key] != manifests[1][key]}
        self.assertEqual(changed, {'model_id', 'asset_bundle', 'closure_constraint_variant', 'usd_path_relative', 'usd_root_sha256'})
        self.assertEqual(manifests[0]['kinematics_sha256'], manifests[1]['kinematics_sha256'])

    def test_mixed_identity_missing_dependencies_and_executed_hash_mismatch_fail(self):
        bundle = self.bundle(4)
        for field, value in (('model_id', 'mkii_fourbar_v3'), ('closure_constraint_variant', 'revolute_5row_v3'),
                             ('usd_root_sha256', 'a'*64), ('kinematics_sha256', 'b'*64),
                             ('bundle_files_sha256', {})):
            wrong = dict(bundle, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                contract.validate_asset_bundle(wrong)
        kinematics = contract.load_kinematics(ROOT/contract.KINEMATICS_PATH)
        for field in ('kinematics_sha256', 'usd_sha256'):
            args = dict(kinematics_sha256=bundle['kinematics_sha256'], usd_sha256=bundle['usd_root_sha256'], asset_bundle=bundle)
            args[field] = 'c'*64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Executed USD/kinematics'):
                contract.runtime_manifest(kinematics, {}, **args)
        clone = contract.validate_asset_bundle(bundle)
        clone['bundle_files_sha256'].clear()
        self.assertEqual(len(bundle['bundle_files_sha256']), 4)

    def test_portable_checkout_and_stale_root_geometry_kinematics_manifest_are_rejected(self):
        relative = Path(contract.ASSET_BUNDLES['mkii_fourbar_v4']['usd_path_relative'])
        expected = self.bundle(4)
        with tempfile.TemporaryDirectory(prefix='bundle_identity_') as temporary:
            base = Path(temporary).resolve()
            original = base/'pristine'
            shutil.copytree((ROOT/relative).parent, original/relative.parent)
            self.assertEqual(self.bundle(4, original), expected)
            for index, filename in enumerate((relative.name, 'geometry.usdc', 'kinematics.json', 'manifest.json')):
                with self.subTest(filename=filename):
                    target = base/str(index)
                    shutil.copytree(original, target)
                    path = target/relative.parent/filename
                    if filename == 'manifest.json':
                        value = json.loads(path.read_text())
                        value['closure_constraint_variant'] = 'revolute_5row_v3'
                        path.write_text(json.dumps(value))
                    else:
                        with path.open('ab') as stream:
                            stream.write(b'\n')
                    with self.assertRaisesRegex(ValueError, 'manifest|variant'):
                        self.bundle(4, target)
            target = base/'redirected'
            shutil.copytree(original, target)
            geometry = target/relative.parent/'geometry.usdc'
            geometry.unlink()
            geometry.symlink_to((original/relative.parent/'geometry.usdc').resolve())
            with self.assertRaisesRegex(ValueError, 'redirected'):
                self.bundle(4, target)

    def test_bundle_resolution_remains_stdlib_only_before_kit(self):
        script = (
            'import sys\n'
            f'sys.path.insert(0, {str(ROOT/"packages/hexapod_core")!r})\n'
            'from hexapod_core import fourbar_v1 as c\n'
            f'c.select_asset_bundle(repo_root={str(ROOT)!r}, environ={{}})\n'
            'assert not {"pxr", "torch", "numpy", "isaaclab"}.intersection(sys.modules)\n'
        )
        result = subprocess.run([sys.executable, '-I', '-c', script], text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
