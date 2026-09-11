"""Owned source stays discoverable; relocated commands work outside the checkout."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('source_inventory', ROOT / 'tools/source_inventory.py')
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


class SourceInventoryTests(unittest.TestCase):
    def fixture(self, root):
        (root / 'tools').mkdir()
        (root / 'configs').mkdir()
        (root / 'packages').mkdir()
        (root / 'tools/example.py').write_text('x = 1\n')
        data = {'tools': [dict(path='tools/example.py', classification='current', owner='lead', purpose='fixture')]}
        (root / inventory.INVENTORY).write_text(json.dumps(data))
        return data

    def test_repository_inventory_covers_all_tools(self):
        self.assertGreater(len(inventory.check()), 0)

    def test_unowned_source_cannot_hide_outside_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / 'tools/extra.py').write_text('x = 2\n')
            with self.assertRaisesRegex(ValueError, 'unowned'):
                inventory.check(root)

    def test_missing_source_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / 'tools/example.py').unlink()
            with self.assertRaisesRegex(ValueError, 'missing'):
                inventory.check(root)

    def test_production_cannot_import_a_frozen_evidence_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / 'packages/consumer.py').write_text('from artifacts.run import model\n')
            with self.assertRaisesRegex(ValueError, 'imports evidence'):
                inventory.check(root)

    def test_relocated_cli_help_works_in_a_fresh_process_from_another_directory(self):
        paths = ['tools/assets/generate_robstride_urdf.py',
                 'experiments/c_length_study/tools/rank_length_study.py',
                 'experiments/terrain/tools/lidar_placement_study.py']
        with tempfile.TemporaryDirectory() as tmp:
            for path in paths:
                with self.subTest(path=path):
                    result = subprocess.run([sys.executable, str(ROOT / path), '--help'],
                                            cwd=tmp, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn('usage:', result.stdout)


if __name__ == '__main__':
    unittest.main()
