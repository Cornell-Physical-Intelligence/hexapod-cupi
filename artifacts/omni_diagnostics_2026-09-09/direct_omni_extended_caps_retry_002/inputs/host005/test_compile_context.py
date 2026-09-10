"""Whole-module CodeType identity, without executing compilation output."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import deadline_adapter as d
import launch_train_spark as h

HERE = Path(__file__).resolve().parent
SUP = HERE.parent/'reference_physics_adapter_009/source_009'
SOURCE = SUP/'tools/launch_reference_physics_spark.py'

class CompileContextTests(unittest.TestCase):
    def test_real_host_load_supervisor_and_full_compilation_without_execution(self):
        calls = []
        original = compile
        def compile_spy(source, filename, mode, *args, **kwargs):
            calls.append((source, filename, mode, kwargs))
            return original(source, filename, mode, *args, **kwargs)
        parent = h.load_supervisor(SimpleNamespace(allocation='pilot', supervisor_source=SUP))
        with patch('builtins.compile', side_effect=compile_spy):
            metadata = d.install(parent, SOURCE)
        matches = [r for r in calls if isinstance(r[0], str) and r[0] == SOURCE.read_text() and r[3].get('dont_inherit') is True]
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0][3]['dont_inherit'])
        self.assertEqual(metadata['source_substitutions'], 3)
        self.assertNotIn('_original_module_source', metadata)
        self.assertIs(parent.run_owned.__globals__, vars(parent))

    def test_changed_real_code_in_same_module_globals_rejected(self):
        parent = h.load_supervisor(SimpleNamespace(allocation='pilot', supervisor_source=SUP))
        code = parent.run_owned.__code__
        self.assertIn(600, code.co_consts)
        parent.run_owned.__code__ = code.replace(co_consts=tuple(601 if type(v) is int and v == 600 else v for v in code.co_consts))
        with self.assertRaisesRegex(ValueError, 'Loaded run_owned differs'):
            d.install(parent, SOURCE)
        self.assertNotIn(d.HELPER_NAME, vars(parent))

    def test_only_adapter_runtime_changes_from_frozen_host004(self):
        parent = HERE.parent/'direct_omni_train_host_004'
        self.assertEqual(hashlib.sha256((parent/'FREEZE_SHA256.json').read_bytes()).hexdigest(), 'b55282c968d64dca218baf10278339ff2013c23e28dc7cbe9a0d0b8dfd05d51f')
        for name in ('launch_train_spark.py', 'run_train_entry.py', 'LEGACY_RUNTIME_SHA256.json'):
            self.assertEqual((HERE/name).read_bytes(), (parent/name).read_bytes())
        old = ast.parse((parent/'deadline_adapter.py').read_text())
        new = ast.parse((HERE/'deadline_adapter.py').read_text())
        old_seams = next(n for n in old.body if isinstance(n, ast.Assign) and n.targets[0].id == 'SEAMS')
        new_seams = next(n for n in new.body if isinstance(n, ast.Assign) and n.targets[0].id == 'SEAMS')
        self.assertEqual(ast.dump(old_seams), ast.dump(new_seams))
        self.assertEqual(d.inspect_supervisor(SOURCE)['original_run_owned_sha256'], json.loads((parent/'DEADLINE_ADAPTER_PROOF.json').read_text())['original_run_owned_sha256'])

if __name__ == '__main__':
    unittest.main()
