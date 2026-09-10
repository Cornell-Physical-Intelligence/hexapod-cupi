"""Only explicit branch selection and future identity pins change in host003."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import launch_train_spark as h

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/'direct_omni_train_host_002'
OLD = '19545653a00e635cc14ed9415f9799978da34674e84f6cad76eb42bd2a961ac4'

class VersionDeltaTests(unittest.TestCase):
    def test_exact_frozen_parent_inventory(self):
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        self.assertEqual(sha(PARENT/'FREEZE_SHA256.json'), OLD)
        manifest = json.loads((PARENT/'FREEZE_SHA256.json').read_text())
        self.assertEqual({p.relative_to(PARENT).as_posix(): sha(p) for p in PARENT.rglob('*') if p.is_file() and p.name != 'FREEZE_SHA256.json'}, manifest)

    def test_all_function_bodies_except_selection_and_cli_unchanged(self):
        def functions(path):
            return {n.name: ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        old, new = functions(PARENT/'launch_train_spark.py'), functions(HERE/'launch_train_spark.py')
        self.assertEqual(set(old), set(new))
        for name in set(old) - {'selected_phases', 'main'}:
            self.assertEqual(old[name], new[name], name)

    def test_main_diff_only_adds_explicit_branch_choice(self):
        old = (PARENT/'launch_train_spark.py').read_text()
        new = (HERE/'launch_train_spark.py').read_text()
        def main(text):
            return ast.dump(next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'main'))
        expected = old.replace("choices=('curriculum','caps')", "choices=('curriculum','caps','quiet_priority')")
        self.assertEqual(main(expected), main(new))

    def test_entry_and_legacy_map_byte_identical(self):
        for name in ('run_train_entry.py', 'LEGACY_RUNTIME_SHA256.json'):
            self.assertEqual((PARENT/name).read_bytes(), (HERE/name).read_bytes())

    def test_unbound_source_or_contract_reject_before_file_reads(self):
        for bound in ('PENDING_ROOT_NATIVE004_SOURCE_BUILD', 'PENDING_NATIVE004_OWNER_FREEZE'):
            with patch.object(h, 'sha') as read_hash:
                with self.assertRaisesRegex(ValueError, 'unbound immutable manifest'):
                    h.verify_tree(Path('/not-a-source'), 'manifest.json', bound)
                read_hash.assert_not_called()

    def test_only_new_quiet_priority_smoke_is_available(self):
        for branch in ('caps', 'curriculum', 'implicit', None):
            with self.subTest(branch=branch), self.assertRaises(ValueError):
                h.selected_phases(SimpleNamespace(allocation='smoke', branch=branch, smoke=None))
        self.assertEqual(h.selected_phases(SimpleNamespace(allocation='smoke', branch='quiet_priority', smoke=None)), h.PHASES['smoke'])

    def test_all_pilots_require_an_explicit_prior_smoke(self):
        for branch in ('caps', 'curriculum', 'quiet_priority'):
            with self.subTest(branch=branch):
                with self.assertRaises(ValueError): h.selected_phases(SimpleNamespace(allocation='pilot', branch=branch, smoke=None))
                self.assertEqual(h.selected_phases(SimpleNamespace(allocation='pilot', branch=branch, smoke=Path('/exact-same-source-smoke'))), h.PHASES['pilot'])

if __name__ == '__main__': unittest.main()
