"""The archived C runtime cannot silently resolve to production compatibility shims."""
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import c_study_runtime as runtime


@contextmanager
def isolated_imports():
    saved = {key: value for key, value in sys.modules.items() if key == "hexapod_rl" or key.startswith("hexapod_rl.")}
    paths, finders = sys.path[:], sys.meta_path[:]
    for name in saved:
        del sys.modules[name]
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == "hexapod_rl" or name.startswith("hexapod_rl."):
                del sys.modules[name]
        sys.modules.update(saved)
        sys.path[:] = paths
        sys.meta_path[:] = finders


class RuntimeBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.package = self.root / "experiments/c_length_study/runtime/hexapod_rl"
        self.package.mkdir(parents=True)
        for name in ("__init__.py", "env.py", "env_cfg.py"):
            (self.package / name).write_text("MARKER = 'validated-study'\n")
        self.files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.package.iterdir()}
        manifest = {"base_commit": runtime.PINNED_BASE_COMMIT, "files": self.files}
        (self.package.parent / "SHA256SUMS.json").write_text(json.dumps(manifest))
        self.shim = self.root / "isaaclab/hexapod_rl"
        self.shim.mkdir(parents=True)
        for name in ("__init__.py", "env.py", "only_modern.py"):
            (self.shim / name).write_text("MARKER = 'production-shim'\n")
        self.pin = patch.object(runtime, "PINNED_RUNTIME_TREE_SHA256", runtime._tree_digest(self.files))
        self.pin.start()

    def tearDown(self):
        self.pin.stop()
        self.temporary.cleanup()

    def test_bootstrap_wins_even_if_shim_path_is_prepended_later(self):
        before = (self.shim / "env.py").read_bytes()
        with isolated_imports():
            first = runtime.bootstrap_c_study_runtime(repo_root=self.root)
            second = runtime.bootstrap_c_study_runtime(repo_root=self.root)
            self.assertEqual(first, second)
            self.assertEqual(sum(isinstance(finder, runtime._PinnedRuntimeFinder) for finder in sys.meta_path), 1)
            sys.path.insert(0, str(self.shim.parent))
            module = importlib.import_module("hexapod_rl.env")
            self.assertEqual(module.MARKER, "validated-study")
            self.assertEqual(Path(module.__file__).resolve(), (self.package / "env.py").resolve())
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module("hexapod_rl.only_modern")
        self.assertEqual((self.shim / "env.py").read_bytes(), before)

    def test_already_imported_wrong_runtime_is_not_reloaded(self):
        with isolated_imports():
            sys.path.insert(0, str(self.shim.parent))
            wrong = importlib.import_module("hexapod_rl.env")
            with self.assertRaisesRegex(RuntimeError, "already imported"):
                runtime.bootstrap_c_study_runtime(repo_root=self.root)
            self.assertIs(sys.modules["hexapod_rl.env"], wrong)
            self.assertEqual(wrong.MARKER, "production-shim")

    def test_source_tampering_and_unmanifested_files_fail_closed(self):
        with isolated_imports():
            (self.package / "env.py").write_text("MARKER = 'tampered'\n")
            with self.assertRaisesRegex(RuntimeError, "bytes changed"):
                runtime.bootstrap_c_study_runtime(repo_root=self.root)
        (self.package / "env.py").write_text("MARKER = 'validated-study'\n")
        (self.package / "extra.py").write_text("pass\n")
        with self.assertRaisesRegex(RuntimeError, "unmanifested"):
            runtime.bootstrap_c_study_runtime(repo_root=self.root)

    def test_import_rechecks_bytes_after_bootstrap(self):
        with isolated_imports():
            runtime.bootstrap_c_study_runtime(repo_root=self.root)
            (self.package / "env.py").write_text("MARKER = 'late-tamper'\n")
            with self.assertRaisesRegex(ImportError, "changed before import"):
                importlib.import_module("hexapod_rl.env")

    def test_missing_vendor_does_not_silently_use_legacy_or_modern_shim(self):
        shutil.rmtree(self.shim)
        shutil.move(str(self.package), str(self.shim))
        hashes = {"isaaclab/hexapod_rl/" + name: digest for name, digest in self.files.items()}
        (self.root / "campaign_source_hashes.json").write_text(json.dumps(hashes))
        with isolated_imports():
            with self.assertRaisesRegex(RuntimeError, "Vendored.*missing"):
                runtime.bootstrap_c_study_runtime(repo_root=self.root)
            binding = runtime.bootstrap_c_study_runtime(repo_root=self.root, mode="legacy-frozen")
            self.assertEqual(binding["mode"], "legacy-frozen")
            self.assertEqual(importlib.import_module("hexapod_rl.env").MARKER, "validated-study")
        (self.shim / "env.py").write_text("MARKER = 'production-shim'\n")
        hashes["isaaclab/hexapod_rl/env.py"] = hashlib.sha256((self.shim / "env.py").read_bytes()).hexdigest()
        (self.root / "campaign_source_hashes.json").write_text(json.dumps(hashes))
        with self.assertRaisesRegex(RuntimeError, "pinned runtime"):
            runtime.bootstrap_c_study_runtime(repo_root=self.root, mode="legacy-frozen")

    def test_real_vendored_runtime_and_entrypoint_order(self):
        import ast
        root = Path(__file__).resolve().parents[2]
        real = json.loads((root / "experiments/c_length_study/runtime/SHA256SUMS.json").read_text())
        self.assertEqual(runtime._tree_digest(real["files"]), "abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280")
        for filename in ("train_length_study.py", "omni_flat_env.py", "length_reference_env.py", "validate_terrain_robot.py"):
            tree = ast.parse((root / "tools" / filename).read_text())
            calls = [index for index, node in enumerate(tree.body)
                     if isinstance(node, (ast.Assign, ast.Expr)) and isinstance(node.value, ast.Call)
                     and isinstance(node.value.func, ast.Name) and node.value.func.id == "bootstrap_c_study_runtime"]
            imports = [index for index, node in enumerate(tree.body) if isinstance(node, ast.ImportFrom)
                       and node.module and (node.module.startswith("hexapod_rl") or node.module == "isaaclab.app")]
            self.assertEqual(len(calls), 1, filename)
            self.assertTrue(all(calls[0] < index for index in imports), filename)


if __name__ == "__main__":
    unittest.main()
