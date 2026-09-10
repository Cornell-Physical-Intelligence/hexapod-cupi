"""Pin the C morphology study to its validated, archived Python runtime.

Call before AppLauncher and before any hexapod_rl import. This bootstrap is
process-local and study-only; the production packages and compatibility shims
are neither edited nor reloaded.
"""
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys


PINNED_BASE_COMMIT = "36769da93f90e0602fb0e3591caeb98bbd7090ed"
PINNED_RUNTIME_TREE_SHA256 = "abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280"
MODE_ENVIRONMENT_VARIABLE = "HEXAPOD_C_STUDY_RUNTIME_MODE"


def _tree_digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _verified_files(package_dir, files):
    if not isinstance(files, dict) or _tree_digest(files) != PINNED_RUNTIME_TREE_SHA256:
        raise RuntimeError("C-study runtime manifest does not match the validated pinned runtime")
    actual = {str(path.relative_to(package_dir).as_posix()) for path in package_dir.rglob("*.py")}
    if actual != set(files):
        raise RuntimeError("C-study runtime has missing or unmanifested Python files")
    for relative, expected in files.items():
        path = package_dir / relative
        if PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
            raise RuntimeError("Unsafe C-study runtime manifest path")
        if path.is_symlink() or not path.resolve().is_relative_to(package_dir):
            raise RuntimeError("C-study runtime Python files must stay inside the pinned package")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"C-study runtime bytes changed: {relative}")


class _PinnedLoader(importlib.machinery.SourceFileLoader):
    """Compile verified source, avoiding stale or unrelated bytecode caches."""
    def __init__(self, fullname, path, expected):
        super().__init__(fullname, str(path))
        self.expected = expected

    def get_code(self, fullname):
        source = Path(self.path).read_bytes()
        if hashlib.sha256(source).hexdigest() != self.expected:
            raise ImportError(f"Pinned C-study source changed before import: {self.path}")
        return self.source_to_code(source, self.path)


class _PinnedRuntimeFinder(importlib.abc.MetaPathFinder):
    def __init__(self, package_dir, files):
        self.package_dir = package_dir
        self.files = dict(files)

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "hexapod_rl" and not fullname.startswith("hexapod_rl."):
            return None
        suffix = fullname.removeprefix("hexapod_rl").lstrip(".").replace(".", "/")
        candidates = (["__init__.py"] if not suffix else [suffix + ".py", suffix + "/__init__.py"])
        relative = next((candidate for candidate in candidates if candidate in self.files), None)
        if relative is None:
            raise ModuleNotFoundError(f"Module is outside the pinned C-study runtime: {fullname}", name=fullname)
        source = self.package_dir / relative
        loader = _PinnedLoader(fullname, source, self.files[relative])
        locations = [str(source.parent)] if source.name == "__init__.py" else None
        return importlib.util.spec_from_file_location(fullname, source, loader=loader, submodule_search_locations=locations)


def bootstrap_c_study_runtime(*, repo_root=None, mode=None):
    """Return an auditable binding, or fail before simulator startup.

    Default mode requires the vendored runtime. ``legacy-frozen`` is an explicit
    compatibility option for a newly packaged legacy layout with a campaign
    manifest containing exactly the same historical runtime hashes. Old frozen
    snapshots without this bootstrap are left untouched.
    """
    repo_root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[1]
    mode = mode if mode is not None else os.environ.get(MODE_ENVIRONMENT_VARIABLE, "vendored")
    if mode == "vendored":
        runtime_root = repo_root / "experiments/c_length_study/runtime"
        package_dir = runtime_root / "hexapod_rl"
        manifest_path = runtime_root / "SHA256SUMS.json"
        if not manifest_path.is_file() or not package_dir.is_dir():
            raise RuntimeError("Vendored C-study runtime is missing; do not fall back to isaaclab compatibility shims")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("base_commit") != PINNED_BASE_COMMIT:
            raise RuntimeError("C-study runtime base commit changed")
        files = manifest.get("files")
    elif mode == "legacy-frozen":
        runtime_root = repo_root / "isaaclab"
        package_dir = runtime_root / "hexapod_rl"
        manifest_path = repo_root / "campaign_source_hashes.json"
        if not manifest_path.is_file():
            raise RuntimeError("Explicit legacy-frozen mode requires a frozen campaign source manifest")
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict):
            raise RuntimeError("Invalid legacy campaign source manifest")
        prefix = "isaaclab/hexapod_rl/"
        files = {key[len(prefix):]: value for key, value in manifest.items()
                 if key.startswith(prefix) and key.endswith(".py")}
    else:
        raise ValueError(f"Unknown C-study runtime mode: {mode!r}")
    package_dir = package_dir.resolve()
    _verified_files(package_dir, files)

    # A mixed process is unrecoverable: replacing sys.path or deleting/reloading
    # modules would leave already-imported classes bound to the wrong model.
    for name, module in tuple(sys.modules.items()):
        if name != "hexapod_rl" and not name.startswith("hexapod_rl."):
            continue
        origin = getattr(module, "__file__", None)
        if not origin:
            raise RuntimeError(f"Cannot establish the origin of already-imported {name}")
        source = Path(origin).resolve()
        if not source.is_relative_to(package_dir) or str(source.relative_to(package_dir).as_posix()) not in files:
            raise RuntimeError(f"Wrong hexapod_rl runtime already imported ({name}: {origin}); use a fresh process")
    active = [finder for finder in sys.meta_path if isinstance(finder, _PinnedRuntimeFinder)]
    if any(finder.package_dir != package_dir or finder.files != files for finder in active):
        raise RuntimeError("A different C-study runtime is already pinned in this process")
    if not active:
        sys.meta_path.insert(0, _PinnedRuntimeFinder(package_dir, files))
    # The finder pins future imports even if another library prepends a source
    # path later. This path entry also makes conventional tooling locate it.
    parent = str(runtime_root.resolve())
    if parent not in sys.path:
        sys.path.insert(0, parent)
    return {"mode": mode, "package_directory": str(package_dir),
            "runtime_tree_sha256": PINNED_RUNTIME_TREE_SHA256, "base_commit": PINNED_BASE_COMMIT,
            "manifest_path": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "python_files_verified": len(files)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--mode", choices=("vendored", "legacy-frozen"))
    arguments = parser.parse_args()
    print(json.dumps(bootstrap_c_study_runtime(repo_root=arguments.repo_root, mode=arguments.mode), indent=2))
