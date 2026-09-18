"""Independent historical Git objects and complete current manifest coverage."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from tools import check_pipeline_lineages as lineage


def encode(records):
    return "".join(f"{digest}  {path}\n" for path, digest in sorted(records.items())).encode()


class PipelineLineageTests(unittest.TestCase):
    def git(self, root, *args):
        return subprocess.run(["git", "-c", "user.name=Lineage Test", "-c", "user.email=test@example.invalid",
                               "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args],
                              cwd=root, capture_output=True, check=True).stdout.decode().strip()

    def historical_fixture(self, directory):
        root = Path(directory)
        self.git(root, "init", "-q")
        contents = {"packages/hexapod_core/pyproject.toml": b"old packaging\n", "frozen.py": b"old contract\n"}
        records = {}
        for relative, content in contents.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            records[relative] = hashlib.sha256(content).hexdigest()
        manifest = root / lineage.ARCHIVED_MANIFEST
        manifest.parent.mkdir(parents=True)
        manifest.write_bytes(b"# historical fixture\n" + encode(records))
        self.git(root, "add", ".")
        self.git(root, "commit", "-qm", "Historical fixture")
        return root, dict(source_ref=self.git(root, "rev-parse", "HEAD"),
                         expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                         expected_count=len(records))

    def test_historical_verification_reads_pinned_git_files_independently_of_current_worktree(self):
        with tempfile.TemporaryDirectory() as directory:
            root, options = self.historical_fixture(directory)
            (root / "frozen.py").write_text("changed current source")
            (root / "packages/hexapod_core/pyproject.toml").write_text("new packaging")
            result = lineage.verify_historical(root, **options)
            self.assertTrue(result["pass"])
            self.assertEqual(result["files_verified"], 2)
            self.assertEqual((root / "frozen.py").read_text(), "changed current source")

    def test_historical_git_bytes_that_disagree_with_manifest_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root, options = self.historical_fixture(directory)
            (root / "frozen.py").write_text("wrong archived source")
            self.git(root, "add", "frozen.py")
            self.git(root, "commit", "-qm", "Invalid historical fixture")
            options["source_ref"] = self.git(root, "rev-parse", "HEAD")
            with self.assertRaisesRegex(ValueError, "do not match all frozen entries"):
                lineage.verify_historical(root, **options)

    def test_historical_missing_git_history_and_changed_manifest_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root, options = self.historical_fixture(directory)
            with self.assertRaisesRegex(ValueError, "unavailable"):
                lineage.verify_historical(root, **dict(options, source_ref="0"*40))
            manifest = root / lineage.ARCHIVED_MANIFEST
            manifest.write_bytes(manifest.read_bytes()+b"# unexpected edit\n")
            with self.assertRaisesRegex(ValueError, "manifest bytes changed"):
                lineage.verify_historical(root, **options)

    def test_historical_symlink_is_not_accepted_as_the_listed_regular_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root, options = self.historical_fixture(directory)
            (root / "frozen.py").unlink()
            (root / "frozen.py").symlink_to("packages/hexapod_core/pyproject.toml")
            self.git(root, "add", "frozen.py")
            self.git(root, "commit", "-qm", "Symlink fixture")
            options["source_ref"] = self.git(root, "rev-parse", "HEAD")
            with self.assertRaisesRegex(ValueError, "nonregular"):
                lineage.verify_historical(root, **options)

    def test_manifest_parser_rejects_duplicates_and_unsafe_paths(self):
        digest = "a"*64
        for content in (f"{digest}  a.py\n"*2, f"{digest}  ../escape\n", f"{digest}  /absolute\n",
                        f"{digest}  a//b\n", f"{digest}  .git/config\n", "bad digest  a.py\n", "# empty\n"):
            with self.subTest(content=content):
                with self.assertRaises(ValueError):
                    lineage.parse_manifest(content.encode())

    def test_current_coverage_rejects_missing_extra_and_changed_files(self):
        actual = {"old.py": "a"*64, "new_model.json": "b"*64}
        content = lineage.CURRENT_HEADER.encode()+encode(actual)
        self.assertEqual(lineage.compare_current_manifest(content, actual), 2)
        for records in ({"old.py": "a"*64}, dict(actual, extra="c"*64), dict(actual, **{"new_model.json": "c"*64})):
            with self.assertRaises(ValueError):
                lineage.compare_current_manifest(lineage.CURRENT_HEADER.encode()+encode(records), actual)
        with self.assertRaisesRegex(ValueError, "lineage header"):
            lineage.compare_current_manifest(content.replace(lineage.ARCHIVED_REF.encode(), b"0"*40), actual)

    def current_fixture(self, root):
        model = {}
        for key in ('urdf', 'model', 'usd'):
            path = root/'robot'/('approved.'+key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('approved '+key).encode())
            model[key] = {'path': path.relative_to(root).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        model['usd_sha256'] = model['usd']['sha256']
        model['usd'] = model['usd']['path']
        (root/'robot/active_model.json').write_text(json.dumps(model))
        return {'robot/active_model.json', *(('robot/approved.'+key) for key in ('urdf', 'model', 'usd'))}

    def test_current_model_identity_is_independent_of_historical_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.current_fixture(root)
            with patch.object(lineage, 'CURRENT_FILES', ('robot/active_model.json',)):
                records, revisions = lineage.current_records(root)
                self.assertEqual(set(records), expected)
                self.assertEqual(revisions, [])
                (root/'robot/approved.urdf').write_text('changed model')
                with self.assertRaisesRegex(ValueError, 'Selected robot identity differs'):
                    lineage.current_records(root)

    def test_generation_checks_history_and_refuses_to_overwrite_any_published_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "new.sha256"
            records = {"old.py": "a"*64, "new.py": "b"*64}
            with patch.object(lineage, "verify_historical", return_value={"pass": True}) as history, \
                 patch.object(lineage, "current_records", return_value=(records, [])):
                result = lineage.generate_current(root, output)
                history.assert_called_once_with(root.resolve())
                self.assertEqual(result["files"], 2)
                self.assertEqual(lineage.parse_manifest(output.read_bytes()), records)
                before = output.read_bytes()
                with self.assertRaises(FileExistsError):
                    lineage.generate_current(root, output)
                self.assertEqual(output.read_bytes(), before)
                with self.assertRaisesRegex(ValueError, "historical manifest cannot"):
                    lineage.generate_current(root, root / lineage.ARCHIVED_MANIFEST)

    def test_current_covers_top_level_prototype_and_cpu_tests_without_nested_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            covered = ("locomotion/env.py", "locomotion/tests/test_env.py")
            excluded = ("locomotion/results/copied_env.py",
                        "locomotion/tests/fixtures/copied_env.py")
            for relative in covered+excluded:
                path = root/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# Source-coverage fixture\n")
            model_paths = self.current_fixture(root)
            with patch.object(lineage, 'CURRENT_FILES', ('robot/active_model.json',)):
                records, _ = lineage.current_records(root)
                self.assertEqual(set(records), set(covered) | model_paths)
                content = lineage.CURRENT_HEADER.encode()+encode(records)
                (root/covered[0]).write_text("# Changed maintained controller fixture\n")
                changed, _ = lineage.current_records(root)
                with self.assertRaisesRegex(ValueError, "hashes differ"):
                    lineage.compare_current_manifest(content, changed)
                (root/covered[1]).unlink()
                (root/covered[1]).symlink_to(root/covered[0])
                with self.assertRaisesRegex(ValueError, "regular file"):
                    lineage.current_records(root)

    def test_generation_cannot_bypass_failed_history(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new.sha256"
            with patch.object(lineage, "verify_historical", side_effect=ValueError("bad history")), \
                 patch.object(lineage, "current_records") as records:
                with self.assertRaisesRegex(ValueError, "bad history"):
                    lineage.generate_current(Path(directory), output)
                records.assert_not_called()
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
