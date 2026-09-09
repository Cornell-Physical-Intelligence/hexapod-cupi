"""Independent historical Git objects and complete current manifest coverage."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import check_pipeline_lineages as lineage


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

    def test_current_retains_all_archived_paths_and_only_allows_named_packaging_revisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archived = {path: hashlib.sha256(b"old").hexdigest() for path in (*lineage.CURRENT_REVISIONS, "frozen.py")}
            for relative in archived:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"new package metadata" if relative in lineage.CURRENT_REVISIONS else b"old")
            runtime = {"new_model.json": "d"*64}
            with patch.object(lineage, "read_archived_manifest", return_value=(b"fixture", archived)), \
                 patch.object(lineage, "identity", return_value={"files": runtime}), \
                 patch.object(lineage, "CURRENT_EXTRA_PATHS", ()):
                records, revisions = lineage.current_records(root)
                self.assertEqual(set(records), set(archived) | set(runtime))
                self.assertEqual(set(revisions), set(lineage.CURRENT_REVISIONS))
                (root / "frozen.py").write_bytes(b"changed contract")
                with self.assertRaisesRegex(ValueError, "outside the explicit packaging revisions"):
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
