"""Check history-free publication and bounded restoration from pinned Git."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from tools import archive

ROOT = Path(__file__).resolve().parents[1]


class ArchiveTests(unittest.TestCase):
    def test_retained_fixture_bytes_match_the_recorded_sources(self):
        manifest = json.loads((ROOT/'tests/fixtures.json').read_text())
        for name, row in manifest['files'].items():
            with self.subTest(path=name):
                self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), row['sha256'])

    def fixture(self, root):
        def git(*args):
            return subprocess.check_output(['git', '-c', 'user.name=Archive Test',
                '-c', 'user.email=test@example.invalid', '-c', 'commit.gpgsign=false',
                '-c', 'core.hooksPath=/dev/null', *args], cwd=root, stderr=subprocess.PIPE).decode().strip()
        git('init', '-q')
        (root/'old').mkdir()
        (root/'old/data').write_bytes(b'original result')
        (root/'old/link').symlink_to('data')
        git('add', 'old')
        git('commit', '-qm', 'Archive fixture')
        commit = git('rev-parse', 'HEAD')
        data = {'repository': 'example/project', 'commit': commit,
            'tree': git('rev-parse', 'HEAD^{tree}'),
            'references': {'old/data': {'commit': commit, 'path': 'old/data',
                'kind': 'blob', 'git_oid': git('rev-parse', 'HEAD:old/data')}},
            'retained': {'old/data': {'path': 'selected',
                'sha256': hashlib.sha256(b'original result').hexdigest()}}}
        (root/'configs').mkdir()
        (root/'configs/archive.json').write_text(json.dumps(data))
        (root/'selected').write_bytes(b'original result')
        return data

    def test_restore_uses_pinned_bytes_and_refuses_overwrite_or_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root/'old/data').write_text('changed working tree')
            archive.check(root, git_objects=True)
            archive.restore('old/data', root/'restored', root)
            self.assertEqual((root/'restored/old/data').read_bytes(), b'original result')
            with self.assertRaises(FileExistsError):
                archive.restore('old/data', root/'restored', root)
            with self.assertRaisesRegex(ValueError, 'regular files'):
                archive.restore('old', root/'links', root)
            self.assertFalse((root/'links').exists())

    def test_publication_resolves_without_git_and_rejects_changed_retained_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = self.fixture(root)
            (root/'.git').rename(root/'history')
            archive.check(root)
            self.assertEqual(archive.retained('old/data', root), root/'selected')
            self.assertIn(data['commit'], archive.url('old/data', root))
            self.assertIsNone(archive.reference('missing', root))
            (root/'selected').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                archive.check(root)

    def test_archive_paths_reject_escapes(self):
        for value in ('../outside', '/absolute', '.git/config', 'a/../b', 'a//b', '-option'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                archive.safe_path(value)
