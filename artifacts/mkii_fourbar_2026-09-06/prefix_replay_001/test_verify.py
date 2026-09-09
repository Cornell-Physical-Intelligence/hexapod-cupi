"""Compact evidence checks fail closed without needing raw NPZs or a GPU."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prefix_evidence_verify', HERE/'verify.py')
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def freeze(folder):
    files = sorted(path for path in folder.iterdir() if path.is_file() and path.name != 'SHA256SUMS')
    (folder/'SHA256SUMS').write_text(''.join(f'{verify.digest(path)}  {path.name}\n' for path in files))


class PrefixEvidenceVerificationTests(unittest.TestCase):
    def fixture(self, directory):
        folder = Path(directory)
        for path in HERE.iterdir():
            if path.is_file():
                shutil.copyfile(path, folder/path.name)
        freeze(folder)
        return folder

    def test_complete_compact_evidence_is_explicit_about_remote_raw_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            result = verify.verify(self.fixture(directory))
            self.assertEqual(result['recorded_remote_npz_files'], 37)
            self.assertEqual(result['raw_npz_bytes_rehashed_this_invocation'], 0)
            self.assertEqual(result['raw_npz_files_rehashed_this_invocation'], 0)
            self.assertTrue(result['diagnostic_complete'])
            self.assertFalse(result['primary_pass'])
            self.assertFalse(result['simulation_training_admission'])

    def test_changed_original_log_fails_local_hash_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = self.fixture(directory)
            with (folder/'container.log').open('ab') as stream:
                stream.write(b'altered evidence')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                verify.verify(folder)

    def test_remote_inventory_cannot_substitute_different_raw_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = self.fixture(directory)
            path = folder/'remote_inventory.json'
            data = json.loads(path.read_text())
            records = data['files']
            row = (next(r for r in records if r['relative_path'] == 'trace_000.npz')
                   if isinstance(records, list) else records['trace_000.npz'])
            row['sha256'] = '0'*64
            path.write_text(json.dumps(data))
            freeze(folder)  # Exercise cross-record consistency, not just the checksum.
            with self.assertRaisesRegex(ValueError, 'Trace hash'):
                verify.verify(folder)

    def test_missing_raw_file_fails_when_rehash_is_explicitly_requested(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as raw:
            folder = self.fixture(directory)
            with self.assertRaises(FileNotFoundError):
                verify.verify(folder, raw)


if __name__ == '__main__':
    unittest.main()
