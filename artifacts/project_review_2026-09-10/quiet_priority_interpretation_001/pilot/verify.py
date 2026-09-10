"""Verify this immutable review and optional local referenced primary inputs."""
import argparse
import hashlib
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--repo', type=Path)
a = ap.parse_args()
root = Path(__file__).resolve().parent
files = json.loads((root / 'FREEZE_SHA256.json').read_text())['files']
assert {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} == set(files) | {'FREEZE_SHA256.json'}
for path, digest in files.items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path
print(f'PASS: {len(files)} review payloads')
if a.repo:
    inputs = {}
    for name, key in [('calculations.json', 'inputs_sha256'), ('physical_comparison.json', 'inputs_sha256'), ('PRIMARY_CODE_CHECK.json', 'bindings')]:
        inputs.update(json.loads((root / name).read_text())[key])
    for path, digest in inputs.items():
        assert hashlib.sha256((a.repo / path).read_bytes()).hexdigest() == digest, path
    print(f'PASS: {len(inputs)} referenced primary input hashes')
