"""Portable read-only receipt verifier, optionally rechecking referenced inputs."""
import argparse
import hashlib
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--repo', type=Path)
a = ap.parse_args()
root = Path(__file__).resolve().parent
files = json.loads((root / 'FREEZE_SHA256.json').read_text())['files']
actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
assert actual == set(files) | {'FREEZE_SHA256.json'}, 'Receipt inventory changed'
for name, digest in files.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
print(f'PASS: {len(files)} immutable review payloads')
if a.repo:
    inputs = json.loads((root / 'calculations_003.json').read_text())['inputs_sha256']
    for name, digest in inputs.items():
        assert hashlib.sha256((a.repo / name).read_bytes()).hexdigest() == digest, name
    print(f'PASS: {len(inputs)} referenced input hashes')
