"""Read-only, standard-library verification of this independent receipt."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'FREEZE_SHA256.json').read_text())['files']
actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
assert actual == set(manifest) | {'FREEZE_SHA256.json'}, 'Receipt inventory changed'
for name, digest in manifest.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
report = json.loads((root / 'report.json').read_text())
owner = root.parent / report['owner_directory']
if owner.is_dir():
    freeze = owner / 'FREEZE_SHA256.json'
    assert hashlib.sha256(freeze.read_bytes()).hexdigest() == report['owner_freeze_sha256']
    files = json.loads(freeze.read_text())['files']
    assert {p.relative_to(owner).as_posix() for p in owner.rglob('*') if p.is_file()} == set(files) | {'FREEZE_SHA256.json'}
    for name, digest in files.items():
        assert hashlib.sha256((owner / name).read_bytes()).hexdigest() == digest, name
    print(f'PASS: {len(manifest)} review payloads and {len(files)} original owner payloads')
else:
    print(f'PASS: {len(manifest)} review payloads; referenced owner bundle is not present')
