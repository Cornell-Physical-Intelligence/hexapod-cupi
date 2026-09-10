"""Portable, read-only verification of this immutable preparation bundle."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'FREEZE_SHA256.json').read_text())
actual={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'FREEZE_SHA256.json'}
assert not any(p.is_symlink() for p in root.rglob('*')), 'Symbolic payload'
assert actual==manifest, 'Missing, added or changed payload'
print(json.dumps({'verified':True,'payloads':len(actual),'freeze_sha256':hashlib.sha256((root/'FREEZE_SHA256.json').read_bytes()).hexdigest()}))
