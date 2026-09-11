"""Verify this immutable compact review with the standard library."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=root/'FREEZE_SHA256.json'
expected=json.loads(manifest.read_text())
actual={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and p!=manifest}
assert actual==expected, 'Payload mismatch'
print(json.dumps({'verified':True,'payloads':len(actual),'freeze_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()},sort_keys=True))
