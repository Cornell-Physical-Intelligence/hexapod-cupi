"""Portable read-only payload integrity verification."""
from pathlib import Path
import hashlib, json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'FREEZE_SHA256.json').read_text())
actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='FREEZE_SHA256.json'}
assert actual==set(manifest['files']), 'Unexpected or missing payload'
for name,expected in manifest['files'].items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
print(json.dumps({'verified_files':len(actual),'scope':manifest['scope']}))
