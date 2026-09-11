"""Portable byte/inventory verification; optional math replay uses NumPy separately."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
expected=json.loads((root/'FREEZE_SHA256.json').read_text())
files={p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='FREEZE_SHA256.json'}
assert set(files)==set(expected),'Unexpected/missing payload'
for name,digest in expected.items():
 p=files[name];assert not p.is_symlink()and p.resolve().is_relative_to(root),'Unsafe path'
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
print(json.dumps({'verified_payloads':len(files),'bytes':sum(p.stat().st_size for p in files.values()),'scope':'CPU proposal only; no native admission'}))
