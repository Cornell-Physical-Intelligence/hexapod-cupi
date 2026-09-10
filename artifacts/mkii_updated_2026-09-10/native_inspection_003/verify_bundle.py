from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
expected=json.loads((r/'BUNDLE_SHA256.json').read_text())
actual={p.relative_to(r).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()for p in r.rglob('*')if p.is_file()and p.name!='BUNDLE_SHA256.json'and '__pycache__'not in p.parts}
assert actual==expected, 'Published payload mismatch'
for directory in ('inspector','host','guard','auditor'):
 root=r/directory;e=json.loads((root/'FREEZE_SHA256.json').read_text())
 a={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()for p in root.rglob('*')if p.is_file()and p.name!='FREEZE_SHA256.json'and '__pycache__'not in p.parts}
 assert a==e,directory
raw=json.loads((r/'terminal/RAW_SHA256.json').read_text())
for name,digest in raw.items():assert hashlib.sha256((r/'terminal'/name).read_bytes()).hexdigest()==digest,name
report=json.loads((r/'terminal/audit.json').read_text())
assert report['audit_verified']and report['inspection_completed']and report['terminal_outcome']=='authentic_completed_inspection'
print('PASS',len(actual),'payloads; completed passive inspection, no physical admission')
