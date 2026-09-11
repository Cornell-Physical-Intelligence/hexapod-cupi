from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
m=json.loads((r/'FREEZE_SHA256.json').read_text())
a={p.relative_to(r).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()for p in r.rglob('*')if p.is_file()and p.name!='FREEZE_SHA256.json'}
assert m==a
print('PASS',len(a),'review payloads')
