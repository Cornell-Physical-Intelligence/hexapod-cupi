from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
e=json.loads((r/'FREEZE_SHA256.json').read_text())
a={p.relative_to(r).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()for p in r.rglob('*')if p.is_file()and p!=r/'FREEZE_SHA256.json'}
assert a==e
print('PASS',len(a),'binding-review payloads; CPU only')
