from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
expected=json.loads((r/'BUNDLE_SHA256.json').read_text())
actual={p.relative_to(r).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()for p in r.rglob('*')if p.is_file()and p.name!='BUNDLE_SHA256.json'and '__pycache__'not in p.parts}
assert actual==expected
for name in ('rejected_draft','accepted_proposal'):
 d=r/name;e=json.loads((d/'FREEZE_SHA256.json').read_text());a={p.relative_to(d).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()for p in d.rglob('*')if p.is_file()and p.name!='FREEZE_SHA256.json'and '__pycache__'not in p.parts};assert a==e,name
print('PASS',len(actual),'payloads; reviewed design, no native actuation admission')
