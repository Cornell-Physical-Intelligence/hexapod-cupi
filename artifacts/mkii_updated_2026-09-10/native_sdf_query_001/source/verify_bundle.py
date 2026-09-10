from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parent
m=json.loads((p/'FREEZE_SHA256.json').read_text())
actual={str(x.relative_to(p))for x in p.rglob('*')if x.is_file()and '__pycache__'not in x.parts and x!=p/'FREEZE_SHA256.json'}
assert set(m)==actual
for name,digest in m.items():
 x=p/name
 assert not x.is_symlink()and x.resolve().is_relative_to(p)
 assert hashlib.sha256(x.read_bytes()).hexdigest()==digest,name
print('PASS',len(m),'payloads',hashlib.sha256((p/'FREEZE_SHA256.json').read_bytes()).hexdigest())
