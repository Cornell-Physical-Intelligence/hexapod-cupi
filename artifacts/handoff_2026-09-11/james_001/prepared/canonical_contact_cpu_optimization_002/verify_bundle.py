from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parent;m=json.loads((p/'FREEZE_SHA256.json').read_text())
a={str(f.relative_to(p))for f in p.rglob('*')if f.is_file()and '__pycache__'not in f.parts and f.name!='FREEZE_SHA256.json'}
assert a==set(m)
for name,want in m.items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==want,name
print('PASS',len(m),'payloads; integrity only')
