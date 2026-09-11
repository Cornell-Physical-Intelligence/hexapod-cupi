from pathlib import Path
import hashlib,json
p=Path(__file__).parent;m=json.loads((p/'FREEZE_SHA256.json').read_text())
assert {str(f.relative_to(p))for f in p.rglob('*')if f.is_file()and f.name!='FREEZE_SHA256.json'}==set(m)
for f,h in m.items():assert hashlib.sha256((p/f).read_bytes()).hexdigest()==h,f
print('PASS',len(m),'payloads')
