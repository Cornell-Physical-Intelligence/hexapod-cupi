"""Portable stdlib integrity verification; not physical admission."""
from pathlib import Path
import hashlib,json
p=Path(__file__).parent;m=json.loads((p/'FREEZE_SHA256.json').read_text())
actual={str(x.relative_to(p))for x in p.rglob('*')if x.is_file()and '__pycache__'not in x.parts and x.name!='FREEZE_SHA256.json'}
assert actual==set(m)
for f,h in m.items():assert hashlib.sha256((p/f).read_bytes()).hexdigest()==h,f
print('PASS',len(m),'payloads')
