from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
expected=json.loads((r/'FREEZE_SHA256.json').read_text())
assert not any(p.is_symlink()for p in r.rglob('*'))
actual={p.relative_to(r).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()for p in r.rglob('*')if p.is_file()and p!=r/'FREEZE_SHA256.json'and '__pycache__'not in p.parts}
assert actual==expected,'Changed/unlisted host input'
print('PASS',len(actual),'host payloads; no standing admission or native allocation')
