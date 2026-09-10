"""Standard-library integrity verifier for this CPU proposal bundle."""
from pathlib import Path
import hashlib,json

root=Path(__file__).resolve().parent
expected=json.loads((root/'FREEZE_SHA256.json').read_text())
actual={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
 for p in root.rglob('*')if p.is_file()and p.name!='FREEZE_SHA256.json'and'__pycache__'not in p.parts}
if actual!=expected:raise SystemExit('Bundle inventory/hash mismatch')
print(f'PASS {len(actual)} exact CPU proposal payloads; no native or policy admission')
