"""Portable stdlib payload/selected-oracle verifier; never imports native code."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
manifest=json.loads((R/'FREEZE_SHA256.json').read_text());files=manifest['files']
actual={str(p.relative_to(R))for p in R.rglob('*')if p.is_file()and p.name!='FREEZE_SHA256.json'and '__pycache__'not in p.parts}
if actual!=set(files):raise SystemExit('Unexpected/missing payload inventory')
for rel,digest in files.items():
 p=R/rel
 if p.is_symlink()or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise SystemExit('Changed payload:'+rel)
for original,entry in json.loads((R/'INPUTS.json').read_text()).items():
 if 'selected_copy'in entry and hashlib.sha256((R/entry['selected_copy']).read_bytes()).hexdigest()!=entry['sha256']:raise SystemExit('Changed selected oracle:'+original)
print(json.dumps({'verified_payloads':len(files),'selected_oracles':5,'native_execution':False,'physical_admission':False}))
