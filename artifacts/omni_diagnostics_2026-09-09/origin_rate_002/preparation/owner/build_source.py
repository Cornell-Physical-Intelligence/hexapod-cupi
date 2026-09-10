"""Fresh933-file successor; only origin-view correction, protocol and lineage differ."""
from pathlib import Path
import ast,hashlib,json,shutil
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent/'reference_origin_adapter_001/source_origin_001';SOURCE=HERE/'source_origin_002'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();expected='47d190425712e7aec39c959fed3fa76aaaef9af6962860593cc0b558cd0df63f'
if SOURCE.exists():raise FileExistsError(SOURCE)
assert sha(PARENT/'campaign_source_hashes.json')==expected
old=json.loads((PARENT/'campaign_source_hashes.json').read_text())
assert {str(p.relative_to(PARENT)) for p in PARENT.rglob('*') if p.is_file()}==set(old)|{'campaign_source_hashes.json'}
for f,h in old.items():assert sha(PARENT/f)==h,f
for f in ['matched_origin.py','origin_contract.py']:ast.parse((HERE/f).read_text())
shutil.copytree(PARENT,SOURCE)
for f in ['matched_origin.py','origin_contract.py']:shutil.copy2(HERE/f,SOURCE/'tools'/f)
origin=json.loads((SOURCE/'source_origin.json').read_text());origin['origin002_correction']={
 'parent_origin001_manifest_sha256':expected,'scope':'cold-reset origin view/storage correction only; no solver, waveform, physical/quiet gate or measurement change',
 'overlays_sha256':{f:sha(HERE/f) for f in ['matched_origin.py','origin_contract.py']},'source001_preserved':True,'fresh32standing_required':True}
(SOURCE/'source_origin.json').write_text(json.dumps(origin,indent=2)+'\n')
new={str(p.relative_to(SOURCE)):sha(p) for p in sorted(SOURCE.rglob('*')) if p.is_file() and p.name!='campaign_source_hashes.json'}
(SOURCE/'campaign_source_hashes.json').write_text(json.dumps(new,indent=2)+'\n')
changed={f:{'before':old.get(f),'after':h} for f,h in new.items() if old.get(f)!=h}
assert len(new)==933 and set(changed)=={'source_origin.json','tools/matched_origin.py','tools/origin_contract.py'}
result={'source':str(SOURCE),'parent_manifest_sha256':expected,'manifest_sha256':sha(SOURCE/'campaign_source_hashes.json'),'files':len(new),'exact_changed_paths':changed,'unchanged930parent_payloads':True}
(HERE/'SOURCE_BUILD.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
