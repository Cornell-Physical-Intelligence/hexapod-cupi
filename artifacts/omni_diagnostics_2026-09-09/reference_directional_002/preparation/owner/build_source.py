"""Narrow successor: three unmeasured directions and strict result finalization."""
from pathlib import Path
import ast,hashlib,json,shutil
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'reference_directional_adapter_001/source_directional_001'
SOURCE=HERE/'source_directional_002'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent='373c9ea08406f6f3593279fa9fda24badf56ae86bd938e140dba7d734b5ee919'
if SOURCE.exists():raise FileExistsError(SOURCE)
assert sha(PARENT/'campaign_source_hashes.json')==parent
old=json.loads((PARENT/'campaign_source_hashes.json').read_text())
assert {str(p.relative_to(PARENT)) for p in PARENT.rglob('*') if p.is_file()}==set(old)|{'campaign_source_hashes.json'}
for f,h in old.items():assert sha(PARENT/f)==h,f
files=['directional_contract.py','directional_metrics.py','run_directional_physics.py','launch_directional_physics_spark.py']
for f in files:ast.parse((HERE/f).read_text());assert (PARENT/'tools'/f).is_file()
shutil.copytree(PARENT,SOURCE)
for f in files:shutil.copy2(HERE/f,SOURCE/'tools'/f)
o=json.loads((SOURCE/'source_origin.json').read_text());o['directional_discriminator002']={
 'parent_directional001_manifest_sha256':parent,'runtime_overlay_sha256':{f:sha(HERE/f) for f in files},
 'scope':'Fresh32standing then three previously unmeasured cold directions; reverse001 remains rejected and is not repeated',
 'changes':['drop already measured reverse from new declared cases','convert NumPy yaw_required scalar to builtin bool','strict serialization and robust failed-state finalization','host result reports actual declared case count'],
 'wave005_physical_gates_and_control_path_unchanged':True,'fresh_exact_source_admission_required':True,
 'reverse001_review_sha256':sha(HERE/'reverse001_replay.json')}
(SOURCE/'source_origin.json').write_text(json.dumps(o,indent=2)+'\n')
new={str(p.relative_to(SOURCE)):sha(p) for p in sorted(SOURCE.rglob('*')) if p.is_file() and p.name!='campaign_source_hashes.json'}
(SOURCE/'campaign_source_hashes.json').write_text(json.dumps(new,indent=2)+'\n')
changed={f:{'before':old.get(f),'after':h} for f,h in new.items() if old.get(f)!=h}
assert len(new)==930 and set(new)==set(old) and set(changed)=={'source_origin.json'}|{'tools/'+f for f in files}
r={'source':str(SOURCE),'parent_manifest_sha256':parent,'manifest_sha256':sha(SOURCE/'campaign_source_hashes.json'),'source_files':len(new),'exact_changed_paths':changed,'all925otherparent_payloads_unchanged':True,'no_parent_files_added_or_removed':True}
(HERE/'SOURCE_BUILD.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
