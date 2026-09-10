"""Two unmeasured cases only; scalar/runtime/physics bytes stay exactly002."""
from pathlib import Path
import ast,hashlib,json,shutil
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent/'reference_directional_adapter_002/source_directional_002';SOURCE=HERE/'source_directional_003'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();parent='7c64602cae91478aebd4b03adcc79f56084db7529fd1c54c0439d061075a1c58'
if SOURCE.exists():raise FileExistsError(SOURCE)
assert sha(PARENT/'campaign_source_hashes.json')==parent
old=json.loads((PARENT/'campaign_source_hashes.json').read_text());assert len(old)==930
assert {str(p.relative_to(PARENT)) for p in PARENT.rglob('*') if p.is_file()}==set(old)|{'campaign_source_hashes.json'}
for f,h in old.items():assert sha(PARENT/f)==h,f
files=['directional_contract.py','launch_directional_physics_spark.py']
for f in files:ast.parse((HERE/f).read_text())
for f in ['directional_metrics.py','run_directional_physics.py']:assert sha(HERE/f)==sha(PARENT/'tools'/f),f
shutil.copytree(PARENT,SOURCE)
for f in files:shutil.copy2(HERE/f,SOURCE/'tools'/f)
o=json.loads((SOURCE/'source_origin.json').read_text());o['directional_discriminator003']={
 'parent_directional002_manifest_sha256':parent,'runtime_overlay_sha256':{f:sha(HERE/f) for f in files},
 'scope':'Fresh32standing then only left_turn and forward_right_arc; rejected reverse001 and left_strafe002 are not repeated',
 'changes':['two-case contract restriction and new identity','host descriptions say two cases'],
 'all_physics_control_metrics_serialization_and_wave005_bytes_unchanged':True,'fresh_exact_source_admission_required':True}
(SOURCE/'source_origin.json').write_text(json.dumps(o,indent=2)+'\n')
new={str(p.relative_to(SOURCE)):sha(p) for p in sorted(SOURCE.rglob('*')) if p.is_file() and p.name!='campaign_source_hashes.json'}
(SOURCE/'campaign_source_hashes.json').write_text(json.dumps(new,indent=2)+'\n')
changed={f:{'before':old.get(f),'after':h} for f,h in new.items() if old.get(f)!=h}
assert len(new)==930 and set(new)==set(old) and set(changed)=={'source_origin.json'}|{'tools/'+f for f in files}
r={'source':str(SOURCE),'parent_manifest_sha256':parent,'manifest_sha256':sha(SOURCE/'campaign_source_hashes.json'),'source_files':len(new),'exact_changed_paths':changed,'all927otherparent_payloads_unchanged':True,'no_parent_files_added_or_removed':True}
(HERE/'SOURCE_BUILD.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
