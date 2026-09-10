"""Fresh directional-only overlay on exact source009; no pair/origin imports."""
from pathlib import Path
import ast,hashlib,json,shutil
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent/'reference_physics_adapter_009/source_009';SOURCE=HERE/'source_directional_001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();parent='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
if SOURCE.exists():raise FileExistsError(SOURCE)
assert sha(PARENT/'campaign_source_hashes.json')==parent
old=json.loads((PARENT/'campaign_source_hashes.json').read_text())
assert {str(p.relative_to(PARENT)) for p in PARENT.rglob('*') if p.is_file()}==set(old)|{'campaign_source_hashes.json'}
for f,h in old.items():assert sha(PARENT/f)==h,f
files=['directional_contract.py','directional_metrics.py','run_directional_physics.py','launch_directional_physics_spark.py']
for f in files:ast.parse((HERE/f).read_text());assert not (PARENT/'tools'/f).exists()
shutil.copytree(PARENT,SOURCE)
for f in files:shutil.copy2(HERE/f,SOURCE/'tools'/f)
o=json.loads((SOURCE/'source_origin.json').read_text());o['directional_discriminator001']={
 'parent_source009_manifest_sha256':parent,'runtime_additions_sha256':{f:sha(HERE/f) for f in files},
 'scope':'Fresh32standing then four independent cold low-speed directional reference cases; no all-direction/PPO qualification',
 'all_original_runtime_bytes_unchanged':True,'no_origin_pose_injection_or_pair_adapter':True,'fresh_exact_source_admission_required':True}
(SOURCE/'source_origin.json').write_text(json.dumps(o,indent=2)+'\n')
new={str(p.relative_to(SOURCE)):sha(p) for p in sorted(SOURCE.rglob('*')) if p.is_file() and p.name!='campaign_source_hashes.json'}
(SOURCE/'campaign_source_hashes.json').write_text(json.dumps(new,indent=2)+'\n')
changed={f:{'before':old.get(f),'after':h} for f,h in new.items() if old.get(f)!=h}
assert len(new)==930 and set(changed)=={'source_origin.json'}|{'tools/'+f for f in files}
r={'source':str(SOURCE),'parent_manifest_sha256':parent,'manifest_sha256':sha(SOURCE/'campaign_source_hashes.json'),'source_files':len(new),'exact_changed_or_new_paths':changed,'all925otherparent_payloads_unchanged':True,'no_parent_files_removed':True}
(HERE/'SOURCE_BUILD.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
