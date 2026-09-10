"""Build a fresh full-C pair diagnostic from exact reference009, not origin reset work."""
from pathlib import Path
import ast,hashlib,json,shutil
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent/'reference_physics_adapter_009/source_009';PAIR=HERE.parent/'reference_load_transfer_001';SOURCE=HERE/'source_pair_001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent_hash='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
pair_hash='6cd797bee322d84328a6bbf9ff4aff289c57af45fbca27bc425d5e8d7a83e334'
if SOURCE.exists():raise FileExistsError(SOURCE)
assert sha(PARENT/'campaign_source_hashes.json')==parent_hash
old=json.loads((PARENT/'campaign_source_hashes.json').read_text())
assert {str(p.relative_to(PARENT)) for p in PARENT.rglob('*') if p.is_file()}==set(old)|{'campaign_source_hashes.json'}
for f,h in old.items():assert sha(PARENT/f)==h,f
assert sha(PAIR/'FREEZE_SHA256.json')==pair_hash
pair_files=json.loads((PAIR/'FREEZE_SHA256.json').read_text());pair_files=pair_files.get('files',pair_files)
for f,h in pair_files.items():assert sha(PAIR/f)==h,f
shared=['serial_geometry.py','wave_math.py','geometry/candidate_c_reference.json','geometry/f050_t060.urdf','reference_residual.py']
for f in shared:assert sha(PAIR/f)==sha(PARENT/'tools'/f),f
owned=['pair_screen_contract.py','run_pair_physics.py','pair_rollout.py','launch_pair_physics_spark.py']
cores=['load_transfer.py','score_transfer.py','quiet_contract.py','source004_wave_helpers.py']
for directory,files in [(HERE,owned),(PAIR,cores)]:
 for f in files:ast.parse((directory/f).read_text());assert not (PARENT/'tools'/f).exists(),f
shutil.copytree(PARENT,SOURCE)
for directory,files in [(HERE,owned),(PAIR,cores)]:
 for f in files:shutil.copy2(directory/f,SOURCE/'tools'/f)
origin=json.loads((SOURCE/'source_origin.json').read_text());origin['pair_diagnostic001']={
 'parent_reference009_manifest_sha256':parent_hash,'pair_owner_freeze_sha256':pair_hash,
 'scope':'Fresh32standing physical+quiet, then proposed four-corner-support LM/RM load-transfer diagnostic. No origin reset writes, wave or PPO.',
 'owned_overlays_sha256':{f:sha(HERE/f) for f in owned},'frozen_pair_dependencies_sha256':{f:sha(PAIR/f) for f in cores},
 'shared_geometry_math_core_verified_byte_identical':{f:sha(PAIR/f) for f in shared},
 'parent926source_payloads_unchanged_except_this_lineage_file':True,'no_existing_wave_gate_changed':True,'fresh_exact_source_admission_required':True}
(SOURCE/'source_origin.json').write_text(json.dumps(origin,indent=2)+'\n')
new={str(p.relative_to(SOURCE)):sha(p) for p in sorted(SOURCE.rglob('*')) if p.is_file() and p.name!='campaign_source_hashes.json'}
(SOURCE/'campaign_source_hashes.json').write_text(json.dumps(new,indent=2)+'\n')
changed={f:{'before':old.get(f),'after':h} for f,h in new.items() if old.get(f)!=h}
assert len(new)==934 and set(changed)=={'source_origin.json'}|{'tools/'+f for f in owned+cores}
assert not any(f in new for f in ['tools/matched_origin.py','tools/run_origin_physics.py'])
result={'source':str(SOURCE),'parent_manifest_sha256':parent_hash,'manifest_sha256':sha(SOURCE/'campaign_source_hashes.json'),'files':len(new),'exact_changed_or_new_paths':changed,'no_parent_paths_removed':True,'shared_dependencies_checked':shared,'pair_owner_freeze_sha256':pair_hash}
(HERE/'SOURCE_BUILD.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
