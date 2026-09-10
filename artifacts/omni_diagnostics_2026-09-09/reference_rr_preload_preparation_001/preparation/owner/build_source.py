"""Create a new exact-source diagnostic; never alter immutable directional002."""
from pathlib import Path
import ast,hashlib,json,shutil
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'reference_directional_adapter_002/source_directional_002'
SOURCE=HERE/'source_rr_preload_001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
PARENT_SHA='7c64602cae91478aebd4b03adcc79f56084db7529fd1c54c0439d061075a1c58'
if SOURCE.exists():raise FileExistsError(SOURCE)
assert sha(PARENT/'campaign_source_hashes.json')==PARENT_SHA
old=json.loads((PARENT/'campaign_source_hashes.json').read_text())
assert {str(p.relative_to(PARENT)) for p in PARENT.rglob('*') if p.is_file()}==set(old)|{'campaign_source_hashes.json'}
for f,h in old.items():assert sha(PARENT/f)==h,f
for filename in ['directional_contract.py','solver_comparison.py']:
 p=HERE/filename;s=p.read_text().replace('WAVE_HASH_UNBOUND',sha(HERE/'wave_reference.py')).replace('HELPER_HASH_UNBOUND',sha(HERE/'rr_preload_diagnostic.py'));p.write_text(s)
files=['wave_reference.py','rr_preload_diagnostic.py','directional_contract.py','launch_directional_physics_spark.py','solver_comparison.py']
for f in files:ast.parse((HERE/f).read_text())
shutil.copytree(PARENT,SOURCE)
for f in files:shutil.copy2(HERE/f,SOURCE/'tools'/f)
o=json.loads((SOURCE/'source_origin.json').read_text());o['rr_first_landing_preload_diagnostic001']={
 'parent_directional002_manifest_sha256':PARENT_SHA,'runtime_overlay_sha256':{f:sha(HERE/f) for f in files},
 'scope':'Fresh32standing then one cold left_strafe; first confirmedRR landing only0.5mm downwardC2 during existing0.3s hold',
 'case_is_an_experiment_not_admission':True,'gait_cadence_body_twist_geometry_and_existing_physical_gates_unchanged':True,
 'old_observation_code_preserved_not_bound_to_new_diagnostic_state':True,'no_actor_loaded_or_trained':True,
 'actual_strafe_diagnosis_freeze_sha256':'8670b8e6acc2becbc7fa55f9de9f8056a5da5902029e53978a1031681303cf23',
 'arc003_has_RM_support_loss_and_is_not_addressed':True}
(SOURCE/'source_origin.json').write_text(json.dumps(o,indent=2)+'\n')
new={str(p.relative_to(SOURCE)):sha(p) for p in sorted(SOURCE.rglob('*')) if p.is_file() and p.name!='campaign_source_hashes.json'}
(SOURCE/'campaign_source_hashes.json').write_text(json.dumps(new,indent=2)+'\n')
changed={f:{'before':old.get(f),'after':h} for f,h in new.items() if old.get(f)!=h}
assert len(new)==931 and set(new)-set(old)=={'tools/rr_preload_diagnostic.py'}
assert set(changed)=={'source_origin.json'}|{'tools/'+f for f in files}
r={'source':str(SOURCE),'parent_manifest_sha256':PARENT_SHA,'manifest_sha256':sha(SOURCE/'campaign_source_hashes.json'),'source_files':len(new),'exact_changed_paths':changed,'all925otherparent_payloads_unchanged':True,'parent_payloads_removed':[],'new_payloads':['tools/rr_preload_diagnostic.py']}
(HERE/'SOURCE_BUILD.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
