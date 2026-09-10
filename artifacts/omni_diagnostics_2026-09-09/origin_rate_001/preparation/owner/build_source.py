"""Build a fresh immutable origin diagnostic from exact frozen009 + explicit overlays."""
from pathlib import Path
import ast,hashlib,json,shutil
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'reference_physics_adapter_009/source_009'
SOURCE=HERE/'source_origin_001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
EXPECTED='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
RUNTIME=['origin_contract.py','matched_origin.py','run_origin_physics.py','launch_origin_physics_spark.py','origin_metrics.py','origin_campaign_review.py','origin_initial_state.json']
if SOURCE.exists():raise FileExistsError(SOURCE)
assert sha(PARENT/'campaign_source_hashes.json')==EXPECTED
mapping=json.loads((PARENT/'campaign_source_hashes.json').read_text())
assert {str(p.relative_to(PARENT)) for p in PARENT.rglob('*') if p.is_file()}==set(mapping)|{'campaign_source_hashes.json'}
for f,h in mapping.items():assert sha(PARENT/f)==h,f
for f in RUNTIME:
 if f.endswith('.py'):ast.parse((HERE/f).read_text())
shutil.copytree(PARENT,SOURCE)
for f in RUNTIME:shutil.copy2(HERE/f,SOURCE/'tools'/f)
origin=json.loads((SOURCE/'source_origin.json').read_text())
origin['matched_origin_diagnostic_001']={'parent_source009_manifest_sha256':EXPECTED,
 'initial_state_sha256':sha(HERE/'origin_initial_state.json'),'new_runtime_overlay_sha256':{f:sha(HERE/f) for f in RUNTIME},
 'scope':'fresh standing plus five matched cold single-env origin measurements; no wave/PPO or production adoption',
 'all_parent_physics_controller_assets_and_gates_preserved':True,'case_reset_only_intervention':'audited009env6 generalized initial state, common across cases exceptglobalXY'}
(SOURCE/'source_origin.json').write_text(json.dumps(origin,indent=2)+'\n')
new={str(p.relative_to(SOURCE)):sha(p) for p in sorted(SOURCE.rglob('*')) if p.is_file() and p.name!='campaign_source_hashes.json'}
(SOURCE/'campaign_source_hashes.json').write_text(json.dumps(new,indent=2)+'\n')
changed={f:{'before':mapping.get(f),'after':h} for f,h in new.items() if mapping.get(f)!=h}
assert set(changed)=={'source_origin.json',*('tools/'+f for f in RUNTIME)}
result={'source':str(SOURCE),'parent_manifest_sha256':EXPECTED,'manifest_sha256':sha(SOURCE/'campaign_source_hashes.json'),'files':len(new),'exact_changed_or_new_paths':changed,'parent_removed_paths':sorted(set(mapping)-set(new)),'ready_for_CPU_preflight_only':True}
(HERE/'SOURCE_BUILD.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
