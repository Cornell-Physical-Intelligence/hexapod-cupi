"""Fresh exact middlepair001 derivative, no mutation of its immutable source."""
from pathlib import Path
import ast,hashlib,json,shutil
H=Path(__file__).resolve().parent;P=H.parent/'reference_pair_physics_adapter_001/source_pair_001';S=H/'source_diagonal_pair_001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not S.exists();parent='69a1a23304448d53f7681dfa87ab4fa652ad682ba0e45406e82187a66c4224f0'
assert sha(P/'campaign_source_hashes.json')==parent;m=json.loads((P/'campaign_source_hashes.json').read_text())
assert len(m)==934 and {str(f.relative_to(P)) for f in P.rglob('*') if f.is_file()}==set(m)|{'campaign_source_hashes.json'}
for f,h in m.items():assert sha(P/f)==h,f
owned=['load_transfer.py','score_transfer.py','pair_screen_contract.py','run_pair_physics.py','launch_pair_physics_spark.py']
for f in owned:ast.parse((H/f).read_text())
shutil.copytree(P,S)
for f in owned:shutil.copy2(H/f,S/'tools'/f)
o=json.loads((S/'source_origin.json').read_text());o['diagonal_static_pair001']={'parent_middlepair001_manifest_sha256':parent,'case_order':['lf_rr','lr_rf'],'scope':'Fresh32 physical+quiet standing then two cold static opposing-diagonal transfer diagnostics. No wave/PPO or prescribed body motion.','owned_overlays':{f:sha(H/f) for f in owned},'only_named_pair_selection_and_case_identity_dispatch_changed':True,'same_numeric_support_quiet_torque_target_and_timing_contract':True,'no_middle_pair_repeat':True,'no_faster_gait_admission':True}
(S/'source_origin.json').write_text(json.dumps(o,indent=2)+'\n')
n={str(f.relative_to(S)):sha(f) for f in sorted(S.rglob('*')) if f.is_file() and f!=S/'campaign_source_hashes.json'}
(S/'campaign_source_hashes.json').write_text(json.dumps(n,indent=2)+'\n');delta={f:{'before':m.get(f),'after':h} for f,h in n.items() if m.get(f)!=h}
assert len(n)==934 and set(delta)=={'source_origin.json'}|{'tools/'+f for f in owned}
r={'parent_manifest_sha256':parent,'manifest_sha256':sha(S/'campaign_source_hashes.json'),'source_files':934,'unchanged_parent_payloads':928,'exact_changed_paths':delta,'source':str(S),'no_parent_paths_removed':True}
(H/'SOURCE_BUILD.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
