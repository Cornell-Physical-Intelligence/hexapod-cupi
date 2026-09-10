"""Assemble immutable actual origin002 evidence without mutating source/results."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'reference_origin_publication_002';RESULT=ROOT/'reference_origin_results_002'
SOURCE=ROOT/'reference_origin_adapter_002/source_origin_002'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if OUT.exists():raise FileExistsError(OUT)
audit=json.loads((RESULT/'remote_audit.json').read_text());assert len(audit['raw_payloads'])==96
for f,h in audit['raw_payloads'].items():assert sha(RESULT/f)==h,f
assert audit['source_unchanged'] and audit['admitted_assets_unchanged'] and audit['source_files']==933 and audit['admitted_asset_files']==550
assert audit['source_manifest_sha256']=='c028664ae782b053f0e126ace1e7a24e6eb8a89cf8dbaa9fd8b5628d28560160'
assert sha(SOURCE/'campaign_source_hashes.json')==audit['source_manifest_sha256']
source_map=json.loads((SOURCE/'campaign_source_hashes.json').read_text())
assert {str(p.relative_to(SOURCE)) for p in SOURCE.rglob('*') if p.is_file()}==set(source_map)|{'campaign_source_hashes.json'}
for f,h in source_map.items():assert sha(SOURCE/f)==h,f
asset_map=json.loads((RESULT/'run/inputs/study_before.sha256.json').read_text());assert len(asset_map)==550
for f,h in asset_map.items():assert sha(SOURCE/'robot/hexapod_mkii_length_study'/f)==h,f
for name,state in audit['owned_containers_absent'].items():assert state['returncode']!=0 and ('no such object' in state['stderr'].lower() or 'no such container' in state['stderr'].lower()),name
assert len(audit['owned_containers_absent'])==12 and 'Result=success' in audit['unit'] and 'ActiveState=inactive' in audit['unit']
assert json.loads((RESULT/'forecast_pause/restored.json').read_text())==audit['pause_restoration']
OUT.mkdir();copied_receipts={}
def copy_file(p,t):
 t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t);assert sha(p)==sha(t)
def frozen(name,path,manifest_name,expected):
 assert sha(path/manifest_name)==expected
 m=json.loads((path/manifest_name).read_text());m=m.get('files',m)
 for f,h in m.items():assert sha(path/f)==h,f;copy_file(path/f,OUT/name/f)
 copy_file(path/manifest_name,OUT/name/manifest_name)
 copied_receipts[name]={'map':manifest_name,'sha256':expected,'payloads':len(m)}
frozen('preparation',ROOT/'reference_origin_preparation_002','BUNDLE_SHA256.json','1c553e9567d678fa9a8ea0581a86c34ac1684fa65ec1514c0789c0c5fe3e9901')
frozen('launch_guard',ROOT/'reference_origin_launch_002','FREEZE_SHA256.json','ec211b0c46fff6010103fe6ca1d822ecde906ba6c90e18d0d8c07b64dbf5f1ad')
frozen('independent_actual_review',ROOT/'reference_origin_actual002_independent_review','FREEZE_SHA256.json','0dfd92a498d4adba120a15c1a704fac81db57836dc806fdec0d6226232daf91b')
for f in audit['raw_payloads']:copy_file(RESULT/f,OUT/'result'/f)
for f in ['remote_audit.json','verify_remote.py','root_review/analyze.py','root_review/report.json']:copy_file(RESULT/f,OUT/'result'/f)
root_review={f:sha(OUT/'result/root_review'/f) for f in ['analyze.py','report.json']}
(OUT/'result/root_review/FREEZE_SHA256.json').write_text(json.dumps(root_review,indent=2)+'\n')
review=json.loads((OUT/'result/root_review/report.json').read_text());independent=json.loads((OUT/'independent_actual_review/report.json').read_text())
assert review['all_existing_physical_passed'] and review['all_existing_quiet_passed'] and all(review['origin_repeat_raw_byte_equality'].values())
assert independent['input_raw_payloads_verified']==96 and independent['local_source_files_verified']==933
assert independent['standing']['all32_quiet_recomputed_pass'] and independent['all_five_executed_joint_targets_exact']
for f in ['trace.npz','physics_substeps.npz','physics_control_integrals.npz']:
 assert sha(OUT/'result/run/origin_a'/f)==sha(OUT/'result/run/origin_repeat'/f)
verification={'scope':'Local packaging verification of preserved remote and independent receipts; no new live GPU/lock check',
 'frozen_input_maps':copied_receipts,'raw96matched_remote_map':True,'remote_audit_sha256':sha(RESULT/'remote_audit.json'),
 'local_source933checked_against_remote_bound_map':True,'source_manifest_sha256':audit['source_manifest_sha256'],
 'all550asset_hashes_match_exact_source_package':True,'root_remote_exact_six_container_id_and_name_absence_retained':True,
 'forecast_pause038_restoration_receipt_sha256':sha(RESULT/'forecast_pause/restored.json'),'root_unit_result':audit['unit'],
 'original_and_repeat_three_npz_byte_equal':True,'standing32_and_allfive_physical_quiet_passed':True,
 'reported_rate_minus_angle_16s_min_max_rad':[review['min_abs_joint_integral_gap_rad'],review['max_abs_joint_integral_gap_rad']],
 'stage2_complete':False,'velocity_fidelity_qualified':False,'original_frozen_source_and_payload_bytes_unchanged':True}
(OUT/'PACKAGING_VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n')
copy_file(Path(__file__),OUT/'assemble.py')
print('Assembled',OUT)
