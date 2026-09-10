"""Assemble a new paired physics source; original009 bytes remain immutable."""
from pathlib import Path
import ast,hashlib,json,shutil
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent/'reference_physics_adapter_009/source_009';SOURCE=HERE/'source_pair_motion_001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
if SOURCE.exists():raise FileExistsError(SOURCE)
assert sha(PARENT/'campaign_source_hashes.json')==parent
old=json.loads((PARENT/'campaign_source_hashes.json').read_text());assert len(old)==926
assert {str(p.relative_to(PARENT)) for p in PARENT.rglob('*') if p.is_file()}==set(old)|{'campaign_source_hashes.json'}
for rel,h in old.items():assert sha(PARENT/rel)==h,rel
files=['pair_motion_contract.py','pair_motion_measurements.py','pair_motion_metrics.py','run_pair_motion.py','launch_pair_motion_spark.py','sensor_freshness.py','sensor_source_contract.json']
for f in files:
 if f.endswith('.py'):ast.parse((HERE/f).read_text())
shutil.copytree(PARENT,SOURCE)
for f in files:shutil.copy2(HERE/f,SOURCE/'tools'/f)
shutil.copytree(HERE/'paired_runtime',SOURCE/'tools/paired_runtime')
o=json.loads((SOURCE/'source_origin.json').read_text());o['paired_contact_motion_physics001']={
 'parent_manifest_sha256':parent,'paired_owner_freeze_sha256':sha(HERE/'paired_runtime/FREEZE_SHA256.json'),
 'runtime_overlay_sha256':{f:sha(HERE/f) for f in files},
 'scope':'Fresh32standing+quiet then1robot4sstartup+2sreferencehold+24sforward.01m/s+18sstop; newexplicit4supportbranch, noPPO',
 'original009_physics_scalar_gates_env_startup_observer_unchanged':True,
 'new_contact_freshness':'Exactsource-bound14sensorclockchecker alreadyexecuted in device001; no sensor.update/reset',
 'new_paired_checks':'Both independentqualifiedflights/landings;4retainedsupport;all400Hztorques;fullrawpairedstatereplay;original50Hz5mmprogressandquiet',
 'old846_849_adopted':False,'fresh_exact_source_admission_required':True}
(SOURCE/'source_origin.json').write_text(json.dumps(o,indent=2)+'\n')
new={str(p.relative_to(SOURCE)):sha(p) for p in sorted(SOURCE.rglob('*')) if p.is_file() and p.name!='campaign_source_hashes.json'}
(SOURCE/'campaign_source_hashes.json').write_text(json.dumps(new,indent=2)+'\n')
changes={f:{'before':old.get(f),'after':h} for f,h in new.items() if old.get(f)!=h}
assert set(old)<=set(new) and {f for f in old if old[f]!=new[f]}=={'source_origin.json'}
r={'source':str(SOURCE),'parent_manifest_sha256':parent,'manifest_sha256':sha(SOURCE/'campaign_source_hashes.json'),
 'source_files':len(new),'exact_changed_paths':changes,'925other_parent_payloads_unchanged':True,'parent_files_removed':[],
 'host_sha256':sha(SOURCE/'tools/launch_pair_motion_spark.py')}
(HERE/'SOURCE_BUILD.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
