"""Import-only successor; source001 remains immutable."""
from pathlib import Path
import ast,hashlib,json,shutil
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent/'reference_rr_preload_diagnostic_001/source_rr_preload_001';SOURCE=HERE/'source_rr_preload_002'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if SOURCE.exists():raise FileExistsError(SOURCE)
assert sha(PARENT/'campaign_source_hashes.json')=='9b71ad4e4be3e75ec34735725e0f815ac4ab0f4c40871150384658e2fdfc9268'
old=json.loads((PARENT/'campaign_source_hashes.json').read_text());assert len(old)==931
assert {str(p.relative_to(PARENT)) for p in PARENT.rglob('*') if p.is_file()}==set(old)|{'campaign_source_hashes.json'}
for f,h in old.items():assert sha(PARENT/f)==h,f
files=['rr_preload_contract.py','rr_preload_diagnostic.py','directional_contract.py']
for f in files:ast.parse((HERE/f).read_text())
shutil.copytree(PARENT,SOURCE)
for f in files:shutil.copy2(HERE/f,SOURCE/'tools'/f)
o=json.loads((SOURCE/'source_origin.json').read_text());o['rr_preload_host_import_correction002']={
 'parent_source_manifest_sha256':sha(PARENT/'campaign_source_hashes.json'),
 'reason':'Actual host001 preflight had no NumPy; pure metadata imported the NumPy controller before any pause/GPU',
 'scope':'Move exact PROPOSAL into stdlib-only module; change helper and host imports/bindings only',
 'runtime_overlay_sha256':{f:sha(HERE/f) for f in files},
 'controller_equations_and_all_solver_and_physical_functions_unchanged':True,
 'no_host_dependency_install_or_physical_run':True}
(SOURCE/'source_origin.json').write_text(json.dumps(o,indent=2)+'\n')
new={str(p.relative_to(SOURCE)):sha(p) for p in sorted(SOURCE.rglob('*')) if p.is_file() and p.name!='campaign_source_hashes.json'}
(SOURCE/'campaign_source_hashes.json').write_text(json.dumps(new,indent=2)+'\n')
changed={f:{'before':old.get(f),'after':h} for f,h in new.items() if old.get(f)!=h}
assert len(new)==932 and set(changed)=={'source_origin.json','tools/rr_preload_contract.py','tools/rr_preload_diagnostic.py','tools/directional_contract.py'}
r={'source_manifest_sha256':sha(SOURCE/'campaign_source_hashes.json'),'parent_manifest_sha256':sha(PARENT/'campaign_source_hashes.json'),'source_payloads':932,'unchanged_parent_payloads':928,'changed_paths':changed,'removed_parent_payloads':[]}
(HERE/'SOURCE_BUILD.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
