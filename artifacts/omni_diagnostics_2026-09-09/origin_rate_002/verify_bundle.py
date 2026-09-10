"""Portable read-only wrapper/hash verification; no SDK, network or evidence writes."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def check(directory,name):
 p=directory/name;m=json.loads(p.read_text());m=m.get('files',m)
 for f,h in m.items():assert sha(directory/f)==h,str(directory/f)
 return len(m)
n=check(ROOT,'BUNDLE_SHA256.json')
assert {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()}==set(json.loads((ROOT/'BUNDLE_SHA256.json').read_text()))|{'BUNDLE_SHA256.json'}
for directory,name in [('preparation','BUNDLE_SHA256.json'),('preparation/owner','FREEZE_SHA256.json'),('preparation/independent_review','FREEZE_SHA256.json'),('launch_guard','FREEZE_SHA256.json'),('independent_actual_review','FREEZE_SHA256.json'),('result/root_review','FREEZE_SHA256.json')]:check(ROOT/directory,name)
a=json.loads((ROOT/'result/remote_audit.json').read_text());assert len(a['raw_payloads'])==96
for f,h in a['raw_payloads'].items():assert sha(ROOT/'result'/f)==h,f
p=json.loads((ROOT/'PACKAGING_VERIFICATION.json').read_text())
assert sha(ROOT/'result/remote_audit.json')==p['remote_audit_sha256']
assert sha(ROOT/'preparation/source_overlay/campaign_source_hashes.json')==a['source_manifest_sha256']
for name,r in p['frozen_input_maps'].items():assert sha(ROOT/name/r['map'])==r['sha256']
for f in ['trace.npz','physics_substeps.npz','physics_control_integrals.npz']:assert sha(ROOT/'result/run/origin_a'/f)==sha(ROOT/'result/run/origin_repeat'/f)
print(json.dumps({'wrapper_payloads_verified':n,'raw_payloads_matched':96,'nested_maps_verified':True,'repeat_npz_byte_equal':True,'live_remote_state_checked':False}))
