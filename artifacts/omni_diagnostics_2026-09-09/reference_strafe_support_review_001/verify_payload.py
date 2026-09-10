from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((R/'BUNDLE_SHA256.json').read_text())
assert {str(p.relative_to(R)) for p in R.rglob('*') if p.is_file() and p.name!='BUNDLE_SHA256.json'}==set(m)
for f,h in m.items():assert sha(R/f)==h,f
s=R/'owner';o=json.loads((s/'FREEZE_SHA256.json').read_text());r=json.loads((R/'ROOT_REVIEW.json').read_text())
assert len(o)==r['owner_payloads']==14
assert sha(s/'FREEZE_SHA256.json')==r['owner_manifest_sha256']
for f,h in o.items():assert sha(s/f)==h,f
for f,h in r['root_numeric_replay_exact'].items():assert sha(s/f)==h,f
assert r['source_payloads_verified']==930 and r['original_rejection_preserved'] and not r['physical_test_performed']
print(json.dumps({'payloads_verified':len(m),'original_owner_payloads':14,'recorded_root_numeric_replay_files':5,'physical_admission':False}))
