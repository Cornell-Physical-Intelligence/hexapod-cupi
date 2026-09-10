from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((R/'BUNDLE_SHA256.json').read_text())
assert {str(p.relative_to(R)) for p in R.rglob('*') if p.is_file() and p!=R/'BUNDLE_SHA256.json'}==set(m)
for f,h in m.items():assert sha(R/f)==h,f
for folder,name,count in [('preparation','BUNDLE_SHA256.json',25),('guard','FREEZE_SHA256.json',2)]:
 q=json.loads((R/folder/name).read_text());assert len(q)==count
 for f,h in q.items():assert sha(R/folder/f)==h,f
r=json.loads((R/'ROOT_REVIEW.json').read_text());p=json.loads((R/'remote_preflight.json').read_text())
assert r['source_manifest_sha256']==p['source_manifest_sha256']
assert p['source_files_verified']==934 and len(p['previous_owned_containers_absent'])==6
assert r['root_focused_tests']['passed'] and r['root_guard_tests']['passed'] and not r['actual_diagonal_results_included']
print(json.dumps({'publication_payloads':len(m),'source_preflight_files':934,'root_tests':12,'guard_tests':8,'actual_diagonal_admission':False}))
