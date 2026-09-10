from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((R/'BUNDLE_SHA256.json').read_text());assert {str(p.relative_to(R)) for p in R.rglob('*') if p.is_file() and p!=R/'BUNDLE_SHA256.json'}==set(m)
for f,h in m.items():assert sha(R/f)==h,f
for folder,name,count in [('preparation','BUNDLE_SHA256.json',36),('host_failure','FREEZE_SHA256.json',4)]:
 q=json.loads((R/folder/name).read_text());assert len(q)==count
 for f,h in q.items():assert sha(R/folder/f)==h,f
r=json.loads((R/'ROOT_REVIEW.json').read_text());f=json.loads((R/'host_failure/failure.json').read_text())
assert r['source_manifest_sha256']==f['source_manifest_sha256']
assert not r['host_preflight_passed'] and not r['GPU_guard_dispatched'] and not f['GPU_guard_dispatched']
assert not any(f['observed_remote_paths_exist'].values())
assert "ModuleNotFoundError: No module named 'numpy'" in (R/'host_failure/stderr.log').read_text()
print(json.dumps({'payloads_verified':len(m),'CPU_tests':16,'host_preflight_passed':False,'GPU_dispatched':False}))
