from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
repo=r.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((r/'BUNDLE_SHA256.json').read_text())
for f,h in m.items():assert sha(r/f)==h,f
for directory in ('frozen','independent_review'):
 for f,h in json.loads((r/directory/'FREEZE_SHA256.json').read_text()).items():assert sha(r/directory/f)==h,f
for f,h in json.loads((r/'frozen/ORACLE_SHA256.json').read_text()).items():assert sha(repo/f)==h,f
print('Verified',len(m),'published payloads, both original freezes and ten oracle dependencies')
