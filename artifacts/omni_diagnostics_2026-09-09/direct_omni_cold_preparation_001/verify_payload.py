from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((r/'BUNDLE_SHA256.json').read_text())
for f,h in m.items():assert sha(r/f)==h,f
for prefix,record in json.loads((r/'PREPARATION.json').read_text())['parts'].items():
 p=r/prefix;assert sha(p/'FREEZE_SHA256.json')==record['freeze_sha256']
 for f,h in json.loads((p/'FREEZE_SHA256.json').read_text()).items():assert sha(p/f)==h,f
print('Verified',len(m),'preparation payloads and four nested freezes')
