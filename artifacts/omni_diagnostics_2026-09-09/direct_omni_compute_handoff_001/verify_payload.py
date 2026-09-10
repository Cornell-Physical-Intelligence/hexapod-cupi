from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((r/'BUNDLE_SHA256.json').read_text())
for f,h in m.items():assert sha(r/f)==h,f
for directory in ('guard','weather_helper','independent_review'):
 for f,h in json.loads((r/directory/'FREEZE_SHA256.json').read_text()).items():assert sha(r/directory/f)==h,f
print('Verified',len(m),'published payloads and three original freezes')
