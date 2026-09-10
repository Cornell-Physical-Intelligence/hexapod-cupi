from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((r/'BUNDLE_SHA256.json').read_text())
for f,h in m.items():assert sha(r/f)==h,f
for directory in ('native','host','guard'):
 for f,h in json.loads((r/directory/'FREEZE_SHA256.json').read_text()).items():assert sha(r/directory/f)==h,f
assert sha(r/'operator/direct_omni_train_source_manifest_001.json')=='37b4b6d5d75f6d697a122fc2a1903f8f625f6a67d1727e1257e1533300e44da6'
print('Verified',len(m),'published payloads, three original freezes and source map')
