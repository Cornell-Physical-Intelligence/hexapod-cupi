from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((r/"BUNDLE_SHA256.json").read_text())
for f,h in m.items():assert sha(r/f)==h,f
f=json.loads((r/"frozen/FREEZE_SHA256.json").read_text());f=f.get("payloads",f)
for p,h in f.items():assert sha(r/"frozen"/p)==h,p
print("Verified",len(m),"published payloads and original freeze")
