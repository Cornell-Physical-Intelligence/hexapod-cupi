from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parent;m=json.loads((p/'BUNDLE_SHA256.json').read_text())
a={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file() and f.name!='BUNDLE_SHA256.json'}
# Nested BUNDLE manifests are ordinary payloads of this wrapper.
for f in p.rglob('BUNDLE_SHA256.json'):
 if f.parent!=p:a[str(f.relative_to(p))]=hashlib.sha256(f.read_bytes()).hexdigest()
assert a==m
r=json.loads((p/'result/remote_audit.json').read_text())
for f,h in r['raw_payloads'].items():assert hashlib.sha256((p/'result'/f).read_bytes()).hexdigest()==h
for f in p.rglob('FREEZE_SHA256.json'):
 for k,h in json.loads(f.read_text()).get('files',json.loads(f.read_text())).items():assert hashlib.sha256((f.parent/k).read_bytes()).hexdigest()==h
print(json.dumps({'payloads':len(m),'raw_payloads':len(r['raw_payloads']),'nested_freezes':True,'read_only':True}))
