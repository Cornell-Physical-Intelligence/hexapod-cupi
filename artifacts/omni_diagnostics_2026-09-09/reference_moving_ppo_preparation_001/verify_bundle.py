from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def verify(p,name):
 m=json.loads((p/name).read_text())
 actual={str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file() and f!=p/name}
 if any(f.is_symlink() for f in p.rglob('*')) or actual!=m:raise ValueError('Changed bundle: '+str(p))
 return len(m)
n=verify(root,'BUNDLE_SHA256.json')
for folder in ('consumer','host','guard','independent_review'):verify(root/folder,'FREEZE_SHA256.json')
b=json.loads((root/'ROOT_BINDINGS.json').read_text());r=json.loads((root/'independent_review/REVIEW.json').read_text())['tested_files_sha256']
h=(root/'host/launch_moving_ppo_spark.py').read_text();g=(root/'guard/launch_guarded_remote.py').read_text()
assert hashlib.sha256(h.replace(b['consumer_freeze_sha256'],'PENDING_FINAL_CONSUMER_FREEZE').encode()).hexdigest()==r['tmp/reference_moving_ppo_launch_001/launch_moving_ppo_spark.py']
assert hashlib.sha256(g.replace(b['host_sha256'],'PENDING_HOST_SHA').replace(b['host_freeze_sha256'],'PENDING_HOST_FREEZE').encode()).hexdigest()==r['tmp/reference_moving_ppo_guard_001/launch_guarded_remote.py']
for key,path in [('consumer_freeze_sha256','consumer/FREEZE_SHA256.json'),('host_sha256','host/launch_moving_ppo_spark.py'),('host_freeze_sha256','host/FREEZE_SHA256.json'),('guard_sha256','guard/launch_guarded_remote.py'),('guard_freeze_sha256','guard/FREEZE_SHA256.json')]:assert sha(root/path)==b[key]
print(json.dumps({'verified_payloads':n,'nested_inputs_and_final_bindings_exact':True,'scope':'preparation and dispatch, not result'},indent=2))
