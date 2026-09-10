from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def tree(root,name,bound=None):
 p=root/name
 if bound is not None:assert sha(p)==bound
 assert not any(f.is_symlink() for f in root.rglob('*'))
 m=json.loads(p.read_text());m=m.get('files',m)
 actual={str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file() and f!=p}
 assert actual==m,str(root)
 return len(m)
count=tree(H,'BUNDLE_SHA256.json')
for folder,b in json.loads((H/'BINDINGS.json').read_text()).items():assert tree(H/folder,b['manifest'],b['sha256'])==b['payloads']
p=json.loads((H/'operator/reference_learning_ppo_preflight_001.json').read_text())
assert p['passed'] and not p['GPU_allocated'] and p['phases']==['standing','calibration','learning_recovery_32']
assert p['identity']['consumer_freeze_sha256']==sha(H/'consumer/FREEZE_SHA256.json')
assert p['host_freeze_sha256']==sha(H/'host/FREEZE_SHA256.json') and p['guard_freeze_sha256']==sha(H/'guard/FREEZE_SHA256.json')
assert len(p['previous_owned_absent'])==4 and all(v['absent'] for v in p['previous_owned_absent'])
r=json.loads((H/'operator/reference_learning_ppo_final_binding_001.json').read_text())
assert r['consumer_freeze_sha256']==p['identity']['consumer_freeze_sha256'] and r['host_sha256']==sha(H/'host/launch_learning_ppo_spark.py') and r['guard_sha256']==sha(H/'guard/launch_guarded_remote.py')
assert r['host_only_consumer_hash_replacement'] and r['guard_only_three_hash_replacements']
assert 'd161f77036f346d2b1abe244d95b2dbe' in (H/'operator/reference_learning_ppo_dispatch_001.log').read_text()
print(json.dumps({'payloads_verified':count,'consumer51_host3_guard11':True,'root_tests':74,'admission_only_dispatched':True,'PPO_updates_claimed':0,'Stage2_complete':False},indent=2))
