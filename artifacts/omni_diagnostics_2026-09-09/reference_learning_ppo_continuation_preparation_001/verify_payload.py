from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root,name):
 m=json.loads((root/name).read_text())
 assert not any(p.is_symlink() for p in root.rglob('*'))
 assert {str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p!=root/name}==m
 return len(m)
n=verify(H,'BUNDLE_SHA256.json');assert verify(H/'guard','FREEZE_SHA256.json')==7
d=json.loads((H/'operator/reference_learning_ppo_decision_001.json').read_text());p=json.loads((H/'operator/reference_learning_ppo_continuation_preflight_001.json').read_text())
assert d['accepted'] and d['approved_updates']==10 and d['schema']=='moving_PPO_learning_admission_review_v1'
assert p['passed'] and not p['GPU_allocated'] and p['decision']['decision_sha256']==sha(H/'operator/reference_learning_ppo_decision_001.json')
assert p['phases']==['train_10','evaluate_initial','evaluate_010','quiet_010'] and p['admitted_asset_files']==550
assert p['guard_sha256']==sha(H/'guard/launch_guarded_remote.py') and p['guard_freeze_sha256']==sha(H/'guard/FREEZE_SHA256.json')
assert '804a727558374268b185325305148b1a' in (H/'operator/reference_learning_ppo_continuation_dispatch_001.log').read_text()
assert json.loads((H/'operator/reference_learning_actual001_root_replay.json').read_text())['integrity_passed']
print(json.dumps({'payloads_verified':n,'approved_PPO_updates':10,'admission_report_preserved':True,'PPO_completion_claimed':False,'Stage2_complete':False},indent=2))
