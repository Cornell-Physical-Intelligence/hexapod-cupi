from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
read=lambda f:json.loads(f.read_text())
assert not any(f.is_symlink() for f in p.rglob('*'))
m=read(p/'BUNDLE_SHA256.json')
assert {str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file() and f.name!='BUNDLE_SHA256.json'}==m
raw=p/'raw';a=read(p/'audit/terminal_audit.json')
local={str(f.relative_to(raw)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in raw.rglob('*') if f.is_file()}
assert local==read(p/'audit/local_raw_map.json')
assert {k:v['sha256'] for k,v in local.items()}==a['raw_payloads']
assert {k:v['bytes'] for k,v in local.items()}==a['raw_sizes_bytes']
assert len(local)==52 and sum(v['bytes'] for v in local.values())==76190510
c=read(raw/'run/campaign.json');t=read(raw/'run/train/training_receipt.json')
assert c['status']=='completed' and c['terminal_inputs_unchanged'] is True
assert set(c['accepted_phases'])=={'standing','train','final_constant','final_stop'}
assert c['PPO_updates_completed']==t['updates_completed']==2 and t['complete'] is True
assert t['reload']['passed'] is True and t['reload']['optimizer_entries']==17
assert sha(raw/'run/train/policy/final.pt')==t['final_checkpoint_sha256']==a['saved_final_checkpoint_sha256']=='4aaf556613e72a80332381c09309cc0ab52a03c6d55527bc76e8a7000a29e1f2'
assert len(a['owned_names_IDs_absent'])==8 and all(v['absent'] is True for v in a['owned_names_IDs_absent'].values())
assert a['quiet_failed_env_ids']==list(range(48))
assert read(raw/'forecast_pause/restored.json')==a['restoration']
print('Verified',len(m),'payloads; completed2-update integration;0/48 quiet passes; Stage2 remains unqualified')
