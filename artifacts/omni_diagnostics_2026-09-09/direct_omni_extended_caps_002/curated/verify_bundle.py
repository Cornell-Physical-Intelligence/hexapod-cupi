"""Portable stdlib verifier for curated bytes and explicit remote-only omissions."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def tree(root):
 assert root.is_dir() and not root.is_symlink() and not any(p.is_symlink() for p in root.rglob('*'))
 return {str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
actual=tree(H);actual.pop('FREEZE_SHA256.json');assert actual==read(H/'FREEZE_SHA256.json')
audit=read(H/'audit/remote_terminal_audit_001.json');assert audit['passed'] is True and not audit['errors'] and all(audit['checks'].values())
full=read(H/'FULL_REMOTE_INVENTORY.json');fetched=read(H/'FETCH_VERIFICATION.json');ordinary=read(H/'REMOTE_ONLY_ORDINARY_AUTOSAVES.json');required=read(H/'REMOTE_ONLY_REQUIRED_TRAINING_RAW.json')
assert full==audit['inventory'] and len(full)==576 and fetched['verified'] is True
sets=[set(fetched['payloads']),set(ordinary),set(required)]
assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]) and set.union(*sets)==set(full)
assert len(ordinary)==500 and set(required)=={'run/train/training_trace.npz'}
assert all(full[k]==v for k,v in {**fetched['payloads'],**ordinary,**required}.items())
assert tree(H/'raw')=={k:v['sha256'] for k,v in fetched['payloads'].items()}
for k,row in fetched['payloads'].items():assert (H/'raw'/k).stat().st_size==row['bytes']
assert len(sets[0])==75 and sum(x['bytes'] for x in fetched['payloads'].values())==260791365
run=H/'raw/run'
for phase in ('standing','initial_constant','initial_stop','train','final_constant','final_stop'):
 expected=read(run/(phase+'_immutable.sha256.json'));present=tree(run/phase)
 missing={k for k in expected if 'run/'+phase+'/'+k in set(ordinary)|set(required)}
 assert present=={k:v for k,v in expected.items() if k not in missing}
 if phase!='train':assert not missing
c=read(run/'campaign.json');receipt=read(run/'train/training_receipt.json')
assert c['status']=='completed' and c['PPO_updates_completed']==500 and c['Stage2_complete'] is False
assert receipt['complete'] is True and receipt['updates_completed']==500 and receipt['reload']['passed'] is True
assert sha(run/'train/policy/final.pt')==audit['campaign']['checkpoint_sha256']==receipt['final_checkpoint_sha256']
assert set(receipt['decision_checkpoints'])=={'1','10','25','50','100','250','500'}
for row in receipt['decision_checkpoints'].values():assert sha(run/'train/policy'/row['file'])==row['sha256']
assert len(audit['owned_absence'])==12 and len(audit['campaign']['quiet_failed_env_ids'])==48
print(json.dumps({'verified':True,'local_raw_files':75,'local_raw_bytes':260791365,'full_remote_files':576,'remote_only_ordinary_autosaves':500,'remote_only_required_training_trace_bytes':required['run/train/training_trace.npz']['bytes'],'final_checkpoint_sha256':receipt['final_checkpoint_sha256'],'actual_updates':500,'quiet_passes':0,'quiet_trials':48,'local_full_training_replay_claimed':False,'Stage2_complete':False},indent=2))
