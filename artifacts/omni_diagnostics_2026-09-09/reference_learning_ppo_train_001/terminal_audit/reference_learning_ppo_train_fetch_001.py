from pathlib import Path
import subprocess,json,hashlib,shutil
root=Path('tmp/reference_learning_ppo_train_results_001');root.mkdir(exist_ok=False)
a=json.loads(Path('tmp/reference_learning_ppo_train_remote_audit_001.json').read_text());shutil.copy2('tmp/reference_learning_ppo_train_remote_audit_001.json',root/'remote_audit.json')
priority={'run/train_10/state.json':0,'run/train_10/raw/trace.npz':1,'run/train_10/raw/physics_substeps.npz':2,'run/train_10/raw/reference_states.npz':3}
for name,digest in sorted(a['raw_payloads'].items(),key=lambda x:(priority.get(x[0],4),x[0])):
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['scp','-q','spark:'+a['remote_paths'][name],str(p)],check=True)
 assert p.stat().st_size==a['raw_sizes_bytes'][name] and hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
 print('verified',name,flush=True)
(root/'local_raw_verification.json').write_text(json.dumps({'all_payloads_verified':True,'files':len(a['raw_payloads']),'bytes':sum(a['raw_sizes_bytes'].values()),'remote_audit_sha256':hashlib.sha256((root/'remote_audit.json').read_bytes()).hexdigest()},indent=2)+'\n')
