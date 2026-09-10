from pathlib import Path
import hashlib,json,runpy
p=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909/direct_omni_train_smoke_guard_004')
m=p/'FREEZE_SHA256.json'
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
assert sha(m)=='d7ecf21be4f9d38d9d5ba50d1d83b844b51ae82656c8e46efe51956d0630becb'
assert not p.is_symlink() and not any(x.is_symlink() for x in p.rglob('*'))
assert {str(x.relative_to(p)):sha(x) for x in p.rglob('*') if x.is_file() and x!=m}==json.loads(m.read_text())
runpy.run_path(str(p/'launch_guarded_remote.py'),run_name='__main__')
