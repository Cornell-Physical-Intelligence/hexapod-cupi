REMOTE='/home/orionh/HEXAPOD_runs/mock_length_study_20260909/direct_omni_train_pilot_quiet_priority_guard_001'
PIN='7e55aecc20fbfe1adf0d62211cf5e39d1d33c7e52c4fee34cf5fa5f0c2b95a70'
from pathlib import Path
import hashlib,json,runpy
p=Path(REMOTE);m=p/'FREEZE_SHA256.json';sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
assert sha(m)==PIN;assert not p.is_symlink() and not any(x.is_symlink() for x in p.rglob('*'))
assert {str(x.relative_to(p)):sha(x) for x in p.rglob('*') if x.is_file() and x!=m}==json.loads(m.read_text())
runpy.run_path(str(p/'launch_guarded_remote.py'),run_name='__main__')
