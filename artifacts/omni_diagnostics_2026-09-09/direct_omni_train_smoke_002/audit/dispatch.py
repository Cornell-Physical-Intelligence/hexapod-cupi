from pathlib import Path
import json,hashlib,runpy
p=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909/direct_omni_train_smoke_guard_003')
m=p/'FREEZE_SHA256.json'
assert hashlib.sha256(m.read_bytes()).hexdigest()=='123c34a1830be1051fe17f35e728bfcf795b0d497fe5d4b043dd78e484906128'
assert {str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file() and f!=m}==json.loads(m.read_text())
runpy.run_path(str(p/'launch_guarded_remote.py'),run_name='__main__')
