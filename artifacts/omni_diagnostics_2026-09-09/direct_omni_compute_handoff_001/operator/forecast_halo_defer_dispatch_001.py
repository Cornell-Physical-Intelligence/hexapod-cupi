from pathlib import Path
import hashlib,runpy,sys
p=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909/forecast_halo_defer_helper_001')
assert hashlib.sha256((p/'FREEZE_SHA256.json').read_bytes()).hexdigest()=='b40bed8e319a6f2bf3e463b415e0574a9f846dd0df2a68a7720394c49b00561f'
sys.argv=[str(p/'defer_halo.py'),'defer'];runpy.run_path(str(p/'defer_halo.py'),run_name='__main__')
