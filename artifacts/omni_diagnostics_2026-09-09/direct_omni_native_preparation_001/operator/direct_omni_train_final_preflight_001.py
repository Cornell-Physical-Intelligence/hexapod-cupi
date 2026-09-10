from pathlib import Path
import hashlib,json,importlib.util,sys
sys.dont_write_bytecode=True
base=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
p=base/'direct_omni_train_guard_001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(p/'FREEZE_SHA256.json')=='137617132f776c23e11dafce893ef2aaf068aaf7f8e5a7037f6e33e1a76e9b4d'
m=json.loads((p/'FREEZE_SHA256.json').read_text());assert {q.relative_to(p).as_posix():sha(q) for q in p.rglob('*') if q.is_file() and q.name!='FREEZE_SHA256.json'}==m
spec=importlib.util.spec_from_file_location('reviewed_guard',p/'launch_guarded_remote.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
g.require_final_bindings();g.verify_previous_owner()
spec=importlib.util.spec_from_file_location('reviewed_host',g.HOST);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
identity=g.validate_train_inputs(h)
assert not g.PAUSE.exists() and not g.OUTPUT.exists()
print(json.dumps({'passed':True,'guard_payloads':len(m),'identity':identity,'previous_owner_and_restoration_verified':True,'no_GPU':True},indent=2))
