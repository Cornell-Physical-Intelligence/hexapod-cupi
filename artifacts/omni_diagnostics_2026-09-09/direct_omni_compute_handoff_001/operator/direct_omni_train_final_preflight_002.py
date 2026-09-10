from pathlib import Path
import hashlib,json,importlib.util,sys
sys.dont_write_bytecode=True
base=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
p=base/'direct_omni_train_guard_002'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(p/'FREEZE_SHA256.json')=='fb457055503fbc2167f3e87f2916d5153ff3b33e672c96d5c973cd3118f15d7b'
m=json.loads((p/'FREEZE_SHA256.json').read_text());assert {q.relative_to(p).as_posix():sha(q) for q in p.rglob('*') if q.is_file() and q.name!='FREEZE_SHA256.json'}==m
spec=importlib.util.spec_from_file_location('reviewed_guard',p/'launch_guarded_remote.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
g.require_final_bindings();g.verify_previous_owner()
assert sha(Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md'))=='22c615d67c6c4dc14ab370b291f8eddd72f8475ad34118b91a4237a34ee05fab'
spec=importlib.util.spec_from_file_location('reviewed_host',g.HOST);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
identity=g.validate_train_inputs(h)
assert not g.PAUSE.exists() and not g.OUTPUT.exists()
print(json.dumps({'passed':True,'guard_payloads':len(m),'identity':identity,'previous_owner_and_restoration_verified':True,'no_GPU':True},indent=2))
