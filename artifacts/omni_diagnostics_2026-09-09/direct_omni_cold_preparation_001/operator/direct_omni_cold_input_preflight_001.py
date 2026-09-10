from pathlib import Path
from types import SimpleNamespace
import hashlib,importlib.util,json,sys,time
sys.dont_write_bytecode=True
base=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
prep=base/'direct_omni_cold_preparation_001';source=base/'direct_omni_cold_source_001';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(prep/'FREEZE_SHA256.json')=='eb87f1dd456771950f7c1bf50ed0a51c1fc6ccb4631217e29d99fa8a84717095'
m=json.loads((prep/'FREEZE_SHA256.json').read_text())
assert {p.relative_to(prep).as_posix() for p in prep.rglob('*') if p.is_file()}==set(m)|{'FREEZE_SHA256.json'}
for name,h in m.items():assert sha(prep/name)==h
spec=importlib.util.spec_from_file_location('cold_contract',prep/'cold_contract.py');contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
identity=contract.verify_inputs(SimpleNamespace(source=source,checkpoint=base/'omni_repair_003/branch_a/inputs/original.pt'))
assert identity['source_manifest_sha256']=='4671eab012aaf79ce49769cfdc1667142237a4b4a45966fd0399a9caa900c51b'
old=json.loads((prep/'remote_base_inventory.json').read_text())['current_complete_files'];new=json.loads((source/'campaign_source_hashes.json').read_text())
changed={n:{'old':h,'new':new.get(n)} for n,h in old.items() if n!='campaign_source_hashes.json' and new.get(n)!=h};added=set(new)-set(old)
assert set(changed)=={'tools/train_length_study.py','robot/hexapod_mkii_length_study/training_plan.json'}
assert added=={'source_origin.json','tools/candidate_asset_audit.py'}
runtime={n.removeprefix('isaaclab/hexapod_rl/'):h for n,h in new.items() if n.startswith('isaaclab/hexapod_rl/') and n.endswith('.py')}
runtime_digest=hashlib.sha256(json.dumps(runtime,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert len(runtime)==16 and runtime_digest=='abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280'
print(json.dumps({'passed':True,'checked_unix':time.time(),'GPU_allocated':False,'source_files':len(new),'identity':identity,'changed_parent_payloads':changed,'added_source_payloads':sorted(added),'legacy_runtime_files':len(runtime),'legacy_runtime_sha256':runtime_digest,'legacy_runtime_hash_method':'basename map, sorted JSON, compact separators; exact same16-file content','no_runtime_bootstrap_claim':True},indent=2))
