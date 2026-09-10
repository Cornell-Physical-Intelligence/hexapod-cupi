from pathlib import Path
from types import SimpleNamespace
import hashlib,json,sys
sys.dont_write_bytecode=True
base=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
contract=base/'direct_omni_train_preparation_001';source=base/'direct_omni_train_source_001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pin='9e591b93ae99ca3a76b7f7500ea20cc778da76c7f6b9b2c86fd22d31aceab3d7'
assert sha(contract/'FREEZE_SHA256.json')==pin
m=json.loads((contract/'FREEZE_SHA256.json').read_text())
assert {p.relative_to(contract).as_posix():sha(p) for p in contract.rglob('*') if p.is_file() and p.name!='FREEZE_SHA256.json'}==m
sys.path.insert(0,str(contract));import direct_contract
identity=direct_contract.verify_inputs(SimpleNamespace(source=source,checkpoint=base/'omni_repair_003/branch_a/inputs/original.pt',allocation='smoke',branch='caps',smoke=None))
assert identity['source_manifest_sha256']=='37b4b6d5d75f6d697a122fc2a1903f8f625f6a67d1727e1257e1533300e44da6'
coord=Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md')
assert sha(coord)=='fdad0bb0126158988dc36ea38538b2f7804299cffb5ff29601ba597089672234'
print(json.dumps({'verified':True,'contract_payloads':len(m),'identity':identity,'coordination_sha256':sha(coord),'no_GPU':True},indent=2))
