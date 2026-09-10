from pathlib import Path
import hashlib,json,subprocess,sys
sys.dont_write_bytecode=True
B=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909');S=B/'reference_diagonal_pair_source_001';G=B/'reference_diagonal_pair_launch_001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(S/'campaign_source_hashes.json')=='26ff2b7f8e57b3a6ea69dae09be58b26c2b0967e92103ed3f2673c0f32ae9625'
m=json.loads((S/'campaign_source_hashes.json').read_text());assert len(m)==934
assert {str(p.relative_to(S)) for p in S.rglob('*') if p.is_file()}==set(m)|{'campaign_source_hashes.json'}
for f,h in m.items():assert sha(S/f)==h,f
assert sha(G/'FREEZE_SHA256.json')=='7bb2c3794e2170a5659637782a0ce500861c9a89483c1728e45647f2e43bc74c'
for f,h in json.loads((G/'FREEZE_SHA256.json').read_text()).items():assert sha(G/f)==h
assert sha(S/'tools/launch_pair_physics_spark.py')=='9e875a7978dfd39dbb7733079fefa0125c9b5b8b59f817a33bf7456b25adcc7c'
sys.path.insert(0,str(S/'tools'));from launch_pair_physics_spark import check_source
check_source(S)
def call(a):return subprocess.check_output(a,text=True,timeout=30).strip()
u=call(['systemctl','--user','show','hexapod-reference-directional-003-20260910.service','-p','ActiveState','-p','SubState','-p','Result','-p','ExecMainStatus','-p','InvocationID'])
assert 'ActiveState=failed' in u and 'ExecMainStatus=1' in u and 'de4c2da0d76e4775bb9c5580e2474303' in u
restored=json.loads((B/'forecast_pause_046/restored.json').read_text());absent={}
jobs=list((B/'reference_directional_003/jobs').glob('*.json'));jobs=[p for p in jobs if not p.name.endswith('_contact_data_audit.json')];assert len(jobs)==3
for p in jobs:
 j=json.loads(p.read_text())
 for token in (j['container_name'],j['container_id']):
  assert token
  q=subprocess.run(['docker','inspect',token],text=True,capture_output=True,timeout=20)
  assert q.returncode!=0 and ('no such object' in q.stderr.lower() or 'no such container' in q.stderr.lower()),token
  absent[token]={'returncode':q.returncode,'stderr':q.stderr}
c=Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md').read_bytes();assert hashlib.sha256(c).hexdigest()=='fdad0bb0126158988dc36ea38538b2f7804299cffb5ff29601ba597089672234'
for p in [B/'forecast_pause_047',B/'reference_diagonal_pair_001',B/'reference_diagonal_pair_preflight_001.json']:assert not p.exists(),p
r={'source_files_verified':934,'source_manifest_sha256':sha(S/'campaign_source_hashes.json'),'guard_manifest_sha256':sha(G/'FREEZE_SHA256.json'),'previous_owner_unit':u,'previous_owned_containers_absent':absent,'pause046_restoration':restored,'coordination_sha256':hashlib.sha256(c).hexdigest(),'gpu_before_guard':call(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits']),'disk':call(['df','-h',str(B)]),'fresh_names':True}
(B/'reference_diagonal_pair_preflight_001.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2));print(c.decode())
