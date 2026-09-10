from pathlib import Path
import hashlib,json,subprocess,sys
sys.dont_write_bytecode=True
B=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909');S=B/'reference_rr_preload_source_001';G=B/'reference_rr_preload_launch_001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(S/'campaign_source_hashes.json')=='9b71ad4e4be3e75ec34735725e0f815ac4ab0f4c40871150384658e2fdfc9268'
m=json.loads((S/'campaign_source_hashes.json').read_text());assert len(m)==931
assert {str(p.relative_to(S)) for p in S.rglob('*') if p.is_file()}==set(m)|{'campaign_source_hashes.json'}
for f,h in m.items():assert sha(S/f)==h,f
assert sha(G/'FREEZE_SHA256.json')=='727374d3f998fa03e22f9490a69804459fcc45d17eda6b5ae30ebe5a05a8c1d1'
for f,h in json.loads((G/'FREEZE_SHA256.json').read_text()).items():assert sha(G/f)==h
assert sha(S/'tools/launch_directional_physics_spark.py')=='e4b16d9e05c08ee44e346baa27b69b426a5a5a1751a60132a1fe0993650194fb'
sys.path.insert(0,str(S/'tools'));from launch_directional_physics_spark import check_source
check_source(S)
def call(a):return subprocess.check_output(a,text=True,timeout=30).strip()
u=call(['systemctl','--user','show','hexapod-diagonal-pairs-001-20260910.service','-p','ActiveState','-p','SubState','-p','Result','-p','ExecMainStatus','-p','InvocationID'])
assert 'ActiveState=inactive' in u and 'ExecMainStatus=0' in u and '5d66c3cbf53343d48988e4f8d2da2682' in u
restored=json.loads((B/'forecast_pause_047/restored.json').read_text());absent={}
jobs=list((B/'reference_diagonal_pair_001/jobs').glob('*.json'));jobs=[p for p in jobs if not p.name.endswith('_contact_data_audit.json')];assert len(jobs)==3
for p in jobs:
 j=json.loads(p.read_text())
 for token in (j['container_name'],j['container_id']):
  assert token
  q=subprocess.run(['docker','inspect',token],text=True,capture_output=True,timeout=20)
  assert q.returncode!=0 and ('no such object' in q.stderr.lower() or 'no such container' in q.stderr.lower()),token
  absent[token]={'returncode':q.returncode,'stderr':q.stderr}
c=Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md').read_bytes();assert hashlib.sha256(c).hexdigest()=='fdad0bb0126158988dc36ea38538b2f7804299cffb5ff29601ba597089672234'
for p in [B/'forecast_pause_048',B/'reference_rr_preload_001',B/'reference_rr_preload_preflight_001.json']:assert not p.exists(),p
r={'source_files_verified':931,'source_manifest_sha256':sha(S/'campaign_source_hashes.json'),'guard_manifest_sha256':sha(G/'FREEZE_SHA256.json'),'previous_owner_unit':u,'previous_owned_containers_absent':absent,'pause047_restoration':restored,'coordination_sha256':hashlib.sha256(c).hexdigest(),'gpu_before_guard':call(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits']),'disk':call(['df','-h',str(B)]),'fresh_names':True}
(B/'reference_rr_preload_preflight_001.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2));print(c.decode().split('### Superseded policy')[0])
