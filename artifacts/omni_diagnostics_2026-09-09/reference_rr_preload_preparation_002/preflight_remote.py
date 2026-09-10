from pathlib import Path
import hashlib,json,subprocess,sys
sys.dont_write_bytecode=True
B=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909');S=B/'reference_rr_preload_source_002';G=B/'reference_rr_preload_launch_002'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(S/'campaign_source_hashes.json')=='bfb0bbde27fdfae1f2ddc7797ff12adf184c7cd68a8e637f5fce87f7e0e3cfa3'
m=json.loads((S/'campaign_source_hashes.json').read_text());assert len(m)==932
assert {str(p.relative_to(S)) for p in S.rglob('*') if p.is_file()}==set(m)|{'campaign_source_hashes.json'}
for f,h in m.items():assert sha(S/f)==h,f
assert sha(G/'FREEZE_SHA256.json')=='af85d3797fdd05022f2bd574282ea8cd6a814e3d1835ddca731131905e478810'
for f,h in json.loads((G/'FREEZE_SHA256.json').read_text()).items():assert sha(G/f)==h
assert sha(S/'tools/launch_directional_physics_spark.py')=='e4b16d9e05c08ee44e346baa27b69b426a5a5a1751a60132a1fe0993650194fb'
sys.path.insert(0,str(S/'tools'));from launch_directional_physics_spark import check_source
check_source(S)
def call(a):return subprocess.check_output(a,text=True,timeout=30).strip()
u=call(['systemctl','--user','show','hexapod-diagonal-pairs-001-20260910.service','-p','ActiveState','-p','SubState','-p','Result','-p','ExecMainStatus','-p','InvocationID'])
fields=dict(line.split('=',1) for line in u.splitlines())
assert fields['ActiveState']=='inactive' and fields['ExecMainStatus']=='0' and fields['Result']=='success'
# systemd drops the live invocation ID after an inactive transient unit is collected.
# An absent live ID is accepted only alongside the exact previously audited terminal job bytes.
assert fields['InvocationID'] in ('','5d66c3cbf53343d48988e4f8d2da2682')
previous_hashes={'run/campaign.json': 'af0f41e2072d478df4d67db292c30aa5b327195012c94425258bbc1f92c7fe4f', 'forecast_pause/restored.json': '35891e0f4a4e59bdb679de100f2e2a7d09736c25cdbc1806189a51a05c5c5108', 'run/jobs/lf_rr.json': 'ef760a65278d6860a9245c327d952af22207918cc33b11778270e97421f0247e', 'run/jobs/lr_rf.json': '66c5afff14f82986176dabb6c55eea479f31697b2f7067b6cd9e78fee519a847', 'run/jobs/standing.json': '3efb21f14f58b1d8db5149db48dedc6a6154988dc9cc7638d437b9830aa8a4d6'}
for f,h in previous_hashes.items():
 q=(B/'forecast_pause_047'/f.removeprefix('forecast_pause/')) if f.startswith('forecast_pause/') else (B/'reference_diagonal_pair_001'/f.removeprefix('run/'))
 assert sha(q)==h,f
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
for p in [B/'forecast_pause_048',B/'reference_rr_preload_002',B/'reference_rr_preload_preflight_002.json']:assert not p.exists(),p
r={'source_files_verified':932,'source_manifest_sha256':sha(S/'campaign_source_hashes.json'),'guard_manifest_sha256':sha(G/'FREEZE_SHA256.json'),'previous_owner_unit':u,'previous_owned_containers_absent':absent,'pause047_restoration':restored,'coordination_sha256':hashlib.sha256(c).hexdigest(),'gpu_before_guard':call(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits']),'disk':call(['df','-h',str(B)]),'fresh_names':True,'previous_terminal_payload_sha256':previous_hashes,'previous_live_invocation_id':fields['InvocationID'],'previous_dispatch_invocation_id':'5d66c3cbf53343d48988e4f8d2da2682'}
(B/'reference_rr_preload_preflight_002.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2));print(c.decode().split('### Superseded policy')[0])
