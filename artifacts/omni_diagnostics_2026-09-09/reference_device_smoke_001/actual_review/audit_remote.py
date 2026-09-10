"""Read-only actual remote identity/absence/restoration audit; no GPU launch."""
from pathlib import Path
import hashlib,json,subprocess,time
B=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(name,manifest,expected,count):
 p=B/name;assert sha(p/manifest)==expected
 recorded=json.loads((p/manifest).read_text());actual={str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file() and f!=p/manifest}
 assert len(recorded)==count and actual==recorded
 return {'sha256':expected,'files':count,'unchanged':True}
verified={}
for name,manifest,digest,count in [('reference_physics_source_009','campaign_source_hashes.json','04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e',926),('reference_device_smoke_adapter_001','FREEZE_SHA256.json','be4870a8f7ca0857ba50ffd3a0c92ac5925b9a816767d66d826af7b59957aa0e',18),('reference_policy_observation_005_001','FREEZE_SHA256.json','22402b0079b0b5ac0acdd4e1d7e819fe206b20ee1883077f7de6eaca55808a63',160),('reference_device_smoke_launch_002','FREEZE_SHA256.json','adbb60b3d77b8d6b268dcf3060232fcf3c493361259a0997582e694d62a35658',8)]:verified[name]=verify(name,manifest,digest,count)
admitted=B/'reference_physics_009';asset=admitted/'inputs/study';receipt=admitted/'inputs/study_before.sha256.json'
assert sha(receipt)=='99fd81ee260bd517e6590fe56b8bb46715827fef1a295553f154524a1978ce00'
m=json.loads(receipt.read_text());assert len(m)==550
assert {str(p.relative_to(asset)):sha(p) for p in asset.rglob('*') if p.is_file()}==m
run=B/'reference_device_smoke_001';pause=B/'forecast_pause_039';raw={}
for root in (run,pause):
 for p in sorted(root.rglob('*')):
  if p.is_file():raw[str(p.relative_to(B))]=sha(p)
absence={}
for p in (run/'jobs').glob('*.json'):
 j=json.loads(p.read_text())
 for token in (j.get('container_id'),j.get('container_name')):
  if not token:continue
  q=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',token],text=True,capture_output=True,timeout=20)
  assert q.returncode and ('no such object' in q.stderr.lower() or 'no such container' in q.stderr.lower()),(token,q.stdout,q.stderr)
  absence[token]={'returncode':q.returncode,'stderr':q.stderr.strip()}
restored=json.loads((pause/'restored.json').read_text())
unit=subprocess.check_output(['systemctl','--user','show','hexapod-reference-device-smoke-001-20260910.service','-p','ActiveState','-p','SubState','-p','Result','-p','ExecMainStatus'],text=True)
assert 'Result=success' in unit and 'ActiveState=inactive' in unit and 'ExecMainStatus=0' in unit
print(json.dumps({'verified_unix':time.time(),'source_bundles':verified,'assets_verified':550,'admitted_asset_receipt_sha256':sha(receipt),'raw_payloads':raw,'owned_containers_absent':absence,'pause_restoration':restored,'unit':unit},indent=2))
