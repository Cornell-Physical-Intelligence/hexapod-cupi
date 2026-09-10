"""Read-only exact001 post-run audit; never signals services or GPU processes."""
from pathlib import Path
import hashlib,json,subprocess,time
BASE=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
run=BASE/'reference_residual_ppo_001';pause=BASE/'forecast_pause_042'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def tree(path,manifest,expected):
 f=path/manifest;assert sha(f)==expected,(path,sha(f))
 m=read(f);a={str(p.relative_to(path)):sha(p) for p in path.rglob('*') if p.is_file() and p!=f}
 assert a==m and not any(p.is_symlink() for p in path.rglob('*'))
 return {'manifest_sha256':sha(f),'payload_files':len(m),'unchanged':True}
source=tree(BASE/'reference_physics_source_009','campaign_source_hashes.json','04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e')
consumer=tree(BASE/'reference_residual_ppo_source_001','FREEZE_SHA256.json','258c304f409373e49667bc7fb653026fcb0754c2c92d6dfd70eb04769c3e754b')
host=tree(BASE/'reference_residual_ppo_launch_001','FREEZE_SHA256.json','54b16b7f75d7b5c6fb65a8040d0535edab7cfab5e8d2e3333e202be0fbfa3985')
bridge=tree(BASE/'reference_device_smoke_adapter_001','FREEZE_SHA256.json','be4870a8f7ca0857ba50ffd3a0c92ac5925b9a816767d66d826af7b59957aa0e')
obs=tree(BASE/'reference_policy_observation_005_001','FREEZE_SHA256.json','22402b0079b0b5ac0acdd4e1d7e819fe206b20ee1883077f7de6eaca55808a63')
assets=run/'inputs/study';am=read(run/'inputs/study_before.sha256.json')
assert {str(p.relative_to(assets)):sha(p) for p in assets.rglob('*') if p.is_file()}==am
raw={}
for p in run.rglob('*'):
 if p.is_file() and assets not in p.parents:raw['run/'+str(p.relative_to(run))]=sha(p)
for p in pause.rglob('*'):
 if p.is_file():raw['forecast_pause/'+str(p.relative_to(pause))]=sha(p)
pref=BASE/'reference_residual_ppo_preflight_002.json';raw['preflight002.json']=sha(pref)
owned={};jobs=[]
for phase in ('standing','smoke'):
 j=read(run/'jobs'/(phase+'.json'));jobs.append({'phase':phase,'container_name':j['container_name'],'container_id':j.get('container_id'),'cleanup_checked':j.get('cleanup_checked')})
 for identity in (j['container_name'],j.get('container_id')):
  if not identity:continue
  r=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',identity],capture_output=True,text=True,timeout=20)
  assert r.returncode==1 and 'no such object' in r.stderr.lower(),(identity,r.returncode,r.stdout,r.stderr)
  owned[identity]={'returncode':r.returncode,'stderr':r.stderr.strip()}
unit=subprocess.run(['systemctl','--user','show','hexapod-reference-residual-ppo-001-20260910.service','-p','Result','-p','ActiveState','-p','SubState','-p','ExecMainStatus'],capture_output=True,text=True,timeout=20)
assert unit.returncode==0
out={'scope':'Read-only completed001 source/input/raw/owned-absence audit; no running002 intervention','verified_unix':time.time(),'source009':source,'consumer001':consumer,'host001':host,'bridge001':bridge,'observation005':obs,'admitted_asset_files':len(am),'admitted_assets_unchanged':True,'raw_payloads':raw,'actual_jobs':jobs,'owned_absence':owned,'unknown_smoke_container_ID_not_invented':jobs[1]['container_id'] is None,'pause_restoration':read(pause/'restored.json'),'unit':unit.stdout.strip(),'preflight002_sha256':sha(pref)}
print(json.dumps(out,indent=2))
