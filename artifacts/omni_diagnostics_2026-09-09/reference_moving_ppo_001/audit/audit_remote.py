"""Read-only terminal audit: no service/container writes or runtime launch."""
from pathlib import Path
import json,hashlib,subprocess,time,sys,importlib.util
from types import SimpleNamespace
sys.dont_write_bytecode=True
B=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909');RUN=B/'reference_moving_ppo_001';PAUSE=B/'forecast_pause_049'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def tree(root,manifest,bound,count):
 assert sha(root/manifest)==bound,root;m=read(root/manifest);assert len(m)==count
 assert not any(p.is_symlink() for p in root.rglob('*'))
 for f,h in m.items():assert sha(root/f)==h,str(root/f)
 assert {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}==set(m)|{manifest}
 return {'path':str(root),'manifest':manifest,'manifest_sha256':bound,'files':count,'unchanged':True}
unit=subprocess.check_output(['systemctl','--user','show','hexapod-moving-ppo-001-20260910.service','-p','ActiveState','-p','SubState','-p','Result','-p','ExecMainStatus','-p','InvocationID'],text=True)
assert 'ActiveState=failed' in unit and 'ExecMainStatus=1' in unit and 'InvocationID=53fa4c8afc7940e0931a77c18d310557' in unit,unit
campaign=read(RUN/'campaign.json');assert campaign['status']=='failed' and campaign['PPO_updates_completed']==0
specs=[('source','reference_physics_source_009','campaign_source_hashes.json','04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e',926),('consumer','reference_moving_ppo_source_001','FREEZE_SHA256.json','dd49fd3e3639e453b067b9d4d67e5465db20865200c6a17616dd6936b527b842',35),('host','reference_moving_ppo_launch_001','FREEZE_SHA256.json','f8cb69359f9a9bbab1b812c603bce25c2de42e8692d44c5552a1374bd6279f49',3),('guard','reference_moving_ppo_guard_001','FREEZE_SHA256.json','7441201d0ff3833049ed1a4047cf0bbc237139a80c62ee1810c2ce4e9096f57a',2),('bridge','reference_device_smoke_adapter_001','FREEZE_SHA256.json','be4870a8f7ca0857ba50ffd3a0c92ac5925b9a816767d66d826af7b59957aa0e',18),('observation','reference_policy_observation_005_001','FREEZE_SHA256.json','22402b0079b0b5ac0acdd4e1d7e819fe206b20ee1883077f7de6eaca55808a63',160)]
inputs={key:tree(B/folder,manifest,h,count) for key,folder,manifest,h,count in specs}
assets=read(RUN/'inputs/study_before.sha256.json');assert len(assets)==550
for f,h in assets.items():assert sha(RUN/'inputs/study'/f)==h,f
assert {str(p.relative_to(RUN/'inputs/study')) for p in (RUN/'inputs/study').rglob('*') if p.is_file()}==set(assets)
# Independently invoke exact reviewed stdlib proof checks; this function never starts an app.
spec=importlib.util.spec_from_file_location('audit_bound_host',B/'reference_moving_ppo_launch_001/launch_moving_ppo_spark.py');host=importlib.util.module_from_spec(spec);spec.loader.exec_module(host)
a=SimpleNamespace(source=B/'reference_physics_source_009',run=B/'reference_physics_009',device_run=B/'reference_device_smoke_001',bridge=B/'reference_device_smoke_adapter_001',consumer=B/'reference_moving_ppo_source_001',observation=B/'reference_policy_observation_005_001',output=RUN)
identity=host.validate_inputs(a,require_standing=True)
raw={};sizes={}
for prefix,root in [('run',RUN),('forecast_pause',PAUSE)]:
 for p in sorted(root.rglob('*')):
  if p.is_file() and not(prefix=='run' and p.is_relative_to(RUN/'inputs/study')):
   name=prefix+'/'+str(p.relative_to(root));raw[name]=sha(p);sizes[name]=p.stat().st_size
for name in ('reference_moving_ppo_preflight_001.py','reference_moving_ppo_preflight_001.json'):
 raw['preflight/'+name]=sha(B/name);sizes['preflight/'+name]=(B/name).stat().st_size
jobs={};absence={}
for p in sorted((RUN/'jobs').glob('*.json')):
 j=read(p)
 if 'container_name' not in j:continue
 assert j.get('cleanup_checked') is True and j.get('container_id'),j
 jobs[p.stem]=j
 for token in (j['container_name'],j['container_id']):
  result=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',token],text=True,capture_output=True,timeout=20)
  assert result.returncode==1 and any(v in result.stderr.lower() for v in ('no such object','no such container')),(token,result.stdout,result.stderr)
  absence[token]={'returncode':result.returncode,'stderr':result.stderr.strip()}
assert set(jobs)=={'standing','calibration','profile_32'} and len(absence)==6
restored=read(PAUSE/'restored.json')
print(json.dumps({'verified_unix':time.time(),'unit':unit,'campaign_status':campaign['status'],'PPO_updates_completed':0,'input_maps':inputs,'assets_files':550,'assets_unchanged':True,'host_readonly_proof_identity':identity,'raw_payloads':raw,'raw_sizes_bytes':sizes,'jobs':{n:{k:j[k] for k in ('status','phase','container_name','container_id','cleanup_checked')} for n,j in jobs.items()},'owned_names_and_ids_absent':absence,'pause_restoration':restored,'audit_actions_read_only':True},indent=2))
