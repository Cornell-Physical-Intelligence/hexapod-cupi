"""Read-only exact terminal preview/source/ownership audit; no signalling."""
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib,importlib.util,json,subprocess,sys,time
sys.dont_write_bytecode=True
BASE=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909');RUN=BASE/'direct_omni_preview_001';PAUSE=BASE/'forecast_pause_058'
UNIT='hexapod-direct-omni-preview-001-20260910.service';INV='7121f177c458460e81de0ef7f4a967f3'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def tree(p):return {f.relative_to(p).as_posix():sha(f) for f in sorted(p.rglob('*')) if f.is_file()}
def checked(root,manifest,bound):
 assert sha(root/manifest)==bound,str(root)
 assert not any(f.is_symlink() for f in root.rglob('*'))
 mapping=tree(root);mapping.pop(manifest);assert mapping==read(root/manifest),str(root)
 return {'manifest_sha256':bound,'payloads':len(mapping),'passed':True}
def call(a):return subprocess.check_output(a,text=True,timeout=30)
unit=dict(line.split('=',1) for line in call(['systemctl','--user','show',UNIT,'-p','ActiveState','-p','SubState','-p','Result','-p','InvocationID','-p','ExecMainStatus']).splitlines() if '=' in line)
assert unit['ActiveState'] in ('inactive','failed') and unit.get('InvocationID') in ('',INV)
journal=call(['journalctl','--user','-u',UNIT,'--no-pager','-o','json'])
rows=[json.loads(line) for line in journal.splitlines() if line.strip()]
assert any(r.get('_SYSTEMD_INVOCATION_ID')==INV or r.get('USER_INVOCATION_ID')==INV for r in rows)
checks={}
for name,manifest,bound in (
 ('direct_omni_train_source_002','campaign_source_hashes.json','64144b2c2466f17c809554ef53f82b736b4199136cc821451d7ac2788752bd2e'),
 ('reference_physics_source_009','campaign_source_hashes.json','04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'),
 ('direct_omni_train_preparation_002','FREEZE_SHA256.json','20fbcd40a869b39d8e3143a6c9715ec62ada5cc6211eaf473f60f8b7167678cb'),
 ('direct_omni_preview_adapter_002','FREEZE_SHA256.json','2f2b807a8c4e265dc25270bceb21734e6f97c72a873f407eccc654b9480af0a4'),
 ('direct_omni_preview_host_001','FREEZE_SHA256.json','640ea3891d8e0d6e83f173f5bc5a2206ccdd2d0ef74c9a324d62ecda4fd9594e'),
 ('direct_omni_preview_guard_001','FREEZE_SHA256.json','daf58f87b42795d89d5d16eddee4031326ec8c36422854d63bd4c13ed7ff4b1a')):
 checks[name]=checked(BASE/name,manifest,bound)
host_path=BASE/'direct_omni_preview_host_001/launch_preview_spark.py'
s=importlib.util.spec_from_file_location('_audited_preview_host',host_path);h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
args=NS(source=BASE/'direct_omni_train_source_002',contract=BASE/'direct_omni_train_preparation_002',supervisor_source=BASE/'reference_physics_source_009',adapter=BASE/'direct_omni_preview_adapter_002',pilot=BASE/'direct_omni_train_pilot_caps_001',checkpoint=BASE/'direct_omni_train_pilot_caps_001/train/policy/final.pt',checkpoint_sha256='ca0545760f81b8b8fb0cafde4b1ef89bfa5d8765a8b43b578e451f648a954415',campaign_sha256='3b166e2e99e954d16af9d5f1b09a5bf429271042eb91ee84e1fa92f37c24b656',output=RUN,host_freeze_sha256=checks['direct_omni_preview_host_001']['manifest_sha256'])
identity=h.verify_inputs(args);campaign=read(RUN/'campaign.json');assert identity==campaign['identity'];assert campaign['terminal_inputs_unchanged'] is True
job=read(RUN/'jobs/recording.json');assert job['cleanup_checked'] is True and job['container_name'] and job['container_id']
absence={}
for identifier in (job['container_name'],job['container_id']):
 r=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',identifier],text=True,capture_output=True,timeout=20)
 assert r.returncode!=0 and any(v in r.stderr.lower() for v in ('no such object','no such container'))
 absence[identifier]={'absent':True,'stderr':r.stderr.strip()}
pause=read(PAUSE/'pause.json');restored=read(PAUSE/'restored.json');timers={k for k,v in pause['units'].items() if k.endswith('.timer') and 'ActiveState=active' in v}
assert pause['unit']==UNIT and set(restored['timers'])==timers and restored['restored_unix']
assert pause['selection_sha256']=='725a9443b1d72e7d78d77eb6031431f590a5d22fab654a3ab638265a2f2d7a30'
state=read(RUN/'recording/state.json');video=read(RUN/'recording/video.json')
assert video['complete'] and video['recorded_control_steps']==1900 and video['frames']==950 and video['checkpoint_sha256']==args.checkpoint_sha256
assert video['video_sha256']==sha(RUN/'recording/rollout.mp4') and video['trace_sha256']==sha(RUN/'recording/trace.npz')
raw={'run/'+k:v for k,v in tree(RUN).items()};raw.update({'forecast_pause/'+k:v for k,v in tree(PAUSE).items()})
second={'run/'+k:v for k,v in tree(RUN).items()};second.update({'forecast_pause/'+k:v for k,v in tree(PAUSE).items()});assert raw==second
result={'checked_unix':time.time(),'read_only':True,'unit':unit,'expected_invocation':INV,'journal':rows,'input_checks':checks,'all_inputs_and_selected_pilot_verified':True,'assets':len(tree(args.source/'robot/hexapod_mkii_length_study')),'legacy_runtime_python_files':len(h.verify_legacy_runtime(args.source)),'identity':identity,'host_campaign_status':campaign['status'],'host_campaign_error':campaign.get('error'),'native_recording_state_status':state['status'],'native_recording_job_status':job['status'],'native_recording_job_exit_code':job.get('exit_code'),'final_integrity_file_present':(RUN/'recording/final_integrity.json').exists(),'video_complete':video['complete'],'video_sha256':video['video_sha256'],'trace_sha256':video['trace_sha256'],'frames':video['frames'],'controls':video['recorded_control_steps'],'owned_names_IDs_absent':absence,'restoration':restored,'raw_payloads':raw,'raw_sizes_bytes':{k:(RUN/k[4:] if k.startswith('run/') else PAUSE/k[len('forecast_pause/'):]).stat().st_size for k in raw},'raw_inventory_stable_after_second_hash_pass':True,'formal_policy_acceptance_claimed':False,'postclose_receipt_success_claimed':False}
print(json.dumps(result,indent=2))
