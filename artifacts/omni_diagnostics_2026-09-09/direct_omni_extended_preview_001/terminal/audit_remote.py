"""Read-only exact terminal preview/source/ownership audit; no signalling."""
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib,importlib.util,json,subprocess,sys,time
sys.dont_write_bytecode=True
BASE=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909');RUN=BASE/'direct_omni_extended_preview_001';PAUSE=BASE/'forecast_pause_064'
UNIT='hexapod-direct-omni-extended-preview-001-20260910.service';INV='3e88eb521861497cb3bd275b60c3c1b8'
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
unit=dict(line.split('=',1) for line in call(['systemctl','--user','show',UNIT,'-p','ActiveState','-p','SubState','-p','Result','-p','InvocationID','-p','ExecMainStatus','-p','MainPID']).splitlines() if '=' in line)
assert unit.get('MainPID')=='0' and unit['ActiveState'] in ('inactive','failed') and unit.get('InvocationID') in ('',INV)
journal=call(['journalctl','--user','-u',UNIT,'--no-pager','-o','json'])
rows=[json.loads(line) for line in journal.splitlines() if line.strip()]
assert any(r.get('_SYSTEMD_INVOCATION_ID')==INV or r.get('USER_INVOCATION_ID')==INV for r in rows)
checks={}
for name,manifest,bound in (
 ('direct_omni_train_host_005','FREEZE_SHA256.json','2db7e44ec40543753a6d1325f6d5dc9ed81be2948931b2ca670deffda68ce959'),
 ('direct_omni_train_source_004','campaign_source_hashes.json','aa70659d33243f1e28e9c71b1a7f9d33341642db770fd7356192573a26fe904e'),
 ('reference_physics_source_009','campaign_source_hashes.json','04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'),
 ('direct_omni_train_preparation_004','FREEZE_SHA256.json','0bdd2abf3811e68ac59e4fbc325eb5d2bb96afd2183eb84c4d37faeb86ff9e9f'),
 ('direct_omni_preview_adapter_004','FREEZE_SHA256.json','fb60da984d73c01725e453521ca861e02b061103dfc2fb266a56aa9dd1fa7a20'),
 ('direct_omni_preview_host_003','FREEZE_SHA256.json','2f4ae1fde7fb85787d700743b7267283fe2a8f150ed8f7d9c9514685039b5ad2'),
 ('direct_omni_extended_preview_guard_001','FREEZE_SHA256.json','4454f8083ad0983f352b9d1e78a7ea537c21835dd53deba4dfe7ec8b2c13c413')):
 checks[name]=checked(BASE/name,manifest,bound)
host_path=BASE/'direct_omni_preview_host_003/launch_preview_spark.py'
s=importlib.util.spec_from_file_location('_audited_preview_host',host_path);h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
args=NS(source=BASE/'direct_omni_train_source_004',contract=BASE/'direct_omni_train_preparation_004',supervisor_source=BASE/'reference_physics_source_009',adapter=BASE/'direct_omni_preview_adapter_004',pilot=BASE/'direct_omni_train_extended_caps_002',checkpoint=BASE/'direct_omni_train_extended_caps_002/train/policy/final.pt',checkpoint_sha256='c376a0a4eb04d54396b4fd6171fe173167767463245213cce7c2cc1d3a2877cf',campaign_sha256='fcf09c70705a02c588ebae0285e8e2737b0ad6f9745d5270bad5db4eff90403c',output=RUN,host_freeze_sha256=checks['direct_omni_preview_host_003']['manifest_sha256'])
identity=h.verify_inputs(args);campaign=read(RUN/'campaign.json');assert identity==campaign['identity'];assert campaign['terminal_inputs_unchanged'] is True
assert {p.stem for p in (RUN/'jobs').glob('*.json') if not p.name.endswith('_contact_data_audit.json')}=={'recording'}
job=read(RUN/'jobs/recording.json');assert job['cleanup_checked'] is True and job['container_name'] and job['container_id']
absence={}
for identifier in (job['container_name'],job['container_id']):
 r=subprocess.run(['docker','inspect','--format','{{.Id}} {{.Name}} {{.State.Running}}',identifier],text=True,capture_output=True,timeout=20)
 assert r.returncode!=0 and any(v in r.stderr.lower() for v in ('no such object','no such container'))
 absence[identifier]={'absent':True,'stderr':r.stderr.strip()}
pause=read(PAUSE/'pause.json');restored=read(PAUSE/'restored.json');timers={k for k,v in pause['units'].items() if k.endswith('.timer') and 'ActiveState=active' in v}
assert pause['unit']==UNIT and set(restored['timers'])==timers and restored['restored_unix']
selection=BASE/'direct_omni_extended_preview_selection_001.json'
assert sha(selection)==pause['selection_sha256'] and read(selection)==pause['selection']
assert pause['output']==str(RUN) and pause['source']==str(args.source)
assert pause['selection_sha256']=='ff96ec258b13ae49110e286d3dd042cabb95d8ae1253dd5a6b08d18f87d88a8a'
state=read(RUN/'recording/state.json') if (RUN/'recording/state.json').is_file() else {}
video=read(RUN/'recording/video.json') if (RUN/'recording/video.json').is_file() else {}
try:
    replay=h.validate_result(RUN/'recording',identity)
    assert campaign.get('recording')==replay and campaign.get('post_exit_original_inputs_reverified') is True
    assert campaign['status']=='completed' and job['status']=='completed' and job['exit_code']==0 and unit['ExecMainStatus']=='0' and unit['Result']=='success'
    recording_passed=True;recording_error=None
except Exception as exc:
    recording_passed=False;recording_error=repr(exc);replay=None
assert not any(p.is_symlink() for root in (RUN,PAUSE) for p in root.rglob('*'))
raw={'run/'+k:v for k,v in tree(RUN).items()};raw.update({'forecast_pause/'+k:v for k,v in tree(PAUSE).items()})
second={'run/'+k:v for k,v in tree(RUN).items()};second.update({'forecast_pause/'+k:v for k,v in tree(PAUSE).items()});assert raw==second
result={'checked_unix':time.time(),'read_only':True,'unit':unit,'expected_invocation':INV,'journal':rows,'input_checks':checks,'all_inputs_and_selected_pilot_verified':True,'assets':len(tree(args.source/'robot/hexapod_mkii_length_study')),'legacy_runtime_python_files':len(h.verify_legacy_runtime(args.source)),'identity':identity,'host_campaign_status':campaign['status'],'host_campaign_error':campaign.get('error'),'native_recording_state_status':state.get('status'),'native_recording_job_status':job['status'],'native_recording_job_exit_code':job.get('exit_code'),'pre_shutdown_integrity_file_present':(RUN/'recording/pre_shutdown_integrity.json').exists(),'video_complete':video.get('complete'),'video_sha256':video.get('video_sha256'),'trace_sha256':video.get('trace_sha256'),'frames':video.get('frames'),'controls':video.get('recorded_control_steps'),'recording_completion_replay_passed':recording_passed,'recording_completion_replay_error':recording_error,'host_validate_result':replay,'actual_postexit_original_inputs_reverified':campaign.get('post_exit_original_inputs_reverified'),'owned_names_IDs_absent':absence,'restoration':restored,'raw_payloads':raw,'raw_sizes_bytes':{k:(RUN/k[4:] if k.startswith('run/') else PAUSE/k[len('forecast_pause/'):]).stat().st_size for k in raw},'raw_inventory_stable_after_second_hash_pass':True,'formal_policy_acceptance_claimed':False,'postexit_recording_receipt_verified':recording_passed,'selection_sha256':sha(selection)}
print(json.dumps(result,indent=2))
