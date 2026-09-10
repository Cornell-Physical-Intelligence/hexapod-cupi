from pathlib import Path
import hashlib,json,subprocess,time
base=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909');run=base/'reference_learning_ppo_001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((run/'campaign.json').read_text());current=json.loads((run/'learning_campaign.json').read_text())
assert sha(run/'campaign.json')=='3175ead5c6e4718abce48ee2b7001187ba46dbff9900eb575f8d21b93de6ee23'
assert old['status']=='completed' and current['status']=='failed' and current['source_and_inputs_unchanged']
unit=subprocess.check_output(['systemctl','--user','show','hexapod-learning-ppo-train-001-20260910.service','-p','ActiveState','-p','SubState','-p','InvocationID','-p','Result'],text=True)
u=dict(l.split('=',1) for l in unit.splitlines());assert u['ActiveState'] in ('failed','inactive') and u['InvocationID'] in ('','804a727558374268b185325305148b1a')
j=json.loads((run/'jobs/train_10.json').read_text());assert j['cleanup_checked'] and j['container_id']
absent={}
for name in (j['container_name'],j['container_id']):
 p=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=20)
 assert p.returncode and any(s in p.stderr.lower() for s in ('no such object','no such container'))
 absent[name]={'absent':True,'stderr':p.stderr.strip()}
raw={}
for root,prefix in ((run/'train_10','run/train_10'),(base/'forecast_pause_052','forecast_pause')):
 for p in sorted(root.rglob('*')):
  if p.is_file():raw[prefix+'/'+str(p.relative_to(root))]=p
for path in ('learning_campaign.json','jobs/train_10.json','jobs/train_10_contact_data_audit.json','logs/train_10.log'):
 p=run/path
 if p.exists():raw['run/'+path]=p
restore=json.loads((base/'forecast_pause_052/restored.json').read_text())
for phase in ('evaluate_initial','evaluate_010','quiet_010'):assert not (run/phase).exists()
s=json.loads((run/'train_10/state.json').read_text());assert s['PPO_updates_completed']==9 and s['status']=='rejected'
report={'checked_unix':time.time(),'unit':u,'expected_invocation':'804a727558374268b185325305148b1a','initial_campaign_unchanged_sha256':sha(run/'campaign.json'),'learning_campaign_sha256':sha(run/'learning_campaign.json'),'training_completed_updates':9,'training_state_status':s['status'],'error':s['error'],'decision010_present':(run/'train_10/decision_010.pt').exists(),'source_and_inputs_unchanged_producer':True,'owned_names_IDs_absent':absent,'pause_restoration':restore,'raw_payloads':{k:sha(p) for k,p in raw.items()},'raw_sizes_bytes':{k:p.stat().st_size for k,p in raw.items()},'remote_paths':{k:str(p) for k,p in raw.items()},'audit_read_only':True,'independent_full_source_recheck_pending':True}
print(json.dumps(report,indent=2))
