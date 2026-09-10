from pathlib import Path
from types import SimpleNamespace
import importlib.util,hashlib,json,subprocess,sys,time
sys.dont_write_bytecode=True
b=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909');r=b/'direct_omni_train_smoke_002';pause=b/'forecast_pause_055'
unit='hexapod-direct-omni-train-smoke-002-20260910.service';inv='6cd738f30a0b4f499026de5cbf8a5bb6'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=importlib.util.spec_from_file_location('host',b/'direct_omni_train_host_002/launch_train_spark.py');h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
a=SimpleNamespace(source=b/'direct_omni_train_source_002',checkpoint=b/'omni_repair_003/branch_a/inputs/original.pt',contract=b/'direct_omni_train_preparation_002',supervisor_source=b/'reference_physics_source_009',output=r,host_freeze_sha256='19545653a00e635cc14ed9415f9799978da34674e84f6cad76eb42bd2a961ac4',allocation='smoke',branch='caps',smoke=None)
identity=h.verify_inputs(a)
h.verify_tree(b/'direct_omni_train_smoke_guard_003','FREEZE_SHA256.json','123c34a1830be1051fe17f35e728bfcf795b0d497fe5d4b043dd78e484906128')
c=h.read(r/'campaign.json');receipt=h.read(r/'train/training_receipt.json')
assert c['status']=='completed' and c['terminal_inputs_unchanged'] and c['PPO_updates_completed']==2
assert receipt['updates_completed']==2 and receipt['complete'] is True
assert receipt['audit']['controls']==48 and receipt['audit']['replicas']==32
assert c['observed_training']['observed_updates_completed']==2
assert h.CONTRACT.verify_smoke_campaign(r,identity)==sha(r/'campaign.json')
for phase in ['standing','train','final_constant','final_stop']:
 assert h.tree_hashes(r/phase)==h.read(r/(phase+'_immutable.sha256.json'))
fields=subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState','-p','SubState','-p','InvocationID','-p','Result'],text=True)
u=dict(x.split('=',1) for x in fields.splitlines());assert u['ActiveState'] in ('inactive','failed')
journal=[json.loads(x) for x in subprocess.check_output(['journalctl','--user','-u',unit,'--no-pager','-o','json'],text=True).splitlines() if x.strip()]
if u.get('InvocationID')!=inv:
 assert u.get('InvocationID')==''
 rows=[x for x in journal if x.get('USER_UNIT')==unit];assert {x.get('USER_INVOCATION_ID') for x in rows}=={inv}
 assert any(x.get('MESSAGE','').startswith('Started '+unit) for x in rows)
assert h.tree_hashes(r/'standing')==h.read(r/'standing_immutable.sha256.json')
standing=h.CONTRACT.validate_result(r/'standing','standing',identity);assert standing==c['accepted_phases']['standing']
absent={};jobs={}
for phase,status in [(p,'completed') for p in ['standing','train','final_constant','final_stop']]:
 j=h.read(r/'jobs'/(phase+'.json'));assert j['cleanup_checked'] and j['status']==status and j['container_id'];jobs[phase]=j
 for token in [j['container_name'],j['container_id']]:
  p=subprocess.run(['docker','inspect',token],capture_output=True,text=True,timeout=20)
  assert p.returncode==1 and any(x in p.stderr.lower() for x in ['no such object','no such container'])
  absent[token]={'absent':True,'stderr':p.stderr.strip()}
assert (r/'final_constant').is_dir() and (r/'final_stop').is_dir()
p=h.read(pause/'pause.json');restore=h.read(pause/'restored.json');assert p['unit']==unit and p['output']==str(r)
assert restore['restored_unix']>=p['created_unix']
assert restore['timers']==[n for n,s in p['units'].items() if n.endswith('.timer') and 'ActiveState=active' in s]
assert p['coordination_sha256']==sha(Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md'))=='22c615d67c6c4dc14ab370b291f8eddd72f8475ad34118b91a4237a34ee05fab'
halo=b/'forecast_halo_defer_001';defer=h.read(halo/'state.json');archive=halo/defer['archive'];am=h.read(archive/'SHA256.json')
assert sha(archive/'SHA256.json')==defer['archive_manifest_sha256']=='53409b319ac4f77ae19be0648b2f8e3b3ffb9b542f7c6a36c6aa597f0b4c650c'
assert {k:v for k,v in h.tree_hashes(archive).items() if k!='SHA256.json'}==am
raw={};sizes={}
for prefix,folder in [('run',r),('forecast_pause',pause)]:
 assert not any(p.is_symlink() for p in folder.rglob('*'))
 for p in folder.rglob('*'):
  if p.is_file():name=prefix+'/'+p.relative_to(folder).as_posix();raw[name]=sha(p);sizes[name]=p.stat().st_size
print(json.dumps({'checked_unix':time.time(),'read_only':True,'unit':u,'expected_invocation':inv,'historical_journal':journal,'identity':identity,'source_files':598,'supervisor_files':926,'legacy_runtime_files':16,'all_input_trees_verified':True,'host_freeze_sha256':a.host_freeze_sha256,'guard_freeze_sha256':'123c34a1830be1051fe17f35e728bfcf795b0d497fe5d4b043dd78e484906128','campaign_sha256':sha(r/'campaign.json'),'campaign_completed':True,'accepted_updates':2,'actual_completed_updates':2,'training_receipt_sha256':sha(r/'train/training_receipt.json'),'saved_final_checkpoint_sha256':sha(r/'train/policy/final.pt'),'reload':receipt['reload'],'quiet_failed_env_ids':c['accepted_phases']['final_stop']['quiet_failed_env_ids'],'standing_unchanged':True,'standing':standing,'jobs':jobs,'owned_names_IDs_absent':absent,'restoration':restore,'halo_archive_unchanged':True,'halo_archive_manifest_sha256':defer['archive_manifest_sha256'],'halo_restarted':(halo/'restored.json').exists(),'raw_payloads':raw,'raw_sizes_bytes':sizes,'formal_policy_acceptance_claimed':False},indent=2))
