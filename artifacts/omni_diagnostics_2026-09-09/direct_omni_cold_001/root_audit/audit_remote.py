from pathlib import Path
from types import SimpleNamespace
import importlib.util,hashlib,json,subprocess,sys,time
sys.dont_write_bytecode=True
b=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909');r=b/'direct_omni_cold_001';pause=b/'forecast_pause_053'
unit='hexapod-direct-omni-cold-001-20260910.service';inv='ae3b95c756db4aa4b639ee4230e8c958'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=importlib.util.spec_from_file_location('host',b/'direct_omni_cold_host_001/launch_cold_spark.py');h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
a=SimpleNamespace(source=b/'direct_omni_cold_source_001',checkpoint=b/'omni_repair_003/branch_a/inputs/original.pt',contract=b/'direct_omni_cold_preparation_001',supervisor_source=b/'reference_physics_source_009',output=r,host_freeze_sha256='efbeac2a0bfb7ebf93efb9ec4dce33363607092b30a7e81ac6351950e1fa6083')
identity=h.verify_inputs(a);h.verify_tree(b/'direct_omni_cold_guard_001','FREEZE_SHA256.json','07d6462ab27120c223e7d12d4b5dfef5d9948afea822236c88f5963f2625cde5')
c=h.read(r/'campaign.json');assert c['status']=='completed' and c['baseline_complete'] and c['terminal_inputs_unchanged'];assert c['PPO_updates_completed']==0
fields=subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState','-p','SubState','-p','InvocationID','-p','Result'],text=True);u=dict(x.split('=',1) for x in fields.splitlines());assert u['ActiveState'] in ('inactive','failed')
journal=[json.loads(x) for x in subprocess.check_output(['journalctl','--user','-u',unit,'--no-pager','-o','json'],text=True).splitlines() if x.strip()]
if u.get('InvocationID')!=inv:
 assert u.get('InvocationID')==''
 rows=[x for x in journal if x.get('USER_UNIT')==unit];assert {x.get('USER_INVOCATION_ID') for x in rows}=={inv}
 assert any(x.get('MESSAGE','').startswith('Started '+unit) for x in rows) and any('Consumed ' in x.get('MESSAGE','') for x in rows)
assert h.tree_hashes(r/'standing')==h.read(r/'standing_immutable.sha256.json')
phases={};absent={}
for phase in ['standing','baseline']:
 phases[phase]=h.CONTRACT.validate_result(r/phase,phase,identity)
 j=h.read(r/'jobs'/(phase+'.json'));assert j['cleanup_checked'] and j['status']=='completed' and j['container_id']
 for token in [j['container_name'],j['container_id']]:
  p=subprocess.run(['docker','inspect',token],capture_output=True,text=True,timeout=20);assert p.returncode==1 and any(x in p.stderr.lower() for x in ['no such object','no such container']);absent[token]={'absent':True,'stderr':p.stderr.strip()}
p=h.read(pause/'pause.json');restore=h.read(pause/'restored.json');assert p['unit']==unit and p['output']==str(r)
assert restore['restored_unix']>=p['created_unix'];assert restore['timers']==[n for n,s in p['units'].items() if n.endswith('.timer') and 'ActiveState=active' in s]
raw={};sizes={};paths={}
for prefix,folder in [('run',r),('forecast_pause',pause)]:
 assert not any(p.is_symlink() for p in folder.rglob('*'))
 for p in folder.rglob('*'):
  if p.is_file():name=prefix+'/'+p.relative_to(folder).as_posix();raw[name]=sha(p);sizes[name]=p.stat().st_size;paths[name]=str(p)
print(json.dumps({'checked_unix':time.time(),'read_only':True,'unit':u,'expected_invocation':inv,'historical_journal':journal,'identity':identity,'source_files':589,'supervisor_files':926,'legacy_runtime_files':16,'all_input_trees_verified':True,'host_freeze_sha256':a.host_freeze_sha256,'guard_freeze_sha256':'07d6462ab27120c223e7d12d4b5dfef5d9948afea822236c88f5963f2625cde5','campaign_sha256':sha(r/'campaign.json'),'campaign_completed':True,'PPO_updates_completed':0,'phases':phases,'standing_unchanged':True,'owned_names_IDs_absent':absent,'restoration':restore,'raw_payloads':raw,'raw_sizes_bytes':sizes,'remote_paths':paths,'formal_policy_acceptance_claimed':False},indent=2))
