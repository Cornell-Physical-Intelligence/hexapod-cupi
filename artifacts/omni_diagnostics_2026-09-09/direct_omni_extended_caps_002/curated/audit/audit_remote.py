"""Read-only terminal/source/inventory audit. Prints failure evidence too."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,importlib.util,json,subprocess,sys,time
sys.dont_write_bytecode=True
B=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909');R=B/'direct_omni_train_extended_caps_002';P=B/'forecast_pause_063'
UNIT='hexapod-direct-omni-train-extended-caps-002-20260910.service';INV='ca9455cefb554c7691e7f8b2ed3ac939'
HOST=B/'direct_omni_train_host_005';HOST_FREEZE='2db7e44ec40543753a6d1325f6d5dc9ed81be2948931b2ca670deffda68ce959'
GUARD=B/'direct_omni_train_extended_caps_guard_002';GUARD_FREEZE='bc372c3564f6aa8ab4e6f9858fe8cf2917e17a76e13b1856191faf1c604e7916'
PHASES=('standing','initial_constant','initial_stop','train','final_constant','final_stop')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def call(argv):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=30)
 return {'argv':argv,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
def tree(root):
 if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):raise ValueError('Symbolic tree: '+str(root))
 return {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
def bundle(root,bound):
 assert sha(root/'FREEZE_SHA256.json')==bound
 values=tree(root);values.pop('FREEZE_SHA256.json');assert values==read(root/'FREEZE_SHA256.json')
 return {'freeze_sha256':bound,'payloads':len(values),'files':values}
out={'schema':'direct_extended_caps002_terminal_audit_v1','observed_unix':time.time(),'read_only':True,'expected_invocation':INV,'errors':[],'checks':{},'formal_policy_acceptance_claimed':False}
def check(label,fn):
 try:value=fn();out[label]=value;out['checks'][label]=True;return value
 except Exception as exc:out['errors'].append(label+': '+repr(exc));out['checks'][label]=False;return None

def owner():
 q=call(['systemctl','--user','show',UNIT,'-p','ActiveState','-p','SubState','-p','InvocationID','-p','MainPID','-p','ExecMainStatus','-p','Result','-p','ExecStart'])
 out['owner_raw']=q;assert q['exit_code']==0
 fields=dict(x.split('=',1) for x in q['stdout'].splitlines() if '=' in x)
 journal=call(['journalctl','--user','-u',UNIT,'--no-pager','-o','json']);out['journal_raw']=journal
 assert journal['exit_code']==0
 rows=[json.loads(x) for x in journal['stdout'].splitlines() if x.strip()]
 if fields.get('InvocationID')!=INV:
  assert fields.get('InvocationID')==''
  unitrows=[r for r in rows if r.get('USER_UNIT')==UNIT]
  assert {r.get('USER_INVOCATION_ID') for r in unitrows}=={INV}
  assert any(r.get('MESSAGE','').startswith('Started '+UNIT) for r in unitrows)
 assert fields['ActiveState']=='inactive' and fields['MainPID']=='0' and fields['ExecMainStatus']=='0' and fields['Result']=='success'
 return fields
check('terminal_owner',owner)
check('host_bundle',lambda:bundle(HOST,HOST_FREEZE));check('guard_bundle',lambda:bundle(GUARD,GUARD_FREEZE))
def inputs():
 assert out['checks']['host_bundle']
 sys.path.insert(0,str(HOST));spec=importlib.util.spec_from_file_location('readonly_host005',HOST/'launch_train_spark.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
 a=SimpleNamespace(source=B/'direct_omni_train_source_004',checkpoint=B/'omni_repair_003/branch_a/inputs/original.pt',contract=B/'direct_omni_train_preparation_004',supervisor_source=B/'reference_physics_source_009',output=R,host_freeze_sha256=HOST_FREEZE,allocation='extended',branch='caps',smoke=B/'direct_omni_train_smoke_004')
 identity=h.verify_inputs(a)
 src=read(a.source/'campaign_source_hashes.json');sup=read(a.supervisor_source/'campaign_source_hashes.json')
 asset={k:v for k,v in src.items() if k.startswith('robot/hexapod_mkii_length_study/')}
 assert len(src)==599 and len(sup)==926 and len(asset)==550 and len(h.verify_legacy_runtime(a.source))==16
 out['input_manifests']={'source':src,'supervisor':sup,'native':read(a.contract/'FREEZE_SHA256.json'),'assets':asset}
 out['_host']=h
 return {'identity':identity,'source_files':len(src),'supervisor_files':len(sup),'asset_files':len(asset),'legacy_files':16,'native_files':len(out['input_manifests']['native']),'checkpoint_sha256':sha(a.checkpoint),'all_verified':True}
check('inputs',inputs)
def predecessor_proofs():
 assert out['checks']['guard_bundle']
 spec=importlib.util.spec_from_file_location('readonly_retry_guard',GUARD/'launch_guarded_remote.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
 g.require_final_bindings();g.verify_previous_owner();g.verify_failed_previous_owner()
 for phase in ('standing','train','final_constant','final_stop'):
  assert tree(g.SMOKE/phase)==read(g.SMOKE/(phase+'_immutable.sha256.json'))
 return {'successful_smoke004_exact_seven_pins':g.PRIOR_PINS,'failed_extended001_exact_three_pins':g.FAILED_PINS,'smoke_invocation':g.PREVIOUS_INVOCATION,'failed_invocation':g.FAILED_INVOCATION,'smoke_eight_identifiers_absent':True,'failed_owner_exact_status_and_absent_output':True}
check('predecessors',predecessor_proofs)
def campaign():
 h=out['_host'];identity=out['inputs']['identity'];c=read(R/'campaign.json');out['campaign_raw']=c
 assert c['identity']==identity and c['status']=='completed' and c['terminal_inputs_unchanged'] is True and c['bounded_campaign_complete'] is True
 assert c['allocation']=='extended' and c['branch']=='caps' and c['planned_phases']==list(PHASES) and set(c['accepted_phases'])==set(PHASES)
 assert c['PPO_updates_completed']==500 and c['Stage2_complete'] is False and c['host_freeze_sha256']==HOST_FREEZE and c['automatic_continuation'] is False
 receipt=read(R/'train/training_receipt.json');assert receipt['complete'] is True and receipt['updates_completed']==500
 assert receipt['audit']['controls']==12000 and receipt['audit']['replicas']==1024
 assert c['observed_training']['observed_updates_completed']==500 and c['observed_training']['training_receipt_complete'] is True
 assert c['observed_training']['training_receipt_sha256']==sha(R/'train/training_receipt.json')
 final=receipt['final_checkpoint_sha256']
 for phase in PHASES:
  accepted=h.CONTRACT.validate_result(R/phase,phase,identity,expected_checkpoint_sha256=final)
  assert accepted==c['accepted_phases'][phase]==read(R/(phase+'_accepted.json'))
  assert h.tree_hashes(R/phase)==read(R/(phase+'_immutable.sha256.json'))
 assert c['accepted_phases']['train']['optimizer_diagnostics']=={'schema':'direct315_actor_gradients_v1','updates':500,'minibatches':10000,'sparse_actor_gradient_rows':14}
 decisions={n:{**row,'path':str(R/'train/policy'/row['file'])} for n,row in receipt['decision_checkpoints'].items()}
 return {'completed':True,'campaign_sha256':sha(R/'campaign.json'),'updates':500,'actual_controls':12000,'actual_transitions':12288000,'receipt_sha256':sha(R/'train/training_receipt.json'),'checkpoint_path':str(R/'train/policy/final.pt'),'checkpoint_sha256':sha(R/'train/policy/final.pt'),'decisions':decisions,'extended_readback':receipt['extended_readback'],'wall_seconds':receipt['wall_seconds'],'reload':receipt['reload'],'phase_trees_unchanged':list(PHASES),'quiet_failed_env_ids':c['accepted_phases']['final_stop']['quiet_failed_env_ids'],'acquisition_is_not_physical_acceptance':True}
check('campaign',campaign)
out['jobs']={};out['owned_absence']={}
for phase in PHASES:
 def job(phase=phase):
  j=read(R/'jobs'/(phase+'.json'));out['jobs'][phase]=j
  assert j['status']=='completed' and j['exit_code']==0 and j['cleanup_checked'] is True and j['container_id'] and j['container_name']
  for token in (j['container_name'],j['container_id']):
   q=call(['docker','inspect',token]);out['owned_absence'][token]=q
   assert q['exit_code']==1 and any(s in q['stderr'].lower() for s in ('no such object','no such container'))
  return {'job_sha256':sha(R/'jobs'/(phase+'.json')),'owned_name_and_id_absent':True}
 check('job_'+phase,job)
def restoration():
 p=read(P/'pause.json');r=read(P/'restored.json');out['pause_raw']=p;out['restored_raw']=r
 assert p['unit']==UNIT and p['output']==str(R) and r['restored_unix']>=p['created_unix']
 assert r['timers']==[n for n,s in p['units'].items() if n.endswith('.timer') and 'ActiveState=active' in s]
 assert p['coordination_sha256']==sha(Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md'))=='35af5100e39980f55a3f5e19654e4550b3c3fb6da114b64fbe0d143817242cf3'
 return r
check('restoration',restoration)
def halo():
 root=B/'forecast_halo_defer_001';state=read(root/'state.json');archive=root/state['archive'];manifest=read(archive/'SHA256.json')
 assert sha(archive/'SHA256.json')==state['archive_manifest_sha256']=='53409b319ac4f77ae19be0648b2f8e3b3ffb9b542f7c6a36c6aa597f0b4c650c'
 actual=tree(archive);actual.pop('SHA256.json');assert actual==manifest
 return {'unchanged':True,'manifest_sha256':state['archive_manifest_sha256'],'payloads':len(manifest),'archive_remains_remote':True}
check('halo_archive',halo)
def inventory():
 records={}
 for label,root in [('run',R),('pause',P)]:
  hashes=tree(root)
  for rel,digest in hashes.items():records[label+'/'+rel]={'sha256':digest,'bytes':(root/rel).stat().st_size}
 return records
check('inventory',inventory)
pins=[R/'campaign.json',*(R/'jobs'/(p+'.json') for p in PHASES),P/'pause.json',P/'launch.json',P/'restored.json']
out['pins']={str(p.relative_to(B)):sha(p) for p in pins if p.is_file()}
out['files']=len(out.get('inventory',{}));out['bytes']=sum(x['bytes'] for x in out.get('inventory',{}).values());out.pop('_host',None)
out['ordinary_autosaves']={k:v for k,v in out.get('inventory',{}).items() if k.startswith('run/train/policy/model_')};out['large_raw_fetched']=False
out['passed']=not out['errors'];print(json.dumps(out,indent=2));sys.exit(0 if out['passed'] else 1)
