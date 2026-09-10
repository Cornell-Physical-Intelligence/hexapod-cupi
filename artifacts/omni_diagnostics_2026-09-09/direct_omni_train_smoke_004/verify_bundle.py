"""Portable byte/receipt verification. No network, GPU or simulator dependencies."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def tree(root):
 assert not root.is_symlink() and not any(p.is_symlink() for p in root.rglob('*'))
 return {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
def verify(root=H):
 actual=tree(root);actual.pop('FREEZE_SHA256.json');assert actual==read(root/'FREEZE_SHA256.json')
 t=root/'terminal';r=t/'run';p=t/'pause';audit=read(t/'remote_terminal_audit_002.json')
 assert audit['passed'] is True and audit['errors']==[] and all(audit['checks'].values())
 assert audit['inputs']['all_verified'] and (audit['inputs']['source_files'],audit['inputs']['asset_files'],audit['inputs']['supervisor_files'],audit['inputs']['legacy_files'],audit['inputs']['native_files'])==(599,550,926,16,28)
 raw={label+'/'+n:v for label,base in [('run',r),('pause',p)] for n,v in tree(base).items()}
 assert raw=={n:x['sha256'] for n,x in audit['inventory'].items()}
 assert len(raw)==52 and sum((t/n).stat().st_size for n in raw)==76234247
 for name,row in audit['inventory'].items():assert (t/name).stat().st_size==row['bytes']
 fetched=read(t/'FETCH_VERIFICATION.json');assert fetched['complete'] and fetched['local_omissions']==[] and fetched['files']==audit['inventory']
 assert fetched['audit_sha256']==sha(t/'remote_terminal_audit_002.json')
 campaign=read(r/'campaign.json');assert campaign['status']=='completed' and campaign['terminal_inputs_unchanged'] is True and campaign['PPO_updates_completed']==2 and campaign['Stage2_complete'] is False
 phases=['standing','train','final_constant','final_stop'];assert campaign['planned_phases']==phases and set(campaign['accepted_phases'])==set(phases)
 for phase in phases:
  assert tree(r/phase)==read(r/(phase+'_immutable.sha256.json'))
  assert sha(r/phase/'state.json')==campaign['accepted_phases'][phase]['state_sha256']
  job=read(r/'jobs'/(phase+'.json'));assert job==audit['jobs'][phase]
  assert job['status']=='completed' and job['exit_code']==0 and job['cleanup_checked'] is True
  for token in (job['container_name'],job['container_id']):
   absent=audit['owned_absence'][token];assert absent['exit_code']==1 and any(x in absent['stderr'].lower() for x in ('no such object','no such container'))
 assert len(audit['owned_absence'])==8
 pins=read(t/'TERMINAL_PINS.json');assert pins['passed'] and pins['pins']==audit['pins'] and len(pins['pins'])==7
 for name,bound in pins['pins'].items():
  parts=Path(name).parts;where=r if parts[0]=='direct_omni_train_smoke_004' else p
  assert parts[0] in ('direct_omni_train_smoke_004','forecast_pause_061') and sha(where.joinpath(*parts[1:]))==bound
 assert read(p/'restored.json')==audit['restoration'] and audit['halo_archive']['unchanged'] is True
 receipt=read(r/'train/training_receipt.json');assert receipt['complete'] is True and receipt['updates_completed']==2
 assert receipt['audit']['controls']==48 and receipt['audit']['replicas']==32 and receipt['reload']['optimizer_entries']==17
 for k in ('passed','exact_actor_critic_normalizer_optimizer','exact_deterministic_action'):assert receipt['reload'][k] is True
 assert sha(r/'train/policy/final.pt')==receipt['final_checkpoint_sha256']=='ffdb347eacea5d87172fdcd122eaa6836fc43ed1bd139f20493fdf3d9672b000'
 analysis=read(root/'analysis/report.json');assert analysis['evidence_verified'] is True and analysis['errors']==[] and analysis['Stage2_complete'] is False
 assert analysis['final_stop']['quiet_passed_replicas']==0 and analysis['training']['updates_completed']==2
 parity=read(t/'SMOKE_BEHAVIOR_PARITY.json');assert all(x['equal'] and x['smoke003']==x['smoke004']==raw[n] for n,x in parity['files'].items())
 for base in (t,root/'analysis',root/'analyzer'):
  values=tree(base);values.pop('FREEZE_SHA256.json');assert values==read(base/'FREEZE_SHA256.json')
 return {'verified':True,'payloads':len(actual),'raw_files':len(raw),'raw_bytes':76234247,'actual_PPO_updates':2,'quiet_passed':0,'quiet_trials':48,'Stage2_complete':False,'remote_recheck_performed':False}
if __name__=='__main__':print(json.dumps(verify(),indent=2))
