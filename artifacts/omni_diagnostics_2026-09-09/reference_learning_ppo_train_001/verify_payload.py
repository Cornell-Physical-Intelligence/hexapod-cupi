"""Portable read-only raw integrity and failure-receipt verification; standard library only."""
from pathlib import Path
import hashlib,json,math,sys
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def tree(root,manifest,bound=None):
 f=root/manifest
 if bound is not None:assert sha(f)==bound
 m=read(f);m=m.get('files',m)
 assert not root.is_symlink() and not any(p.is_symlink() for p in root.rglob('*'))
 actual={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p!=f}
 assert actual==m,str(root)
 return len(m)
def main():
 total=tree(H,'BUNDLE_SHA256.json');recon=read(H/'RECONSTRUCTION.json')
 for name,r in recon['copied_frozen_bundles'].items():assert tree(H/name,r['manifest'],r['sha256'])==r['payload_count']
 for name,r in recon['referenced_maps'].items():
  p=H/'references'/name;assert sha(p)==r['sha256'] and len(read(p))==r['payload_count']
 audit=read(H/'terminal_audit/remote_audit.json')
 storage=read(H/'RAW_STORAGE.json');assert set(storage)==set(audit['raw_payloads']) and len(storage)==46
 assert read(H/'terminal_audit/local_raw_verification.json')['remote_audit_sha256']==sha(H/'terminal_audit/remote_audit.json')
 sys.path.insert(0,str(H/'chunk_tool'));from chunk_artifact import consume
 chunks=0
 for logical,r in storage.items():
  assert not Path(logical).is_absolute() and '..' not in Path(logical).parts
  assert r['sha256']==audit['raw_payloads'][logical] and r['bytes']==audit['raw_sizes_bytes'][logical]
  if r['kind']=='file':
   p=H/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'] and r['bytes']<=100*1024*1024
  else:
   assert r['kind']=='lossless_chunks' and r['bytes']>100*1024*1024
   p=H/r['manifest'];assert sha(p)==r['manifest_sha256'];m=consume(p.parent)
   assert m['logical_path']==logical and m['original_bytes']==r['bytes'] and m['original_sha256']==r['sha256']
   assert all(c['bytes']<=64*1024*1024 for c in m['chunks']);chunks+=1
 assert chunks==2 and sum(r['bytes'] for r in storage.values())==460583084
 def raw(name):
  r=storage[name];assert r['kind']=='file';return read(H/r['path'])
 c=raw('run/learning_campaign.json');s=raw('run/train_10/state.json')
 assert c['status']=='failed' and c['phase_group']=='learn' and c['PPO_updates_completed']==0
 assert c['training_reported_PPO_updates_completed']==s['PPO_updates_completed']==audit['training_completed_updates']==9
 assert c['training_reported_policy_training_started'] and s['policy_training_started']
 assert s['status']=='rejected' and s['session']['controls']==2600 and s['mode']=='train_10'
 assert 'Recovered canonical stance failed unchanged contact/motor checks' in s['error']
 assert c['source_and_inputs_unchanged'] and s['source_inputs_unchanged'] and audit['source_and_inputs_unchanged_producer']
 assert not s['Stage2_complete'] and not c['Stage2_complete'] and not audit['decision010_present']
 assert not any('.pt' in k or '/quiet' in k or '/retention' in k for k in storage)
 updates=[json.loads(x) for x in (H/storage['run/train_10/updates.jsonl']['path']).read_text().splitlines()]
 assert [x['updates_completed'] for x in updates]==list(range(1,10))
 assert all(all(math.isfinite(x) for x in r['losses'].values()) for r in updates)
 decision=read(H/'reviewed_decision001.json');decision_sha=sha(H/'reviewed_decision001.json')
 assert decision_sha==c['learning_decision']['decision_sha256']==s['learning_decision']['decision_sha256']
 assert decision['accepted'] and decision['approved_updates']==10
 assert s['identity']==decision['identity']
 assert s['input_checkpoint_sha256']==recon['separate_prior_admission']['initial_checkpoint_sha256']
 for field,name in [('consumer_freeze_sha256','consumer003.json'),('physical_source_sha256','source009.json'),('observation_freeze_sha256','observation005.json'),('bridge_freeze_sha256','bridge001.json')]:
  assert s['identity'][field]==recon['referenced_maps'][name]['sha256']
 assert c['host_freeze_sha256']==recon['referenced_maps']['host001.json']['sha256']
 job=raw('run/jobs/train_10.json');assert job['status']=='failed' and job['cleanup_checked'] and job['phase']=='train_10'
 assert not job['no_policy_loaded'] and job['container_id']
 assert set(audit['owned_names_IDs_absent'])=={job['container_name'],job['container_id']}
 for token in (job['container_name'],job['container_id']):assert audit['owned_names_IDs_absent'][token]['absent']
 assert audit['unit']['ActiveState']=='failed' and audit['unit']['Result']=='exit-code'
 assert audit['unit']['InvocationID']==audit['expected_invocation']=='804a727558374268b185325305148b1a'
 restored=raw('forecast_pause/restored.json');assert restored==audit['pause_restoration'] and restored['restored_unix']==1789049775.7754648
 assert restored['owned_cleanup_checked']==[] # Historical restorer receipt; direct owned-name/ID inspection is the cleanup evidence.
 n=read(H/'failure_review/report.json')
 assert n['PPO_updates_completed']==9 and n['controls']==2600 and n['failed_rows']==[8] and not n['qualifying']
 assert not n['decision_checkpoint_produced'] and n['recovery_controls']==200 and n['timeout_control']==2400
 assert n['six_contacts_at_failure_all_rows'] and not n['recovered_rows_nonfoot'] and not n['recovered_rows_native_termination']
 assert n['failed_joint']['runtime_name']=='revolute_2_1' and n['failed_joint']['named_leg']=='RM'
 assert n['failed_joint']['last_eight_substep_peak_nm']==1.6276484727859497
 assert n['reset_contract_checks']['recovery_raw_action_and_residual_exact_zero']
 for logical,h in n['input_sha256'].items():assert storage[logical]['sha256']==h
 full=recon['independent_remote_source_audit'];assert full['status']=='passed'
 assert sha(H/full['path'])==full['sha256'] and sha(H/full['copy_manifest'])==full['copy_manifest_sha256']
 files=read(H/full['copy_manifest']);assert len(files)==6
 for name,h in files.items():assert sha(H/'terminal_audit/full_integrity'/name)==h
 f=read(H/full['path']);assert f['audit_read_only'] and f['assets_files']==550 and f['assets_unchanged']
 map_names={'source':('source009.json',926),'consumer':('consumer003.json',51),'host':('host001.json',3),'guard':('initial_guard001.json',11),'bridge':('bridge001.json',18),'observation':('observation005.json',160)}
 for key,(name,count) in map_names.items():
  row=f['input_maps'][key];assert row['unchanged'] and row['files']==count and row['manifest_sha256']==recon['referenced_maps'][name]['sha256']
 row=f['input_maps']['continuation_guard'];assert row['unchanged'] and row['files']==7 and row['manifest_sha256']==recon['copied_frozen_bundles']['continuation_guard']['sha256']
 assert f['decision_sha256']==decision_sha and f['host_readonly_identity']==s['identity']
 assert f['initial_campaign_sha256']==audit['initial_campaign_unchanged_sha256']
 for key,count in [('initial',74),('learning',46)]:
  check=f['raw_snapshot_checks'][key];assert check['unchanged'] and check['files']==count
  assert sha(H/'terminal_audit/full_integrity'/(key+'_snapshot.json'))==check['snapshot_sha256']
 assert read(H/'terminal_audit/full_integrity/learning_snapshot.json')==audit
 initial=read(H/'terminal_audit/full_integrity/initial_snapshot.json')
 assert initial['campaign_sha256']==f['initial_campaign_sha256'] and len(initial['raw_payloads'])==74
 assert initial['raw_payloads']['run/calibration/initial.pt']==s['input_checkpoint_sha256']
 assert initial['raw_payloads']['run/inputs/study_before.sha256.json']==recon['referenced_maps']['study550.json']['sha256']
 print(json.dumps({'bundle_payloads_verified':total,'raw_original_files_verified':46,'raw_original_bytes':460583084,'losslessly_chunked_originals':2,
 'PPO_updates_actual':9,'host_completed_allocation_updates':0,'fatal_control':2600,'failed_row':8,'decision010_checkpoint_produced':False,
 'cold_or_quiet_evaluation_produced':False,'owned_training_name_and_ID_absent':True,'pause052_restored_unix':restored['restored_unix'],
 'independent_remote_source_audit':recon['independent_remote_source_audit'],'Stage2_complete':False,'read_only':True},indent=2))
if __name__=='__main__':main()
