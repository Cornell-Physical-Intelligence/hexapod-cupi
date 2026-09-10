"""Portable read-only integrity/verdict verification; no numerical simulation imports."""
from pathlib import Path
import hashlib,json,sys
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
 audit=read(H/'terminal_audit/remote_audit.json');after=read(H/'terminal_audit/remote_audit_after_fetch.json')
 for key in ('raw_payloads','raw_sizes_bytes','input_maps','assets_files','assets_unchanged','jobs','owned_names_and_ids_absent','pause_restoration','campaign_sha256'):
  assert audit[key]==after[key],key
 storage=read(H/'RAW_STORAGE.json');assert set(storage)==set(audit['raw_payloads']) and len(storage)==74
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
 assert chunks==2 and sum(r['bytes'] for r in storage.values())==540347369
 def raw(name):
  r=storage[name];assert r['kind']=='file';return read(H/r['path'])
 c=raw('run/campaign.json')
 assert c['status']=='completed' and c['phase_group']=='admission' and c['PPO_updates_completed']==0
 assert c['learning_admission_only_complete'] and not c['automatic_continuation_allowed'] and not c['Stage2_complete']
 assert c['source_and_inputs_unchanged'] and set(c['accepted_phases'])=={'standing','calibration','learning_recovery_32'}
 assert c['last_completed_phase']=='learning_recovery_32'
 for name in c['accepted_phases']:
  assert raw('run/'+name+'_accepted.json')==c['accepted_phases'][name]
  mapping={k[len('run/'+name+'/'):]:v['sha256'] for k,v in storage.items() if k.startswith('run/'+name+'/')}
  assert mapping==raw('run/'+name+'_immutable.sha256.json')
 for key,count in [('source',926),('consumer',51),('host',3),('guard',11),('bridge',18),('observation',160)]:
  assert audit['input_maps'][key]['files']==count and audit['input_maps'][key]['unchanged']
 assert audit['assets_files']==550 and audit['assets_unchanged'] and len(raw('run/inputs/study_before.sha256.json'))==550
 assert set(audit['jobs'])=={'standing','calibration','learning_recovery_32'} and len(audit['owned_names_and_ids_absent'])==6
 for job in audit['jobs'].values():
  assert job['cleanup_checked'] and job['container_id'] and job['status']=='completed'
  for token in (job['container_name'],job['container_id']):assert audit['owned_names_and_ids_absent'][token]['absent']
 assert audit['unit']['ActiveState']=='inactive' and audit['unit']['ExecMainStatus']=='0' and audit['unit']['InvocationID']==''
 journal=audit['historical_invocation_journal'];u='hexapod-learning-ppo-admission-001-20260910.service'
 selected=[r for r in journal if r.get('USER_UNIT')==u]
 assert {r['USER_INVOCATION_ID'] for r in selected}=={'d161f77036f346d2b1abe244d95b2dbe'}
 assert any(r['MESSAGE'].startswith('Started '+u) for r in selected) and any('Consumed ' in r['MESSAGE'] for r in selected)
 restored=raw('forecast_pause/restored.json');assert restored==audit['pause_restoration'] and restored['restored_unix']==1789048842.716838
 s=raw('run/learning_recovery_32/state.json');assert s['learning_recovery_passed'] and s['PPO_updates_completed']==0 and not s['policy_training_started']
 n=read(H/'independent_numerical_review/report.json')
 assert n['integrity_passed'] and not n['physical_admission'] and not n['training_allocation_approved']
 assert n['raw']['completed_controls']==712 and n['raw']['physical_samples']==5697 and n['raw']['active_transitions']==15443 and n['raw']['recovery_transitions_excluded']==941
 assert n['ledger']['completed_finite_recovery_rows']==2 and n['ledger']['bootstrap_records']==5
 assert n['raw']['finite_terminal_transitions']==6 and n['raw']['PPO_updates_replayed']==0 and n['raw']['all_14_sensor_clocks_replayed']
 assert n['ledger']['actual_final_packet_and_reset_history_replayed'] and n['ledger']['terminal_bootstrap_zero_replayed'] and not n['ledger']['GPU_timeout_bootstrap_observed']
 assert (H/'terminal_audit/audit_attempt001/empty_remote_audit.json').stat().st_size==0
 assert 'Wrong or unavailable invocation identity' in (H/'terminal_audit/audit_attempt001/audit_stderr.log').read_text()
 print(json.dumps({'bundle_payloads_verified':total,'raw_original_files_verified':74,'losslessly_chunked_originals':2,
  'all_three_admission_phases_completed':True,'controls':712,'active_transitions':15443,'excluded_recovery_transitions':941,
  'completed_recovery_rows':2,'finite_terminal_rows':6,'PPO_updates':0,'all_three_owned_names_and_IDs_absent':True,
  'pause051_restored_unix':restored['restored_unix'],'physical_admission':False,'automatic_PPO_allocation':False,'read_only':True},indent=2))
if __name__=='__main__':main()
