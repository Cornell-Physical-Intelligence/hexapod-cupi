"""Saved-evidence checks only: no remote, native execution or operator script import."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
W='/home/orionh/ithaca-reconstruction/queue-worker.py'
B='/home/orionh/HEXAPOD_runs/canonical_direct_20260910/exclusive_reconstruction_002'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads((R/p).read_text())
def need(ok,message):
 if not ok:raise ValueError(message)
def validate():
 files=read('BUNDLE_SHA256.json')['files']
 actual={str(p.relative_to(R))for p in R.rglob('*')if p.is_file()and p.name!='BUNDLE_SHA256.json'and '__pycache__'not in p.parts}
 need(actual==set(files),'Wrong public inventory')
 for p,h in files.items():need(not(R/p).is_symlink()and sha(R/p)==h,'Changed public payload:'+p)
 for item in read('SOURCE_SELECTION.json')['selected_copies']:
  p=R/item['public_copy'];need(sha(p)==item['sha256']and p.stat().st_size==item['bytes'],'Copy binding changed')
 prior=read('ancestry/reconstruction_block_receipt.json')
 oldmap=read('ancestry/ops001_BUNDLE_SHA256.json')['files']
 need(sha(R/'ancestry/ops001_BUNDLE_SHA256.json')=='c691f1aa5c975f9499701ecafacaca3cca0df50dfbdd745561d5b71d079f046f','Wrong ops001 ancestry')
 need(sha(R/'ancestry/reconstruction_block_receipt.json')==oldmap['actual/reconstruction_block_receipt.json'],'Old receipt changed')
 probe=read('rejected_probe_source/FREEZE_SHA256.json')
 need(len(probe)==2,'Wrong rejected source')
 for p,h in probe.items():need(sha(R/'rejected_probe_source'/p)==h,'Changed rejected probe:'+p)
 transfer=read('failed_preflight/reservation_probe_transfer.json');need(transfer['returncode']==0,'Probe transfer did not finish')
 payload=json.loads(transfer['stdout']);need(payload['verified_files']==3 and payload['manifest_sha256']==sha(R/'rejected_probe_source/FREEZE_SHA256.json'),'Probe transfer/freeze mismatch')
 policy=read('rejected_probe_source/inputs/reservation_policy.json')
 need(policy['files'][W]==prior['worker_source_blocked_sha256'],'Old expected gate mismatch')
 failure=(R/'failed_preflight/reservation_probe.stderr').read_text()
 need('RuntimeError: Persistent automation/reconstruction binding changed: '+W in failure,'Actual failed preflight missing')
 current=read('actual/reconstruction_entry_block_receipt.json');before=read('actual/replacement_before.json')
 need(current['verified']is True and current['blocked_entry_directory']==W and current['entry_directory_is_symlink']is False,'New entry receipt invalid')
 need(sha(R/'actual/replacement_before.json')==current['files'][B+'/before.json']['sha256'],'Replacement observation changed')
 need(before['entry_sha256']==prior['worker_source_original_sha256'] and before['prior_gate_sha256']==policy['files'][W],'Replacement/original mismatch')
 for name in ['uploader_restored_original.py','removed_entry.py']:
  meta=current['files'][B+'/'+name];need(meta['sha256']==prior['worker_source_original_sha256']and meta['size_bytes']==6325,'Original backup mismatch')
 blocker=current['files'][W+'/__main__.py'];need(blocker['sha256']=='2ff83c0fada4bb18a0f568bec267fadf7ab9e7c7a60f83a8ecce38463ac9537c'and blocker['size_bytes']==140,'Wrong blocker identity')
 python=current['python_entry_probe']
 need(python['returncode']==1 and python['stdout']=='' and python['isolated_stdlib_only']is True and 'Spark external automation is blocked'in python['stderr'],'Python probe did not reject')
 need(current['ordinary_file_open_blocked']is True and current['queue_lock_nonblocking_probe_returncode']==1,'File/lock refusal absent')
 unit=current['queue_lock_unit'];need(unit['ActiveState']=='active'and unit['SubState']=='running'and unit['UnitFileState']=='enabled'and unit['MainPID']==prior['unit']['MainPID'],'Queue holder identity/state differs')
 need(not current['cuda_processes'].strip()and current['original_source_preserved']is True and current['queue_outputs_untouched']is True and current['release_only_on_user_instruction']is True,'Preservation/snapshot scope differs')
 passed_source=read('accepted_probe_source/FREEZE_SHA256.json')
 need(len(passed_source)==2,'Wrong accepted source inventory')
 for p,h in passed_source.items():need(sha(R/'accepted_probe_source'/p)==h,'Changed accepted probe source:'+p)
 pt=read('successful_preflight/reservation_probe002_transfer.json');need(pt['returncode']==0,'Accepted probe transfer failed')
 transfer2=json.loads(pt['stdout']);need(transfer2['verified_files']==3 and transfer2['manifest_sha256']==sha(R/'accepted_probe_source/FREEZE_SHA256.json'),'Accepted transfer/freeze mismatch')
 passed=read('successful_preflight/reservation_probe002.json');policy2=read('accepted_probe_source/inputs/reservation_policy.json')
 need(passed['verified']is True and passed['source_sha256']==passed_source['launch_guarded_remote.py']and passed['policy_sha256']==passed_source['inputs/reservation_policy.json'],'Accepted probe source/policy binding differs')
 need(passed['native_calls']==0 and passed['output_created']is False and passed['pause_created']is False,'Reservation probe mislabeled as no native launch')
 res=passed['reservation'];need(res['masked_user_units']==31 and len(res['mask_reload_flags'])==31 and sum(v=='yes'for v in res['mask_reload_flags'].values())==11,'Mask/reload metadata changed')
 need(res['queue_lock_held']is True and res['queue_lock_owner_pid']==int(unit['MainPID'])and res['reservation_active_at_preflight']is True and res['release_attempted']is False,'Reservation/queue-lock verification differs')
 need(res['mask_policy_sha256']==passed['policy_sha256']and res['marker_sha256']==prior['persistent_reservation_unchanged'],'Accepted reservation identity changed')
 need(policy2['files'][W+'/__main__.py']==blocker['sha256'],'Accepted policy did not bind new entry blocker')
 return {'public_payloads':len(files),'rejected_probe_payloads':2,'accepted_probe_payloads':2,'successful_reservation_only_probe':True,'old_file_gate_replacement_preserved':True,'new_python_fileopen_lock_probes_verified':True,'remote_backup_contents_replayed':False,'retry_launch_claim':False,'general_privileged_GPU_partition_claim':False}
if __name__=='__main__':print(json.dumps(validate(),indent=2))
