"""Portable saved-evidence verifier. No SSH, subprocess, GPU or operator-script execution."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with path.open('rb')as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(rel):return json.loads((ROOT/rel).read_text())
def require(condition,message):
 if not condition:raise ValueError(message)
def validate():
 manifest=read('BUNDLE_SHA256.json');files=manifest.get('files',manifest)
 actual={str(p.relative_to(ROOT))for p in ROOT.rglob('*')if p.is_file()and p.name!='BUNDLE_SHA256.json'and '__pycache__'not in p.parts}
 require(actual==set(files),'Public inventory differs')
 for rel,digest in files.items():require(not(ROOT/rel).is_symlink()and sha(ROOT/rel)==digest,'Changed public payload:'+rel)
 selected=read('SELECTION.json');copy=read('COPY_VERIFICATION.json')
 for row in copy['files']:
  p=ROOT/row['public_copy'];entry=selected[row['source_group']]['inventory'][row['source_path']]
  require(sha(p)==entry['sha256']==row['sha256']and p.stat().st_size==entry['bytes']==row['bytes'],'Selected copy mismatch')
 for group,subdir in [('scheduler','scheduler'),('git','github')]:
  parent=read(subdir+'/FREEZE_SHA256.json');parent=parent.get('files',parent)
  inv=selected[group]['inventory']
  require(set(inv)-{'FREEZE_SHA256.json'}==set(parent),'Recorded parent inventory mismatch')
  require(all(inv[p]['sha256']==v for p,v in parent.items()),'Recorded parent hash mismatch')
  require(sha(ROOT/subdir/'FREEZE_SHA256.json')==copy['complete_source_checks'][group]['freeze_sha256'],'Parent freeze identity differs')
 # Original local parent checks are recorded; excluded source payload bytes cannot be replayed here.
 a=read('actual/automation_block_verified.json');r=read('actual/reconstruction_block_receipt.json')
 before=read('actual/exclusive_automation_block_001/before.json');backup=read('actual/backup_inventory.json')
 require(a['verified']is True and r['verified']is True and a['original_unit_files_verified']is True,'Missing successful receipts or original-file verification')
 require(a['user_unit_count']==len(a['user_units'])==31,'Wrong masked-unit count')
 require(set(a['user_units'])==set(before['units'])==set(before['files']),'Original/current exact unit mismatch')
 require(len(a['mask_paths'])==31 and {Path(p).name for p in a['mask_paths']}==set(a['user_units']),'Mask path mismatch')
 for name,state in a['user_units'].items():
  require(state['LoadState']==state['UnitFileState']=='masked'and state['ActiveState']=='inactive','Unit not verified masked/inactive:'+name)
  if name.endswith('.service'):require(state['MainPID']=='0','Service PID remains:'+name)
  else:require('MainPID'not in state,'Unexpected timer PID field')
 require(sum(s['NeedDaemonReload']=='yes'for s in a['user_units'].values())==11,'Reload caveat changed')
 system=a['system_units_already_masked'];require(len(system)==4,'Wrong original system units')
 for name,state in system.items():
  require(state['LoadState']==state['UnitFileState']=='masked'and state['ActiveState']=='inactive','System state differs')
  if name.endswith('.service'):require(state['MainPID']=='0','System service PID remains')
 probes=a['direct_start_and_reenable_probes'];require(len(probes)==5,'Missing direct/refuse probes')
 require(all(p['returncode']!=0 and 'masked'in p['stderr'].lower()for p in probes),'Probe did not reject')
 require(sha(ROOT/'actual/exclusive_automation_block_001/before.json')==a['before_sha256'],'Original unit receipt changed')
 require(len(backup)==32,'Wrong remote backup inventory count')
 for name,meta in before['files'].items():
  item=backup[a['backup_root']+'/'+name]
  require(meta['kind']==item['kind']=='file'and meta['sha256']==item['sha256']and meta['mode']==item['mode'],'Remote unit backup identity mismatch:'+name)
 workerbackup=backup[r['worker_source_backup']]
 require(workerbackup['sha256']==r['worker_source_original_sha256'],'Worker original backup differs')
 original=read('actual/exclusive_reconstruction_001/before.json')
 bkey='/home/orionh/HEXAPOD_runs/canonical_direct_20260910/exclusive_reconstruction_001/before.json'
 require(sha(ROOT/'actual/exclusive_reconstruction_001/before.json')==r['files'][bkey],'Worker original receipt changed')
 observed=read('actual/observed.json')['producers'][0][0]
 require(original['worker_pid']==observed['pid']and original['worker_starttime']==observed['starttime'],'Stopped original worker identity differs')
 require(original['worker_source_sha256']==r['worker_source_original_sha256'],'Worker source mismatch')
 require(r['exact_worker_exited']is True and r['signal']=='SIGTERM via bound pidfd','Exact stop missing')
 require(r['unit']['ActiveState']=='active'and r['unit']['UnitFileState']=='enabled'and int(r['unit']['MainPID'])==r['queue_lock']['pid'],'Persistent queue holder mismatch')
 require(r['queue_lock']['queue_mutated']is False and r['queue_preserved']is True,'Queue preservation not recorded')
 require(r['queue_lock']['reservation_sha256']==r['persistent_reservation_unchanged'],'Reservation identity changed')
 require(a['release_only_on_user_instruction']is True and a['own_HEXAPOD_units_untouched']is True,'Reservation scope changed')
 require(not a['cuda_processes'].strip()and not r['cuda_processes'].strip(),'CUDA context remained at saved observations')
 require("KeyError: 'MainPID'"in(ROOT/'failures/automation_block.stderr').read_text(),'Original failure not preserved')
 require('automation_block_receipt.json'not in files,'Empty placeholder represented as receipt')
 for name,key in [('forecast_workflows','workflows'),('forecast_runs','workflow_runs')]:
  data=read('github/'+name+'.json');require(data['method']=='GET'and data['exit_code']==0 and data['data']['total_count']==0 and data['data'][key]==[],'GitHub empty inventory changed')
 git=read('github/REPORT.json');require(git['read_only']is True and git['mutations']==0,'GitHub mutation falsely recorded')
 require(git['github_workflow_count']==git['github_run_count']==0,'GitHub report mismatch')
 tree=read('github/forecast_tree.json');require(tree['data']['truncated']is False,'Incomplete recorded Git tree')
 require(not any(p['path'].startswith('.github/workflows/')for p in tree['data']['tree']),'Workflow path present')
 return {'public_payloads':len(files),'user_masks_verified':31,'already_masked_system_units':4,'rejected_probes':5,'reload_yes_preserved':11,'remote_backup_metadata_entries':32,'github_mutations':0,'live_remote_query':False,'remote_backup_contents_replayed':False,'no_general_manual_GPU_partition_claim':True}
if __name__=='__main__':print(json.dumps(validate(),indent=2))
