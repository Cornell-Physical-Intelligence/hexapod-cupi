"""Resume only missing curated files; original failure and verified inputs preserved."""
from pathlib import Path
import hashlib,json,shutil,subprocess,time
ROOT=Path('tmp/direct_omni_train_pilot_curriculum_results_001');plan=json.loads((ROOT/'FETCH_PLAN.json').read_text());selected=plan['selected'];base=plan['remote_base'];reserve=512<<20
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
missing=[];verified={}
for key,bound in selected.items():
 p=ROOT/key
 if p.exists():
  if sha(p)!=bound:raise RuntimeError('Already-fetched file changed: '+key)
  verified[key]=bound
 else:missing.append(key)
needed=sum(plan['all_remote_files'][k]['size_bytes'] for k in missing);free=shutil.disk_usage(ROOT).free
if free<needed+reserve:raise RuntimeError(f'Resume needs{needed}+{reserve};free{free}')
print('RESUME',len(missing),needed,'FREE',free,flush=True)
for key in sorted(missing,key=lambda k:plan['all_remote_files'][k]['size_bytes'],reverse=True):
 size=plan['all_remote_files'][key]['size_bytes']
 if shutil.disk_usage(ROOT).free<size+reserve:raise RuntimeError('Disk headroom changed before '+key)
 prefix,relative=key.split('/',1);remote=base+'/'+('direct_omni_train_pilot_curriculum_001' if prefix=='run' else 'forecast_pause_056')+'/'+relative
 target=ROOT/key;target.parent.mkdir(parents=True,exist_ok=True);part=target.with_name(target.name+'.part')
 subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8','spark:'+remote,str(part)],check=True,timeout=180)
 if part.stat().st_size!=size or sha(part)!=selected[key]:raise RuntimeError('Fetched mismatch '+key)
 part.replace(target);verified[key]=selected[key];print('VERIFIED',size,key,flush=True)
(ROOT/'LOCAL_SHA256.json').write_text(json.dumps(verified,indent=2)+'\n')
(ROOT/'FETCH_COMPLETE.json').write_text(json.dumps({'completed_unix':time.time(),'selected_files':len(verified),'selected_bytes':plan['selected_bytes'],'remote_files':len(plan['all_remote_files']),'remote_bytes':sum(x['size_bytes'] for x in plan['all_remote_files'].values()),'omitted_files':len(plan['omitted_remote_only']),'omitted_bytes':sum(v['size_bytes'] for v in plan['omitted_remote_only'].values()),'all_selected_hashes_verified':True,'free_bytes_after':shutil.disk_usage(ROOT).free,'minimum_reserve_bytes':reserve,'no_remote_files_modified':True,'first_attempt_failed_transient_local_ENOSPC':True,'resume_selected_files':len(missing)},indent=2)+'\n')
print('CURATED_FETCH_COMPLETE',len(verified),flush=True)
