"""Curated, hash-verified download; all intermediate autosaves remain remote."""
from pathlib import Path
import hashlib,json,shutil,subprocess,time
ROOT=Path('tmp/direct_omni_train_pilot_curriculum_results_001');ROOT.mkdir(exist_ok=False)
AUDIT=Path('tmp/direct_omni_train_pilot_curriculum_terminal_audit_001.json');a=json.loads(AUDIT.read_text());BASE='/home/orionh/HEXAPOD_runs/mock_length_study_20260909'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
selected={k:v for k,v in a['raw_payloads'].items() if '/policy/model_' not in k};omitted={k:{'sha256':v,'size_bytes':a['raw_sizes_bytes'][k],'reason':'Intermediate native autosave excluded from local copy; full bytes preserved remotely and decision10/25/50 plus final are selected.'} for k,v in a['raw_payloads'].items() if k not in selected}
needed=sum(a['raw_sizes_bytes'][k] for k in selected);free=shutil.disk_usage(ROOT).free;reserve=128<<20
if needed+reserve>free:raise RuntimeError(f'Insufficient disk headroom: selected={needed} free={free} reserve={reserve}')
shutil.copyfile(AUDIT,ROOT/'remote_audit.json')
plan={'remote_base':BASE,'all_remote_files':{k:{'sha256':v,'size_bytes':a['raw_sizes_bytes'][k]} for k,v in a['raw_payloads'].items()},'selected':selected,'selected_bytes':needed,'omitted_remote_only':omitted,'free_bytes_before':free,'minimum_reserve_bytes':reserve,'partial_local_tree_is_not_full_remote_immutable_tree':True}
(ROOT/'FETCH_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
def priority(k):
 if k.endswith('.npz'):return 2 if k.startswith('run/train/') else 1
 if k.endswith('.pt'):return 3
 return 0
verified={}
for key in sorted(selected,key=lambda k:(priority(k),a['raw_sizes_bytes'][k],k)):
 size=a['raw_sizes_bytes'][key]
 if shutil.disk_usage(ROOT).free<size+reserve:raise RuntimeError('Disk headroom changed before '+key)
 prefix,relative=key.split('/',1);remote=BASE+'/'+('direct_omni_train_pilot_curriculum_001' if prefix=='run' else 'forecast_pause_056')+'/'+relative
 target=ROOT/key;target.parent.mkdir(parents=True,exist_ok=True);partial=target.with_name(target.name+'.part')
 subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=8','spark:'+remote,str(partial)],check=True,timeout=180)
 if partial.stat().st_size!=size or sha(partial)!=selected[key]:raise RuntimeError('Fetched size/hash mismatch: '+key)
 partial.replace(target);verified[key]=selected[key];print('VERIFIED',size,key,flush=True)
(ROOT/'LOCAL_SHA256.json').write_text(json.dumps(verified,indent=2)+'\n')
(ROOT/'FETCH_COMPLETE.json').write_text(json.dumps({'completed_unix':time.time(),'selected_files':len(verified),'selected_bytes':needed,'remote_files':len(a['raw_payloads']),'remote_bytes':a['raw_total_bytes'],'omitted_files':len(omitted),'omitted_bytes':sum(v['size_bytes'] for v in omitted.values()),'all_selected_hashes_verified':True,'free_bytes_after':shutil.disk_usage(ROOT).free,'no_remote_files_modified':True},indent=2)+'\n')
print('CURATED_FETCH_COMPLETE',len(verified),needed,flush=True)
