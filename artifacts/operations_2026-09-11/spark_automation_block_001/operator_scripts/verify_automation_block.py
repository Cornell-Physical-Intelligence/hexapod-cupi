from pathlib import Path
import json,subprocess,hashlib,os,time
base=Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910');root=base/'exclusive_automation_block_001'
before=json.loads((root/'before.json').read_text());names=list(before['units']);assert len(names)==31
config=Path('/home/orionh/.config/systemd/user')
props=['LoadState','ActiveState','SubState','MainPID','UnitFileState','FragmentPath','NeedDaemonReload','DropInPaths']
def state(n):
 cmd=['systemctl','--user','show',n]
 for p in props:cmd+=['-p',p]
 return dict(l.split('=',1)for l in subprocess.check_output(cmd,text=True,timeout=20).splitlines())
after={n:state(n)for n in names}
for n,v in after.items():
 assert (config/n).is_symlink()and os.readlink(config/n)=='/dev/null'
 assert v['LoadState']=='masked'and v['UnitFileState']=='masked'and v['ActiveState']=='inactive'
 if n.endswith('.service'):assert v['MainPID']=='0'
 else:assert 'MainPID'not in v # Timer units expose no process property.
 meta=before['files'][n];p=root/'original_units'/n
 if meta['kind']=='file':assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256']
 elif meta['kind']=='symlink':assert p.is_symlink()and os.readlink(p)==meta['target']
probes=[]
for n in ('stormscope-dispatch.timer','stormscope-private-scout.service','stormscope-private-v2-dispatch.timer','ollama.service'):
 r=subprocess.run(['systemctl','--user','start',n],capture_output=True,text=True,timeout=20)
 assert r.returncode!=0 and 'masked'in r.stderr.lower();probes.append({'unit':n,'returncode':r.returncode,'stderr':r.stderr})
r=subprocess.run(['systemctl','--user','enable','--now','stormscope-dispatch.timer'],capture_output=True,text=True,timeout=20)
assert r.returncode!=0 and 'masked'in r.stderr.lower();probes.append({'command':'enable --now stormscope-dispatch.timer','returncode':r.returncode,'stderr':r.stderr})
system={n:dict(l.split('=',1)for l in subprocess.check_output(['systemctl','show',n,'-p','LoadState','-p','ActiveState','-p','UnitFileState','-p','MainPID'],text=True,timeout=20).splitlines())for n in ('wx-forecast.timer','wx-forecast.service','wx-forecast-pm.timer','wx-forecast-pm.service')}
assert all(v['LoadState']=='masked'and v['ActiveState']=='inactive'and (not n.endswith('.service')or v['MainPID']=='0')for n,v in system.items())
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True,timeout=20);assert not gpu.strip()
receipt={'verified':True,'observed_unix':time.time(),'user_unit_count':31,'user_units':after,'system_units_already_masked':system,'direct_start_and_reenable_probes':probes,'mask_paths':[str(config/n)for n in names],'backup_root':str(root/'original_units'),'before_sha256':hashlib.sha256((root/'before.json').read_bytes()).hexdigest(),'cuda_processes':gpu,'release_only_on_user_instruction':True,'own_HEXAPOD_units_untouched':True,'original_unit_files_verified':True,'initial_verifier_failure':'Timer units do not expose MainPID; original installer completed all31 masks before failing its timer property lookup. Exact error is retained.','reload_metadata':'Eleven masked units retain historical Condition drop-ins and report NeedDaemonReload=yes even after explicit daemon-reload. Both their loaded state and file state are masked and direct start/reenable rejects. Preserve this readback; do not falsely report no reload flag.'}
(root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
