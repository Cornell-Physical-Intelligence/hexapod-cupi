from pathlib import Path
import subprocess,json,hashlib,os,time
base=Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910');marker=base/'exclusive_reservation_001/ACTIVE'
assert hashlib.sha256(marker.read_bytes()).hexdigest()=='d6135e574b033fbbbe716e7f6b34876a42f1897f560d0c97ab3ae4da33532dc2'
unit_names=[]
for prefix,kinds in [('stormscope-',('dispatch','scout','monitor','publish','verify')),('stormscope-private-',('dispatch','scout','monitor','prepare','verify')),('stormscope-private-v2-',('dispatch','scout','monitor','publish','verify'))]:
 unit_names.extend(prefix+k+'.'+suffix for k in kinds for suffix in ('timer','service'))
unit_names.append('ollama.service');assert len(set(unit_names))==31
props=['LoadState','ActiveState','SubState','MainPID','UnitFileState','FragmentPath','NeedDaemonReload','DropInPaths']
def state(name):
 cmd=['systemctl','--user','show',name]
 for p in props:cmd+=['-p',p]
 return dict(l.split('=',1)for l in subprocess.check_output(cmd,text=True,timeout=20).splitlines())
root=base/'exclusive_automation_block_001';root.mkdir(exist_ok=False)
backup=root/'original_units';backup.mkdir()
config=Path('/home/orionh/.config/systemd/user')
before={n:state(n)for n in unit_names};assert all(v['LoadState']!='not-found' for v in before.values())
meta={}
for n in unit_names:
 p=config/n
 if p.is_symlink():meta[n]={'kind':'symlink','target':os.readlink(p),'sha256_of_target_bytes':hashlib.sha256(p.read_bytes()).hexdigest()}
 elif p.exists():meta[n]={'kind':'file','mode':oct(p.stat().st_mode & 0o777),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 else:meta[n]={'kind':'absent_override','fragment_path':before[n]['FragmentPath']}
(root/'before.json').write_text(json.dumps({'observed_unix':time.time(),'units':before,'files':meta,'authority':'User explicitly requested complete shutdown of automated Spark jobs, including external triggers, with HEXAPOD-only priority. Original units and enable states retained for user-authorized release.'},indent=2)+'\n')
subprocess.run(['systemctl','--user','stop',*unit_names],check=True,timeout=90,capture_output=True)
subprocess.run(['systemctl','--user','disable',*unit_names],check=True,timeout=30,capture_output=True)
for n in unit_names:
 p=config/n
 if p.is_symlink() and os.readlink(p)=='/dev/null':continue
 if p.exists() or p.is_symlink():os.replace(p,backup/n)
 os.symlink('/dev/null',p)
subprocess.run(['systemctl','--user','daemon-reload'],check=True,timeout=30,capture_output=True)
subprocess.run(['systemctl','--user','reset-failed',*unit_names],timeout=30,capture_output=True)
after={n:state(n)for n in unit_names}
assert all(v['LoadState']=='masked' and v['UnitFileState']=='masked' and v['ActiveState']=='inactive' and v['MainPID']=='0' and v['NeedDaemonReload']=='no' for v in after.values()),after
assert all((config/n).is_symlink() and os.readlink(config/n)=='/dev/null'for n in unit_names)
probes=[]
for n in ('stormscope-dispatch.timer','stormscope-private-scout.service','stormscope-private-v2-dispatch.timer','ollama.service'):
 r=subprocess.run(['systemctl','--user','start',n],capture_output=True,text=True,timeout=20)
 assert r.returncode!=0 and 'masked' in r.stderr.lower(),(n,r.returncode,r.stderr)
 probes.append({'unit':n,'returncode':r.returncode,'stderr':r.stderr})
r=subprocess.run(['systemctl','--user','enable','--now','stormscope-dispatch.timer'],capture_output=True,text=True,timeout=20)
assert r.returncode!=0 and 'masked' in r.stderr.lower()
probes.append({'command':'enable --now stormscope-dispatch.timer','returncode':r.returncode,'stderr':r.stderr})
system={n:dict(l.split('=',1)for l in subprocess.check_output(['systemctl','show',n,'-p','LoadState','-p','ActiveState','-p','UnitFileState','-p','MainPID'],text=True,timeout=20).splitlines())for n in ('wx-forecast.timer','wx-forecast.service','wx-forecast-pm.timer','wx-forecast-pm.service')}
assert all(v['LoadState']=='masked' and v['ActiveState']=='inactive'and v['MainPID']=='0'for v in system.values())
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True,timeout=20);assert not gpu.strip()
receipt={'verified':True,'observed_unix':time.time(),'user_unit_count':31,'user_units':after,'system_units_already_masked':system,'direct_start_and_reenable_probes':probes,'mask_paths':[str(config/n)for n in unit_names],'backup_root':str(backup),'before_sha256':hashlib.sha256((root/'before.json').read_bytes()).hexdigest(),'cuda_processes':gpu,'reservation_marker_unchanged_sha256':hashlib.sha256(marker.read_bytes()).hexdigest(),'release_only_on_user_instruction':True,'own_HEXAPOD_units_untouched':True}
(root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
