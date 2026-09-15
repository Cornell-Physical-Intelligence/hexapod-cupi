"""Reestablish bounded, reversible compute controls under 14 September consent."""
from pathlib import Path
import datetime, fcntl, hashlib, json, os, subprocess, time
R=Path('/home/orionh/HEXAPOD_runs/restart_20260914/spark_ownership_001')
OLD=Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910')
MARKER=R/'ACTIVE'
ENTRY=Path('/home/orionh/ithaca-reconstruction/queue-worker.py')
UNIT=Path('/home/orionh/.config/systemd/user/hexapod-exclusive-reconstruction-queue.service')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(args,check=True):
 p=subprocess.run(args,text=True,capture_output=True,timeout=45)
 if check and p.returncode:raise RuntimeError(f'{args[0]}: {p.stderr[:500]}')
 return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
def show(unit,user=True):
 args=['systemctl']+(['--user'] if user else [])+['show',unit]
 fields=['Id','LoadState','ActiveState','SubState','MainPID','UnitFileState','FragmentPath','DropInPaths','NeedDaemonReload']
 d=run(args+['--property='+v for v in fields]);return dict(s.split('=',1) for s in d['stdout'].splitlines() if '=' in s)
def save(name,obj):
 with (R/name).open('x') as f:json.dump(obj,f,indent=2);f.write('\n')
def private_copy(p,name):
 dest=R/name
 with dest.open('xb') as f:f.write(p.read_bytes())
 os.chmod(dest,p.stat().st_mode&0o777)
 return {'original':str(p),'backup':str(dest),'sha256':sha(dest),'mode':oct(p.stat().st_mode&0o777)}
assert (R/'receipt.json').exists() and not MARKER.exists()
assert ENTRY.is_file() and not ENTRY.is_symlink()
assert UNIT.is_file() and not UNIT.is_symlink()
assert show(UNIT.name)['ActiveState']=='inactive'
assert not run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'])['stdout'].strip()
assert not run(['docker','ps','-q'])['stdout'].strip()
# Refuse to alter a launcher that still has a live process.
entry_processes=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdecimal():continue
 try:args=(proc/'cmdline').read_bytes().split(b'\0')
 except (OSError,PermissionError):continue
 if str(ENTRY).encode() in args:entry_processes.append(int(proc.name))
assert not entry_processes, entry_processes
lock=Path('/home/orionh/ithaca-reconstruction/queue-worker.lock')
fd=os.open(lock,os.O_RDWR|os.O_CREAT,0o600)
fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
baseunits=[]
for prefix,kinds in [('stormscope',('dispatch','scout','monitor','publish','verify')),('stormscope-private',('dispatch','scout','monitor','prepare','verify')),('stormscope-private-v2',('dispatch','scout','monitor','publish','verify'))]:
 for kind in kinds:
  for suffix in ('timer','service'):baseunits.append(f'{prefix}-{kind}.{suffix}')
baseunits+=['ollama.service','dsv41-inference.service']
maskpaths=[Path('/home/orionh/.config/systemd/user')/n for n in baseunits]
maskstates={}
for path in maskpaths:
 s=show(path.name);maskstates[path.name]=s
 assert path.is_symlink() and os.readlink(path)=='/dev/null',str(path)
 assert s['LoadState']==s['UnitFileState']=='masked' and s['ActiveState']=='inactive',s
 if path.name.endswith('.service'):assert s['MainPID']=='0',s
systemstates={}
for unit in ('wx-forecast.timer','wx-forecast.service','wx-forecast-pm.timer','wx-forecast-pm.service'):
 s=show(unit,False);systemstates[unit]=s
 assert s['LoadState']==s['UnitFileState']=='masked' and s['ActiveState']=='inactive',s
backup_entry=private_copy(ENTRY,'queue-worker.before.py')
backup_unit=private_copy(UNIT,'queue-holder.service.before')
before={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'entry':backup_entry,'queue_unit':backup_unit,'old_marker_absent':not (OLD/'exclusive_reservation_001/ACTIVE').exists(),'mask_states':maskstates,'system_mask_states':systemstates,'entry_processes':entry_processes,'queue_unlocked_before':True}
save('reservation_before.json',before)
marker={'exclusive':True,'authorization_date':'2026-09-14','project':'HEXAPOD canonical mass-corrected restart','dispatcher':'root current Codex task','release_only_on_user_instruction':True,'prior_release_preserved':str(OLD/'exclusive_reservation_001/ACTIVE.released-20260911')}
save('ACTIVE',marker)
blocker="raise SystemExit('HEXAPOD exclusive Spark reservation: reconstruction deferred until explicit user release; original launcher preserved in restart_20260914/spark_ownership_001.')\n"
assert sha(ENTRY)==backup_entry['sha256']
ENTRY.rename(R/'queue-worker.removed.py')
ENTRY.mkdir()
(ENTRY/'__main__.py').write_text(blocker)
(ENTRY/'HEXAPOD_RESERVATION.txt').write_text('User-authorized 2026-09-14 exclusive HEXAPOD reservation. Restore only after explicit release and exact backup/mask verification. No queue/output deletion. Receipt: '+str(R)+'\n')
helper='''from pathlib import Path
import datetime,fcntl,hashlib,json,os,time
root=Path(__file__).resolve().parent
marker=root/'ACTIVE'
lock=Path('/home/orionh/ithaca-reconstruction/queue-worker.lock')
fd=os.open(lock,os.O_RDWR|os.O_CREAT,0o600)
fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
record={'pid':os.getpid(),'acquired_unix':time.time(),'lock_path':str(lock),'reservation_path':str(marker),'reservation_sha256':hashlib.sha256(marker.read_bytes()).hexdigest(),'queue_mutated':False}
(root/'acquisitions').mkdir(exist_ok=True)
with (root/'acquisitions'/f'{time.time_ns()}-{os.getpid()}.json').open('x') as f:json.dump(record,f,indent=2)
tmp=root/'lock_acquired.json.tmp';tmp.write_text(json.dumps(record,indent=2)+'\\n');tmp.replace(root/'lock_acquired.json')
while marker.exists():time.sleep(5)
os.close(fd)
'''
(R/'hold_queue_lock.py').write_text(helper)
unittext='[Unit]\nDescription=Reserve reconstruction queue for HEXAPOD exclusive Spark use\nConditionPathExists='+str(MARKER)+'\n\n[Service]\nType=simple\nExecStart=/usr/bin/python3 -B '+str(R/'hold_queue_lock.py')+'\nRestart=on-failure\nRestartSec=2\n\n[Install]\nWantedBy=default.target\n'
assert sha(UNIT)==backup_unit['sha256']
UNIT.write_text(unittext)
run(['systemctl','--user','daemon-reload'])
fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
run(['systemctl','--user','enable','--now',UNIT.name])
for attempt in range(10):
 state=show(UNIT.name)
 if state['ActiveState']=='active' and (R/'lock_acquired.json').exists():break
 time.sleep(.2)
assert state['ActiveState']=='active' and state['SubState']=='running' and state['UnitFileState']=='enabled',state
lock_record=json.loads((R/'lock_acquired.json').read_text());pid=int(state['MainPID'])
assert lock_record['pid']==pid and lock_record['reservation_sha256']==sha(MARKER)
assert (Path('/proc')/str(pid)/'cmdline').read_bytes()==('/usr/bin/python3\0-B\0'+str(R/'hold_queue_lock.py')+'\0').encode()
st=lock.stat();inode=f'{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}'
owned=[line for line in Path('/proc/locks').read_text().splitlines() if f' {pid} {inode} ' in line]
assert len(owned)==1 and 'FLOCK  ADVISORY  WRITE' in owned[0],owned
probe=run(['/usr/bin/python3','-I','-S','-B',str(ENTRY),'--queue','/home/orionh/ithaca-reconstruction/queue'],False)
assert probe['returncode']==1 and 'HEXAPOD exclusive Spark reservation' in probe['stderr']
coord=Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md')
header='# Exclusive HEXAPOD Spark ownership renewed — 14 September 2026 UTC\n\nThe user explicitly confirmed the mass-corrected robot, authorized restarting standing → walking/stopping → terrain → survey, and said “you can stop all other processes on the spark” and “take full ownership of the spark, you have my permissions”. Root is the sole dispatcher. Identified competing user compute is deferred across allocations; preserve outputs and exact restart state. SSH, OS services, networking and host health remain available.\n\nCurrent reservation: '+str(R)+'. DeepSeek service is stopped and persistently masked; reconstruction entry and queue are blocked; existing forecast/Ollama masks remain. Only a later explicit user release permits restoring competing compute. Hold both GPU locks for each allocation, verify live resources and controls before and during launch, and clean only exact owned containers. Authorization does not admit physics or complete a stage. Historical release/pause receipts below remain unchanged.\n\n---\n\n'
original=coord.read_bytes();assert original==(R/'SPARK_COMPUTE_COORDINATION.before.md').read_bytes()
coord.write_bytes(header.encode()+original)
filepaths=[MARKER,UNIT,R/'hold_queue_lock.py',ENTRY/'__main__.py',ENTRY/'HEXAPOD_RESERVATION.txt',R/'dsv41-inference.service.original',R/'queue-worker.before.py',R/'queue-holder.service.before']
policy={'schema':'canonical_exclusive_mask_reservation_v2','release_only_on_user_instruction':True,'marker_sha256':sha(MARKER),'blocked_entry_directories':[str(ENTRY)],'files':{str(p):sha(p) for p in filepaths},'mask_paths':[str(p) for p in maskpaths],'system_masked_units':list(systemstates),'original_units_before':str(OLD/'exclusive_automation_block_001/before.json'),'original_units_backup':str(OLD/'exclusive_automation_block_001/original_units'),'queue_unit':UNIT.name,'queue_unit_path':str(UNIT),'queue_helper':str(R/'hold_queue_lock.py'),'queue_lock_path':str(lock),'queue_lock_record':str(R/'lock_acquired.json')}
save('reservation_policy.json',policy)
receipt={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reservation_path':str(R),'marker_sha256':sha(MARKER),'coordination_sha256':sha(coord),'policy_sha256':sha(R/'reservation_policy.json'),'masked_user_units':len(maskpaths),'system_masked_units':len(systemstates),'queue_unit':state,'queue_lock_record':lock_record,'proc_locks':owned,'entry_probe':probe,'backups':[backup_entry,backup_unit],'queue_outputs_mutated':False,'gpu':run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader']),'native_run_started':False}
save('reservation_receipt.json',receipt)
print(json.dumps(receipt,indent=2))
