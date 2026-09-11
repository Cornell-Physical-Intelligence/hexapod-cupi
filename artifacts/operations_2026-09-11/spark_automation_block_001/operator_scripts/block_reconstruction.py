from pathlib import Path
import hashlib,json,subprocess,time,signal,os
base=Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910')
marker=base/'exclusive_reservation_001/ACTIVE'
assert hashlib.sha256(marker.read_bytes()).hexdigest()=='d6135e574b033fbbbe716e7f6b34876a42f1897f560d0c97ab3ae4da33532dc2'
worker=Path('/home/orionh/ithaca-reconstruction/queue-worker.py')
original=worker.read_bytes();assert hashlib.sha256(original).hexdigest()=='0b5ae6e035d4f355017cf170955ce6de13a99ae98dd4ad5b8c767a2bd870ad78'
pid=3803709;expected_start='140218694';expected_cmd=b'/home/orionh/ithaca-reconstruction/env/bin/python\0-u\0/home/orionh/ithaca-reconstruction/queue-worker.py\0--queue\0/home/orionh/ithaca-reconstruction/queue\0'
proc=Path('/proc')/str(pid)
fd=os.pidfd_open(pid)
def verify_target():
 s=(proc/'stat').read_text();tail=s[s.rfind(')')+2:].split()
 assert tail[19]==expected_start and (proc/'cmdline').read_bytes()==expected_cmd
 assert (proc/'status').read_text().split('Uid:\t',1)[1].splitlines()[0].split()==['1000']*4
verify_target()
root=base/'exclusive_reconstruction_001';root.mkdir(exist_ok=False)
unit=Path('/home/orionh/.config/systemd/user/hexapod-exclusive-reconstruction-queue.service');assert not unit.exists() and not unit.is_symlink()
(root/'queue-worker.original.py').write_bytes(original)
queue=Path('/home/orionh/ithaca-reconstruction/queue')
before={'observed_unix':time.time(),'worker_pid':pid,'worker_starttime':expected_start,'worker_source_sha256':hashlib.sha256(original).hexdigest(),'queue_pending':[{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size':p.stat().st_size}for p in queue.glob('*.request.json')],'authority':'User explicitly ordered all automated Spark jobs stopped and HEXAPOD-only priority; preserve queue and prior source for release.'}
(root/'before.json').write_text(json.dumps(before,indent=2)+'\n')
first,rest=original.split(b'\n',1)
gate=('\n# HEXAPOD exclusive reservation: restore original source only on user release.\nfrom pathlib import Path as _hexapod_reservation_path\nif _hexapod_reservation_path('+repr(str(marker))+').exists():\n    raise SystemExit("Spark user compute is reserved for HEXAPOD; reconstruction queue deferred.")\n\n').encode()
patched=first+b'\n'+gate+rest
compile(patched,str(worker),'exec')
helper='''from pathlib import Path
import fcntl,hashlib,json,os,time
root=Path(__file__).resolve().parent
marker=Path(%r)
lock=Path('/home/orionh/ithaca-reconstruction/queue-worker.lock')
fd=os.open(lock,os.O_RDWR|os.O_CREAT,0o600)
fcntl.flock(fd,fcntl.LOCK_EX)
(root/'lock_acquired.json').write_text(json.dumps({'pid':os.getpid(),'acquired_unix':time.time(),'lock_path':str(lock),'reservation_path':str(marker),'reservation_sha256':hashlib.sha256(marker.read_bytes()).hexdigest(),'queue_mutated':False},indent=2)+'\\n')
while marker.exists():time.sleep(5)
os.close(fd)
'''%str(marker)
(root/'hold_queue_lock.py').write_text(helper)
unittext='''[Unit]
Description=Reserve reconstruction queue for HEXAPOD exclusive Spark use
ConditionPathExists=%s

[Service]
Type=simple
ExecStart=/usr/bin/python3 -B %s
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
'''%(marker,root/'hold_queue_lock.py')
(root/'installed.service').write_text(unittext)
unit.write_text(unittext)
assert worker.read_bytes()==original
worker.write_bytes(patched)
subprocess.run(['systemctl','--user','daemon-reload'],check=True,timeout=30,capture_output=True)
subprocess.run(['systemctl','--user','enable','--now',unit.name],check=True,timeout=30,capture_output=True)
verify_target()
signal.pidfd_send_signal(fd,signal.SIGTERM,None,0);os.close(fd)
end=time.monotonic()+25
while time.monotonic()<end:
 if (root/'lock_acquired.json').exists() and not proc.exists():break
 time.sleep(.5)
assert (root/'lock_acquired.json').exists() and not proc.exists(),'Exact worker exit/queue lock acquisition not yet verified'
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True,timeout=20)
assert not gpu.strip(),'Another CUDA process remains'
state=dict(line.split('=',1)for line in subprocess.check_output(['systemctl','--user','show',unit.name,'-p','ActiveState','-p','SubState','-p','MainPID','-p','UnitFileState','-p','NeedDaemonReload','-p','FragmentPath'],text=True).splitlines())
assert state['ActiveState']=='active' and state['SubState']=='running' and state['UnitFileState']=='enabled' and state['NeedDaemonReload']=='no'
receipt={'verified':True,'observed_unix':time.time(),'exact_worker_exited':True,'signal':'SIGTERM via bound pidfd','worker_source_original_sha256':hashlib.sha256(original).hexdigest(),'worker_source_blocked_sha256':hashlib.sha256(worker.read_bytes()).hexdigest(),'worker_source_backup':str(root/'queue-worker.original.py'),'queue_preserved':True,'queue_pending_after':len(list(queue.glob('*.request.json'))),'cuda_processes':gpu,'unit':state,'queue_lock':json.loads((root/'lock_acquired.json').read_text()),'persistent_reservation_unchanged':hashlib.sha256(marker.read_bytes()).hexdigest(),'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [worker,unit,root/'hold_queue_lock.py',root/'installed.service',root/'before.json']}}
(root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
