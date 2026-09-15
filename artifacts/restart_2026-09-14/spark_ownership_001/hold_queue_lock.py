from pathlib import Path
import datetime,fcntl,hashlib,json,os,time
root=Path(__file__).resolve().parent
marker=root/'ACTIVE'
lock=Path('/home/orionh/ithaca-reconstruction/queue-worker.lock')
fd=os.open(lock,os.O_RDWR|os.O_CREAT,0o600)
fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
record={'pid':os.getpid(),'acquired_unix':time.time(),'lock_path':str(lock),'reservation_path':str(marker),'reservation_sha256':hashlib.sha256(marker.read_bytes()).hexdigest(),'queue_mutated':False}
(root/'acquisitions').mkdir(exist_ok=True)
with (root/'acquisitions'/f'{time.time_ns()}-{os.getpid()}.json').open('x') as f:json.dump(record,f,indent=2)
tmp=root/'lock_acquired.json.tmp';tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(root/'lock_acquired.json')
while marker.exists():time.sleep(5)
os.close(fd)
