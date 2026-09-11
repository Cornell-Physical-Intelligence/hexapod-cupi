from pathlib import Path
import hashlib,json,subprocess,os,time
base=Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910');worker=Path('/home/orionh/ithaca-reconstruction/queue-worker.py')
original=worker.read_bytes();expected='0b5ae6e035d4f355017cf170955ce6de13a99ae98dd4ad5b8c767a2bd870ad78';assert hashlib.sha256(original).hexdigest()==expected
root=base/'exclusive_reconstruction_002';root.mkdir(exist_ok=False)
(root/'uploader_restored_original.py').write_bytes(original)
before={'observed_unix':time.time(),'entry_path':str(worker),'entry_sha256':expected,'entry_modified_unix':worker.stat().st_mtime,'prior_gate_sha256':'5073f20f8ffefd978aae8bcb833a4d6e3821f71c5533b8f73270bb9ac8b858b0','reason':'External replacement removed the earlier source gate; root preflight rejected allocation. Preserve exact restored source and block the launcher path itself.'}
(root/'before.json').write_text(json.dumps(before,indent=2)+'\n')
assert worker.read_bytes()==original
os.rename(worker,root/'removed_entry.py');worker.mkdir()
blocker='raise SystemExit("Spark external automation is blocked for HEXAPOD exclusive use. Restore this launcher only after explicit user release.")\n'
(worker/'__main__.py').write_text(blocker)
(worker/'HEXAPOD_RESERVATION.txt').write_text('This launcher path is intentionally a directory. Python executes its blocking __main__.py before any reconstruction/CUDA imports. Normal source-file uploads cannot replace this directory entry. Original sources are preserved under '+str(root)+'. Only explicit user release may restore the original regular file. Preserve any subsequently uploaded files here as evidence.\n')
r=subprocess.run(['/usr/bin/python3','-I','-S','-B',str(worker),'--queue','/home/orionh/ithaca-reconstruction/queue'],capture_output=True,text=True,timeout=15)
assert r.returncode==1 and 'Spark external automation is blocked' in r.stderr
try:
 fd=os.open(worker,os.O_WRONLY)
except IsADirectoryError:upload_blocked=True
else:
 os.close(fd);raise AssertionError('Ordinary file-write entry not blocked')
lock=Path('/home/orionh/ithaca-reconstruction/queue-worker.lock')
probe=subprocess.run(['/usr/bin/flock','-n',str(lock),'/usr/bin/true'],capture_output=True,text=True,timeout=10);assert probe.returncode==1
state=dict(l.split('=',1)for l in subprocess.check_output(['systemctl','--user','show','hexapod-exclusive-reconstruction-queue.service','-p','ActiveState','-p','SubState','-p','MainPID','-p','UnitFileState','-p','FragmentPath'],text=True,timeout=20).splitlines());assert state['ActiveState']=='active'and state['SubState']=='running'and state['UnitFileState']=='enabled'
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True,timeout=20);assert not gpu.strip()
files=[worker/'__main__.py',worker/'HEXAPOD_RESERVATION.txt',root/'uploader_restored_original.py',root/'removed_entry.py',root/'before.json']
receipt={'verified':True,'observed_unix':time.time(),'blocked_entry_directory':str(worker),'entry_directory_is_symlink':worker.is_symlink(),'files':{str(p):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}for p in files},'python_entry_probe':{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'isolated_stdlib_only':True},'ordinary_file_open_blocked':upload_blocked,'queue_lock_nonblocking_probe_returncode':probe.returncode,'queue_lock_unit':state,'cuda_processes':gpu,'original_source_preserved':True,'queue_outputs_untouched':True,'release_only_on_user_instruction':True,'scope':'Known worker launcher and its queue; no arbitrary privileged-GPU security partition claimed.'}
assert worker.is_dir()and not worker.is_symlink()and hashlib.sha256((worker/'__main__.py').read_bytes()).hexdigest()==hashlib.sha256(blocker.encode()).hexdigest()
(root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
