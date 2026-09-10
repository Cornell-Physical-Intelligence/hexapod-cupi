"""Acquire only the root-approved inventory selection; SSH performs reads only."""
from pathlib import Path
import hashlib,json,shutil,subprocess,tarfile,time
H=Path(__file__).resolve().parent
plan=json.loads((H/'ACQUISITION_PLAN.json').read_text())
selected={r['path']:r for r in plan['inventory_rows'] if r['disposition']=='local_required'}
reserve=256*1024*1024
if shutil.disk_usage(H).free<sum(r['bytes'] for r in selected.values())+reserve:raise RuntimeError('Current free-space budget changed; no fetch')
remote='''from pathlib import Path
import sys,tarfile
B=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
roots={'run':B/'direct_omni_train_extended_caps_002','pause':B/'forecast_pause_063'}
rows=REPLACE_ROWS
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 for name,size in rows:
  prefix,rel=name.split('/',1);p=roots[prefix]/rel
  if p.is_symlink() or not p.is_file() or p.stat().st_size!=size:raise RuntimeError('Selected remote file changed: '+name)
  tar.add(p,arcname=name,recursive=False)
'''.replace('REPLACE_ROWS',repr([(k,v['bytes']) for k,v in selected.items()]))
proc=subprocess.Popen(['ssh','spark','python3','-B','-'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
proc.stdin.write(remote.encode());proc.stdin.close()
verified={};started=time.time()
try:
 with tarfile.open(fileobj=proc.stdout,mode='r|') as tar:
  for member in tar:
   name=member.name
   if name not in selected or name in verified or not member.isfile() or member.size!=selected[name]['bytes']:raise RuntimeError('Unexpected streamed entry '+name)
   target=H/'raw'/name
   if target.exists():raise RuntimeError('Refusing to overwrite selected raw')
   if shutil.disk_usage(H).free<member.size+reserve:raise RuntimeError('Concurrent disk use exhausted reserve')
   target.parent.mkdir(parents=True,exist_ok=True);partial=target.with_name(target.name+'.part')
   h=hashlib.sha256();count=0
   with tar.extractfile(member) as src,partial.open('xb') as dst:
    for block in iter(lambda:src.read(4*1024*1024),b''):dst.write(block);h.update(block);count+=len(block)
   if h.hexdigest()!=selected[name]['sha256'] or count!=selected[name]['bytes']:raise RuntimeError('Fetched hash/size mismatch '+name)
   partial.replace(target);verified[name]={'sha256':h.hexdigest(),'bytes':count}
   (H/'FETCH_PROGRESS.json').write_text(json.dumps({'verified_files':len(verified),'verified_bytes':sum(x['bytes'] for x in verified.values()),'last':name},indent=2)+'\n')
 stderr=proc.stderr.read().decode(errors='replace');code=proc.wait();(H/'fetch.stderr.txt').write_text(stderr)
 if code!=0 or set(verified)!=set(selected):raise RuntimeError('Remote stream incomplete '+str(code))
 result={'schema':'extended_curated_fetch_v1','started_unix':started,'finished_unix':time.time(),'verified':True,'remote_reads_only':True,'files':len(verified),'bytes':sum(x['bytes'] for x in verified.values()),'payloads':verified,'remaining_free_bytes':shutil.disk_usage(H).free,'required_reserve_bytes':reserve,'full_remote_inventory_files':len(plan['inventory_rows']),'local_training_replay_verified':False,'omitted_training_trace_blocks_full_local_analysis':True}
 (H/'FETCH_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='payloads'},indent=2))
finally:
 if proc.poll() is None:proc.terminate();proc.wait()
