"""Fetch only the immutable terminal inventory; remote cat only, no remote writes."""
import argparse,hashlib,json,shlex,shutil,subprocess,time
from pathlib import Path,PurePosixPath
B=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
REMOTE={'run':B/'direct_omni_train_smoke_004','pause':B/'forecast_pause_061'}
RESERVE=512*1024**2

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 audit=json.loads(a.audit.read_text());assert audit['passed'] is True and not audit['errors']
 a.output.mkdir(parents=True,exist_ok=True);report={'audit_sha256':digest(a.audit),'started_unix':time.time(),'free_bytes_before':shutil.disk_usage(a.output).free,'files':{},'local_omissions':[],'remote_writes':False}
 for name,row in audit['inventory'].items():
  parts=PurePosixPath(name).parts
  assert len(parts)>=2 and parts[0] in REMOTE and '..' not in parts and not PurePosixPath(name).is_absolute()
  target=a.output.joinpath(*parts);target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():assert target.is_file() and target.stat().st_size==row['bytes'] and digest(target)==row['sha256']
  else:
   assert shutil.disk_usage(a.output).free-row['bytes']>=RESERVE,'Must retain512MiB free'
   part=target.with_name(target.name+'.part');assert not part.exists()
   remote=REMOTE[parts[0]].joinpath(*parts[1:])
   try:
    with part.open('xb') as out:
     q=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','spark',shlex.join(['cat',str(remote)])],stdout=out,stderr=subprocess.PIPE,timeout=180)
    if q.returncode:raise RuntimeError(q.stderr.decode())
    assert part.stat().st_size==row['bytes'] and digest(part)==row['sha256'],'Fetched bytes mismatch: '+name
    part.rename(target)
   except BaseException:
    if part.exists():part.unlink()
    raise
  report['files'][name]=row
 report.update(complete=True,completed_unix=time.time(),total_bytes=sum(r['bytes'] for r in report['files'].values()),free_bytes_after=shutil.disk_usage(a.output).free)
 (a.output/'FETCH_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:report[k] for k in ('complete','total_bytes','free_bytes_after')}))
if __name__=='__main__':main()
