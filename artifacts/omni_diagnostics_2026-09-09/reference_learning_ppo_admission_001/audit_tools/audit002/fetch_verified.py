"""Fetch only terminal-inventoried bytes; immutable final paths after verification."""
from pathlib import Path,PurePosixPath
import argparse,concurrent.futures,hashlib,json,subprocess
BASE='/home/orionh/HEXAPOD_runs/mock_length_study_20260909/'
def remote(name):
 p=PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts or str(p)!=name:raise ValueError('Unsafe path')
 for prefix,folder in [('run/','reference_learning_ppo_001/'),('forecast_pause/','forecast_pause_051/'),('preflight/','')]:
  if name.startswith(prefix):return BASE+folder+name[len(prefix):]
 raise ValueError('Unknown prefix')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 home=args.output;record=json.loads((home/'remote_audit.json').read_text())
 def fetch(item):
  name,h=item;dst=home/name;dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists():
   if sha(dst)!=h:raise ValueError('Existing final payload differs: '+name)
   return
  part=dst.with_name(dst.name+'.fetch-part')
  if part.exists():raise FileExistsError('Review incomplete earlier transfer: '+str(part))
  subprocess.run(['scp','-q','-o','BatchMode=yes','spark:'+remote(name),str(part)],check=True,timeout=900)
  if part.stat().st_size!=record['raw_sizes_bytes'][name] or sha(part)!=h:raise ValueError('Fetched hash/size differs: '+name)
  part.replace(dst);print('verified',name,flush=True)
 items=sorted(record['raw_payloads'].items(),key=lambda kv:(0 if '/learning_recovery_32/' in kv[0] else 1,record['raw_sizes_bytes'][kv[0]]))
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(fetch,items))
 result={'all_remote_payloads_match':True,'raw_payloads':len(items),'remote_audit_sha256':sha(home/'remote_audit.json')}
 out=home/'local_raw_verification.json'
 if out.exists():raise FileExistsError(out)
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
