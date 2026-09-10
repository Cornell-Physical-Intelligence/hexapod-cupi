"""Fetch terminal-inventoried bytes only; never rewrite an existing wrong payload."""
from pathlib import Path
import concurrent.futures,hashlib,json,subprocess
H=Path(__file__).resolve().parent;B='/home/orionh/HEXAPOD_runs/mock_length_study_20260909/'
a=json.loads((H/'remote_audit.json').read_text())
def remote(f):
 if f.startswith('run/'):return B+'reference_moving_ppo_001/'+f[4:]
 if f.startswith('forecast_pause/'):return B+'forecast_pause_049/'+f[len('forecast_pause/'):]
 if f.startswith('preflight/'):return B+f[len('preflight/'):]
 raise ValueError(f)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
 return h.hexdigest()
def fetch(item):
 f,h=item;p=H/f;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert sha(p)==h;return
 subprocess.run(['scp','-q','-o','BatchMode=yes','spark:'+remote(f),str(p)],check=True,timeout=600)
 assert sha(p)==h,f
 print('verified',f,flush=True)
items=sorted(a['raw_payloads'].items(),key=lambda x:(0 if '/profile_32/' in x[0] else 1,a['raw_sizes_bytes'][x[0]]))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,items))
print('ALL',len(items),'raw payloads verified')
