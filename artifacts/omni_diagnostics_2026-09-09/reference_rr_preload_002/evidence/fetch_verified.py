"""Read-only fetch of paths inventoried only after actual terminal/restoration."""
from pathlib import Path
import concurrent.futures,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parent;BASE='/home/orionh/HEXAPOD_runs/mock_length_study_20260909/'
a=json.loads((ROOT/'remote_audit.json').read_text())
def remote(name):
 if name.startswith('run/'):return BASE+'reference_rr_preload_002/'+name[4:]
 if name.startswith('forecast_pause/'):return BASE+'forecast_pause_048/'+name[len('forecast_pause/'):]
 if name=='remote_preflight.json':return BASE+'reference_rr_preload_preflight_002.json'
 raise ValueError(name)
def fetch(item):
 name,h=item;p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==h;return
 subprocess.run(['scp','-q','-o','BatchMode=yes','spark:'+remote(name),str(p)],check=True,timeout=300)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h,name
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,a['raw_payloads'].items()))
print('Verified',len(a['raw_payloads']),'remote raw payloads')
