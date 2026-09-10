"""Fetch only the terminal payload paths already inventoried by read-only audit."""
from pathlib import Path
import concurrent.futures,hashlib,json,subprocess
root=Path(__file__).resolve().parent;a=json.loads((root/'remote_audit.json').read_text());base='/home/orionh/HEXAPOD_runs/mock_length_study_20260909/'
def source(name):
 if name.startswith('run/'):return base+'reference_residual_ppo_002/'+name[4:]
 if name.startswith('forecast_pause/'):return base+'forecast_pause_043/'+name[len('forecast_pause/'):]
 if name=='directional_preflight002.json':return base+'reference_directional_preflight_002.json'
 assert name=='preflight002.json';return base+'reference_residual_ppo_preflight_002.json'
def fetch(item):
 name,h=item;p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==h;return
 subprocess.run(['scp','-q','-o','BatchMode=yes','spark:'+source(name),str(p)],check=True,timeout=300)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h,name
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,a['raw_payloads'].items()))
print('Verified',len(a['raw_payloads']),'remote raw payloads')
