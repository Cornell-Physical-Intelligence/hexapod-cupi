from pathlib import Path
import runpy,hashlib,json,time,subprocess
p=Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910/standing32_guard_005_reservation_probe002/launch_guarded_remote.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='e80490c0f46aed03610e16d5140fc80a2376536409ced4715a1191b3267b5d10'
g=runpy.run_path(str(p),run_name='handoff_readback_only');reservation=g['verify_reservation']()
def call(cmd):
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
 if r.returncode:raise RuntimeError(str(cmd)+': '+r.stderr)
 return r.stdout
units=json.loads(call(['systemctl','--user','list-units','--all','--output=json','--no-pager']))
hexunits=[x for x in units if 'hexapod' in x['unit'].lower()]
timers=json.loads(call(['systemctl','--user','list-timers','--all','--output=json','--no-pager']))
hextimers=[x for x in timers if 'hexapod' in json.dumps(x).lower()]
containers=call(['docker','ps','--format','{{.ID}} {{.Names}}']).strip().splitlines()
gpu=call(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader']).strip().splitlines()
active=[x for x in hexunits if x.get('active') in ['active','activating','reloading'] and x['unit']!='hexapod-exclusive-reconstruction-queue.service']
print(json.dumps({'observed_unix':time.time(),'reservation':reservation,'hexapod_units':hexunits,'hexapod_timers':hextimers,'active_research_units':active,'running_containers':containers,'gpu_compute_processes':gpu,'readonly':True},indent=2))
