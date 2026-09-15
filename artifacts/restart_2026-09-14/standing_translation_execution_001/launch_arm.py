"""Dispatch one fresh reviewed arm under bounded systemd ownership."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys
root=Path('/home/orionh/HEXAPOD_runs/restart_20260914/standing_translation_001')
arm=sys.argv[1];assert arm in ('origin','xy14_4')
unit='hexapod-placement-'+arm.replace('_','-')+'-001-20260914.service'
binding=root/'bindings'/(arm+'_guard.json');output=root/(arm+'_001')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected={'origin':'5e80ac3e673224e499c69cbbe42f45511641b63771214ad4a440ee8c6faa3c79','xy14_4':'2b38e7bd42cc03fa2a8b0cc556245ac091190127ade72410037cd043ecba6682'}
assert sha(binding)==expected[arm] and not output.exists()
state=subprocess.check_output(['systemctl','--user','show',unit,'-p','LoadState','-p','ActiveState'],text=True)
assert 'LoadState=not-found' in state and 'ActiveState=inactive' in state
base=['/usr/bin/python3','-B',str(root/'guard/launch_guarded_diagnostic.py'),'--bindings',str(binding),'--bindings-sha256',expected[arm],'--guard-freeze-sha256','2f2c8ec8317414a6087c5196033f99b9e55981a873a5b6b9eeaee9a1c671e1ec','--placement',arm,'--output',str(output)]
# Paths/arguments are all fixed simple tokens; systemd ExecStopPost runs no shell.
assert all(' ' not in a and '\n' not in a for a in base)
cmd=['systemd-run','--user','--unit='+unit,'--property=RuntimeMaxSec=1320','--property=TimeoutStopSec=180','--property=KillMode=process','--property=Environment=PYTHONDONTWRITEBYTECODE=1','--property=Environment=PYTHONUNBUFFERED=1','--property=ExecStopPost='+' '.join(base+['--cleanup-only']),*base]
p=subprocess.run(cmd,text=True,capture_output=True,timeout=30)
receipt={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'unit':unit,'arm':arm,'output':str(output),'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'meaning':'Dispatch accepted is not proof of completed native initialization or standing qualification.'}
with (root/(arm+'_launch.json')).open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt,indent=2));sys.exit(p.returncode)
