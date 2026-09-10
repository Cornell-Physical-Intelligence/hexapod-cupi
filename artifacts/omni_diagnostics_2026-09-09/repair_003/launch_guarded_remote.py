"""User-authorized forecasting pause and bounded new repair campaign only."""
from pathlib import Path
import fcntl,json,os,subprocess,time
base=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
source=base/'omni_repair_source_003_a';pause=base/'forecast_pause_012';output=base/'omni_repair_003'
unit='hexapod-omni-repair-pair-003-20260909.service'
def call(args):return subprocess.check_output(args,text=True,timeout=30).strip()
if pause.exists() or output.exists():raise RuntimeError('Fresh pause/run paths required')
units={name:call(['systemctl','--user','show',name,'-p','ActiveState','-p','SubState','-p','MainPID']) for name in
       ('stormscope-dispatch.timer','stormscope-scout.timer','stormscope-dispatch.service','stormscope-scout.service')}
if any('ActiveState=active' in value or 'ActiveState=activating' in value for name,value in units.items() if name.endswith('.service')):
    raise RuntimeError('Forecasting service currently active; inspect exact job before any pause')
gpu=call(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'])
if gpu:raise RuntimeError('Unrelated CUDA workload: '+gpu)
locks=[]
try:
    for path in ('/opt/wx/gpu.lock','/tmp/hexapod-isaac-gpu.lock'):
        fd=os.open(path,os.O_RDONLY);locks.append(fd);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pause.mkdir()
    record={'user_authorized_pause':True,'reason':'Two bounded 50-update quiet-stand PPO comparisons; restore forecasting afterward',
            'created_unix':time.time(),'units':units,'gpu_before':gpu,'containers_before':call(['docker','ps','--format','{{.ID}} {{.Names}} {{.Image}}']),
            'forecast_processes':[],'unit':unit}
    (pause/'pause.json').write_text(json.dumps(record,indent=2)+'\n')
    active=[name for name,value in units.items() if name.endswith('.timer') and 'ActiveState=active' in value]
    restore='''from pathlib import Path
import subprocess,json,time
p=Path(__file__).parent
if not (p/'restored.json').exists():
    r=json.loads((p/'pause.json').read_text())
    names=[name for name,state in r['units'].items() if name.endswith('.timer') and 'ActiveState=active' in state]
    if names: subprocess.run(['systemctl','--user','start',*names],check=True)
    (p/'restored.json').write_text(json.dumps({'restored_unix':time.time(),'timers':names})+'\\n')
'''
    restorer=pause/'resume_forecasting.py';restorer.write_text(restore)
    # Arm independent recovery before the short timer pause, then normal service exit restores sooner.
    subprocess.run(['systemd-run','--user','--unit=hexapod-forecast-restore-012','--on-active=75m','/usr/bin/python3',str(restorer)],check=True)
    if active:subprocess.run(['systemctl','--user','stop',*active],check=True)
    # Reject a race where a dispatcher started before its timer stopped. Never kill it.
    if any('ActiveState=active' in call(['systemctl','--user','show',name,'-p','ActiveState']) for name in units if name.endswith('.service')):
        raise RuntimeError('Forecasting service started during pause preflight')
    record['paused_unix']=time.time();(pause/'pause.json').write_text(json.dumps(record,indent=2)+'\n')
finally:
    for fd in reversed(locks):os.close(fd)
try:
    cmd=['systemd-run','--user','--unit='+unit,'--property=RuntimeMaxSec=3600','--property=TimeoutStopSec=180',
         '--property=KillMode=process','--property=ExecStopPost=/usr/bin/python3 '+str(pause/'resume_forecasting.py'),
         str(source/'isaaclab/deploy/hexapod-rl'),'omni-repair-pair','--sources',str(source),str(base/'omni_repair_source_003_b'),
         '--checkpoint',str(base/'omni_flat_002/stage_002/f050_t060/train/policy/final.pt'),
         '--checkpoint-sha256','1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8','--output',str(output)]
    subprocess.run(cmd,check=True)
    (pause/'launch.json').write_text(json.dumps({'command':cmd,'launched_unix':time.time()},indent=2)+'\n')
except Exception:
    subprocess.run(['/usr/bin/python3',str(pause/'resume_forecasting.py')],check=True);raise
print(json.dumps({'unit':unit,'output':str(output),'pause':str(pause)}))
