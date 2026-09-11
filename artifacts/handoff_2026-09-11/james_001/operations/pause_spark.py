from pathlib import Path
import subprocess,json,time,hashlib
base=Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910');out=base/'james_handoff_pause_001';out.mkdir(exist_ok=False)
timer='hexapod-restore-forecasting-20260909.timer';service='hexapod-restore-forecasting-20260909.service'
def run(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=30)
 if r.returncode:raise RuntimeError(str(args)+': '+r.stderr)
 return r.stdout
before={n:run(['systemctl','--user','show',n,'-p','Id','-p','ActiveState','-p','SubState','-p','FragmentPath','-p','Transient','-p','MainPID','-p','ExecStart'])for n in [timer,service]}
assert '/home/orionh/HEXAPOD_runs/mock_length_study_20260909/forecast_pause_001/resume_forecasting.py' in before[service]
assert 'MainPID=0' in before[service]
(out/'stale_restore_before.json').write_text(json.dumps(before,indent=2)+'\n')
run(['systemctl','--user','stop',timer,service])
after={n:run(['systemctl','--user','show',n,'-p','Id','-p','ActiveState','-p','SubState','-p','MainPID'])for n in [timer,service]}
assert all('ActiveState=inactive' in x or 'ActiveState=failed' in x for x in after.values())
marker={'schema':'james_handoff_pause_v1','paused_at_unix':time.time(),'reason':'User paused all research for handoff to James. No autonomous training, validation, research or recording restart.','resume_owner':'James / explicit user instruction','research_paused':True,'external_automation_reservation_released':False,'continuation_automation_id':'advance-hexapod-stage-2','continuation_status':'PAUSED','previous_main':'e6fc64444ee6b02c973e454ebe285843dcbe7fc2','stopped_stale_timer':timer}
(out/'PAUSED.json').write_text(json.dumps(marker,indent=2)+'\n')
receipt={'observed_unix':time.time(),'marker':marker,'remote_pause_path':str(out),'stale_timer_before':before,'stale_timer_after':after,'marker_sha256':hashlib.sha256((out/'PAUSED.json').read_bytes()).hexdigest(),'no_training_process_signaled':True}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
