from pathlib import Path
import subprocess,json,time,hashlib
out=Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910/james_handoff_pause_001')
before=json.loads((out/'stale_restore_before.json').read_text());timer='hexapod-restore-forecasting-20260909.timer';service='hexapod-restore-forecasting-20260909.service'
assert '/home/orionh/HEXAPOD_runs/mock_length_study_20260909/forecast_pause_001/resume_forecasting.py' in before[service]
after={}
for name in [timer,service]:
 r=subprocess.run(['systemctl','--user','show',name,'-p','Id','-p','LoadState','-p','ActiveState','-p','SubState','-p','MainPID'],capture_output=True,text=True,timeout=30)
 props=dict(line.split('=',1)for line in r.stdout.splitlines()if '='in line)
 assert props.get('ActiveState')in ['inactive','failed'] and props.get('MainPID','0')=='0',props
 assert r.returncode==0 or props.get('LoadState')=='not-found',(r.returncode,props)
 after[name]={'properties':props,'returncode':r.returncode,'stderr':r.stderr}
r=subprocess.run(['systemctl','--user','list-timers','--all','--output=json','--no-pager'],capture_output=True,text=True,check=True,timeout=30);timers=json.loads(r.stdout)
assert not any(x.get('unit')==timer for x in timers)
marker={'schema':'james_handoff_pause_v1','paused_at_unix':time.time(),'reason':'User paused all research for handoff to James. No autonomous training, validation, research or recording restart.','resume_owner':'James / explicit user instruction','research_paused':True,'external_automation_reservation_released':False,'continuation_automation_id':'advance-hexapod-stage-2','continuation_status':'PAUSED','previous_main':'e6fc64444ee6b02c973e454ebe285843dcbe7fc2','stopped_stale_timer':timer}
assert not(out/'PAUSED.json').exists();(out/'PAUSED.json').write_text(json.dumps(marker,indent=2)+'\n')
receipt={'observed_unix':time.time(),'marker':marker,'remote_pause_path':str(out),'stale_timer_before':before,'stale_timer_after':after,'marker_sha256':hashlib.sha256((out/'PAUSED.json').read_bytes()).hexdigest(),'no_training_process_signaled':True,'first_stop_outcome':'Timer stopped and transient service disappeared; combined stop returned nonzero for the unloaded service. This separate readback verifies both inactive and no scheduled timer.'}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
