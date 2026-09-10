from pathlib import Path
import json,subprocess,time,hashlib
B=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909');p=B/'direct_omni_train_pilot_quiet_priority_001'
files={}
for root,rel in [(p,'campaign.json'),(B,'forecast_pause_060/pause.json'),(B,'forecast_pause_060/launch.json')]:
 f=root/rel;files[rel]={'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'value':json.loads(f.read_text())}
q=subprocess.run(['systemctl','--user','show','hexapod-direct-omni-train-pilot-quiet-priority-001-20260910.service','-p','ActiveState','-p','SubState','-p','InvocationID','-p','MainPID'],capture_output=True,text=True,timeout=20)
assert q.returncode==0 and 'InvocationID=721f1886478a4e518c7e0b741256413f' in q.stdout
print(json.dumps({'observed_unix':time.time(),'owner':q.stdout,'files':files}))
