from pathlib import Path
import json,subprocess,time,hashlib
B=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
def call(a):
 q=subprocess.run(a,capture_output=True,text=True,timeout=20);return {'exit_code':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
p=B/'direct_omni_train_smoke_003'
files={}
for rel in ('campaign.json','jobs/standing.json'):
 f=p/rel
 if f.exists():files[rel]={'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'value':json.loads(f.read_text())}
for rel in ('forecast_pause_059/pause.json','forecast_pause_059/launch.json'):
 f=B/rel;files[rel]={'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'value':json.loads(f.read_text())}
print(json.dumps({'observed_unix':time.time(),'run':files,'owner':call(['systemctl','--user','show','hexapod-direct-omni-train-smoke-003-20260910.service','-p','ActiveState','-p','SubState','-p','InvocationID','-p','MainPID','-p','ExecMainStatus']),'weather':call(['systemctl','--user','show','stormscope-halo104-1010-20260910.service','-p','ActiveState','-p','SubState','-p','InvocationID','-p','MainPID','-p','ExecMainStatus']),'weather_log_tail':Path('/home/orionh/stormscope-halo-tests-20260910/halo104-1010.log').read_text().splitlines()[-4:],'gpu':call(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])}))
