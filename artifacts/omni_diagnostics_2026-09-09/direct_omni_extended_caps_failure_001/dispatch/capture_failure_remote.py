from pathlib import Path
import hashlib,json,subprocess,time
B=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
UNIT='hexapod-direct-omni-train-extended-caps-001-20260910.service'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def call(argv):
    r=subprocess.run(argv,text=True,capture_output=True,timeout=30)
    return dict(argv=argv,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr)
r={'schema':'extended_caps001_pre_simulator_failure_v1','observed_unix':time.time(),'expected_invocation':'75ddb0b75a8a4f0d8fc4a34c92e87486','read_only':True}
r['owner']=call(['systemctl','--user','show',UNIT,'-p','ActiveState','-p','SubState','-p','InvocationID','-p','MainPID','-p','ExecMainStatus','-p','Result'])
r['journal']=call(['journalctl','--user','-u',UNIT,'--no-pager','-o','json'])
r['GPU']=call(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'])
r['containers']=call(['docker','ps','--format','{{.ID}} {{.Names}} {{.Image}}'])
out=B/'direct_omni_train_extended_caps_001';r['output_exists']=out.exists()
pause=B/'forecast_pause_062';r['pause_files']={p.relative_to(pause).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in pause.rglob('*') if p.is_file()}
for name in ['pause.json','launch.json','restored.json']:
    r[name]=json.loads((pause/name).read_text())
for key,folder,pin in [('host','direct_omni_train_host_004','b55282c968d64dca218baf10278339ff2013c23e28dc7cbe9a0d0b8dfd05d51f'),('guard','direct_omni_train_extended_caps_guard_001','a9507bada19f9aa9fbb907b55c9a2f9c7cd0413a84e3600e30582e0794ebf19a')]:
    p=B/folder; f=p/'FREEZE_SHA256.json';assert sha(f)==pin; m=json.loads(f.read_text());assert {x.relative_to(p).as_posix():sha(x) for x in p.rglob('*') if x.is_file() and x!=f}==m;r[key]={'freeze_sha256':pin,'verified_payloads':len(m)}
assert 'ActiveState=failed' in r['owner']['stdout'] and 'MainPID=0' in r['owner']['stdout'] and 'ExecMainStatus=1' in r['owner']['stdout']
assert 'Loaded run_owned differs from exact verified source' in r['journal']['stdout']
assert not r['output_exists'] and r['restored.json'].get('restored_unix')
r['verified_pre_simulator_failure']=True
print(json.dumps(r,indent=2))
