from pathlib import Path
import hashlib,json,subprocess,datetime,sys
root=Path('/home/orionh/HEXAPOD_runs/restart_20260914/standing_translation_001')
expected={'source':'6e11c382b2c942aad3050e9827624e6e9bb529aef8f5608f07054656b70d7163','host':'5932b73daab172e78e0edea0713a4b89e56a4aa14b76a891e0f2c254c788ef09','guard':'2f2c8ec8317414a6087c5196033f99b9e55981a873a5b6b9eeaee9a1c671e1ec'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,want in expected.items():
 folder=root/name;manifest=folder/'FREEZE_SHA256.json'
 assert sha(manifest)==want
 assert not any(p.is_symlink() for p in folder.rglob('*'))
 assert json.loads(manifest.read_text())=={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file() and p!=manifest}
record={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'transferred_trees_verified':expected,'native_run_started':False,'arms':{}}
for arm in ('origin','xy14_4'):
 binding=root/'bindings'/(arm+'_guard.json');output=root/(arm+'_001')
 cmd=['/usr/bin/python3','-B',str(root/'guard/launch_guarded_diagnostic.py'),'--bindings',str(binding),'--bindings-sha256',sha(binding),'--guard-freeze-sha256',expected['guard'],'--placement',arm,'--output',str(output),'--preflight-only']
 p=subprocess.run(cmd,cwd='/tmp',capture_output=True,text=True,timeout=120)
 record['arms'][arm]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 if p.returncode:break
with (root/'preflight_001.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record,indent=2))
sys.exit(0 if len(record['arms'])==2 and all(v['returncode']==0 for v in record['arms'].values()) else 1)
