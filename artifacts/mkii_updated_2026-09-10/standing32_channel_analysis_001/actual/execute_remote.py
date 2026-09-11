from pathlib import Path
import subprocess,json,hashlib,time
base=Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910')
src=base/'standing32_channel_analysis_source_001'
assert hashlib.sha256((src/'FREEZE_SHA256.json').read_bytes()).hexdigest()=='d1eb068b71ccf01bae0fe26d16f8f24e83be260c5ca4556ec75ba1d0d678faab'
m=json.loads((src/'FREEZE_SHA256.json').read_text())
assert {str(p.relative_to(src)) for p in src.rglob('*') if p.is_file()}==set(m)|{'FREEZE_SHA256.json'}
for rel,h in m.items():assert hashlib.sha256((src/rel).read_bytes()).hexdigest()==h,rel
out=base/'standing32_channel_analysis_result_001';out.mkdir(exist_ok=False)
cmd=['/usr/bin/python3','-B','-S',str(src/'analyze_remote.py'),'--run',str(base/'native_standing32_005'),'--source',str(base/'standing_source_005'),'--audit',str(src/'inputs/audit.json')]
started=time.time();rc=None;error=None
with (out/'analysis.json').open('xb')as stdout,(out/'analysis.stderr').open('xb')as stderr:
 try:rc=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=600,check=False).returncode
 except subprocess.TimeoutExpired:error='Owned CPU analyzer exceeded 600-second wall bound'
files={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}for p in out.iterdir()if p.is_file()}
receipt={'schema':'owned_cpu_analysis_execution_v1','command':cmd,'started_unix':started,'completed_unix':time.time(),'returncode':rc,'error':error,'source_freeze_sha256':'d1eb068b71ccf01bae0fe26d16f8f24e83be260c5ca4556ec75ba1d0d678faab','output':str(out),'files':files}
(out/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
raise SystemExit(0 if rc==0 and error is None else 1)
