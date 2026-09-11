"""Root-owned exact immutable bundle transfer, refusing existing destinations."""
from pathlib import Path
import sys,json,hashlib,base64,subprocess,shlex,time
local=Path(sys.argv[1]);name=sys.argv[2];bound=sys.argv[3];receipt=Path(sys.argv[4])
assert '/'not in name and name.replace('_','').isalnum()
manifest=local/'FREEZE_SHA256.json';assert hashlib.sha256(manifest.read_bytes()).hexdigest()==bound
m=json.loads(manifest.read_text());actual={str(p.relative_to(local))for p in local.rglob('*')if p.is_file()};assert actual==set(m)|{'FREEZE_SHA256.json'}
assert not any(p.is_symlink()for p in local.rglob('*'))
files={}
for rel in [*m,'FREEZE_SHA256.json']:
 b=(local/rel).read_bytes();h=hashlib.sha256(b).hexdigest()
 if rel in m:assert h==m[rel],rel
 files[rel]={'b64':base64.b64encode(b).decode(),'sha256':h}
code='''import json,sys,base64,hashlib
from pathlib import Path
a=json.load(sys.stdin);p=Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910')/a['name'];p.mkdir(exist_ok=False)
for rel,v in a['files'].items():
 assert not Path(rel).is_absolute() and '..'not in Path(rel).parts
 b=base64.b64decode(v['b64']);assert hashlib.sha256(b).hexdigest()==v['sha256'];q=p/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b)
actual={str(q.relative_to(p)):hashlib.sha256(q.read_bytes()).hexdigest()for q in p.rglob('*')if q.is_file()}
assert actual=={k:v['sha256']for k,v in a['files'].items()}
print(json.dumps({'destination':str(p),'verified_files':len(actual),'manifest_sha256':actual['FREEZE_SHA256.json']}))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','spark','python3 -B -S -c '+shlex.quote(code)],input=json.dumps({'name':name,'files':files}),capture_output=True,text=True,timeout=120)
s={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'observed_unix':time.time()}
receipt.parent.mkdir(parents=True,exist_ok=True);assert not receipt.exists();receipt.write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s));assert r.returncode==0
