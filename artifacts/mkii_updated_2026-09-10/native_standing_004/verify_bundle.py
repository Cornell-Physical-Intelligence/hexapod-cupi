from pathlib import Path
import gzip,hashlib,importlib.util,json,os,shutil,sys,tempfile
sys.dont_write_bytecode=True
r=Path(__file__).resolve().parent
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
expected=json.loads((r/'BUNDLE_SHA256.json').read_text())
actual={str(p.relative_to(r)):digest(p)for p in r.rglob('*')if p.is_file()and p!=r/'BUNDLE_SHA256.json'and '__pycache__'not in p.parts}
assert actual==expected,'Publication inventory changed'
for name in ['source','host','guard','auditor','geometry_review']:
 p=r/name;m=json.loads((p/'FREEZE_SHA256.json').read_text())
 assert {str(f.relative_to(p)):digest(f)for f in p.rglob('*')if f.is_file()and f!=p/'FREEZE_SHA256.json'and '__pycache__'not in f.parts}==m,name
a=json.loads((r/'terminal/audit.json').read_text());raw=json.loads((r/'terminal/RAW_SHA256.json').read_text());enc=json.loads((r/'terminal/RAW_ENCODING.json').read_text())
assert a['audit_verified']and a['standing_completed']and not a['training_allowed']and a['terminal_outcome']=='authentic_completed_standing'
assert raw==a['raw_inventory']and set(enc)==set(raw)and len(raw)==38
with tempfile.TemporaryDirectory(prefix='hexapod-standing-verify-')as td:
 t=Path(td)
 for rel,pin in raw.items():
  e=enc[rel];src=r/'terminal'/e['storage_path'];dst=t/rel;dst.parent.mkdir(parents=True,exist_ok=True)
  assert e['sha256']==pin['sha256']and e['size_bytes']==pin['size_bytes']
  if e['encoding']=='gzip':
   with gzip.open(src,'rb')as fi,dst.open('wb')as fo:shutil.copyfileobj(fi,fo)
  else:
   assert e['encoding']=='identity'
   try:os.link(src,dst)
   except OSError:shutil.copyfile(src,dst)
  assert dst.stat().st_size==pin['size_bytes']and digest(dst)==pin['sha256'],rel
 spec=importlib.util.spec_from_file_location('_published_standing_contract',r/'source/standing_contract.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 s=json.loads((t/'run/standing/state.json').read_text());receipt=m.validate_result(t/'run/standing',s['identity'])
 assert receipt==a['native_validation']['receipt']
print('PASS',len(actual),'payloads; all38 original raw files; supported standing only, no PPO or Stage2 admission')
