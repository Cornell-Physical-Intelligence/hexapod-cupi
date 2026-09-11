"""Portable original-byte reconstruction and exact positive source005 validation."""
from pathlib import Path,PurePosixPath
import gzip,hashlib,importlib.util,json,os,shutil,sys,tempfile
sys.dont_write_bytecode=True

def require(ok,why):
 if not ok:raise ValueError(why)
def read(p):return json.loads(p.read_text())
def digest(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def relpath(s):
 p=PurePosixPath(s);require(not p.is_absolute()and '..'not in p.parts and s!='','Unsafe relative path');return Path(s)
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def verify(r):
 r=r.resolve();require(not any(p.is_symlink()for p in r.rglob('*')),'Symbolic publication payload')
 expected=read(r/'BUNDLE_SHA256.json');actual={p.relative_to(r).as_posix():digest(p)for p in r.rglob('*')if p.is_file()and p!=r/'BUNDLE_SHA256.json'};require(actual==expected,'Publication inventory changed')
 for name,pin in read(r/'COMPONENTS.json').items():
  d=r/name;require(digest(d/'FREEZE_SHA256.json')==pin['freeze_sha256'],'Component freeze differs: '+name);m=read(d/'FREEZE_SHA256.json');require(len(m)==pin['payloads'],'Wrong component count')
  require({f.relative_to(d).as_posix():digest(f)for f in d.rglob('*')if f.is_file()and f!=d/'FREEZE_SHA256.json'}==m,'Component payloads differ: '+name)
 require({f.relative_to(r/'root_checks').as_posix():digest(f)for f in (r/'root_checks').rglob('*')if f.is_file()}==read(r/'ROOT_CHECKS_SHA256.json'),'Root readback snapshot differs')
 a=read(r/'terminal/audit.json');raw=read(r/'terminal/RAW_SHA256.json');enc=read(r/'terminal/RAW_ENCODING.json')
 require(a['audit_verified']is True and a['errors']==[]and a['standing_completed']is True and a['terminal_outcome']=='authentic_completed_standing','Not an authentic supported standing pass')
 require(a['expected_invocation']=='64a373e301c24b70aa3790c18aca147c','Wrong actual invocation');require(raw==a['raw_inventory']and set(enc)==set(raw)and len(raw)==40,'Raw transport inventory differs')
 with tempfile.TemporaryDirectory(prefix='native6-positive-replay-')as td:
  t=Path(td)
  for rel,pin in raw.items():
   e=enc[rel];src=r/'terminal'/relpath(e['storage_path']);dst=t/relpath(rel);dst.parent.mkdir(parents=True,exist_ok=True)
   require(e['sha256']==pin['sha256']and e['size_bytes']==pin['size_bytes'],'Wrong encoding identity')
   if e['encoding']=='gzip':
    with gzip.open(src,'rb')as fi,dst.open('wb')as fo:shutil.copyfileobj(fi,fo,8<<20)
   else:
    require(e['encoding']=='identity','Unknown transport encoding')
    try:os.link(src,dst)
    except OSError:shutil.copyfile(src,dst)
   require(dst.stat().st_size==pin['size_bytes']and digest(dst)==pin['sha256'],'Raw reconstruction differs: '+rel)
  d=t/'run/standing';state=read(d/'state.json');report=read(d/'standing_report.json');campaign=read(t/'run/campaign.json');job=read(t/'run/jobs/standing.json')
  require(state==a['native_state']and report==a['standing_report']and campaign==a['campaign']and job==a['job'],'Audit and original JSON differ')
  require(state['identity']['inspector_freeze_sha256']==read(r/'COMPONENTS.json')['source']['freeze_sha256'],'Wrong source lineage')
  contract=load('_published_native6_contract',r/'source/standing_contract.py');receipt=contract.validate_result(d,state['identity']);require(receipt==a['native_validation']['receipt']and a['native_validation']['passed']is True,'Original positive validator receipt differs')
  require(receipt['num_envs']==1 and receipt['standing_pass']is True,'This artifact is only the one-robot standing screen')
  auditor=load('_published_native6_auditor',r/'auditor/audit_remote.py');require(auditor.solver_check(read(d/'solver_readback.json'),state['identity'],state)==a['solver_verification'],'Original solver semantic check differs')
  outcome=auditor.classify(campaign,job,state,a['unit'],a['native_validation']);require(outcome=='authentic_completed_standing','Original owner/acquisition classification differs')
  require(auditor.restoration(read(t/'forecast_pause/pause.json'),read(t/'forecast_pause/restored.json'),read(t/'forecast_pause/launch.json'))==a['restoration'],'Original per-job restoration differs')
  phase_map=read(t/'run/standing_immutable.sha256.json');require(digest(t/'run/standing_immutable.sha256.json')==campaign['standing_payload_manifest_sha256'],'Post-exit seal hash differs');require(phase_map=={f.relative_to(d).as_posix():digest(f)for f in d.rglob('*')if f.is_file()},'Complete post-exit phase seal differs')
 require(len(a['owned_absence']['identifiers'])==2 and all(v['absent']is True for v in a['owned_absence']['identifiers'].values()),'Missing historical exact owned name/ID absence')
 require(a['exclusive_reservation']['verified']is True and a['operational_deadline']['passed']is True,'Missing reservation or deadline audit')
 require(report['gates']==read(r/'EXPECTED_GATES.json'),'Gate dictionary changed')
 return {'verified':True,'payloads':len(actual),'raw_payloads':40,'raw_original_bytes':sum(v['size_bytes']for v in raw.values()),'original_positive_contract_replayed':True,'original_receipt':receipt,'same_source_one_robot_standing_pass':True,'training_allowed':False,'physical_admission':False,'stage2_complete':False}
if __name__=='__main__':print(json.dumps(verify(Path(__file__).resolve().parent),indent=2))
