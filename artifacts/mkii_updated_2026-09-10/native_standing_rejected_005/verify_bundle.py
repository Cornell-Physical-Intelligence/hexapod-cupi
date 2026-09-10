"""Portable original-byte verifier. Optional exact reconstruction/negative contract replay."""
from pathlib import Path,PurePosixPath
import argparse,gzip,hashlib,importlib.util,json,os,shutil,sys
sys.dont_write_bytecode=True

def require(ok,msg):
 if not ok:raise ValueError(msg)
def read(p):return json.loads(Path(p).read_text())
def digest_stream(f):
 h=hashlib.sha256();size=0
 for b in iter(lambda:f.read(8<<20),b''):h.update(b);size+=len(b)
 return h.hexdigest(),size
def digest(p):
 with Path(p).open('rb')as f:return digest_stream(f)[0]
def safe_rel(p):
 q=PurePosixPath(p);require(not q.is_absolute()and '..'not in q.parts,'Unsafe storage path');return Path(p)
def source_module(root):
 spec=importlib.util.spec_from_file_location('_published_rejected_standing_contract',root/'source/standing_contract.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def verify(root,replay=None):
 root=root.resolve();require(not any(p.is_symlink()for p in root.rglob('*')),'Symbolic publication payload')
 expected=read(root/'BUNDLE_SHA256.json');actual={p.relative_to(root).as_posix():digest(p)for p in root.rglob('*')if p.is_file()and p!=root/'BUNDLE_SHA256.json'and '__pycache__'not in p.parts};require(actual==expected,'Publication inventory differs')
 for name,pin in read(root/'COMPONENTS.json').items():
  d=root/name;require(digest(d/'FREEZE_SHA256.json')==pin['freeze_sha256'],'Component freeze differs:'+name);m=read(d/'FREEZE_SHA256.json');require(len(m)==pin['payloads'],'Component count differs:'+name);require({p.relative_to(d).as_posix():digest(p)for p in d.rglob('*')if p.is_file()and p!=d/'FREEZE_SHA256.json'and '__pycache__'not in p.parts}==m,'Component bytes differ:'+name)
 t=root/'terminal';audit=read(t/'audit.json');raw=read(t/'RAW_SHA256.json');enc=read(t/'RAW_ENCODING.json')
 require(audit['audit_verified']is True and audit['errors']==[]and audit['terminal_outcome']=='authentic_terminal_failure'and audit['standing_completed']is False,'Not an authentic audited rejection')
 require(audit['expected_invocation']=='f70b9d4b929244699a5725e018b38ba9','Wrong actual invocation');require(raw==audit['raw_inventory']and set(enc)==set(raw),'Raw inventory/encoding differs')
 total=0
 for rel,pin in raw.items():
  e=enc[rel];path=t/safe_rel(e['storage_path']);require(e['sha256']==pin['sha256']and e['size_bytes']==pin['size_bytes'],'Wrong original encoding identity:'+rel)
  if e['encoding']=='gzip':
   with gzip.open(path,'rb')as f:h,n=digest_stream(f)
  else:
   require(e['encoding']=='identity','Unknown transport encoding')
   with path.open('rb')as f:h,n=digest_stream(f)
  require((h,n)==(pin['sha256'],pin['size_bytes']),'Original raw bytes differ:'+rel);total+=n
 def raw_json(rel):
  e=enc[rel];path=t/safe_rel(e['storage_path']);require(e['encoding']=='identity','Small JSON must retain original direct storage');return read(path)
 state=raw_json('run/standing/state.json');report=raw_json('run/standing/standing_report.json');campaign=raw_json('run/campaign.json');job=raw_json('run/jobs/standing.json');restored=raw_json('forecast_pause/restored.json')
 require(state==audit['native_state']and report==audit['standing_report']and campaign==audit['campaign']and job==audit['job'],'Root audit differs from original JSON')
 require(state['status']=='completed'and state['explicit_steps_completed']==8000 and state['standing_pass']is False and report['all_pass']is False and report['num_envs']==1,'Original acquisition/rejection changed')
 require(campaign['status']=='failed'and job['cleanup_checked']is True and audit['native_validation']['passed']is False,'Failed host/native verdict changed')
 require(state['identity']['inspector_freeze_sha256']==read(root/'COMPONENTS.json')['source']['freeze_sha256'],'Wrong source lineage')
 require(all(state[k]is False for k in ('physical_admission','physics_admitted','training_allowed')),'Broader admission overclaim')
 for rel,h in state['outputs'].items():require(raw['run/standing/'+rel]['sha256']==h,'Native output receipt differs:'+rel)
 require(all(x['absent']is True for x in audit['owned_absence']['identifiers'].values())and len(audit['owned_absence']['identifiers'])==2,'Missing historical owned-name/ID absence')
 require(restored['scope']=='per_job_snapshot_only'and restored['persistent_reservation_release_attempted']is False and audit['exclusive_reservation']['verified']is True,'Persistent reservation release overclaimed')
 require(audit['solver_verification']['passed']is True and audit['operational_deadline']['passed']is True,'Actual solver/deadline proof missing')
 spec=importlib.util.spec_from_file_location('_frozen_auditor',root/'auditor/audit_remote.py');aud=importlib.util.module_from_spec(spec);spec.loader.exec_module(aud)
 require(aud.solver_check(raw_json('run/standing/solver_readback.json'),state['identity'],state)==audit['solver_verification'],'Original solver semantic check differs')
 # Same source-owned quiet gate dictionary remains present; original failed values are untouched.
 require(report['gates']==read(root/'EXPECTED_GATES.json'),'Acceptance thresholds changed')
 result={'verified':True,'payloads':len(actual),'original_raw_payloads':len(raw),'original_raw_bytes':total,'authentic_native_rejection':True,'source_contract_negative_replay_performed':False,'physical_admission':False,'training_allowed':False,'stage2_complete':False}
 if replay is not None:
  replay=replay.resolve();require(not replay.exists()and not replay.is_symlink(),'Replay destination must be fresh');require(root not in replay.parents and replay not in root.parents,'Replay destination overlaps immutable publication');replay.mkdir(parents=True)
  for rel,e in enc.items():
   src=t/safe_rel(e['storage_path']);dst=replay/safe_rel(rel);dst.parent.mkdir(parents=True,exist_ok=True)
   if e['encoding']=='gzip':
    with gzip.open(src,'rb')as fi,dst.open('wb')as fo:shutil.copyfileobj(fi,fo,8<<20)
   else:
    try:os.link(src,dst)
    except OSError:shutil.copyfile(src,dst)
   require(dst.stat().st_size==raw[rel]['size_bytes']and digest(dst)==raw[rel]['sha256'],'Reconstruction mismatch:'+rel)
  try:source_module(root).validate_result(replay/'run/standing',state['identity'])
  except ValueError as e:
   require(repr(e)==audit['native_validation']['error'],'Original source rejects for different reason:'+repr(e));result['source_contract_negative_replay_performed']=True;result['source_contract_rejection']=repr(e)
  else:raise ValueError('Original rejected source unexpectedly admitted run')
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--replay-directory',type=Path);a=p.parse_args();print(json.dumps(verify(Path(__file__).resolve().parent,a.replay_directory),indent=2))
