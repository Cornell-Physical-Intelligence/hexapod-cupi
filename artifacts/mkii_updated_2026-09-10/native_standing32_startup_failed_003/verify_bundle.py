"""Portable read-only verification of the original pre-AppReady failure."""
from pathlib import Path,PurePosixPath
import hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True

def require(ok,msg):
 if not ok:raise ValueError(msg)
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def verify(root):
 r=root.resolve();require(not any(p.is_symlink()for p in r.rglob('*')),'Symbolic bundle payload');m=read(r/'BUNDLE_SHA256.json');actual={p.relative_to(r).as_posix():sha(p)for p in r.rglob('*')if p.is_file()and p!=r/'BUNDLE_SHA256.json'};require(actual==m,'Publication bytes differ')
 for name,pin in read(r/'COMPONENTS.json').items():
  p=r/name;require(sha(p/'FREEZE_SHA256.json')==pin['freeze_sha256'],'Wrong component freeze');own=read(p/'FREEZE_SHA256.json');require(len(own)==pin['payloads']and{f.relative_to(p).as_posix():sha(f)for f in p.rglob('*')if f.is_file()and f!=p/'FREEZE_SHA256.json'}==own,'Missing/changed component bytes')
 require({p.relative_to(r/'root_checks').as_posix():sha(p)for p in (r/'root_checks').rglob('*')if p.is_file()}==read(r/'ROOT_CHECKS_SHA256.json'),'Root snapshot changed')
 t=r/'terminal';a=read(t/'audit.json');raw=read(t/'RAW_SHA256.json');enc=read(t/'RAW_ENCODING.json')
 require(a['audit_verified']is True and a['errors']==[]and a['terminal_outcome']=='authentic_terminal_failure'and a['standing_completed']is False and a['raw_acquisition_completed']is False,'Wrong historical failure verdict')
 require(a['expected_invocation']=='6eac4972e2e64f9991955a1b7bb73afc','Wrong invocation');require(raw==a['raw_inventory']and len(raw)==9 and set(enc)==set(raw),'Wrong nine-file raw inventory')
 for rel,pin in raw.items():
  e=enc[rel];path=PurePosixPath(e['storage_path']);require(not path.is_absolute()and '..'not in path.parts and e['storage_path']==rel and e['encoding']=='identity','Changed direct raw transport');p=t/rel
  require(e['sha256']==pin['sha256']and e['size_bytes']==pin['size_bytes']and sha(p)==pin['sha256']and p.stat().st_size==pin['size_bytes'],'Original raw differs: '+rel)
 state=read(t/'run/standing/state.json');job=read(t/'run/jobs/standing.json');campaign=read(t/'run/campaign.json')
 require(state==a['native_state']and job==a['job']and campaign==a['campaign'],'Audit differs from exact original JSON')
 require(state['status']=='running'and state['explicit_steps_completed']==0 and state['checks']=={}and state['errors']==[],'Unfinalized initial state was rewritten')
 require({k for k in raw if k.startswith('run/standing/')}=={'run/standing/state.json'},'Unexpected physical evidence')
 require(job['status']=='failed'and job['startup_failure_kind']=='no_reference_AppReady_by90s'and job['app_ready_deadline_seconds']==90 and job['deadline_seconds']==1200 and job.get('exit_code')is None and job['cleanup_checked']is True,'Wrong startup failure or cleanup')
 require(campaign['error']==job['error']=="TimeoutError('No AppReady by90s; preserve startup log and45s traceback')",'Original failure text changed')
 require(state['identity']['inspector_freeze_sha256']==read(r/'COMPONENTS.json')['source']['freeze_sha256'],'Wrong native source identity')
 require(all(state[k]is False for k in ('physical_admission','physics_admitted','training_allowed'))and campaign['stage2_complete']is False,'Broader admission overclaim')
 aud=load('_published_startup_auditor',r/'auditor/audit_remote.py');require(aud.classify(campaign,job,state,a['unit'],a['native_validation'])=='authentic_terminal_failure','Original terminal classification differs')
 require(aud.restoration(read(t/'forecast_pause/pause.json'),read(t/'forecast_pause/restored.json'),read(t/'forecast_pause/launch.json'))==a['restoration'],'Original restoration differs')
 require(len(a['owned_absence']['identifiers'])==2 and all(v['absent']is True for v in a['owned_absence']['identifiers'].values())and a['exclusive_reservation']['verified']is True,'Historical cleanup/reservation proof missing')
 # This negative call stops at the original incomplete-acquisition check. It does
 # not score absent physics or assert that native finalization happened.
 contract=load('_published_startup_contract',r/'source/standing_contract.py')
 try:contract.validate_result(t/'run/standing',state['identity'])
 except ValueError as error:require(str(error)=='Incomplete standing acquisition','Unexpected original contract rejection: '+repr(error))
 else:raise ValueError('Incomplete native state was admitted')
 log=(t/'run/logs/standing.log').read_text();require('REFERENCE_SCREEN_APP_READY'not in log,'Unexpected AppReady marker')
 return {'verified':True,'payloads':len(actual),'raw_files':9,'raw_original_bytes':sum(v['size_bytes']for v in raw.values()),'authentic_startup_failure':True,'original_contract_rejects_incomplete_acquisition':True,'physical_verdict_available':False,'standing_completed':False,'training_allowed':False,'physical_admission':False,'stage2_complete':False}
if __name__=='__main__':print(json.dumps(verify(Path(__file__).resolve().parent),indent=2))
