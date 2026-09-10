"""Read-only verification of curated raw evidence, copied freezes and checkpoint identity."""
from pathlib import Path
import hashlib,json,sys
P=Path(__file__).resolve().parent

def require(ok,message):
 if not ok:raise RuntimeError(message)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def check_map(root,m):
 for name,digest in m.items():
  p=root/name
  require(p.is_file() and p.resolve().is_relative_to(root.resolve()),f'Absent or external file: {p}')
  require(sha(p)==digest,f'Hash mismatch: {p}')

freeze=read(P/'FREEZE_SHA256.json')
actual={str(p.relative_to(P)) for p in P.rglob('*') if p.is_file() and p.name!='FREEZE_SHA256.json'}
# Nested original manifests are payloads and must remain in the primary inventory.
actual|={str(p.relative_to(P)) for p in P.rglob('FREEZE_SHA256.json') if p!=P/'FREEZE_SHA256.json'}
require(set(freeze)==actual,'Publication file inventory differs')
check_map(P,freeze)
preps={}
for f in (P/'preparation').glob('*/FREEZE_SHA256.json'):
 m=read(f);check_map(f.parent,m);preps[f.parent.name]=len(m)
source=read(P/'preparation/source/campaign_source_hashes.json')
require(len(source)==598,'Wrong source count')
require(len([k for k in source if k.startswith('robot/hexapod_mkii_length_study/')])==550,'Wrong asset count')
latest=read(P/'LATEST_CHECKPOINTS.json');coverage={}
for branch in ['curriculum','caps']:
 d=P/branch;m=read(d/'audit/LOCAL_SHA256.json');plan=read(d/'audit/FETCH_PLAN.json');audit=read(d/'audit/remote_audit.json');report=read(d/'analysis/report.json')
 require(len(m)==72 and len(plan['all_remote_files'])==122 and len(plan['omitted_remote_only'])==50,'Wrong curated coverage')
 require(set(m)==set(plan['selected']),'Selected inventory mismatch')
 check_map(d/'raw',m)
 raw_set={str(x.relative_to(d/'raw')) for x in (d/'raw').rglob('*') if x.is_file()}
 require(raw_set==set(m),'Curated raw inventory differs')
 for k,v in m.items():require(plan['all_remote_files'][k]['sha256']==v,'Remote/local map differs')
 total=sum((d/'raw'/k).stat().st_size for k in m)
 require(total==plan['selected_bytes'],'Curated byte count differs')
 require(audit['all_input_trees_verified'] and audit['campaign_completed'] and audit['actual_completed_updates']==50,'Actual campaign not completed')
 require(len(audit['owned_names_IDs_absent'])==12 and all(x['absent'] for x in audit['owned_names_IDs_absent'].values()),'Owned cleanup unverified')
 require(audit['halo_archive_unchanged'] and not audit['formal_policy_acceptance_claimed'],'Invalid cleanup/admission statement')
 t=report['training'];cp=latest['latest_saved'][branch]
 require(not report['errors'] and t['updates_completed']==50 and t['strict_reload']['passed'],'Training/reload proof failed')
 require(cp['sha256']==audit['saved_final_checkpoint_sha256']==t['checkpoint_sha256']==sha(P/cp['path']),'Final checkpoint mismatch')
 require(report['final_stop']['total_replicas']==48 and report['final_stop']['quiet_passed_replicas']==0,'Quiet verdict changed')
 require(cp['updates']==50 and not latest['Stage2_complete'],'Latest checkpoint/admission mislabeled')
 check_map(d/'analysis',read(d/'analysis/REMOTE_REPORT_SHA256.json'))
 coverage[branch]={'selected_files':72,'selected_bytes':total,'remote_files':122,'omitted_autosaves':50,'final_checkpoint_sha256':cp['sha256'],'quiet_passed':0}
a=read(P/'curriculum/analysis/report.json');b=read(P/'caps/analysis/report.json');paired=read(P/'comparison/report.json')
require(a['initial_constant']==b['initial_constant'] and a['initial_stop']==b['initial_stop'],'Initial evidence differs')
require(paired['initial_constant_evidence_exact_equal'] and paired['initial_stop_evidence_exact_equal'] and not paired['Stage2_complete'],'Paired comparison labels differ')
print(json.dumps({'passed':True,'publication_files':len(freeze),'preparation_payloads':preps,'branches':coverage,'freeze_sha256':sha(P/'FREEZE_SHA256.json')},indent=2))
