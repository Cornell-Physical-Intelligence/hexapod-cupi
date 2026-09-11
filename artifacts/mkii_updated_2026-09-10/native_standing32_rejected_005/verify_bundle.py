"""Portable report/byte audit only; deliberately no native/full-raw scoring replay."""
from pathlib import Path
import hashlib,json,math
from summarize_report import summarize,csv_text
R=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads((R/p).read_text())
def need(ok,msg):
 if not ok:raise ValueError(msg)
def validate():
 m=read('BUNDLE_SHA256.json')['files']
 actual={str(p.relative_to(R))for p in R.rglob('*')if p.is_file()and p!=R/'BUNDLE_SHA256.json'and '__pycache__'not in p.parts}
 need(actual==set(m),'Unexpected public inventory')
 for p,h in m.items():need(not(R/p).is_symlink()and sha(R/p)==h,'Changed public payload:'+p)
 copies=read('COPY_VERIFICATION.json')
 for c in copies['files']:
  p=R/c['public_copy'];need(sha(p)==c['sha256']and p.stat().st_size==c['bytes'],'Copied-byte mismatch')
 audit=read('terminal/audit.json');raw=read('terminal/RAW_SHA256.json');encoding=read('terminal/RAW_ENCODING.json');remote=read('REMOTE_ONLY.json')
 need(audit['audit_verified']is True and audit['errors']==[]and audit['raw_inventory_stable']is True,'Terminal audit failed')
 need(raw==audit['raw_inventory']and len(raw)==39 and sum(v['size_bytes']for v in raw.values())==5019294627,'Full inventory mismatch')
 need(len(encoding)==27 and sum(v['size_bytes']for v in encoding.values())==3517992,'Curated selection mismatch')
 for p,v in encoding.items():
  need(v['encoding']=='identity'and v['sha256']==raw[p]['sha256']and v['size_bytes']==raw[p]['size_bytes'],'Encoding metadata mismatch')
  path=R/'terminal'/v['storage_path'];need(path.stat().st_size==v['size_bytes']and sha(path)==v['sha256'],'Raw copied-byte mismatch:'+p)
 need(set(raw)-set(encoding)==set(remote['files'])and remote['count']==12 and remote['bytes']==5015776635,'Remote-only exclusions incomplete')
 for p,v in remote['files'].items():
  need(v['sha256']==raw[p]['sha256']and v['size_bytes']==raw[p]['size_bytes'],'Remote entry changed')
  need(not(R/'terminal'/p).exists(),'Remote-only file silently added')
 for key,info in copies['lineages'].items():
  folder=R/'preparation'/key;freeze=folder/'FREEZE_SHA256.json';fm=json.loads(freeze.read_text());fm=fm.get('files',fm)
  need(sha(freeze)==info['freeze_sha256']and len(fm)==info['payloads'],'Preparation freeze mismatch:'+key)
  payloads={str(p.relative_to(folder))for p in folder.rglob('*')if p.is_file()and p!=freeze and '__pycache__'not in p.parts}
  need(payloads==set(fm),'Preparation inventory mismatch:'+key)
  for p,h in fm.items():need(sha(folder/p)==h,'Preparation hash mismatch:'+key+'/'+p)
  if key!='auditor':need(audit['input_'+key]['manifest_sha256']==info['freeze_sha256'],'Actual lineage differs:'+key)
 for kind in ['source','host','guard','ownership_supervisor','asset']:
  need(audit['input_'+kind]['passed']is True and audit['final_input_'+kind]==audit['input_'+kind],'Start/end input audit mismatch:'+kind)
 state=read('terminal/run/standing/state.json');session=read('terminal/run/standing/session.json');job=read('terminal/run/jobs/standing.json');campaign=read('terminal/run/campaign.json');report=read('terminal/run/standing/standing_report.json')
 need(audit['raw_acquisition_completed']is True and audit['standing_completed']is False and audit['training_allowed']is False and audit['physical_admission']is False,'Acquisition/admission semantics changed')
 need(state['status']=='completed'and state['explicit_steps_completed']==8000 and state['errors']==[]and state['standing_pass']is False,'Native completion/rejection changed')
 need(session['steps']==session['captured_steps']==8000 and session['controls']==1000 and session['reset_count']==1 and session['failure']is None and session['all_rows_recorded']is True,'Incomplete session')
 need(job['status']=='completed'and job['exit_code']==0 and job['cleanup_checked']is True and job['deadline_seconds']==1200 and job['app_ready_deadline_seconds']==90,'Operational completion changed')
 need(not job.get('competitors')and not job.get('error')and job['no_policy_loaded']is True,'Unexpected interference/policy')
 need(campaign['status']=='failed'and campaign['error']=="ValueError('Standing physics/quiet rejected; acquisition is not admission')"and campaign['training_allowed']is False,'Host rejection changed')
 need(audit['native_validation']['attempted']is True and audit['native_validation']['passed']is False,'Original validator rejection missing')
 need(audit['unit']['MainPID']=='0'and audit['unit']['ExecMainStatus']=='1'and audit['unit']['InvocationID']==audit['expected_invocation'],'Terminal unit identity changed')
 identifiers=audit['owned_absence']['identifiers'];need(set(identifiers)=={job['container_name'],job['container_id']}and all(v['absent']is True for v in identifiers.values()),'Exact owned absence missing')
 need(report==audit['standing_report']and report==read('terminal/report.json'),'Saved report differs')
 summary=summarize(report);need(summary==read('SUMMARY.json')and csv_text(summary)==(R/'PER_ENVIRONMENT.csv').read_text(),'Summary reproduction differs')
 need(summary['combined_passes']==summary['physical_passes']==10 and summary['quiet_passes']==24 and len(summary['support_failure_envs'])==22 and summary['missing_support_samples_total']==73,'Report counts differ')
 for row in report['replicas']:
  need(row['failed_physical_bounds']==(['six_toe_support']if row['physical']['post_settle_missing_six_toe_substeps']else []),'Other physical failure present')
  quiet=row['quiet'];need(quiet['failed_bounds']==([]if quiet['pass']else ['max_joint_velocity_rms_rad_s']),'Other quiet failure present')
  need(row['pass']==(not row['failed_physical_bounds']and quiet['pass']),'Combined flag inconsistent')
  for k,bound in report['gates'].items():
   v=quiet[k];need(type(v)in(int,float)and math.isfinite(v),'Nonfinite reported metric')
   need((v>bound)==(k in quiet['failed_bounds']),'Report metric/gate contradiction:'+k)
 solver=read('terminal/run/standing/solver_readback.json')
 for stage in ['after_authoring','after_reset','after_controlled_steps']:
  need(len(solver[stage])==32 and all(v['position_iterations']==32 and v['velocity_iterations']==0 for v in solver[stage].values()),'Solver readback mismatch')
 need(audit['solver_verification']['passed']is True and audit['operational_deadline']['passed']is True,'Solver/operational audit failed')
 reservation=audit['exclusive_reservation'];need(reservation['verified']is True and reservation['persistent_release_attempted']is False,'Persistent reservation not verified')
 need(reservation['guard_verification']['queue_lock_held']is True and reservation['guard_verification']['masked_user_units']==31,'Reservation checks missing')
 restore=read('terminal/forecast_pause/restored.json');need(restore['persistent_reservation_release_attempted']is False and restore['timers']==[]and audit['restoration']['passed']is True,'Per-job restoration scope changed')
 return {'payloads':len(m),'curated_raw_verified':27,'full_recorded_raw':39,'remote_only':12,'source_payloads':109,'host_payloads':63,'guard_payloads':116,'auditor_payloads':24,'combined_passes':10,'quiet_passes':24,'raw_acquisition_completed':True,'native_admission':False,'local_full_raw_replay':False,'remote_or_GPU_calls':0}
if __name__=='__main__':print(json.dumps(validate(),indent=2))
