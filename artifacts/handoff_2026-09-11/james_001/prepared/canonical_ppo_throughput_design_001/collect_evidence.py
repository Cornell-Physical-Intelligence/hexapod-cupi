#!/usr/bin/env python3
"""Read only local pinned receipts; no simulator, network or raw5GB input."""
import argparse,hashlib,json
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.repo.resolve();bindings={}
 def read(name):
  f=r/name;bindings[name]={'sha256':sha(f),'bytes':f.stat().st_size};return json.loads(f.read_text())
 checked={}
 for source,expected in [('tmp/updated_native_standing_004','037013353a30c5f3751235634aafdb3f204b676837962b14df72c3051c0a2786'),('tmp/canonical_ppo_integration_002','1e173946fe2548a82207791528f503ac6d12766fd5b50f25746e6edf92704617')]:
  name=source+'/FREEZE_SHA256.json';m=read(name);assert bindings[name]['sha256']==expected
  for rel,digest in m.items():assert sha(r/source/rel)==digest,(source,rel)
  checked[source]={'manifest_sha256':expected,'verified_payloads':len(m)}
 audit=read('tmp/canonical_native_standing32_terminal_002/audit.json')
 session=read('artifacts/mkii_updated_2026-09-10/native_standing32_rejected_002/curated/run/standing/session.json')
 assert audit['raw_inventory']['run/standing/session.json']['sha256']==bindings['artifacts/mkii_updated_2026-09-10/native_standing32_rejected_002/curated/run/standing/session.json']['sha256']
 inv=audit['raw_inventory'];assert sum(x['size_bytes']for x in inv.values())==audit['raw_total_bytes']
 profile=read('tmp/updated_native_standing_004/contact_optimization_review/RESULT.json')
 modules=['standing_session.py','standing_math.py','standing_score.py']
 same={f:sha(r/'tmp/updated_native_standing_003'/f)==sha(r/'tmp/updated_native_standing_004'/f)for f in modules};assert all(same.values())
 for source,files in [('tmp/updated_native_standing_004',modules+['standing_geometry.py']),('tmp/canonical_ppo_integration_002',['canonical_direct_ppo/native_bridge.py','canonical_direct_ppo/neutral_prefix.py','canonical_direct_ppo/runner.py'])]:
  for f in files:
   path=r/source/f
   if path.exists():bindings[source+'/'+f]={'sha256':sha(path),'bytes':path.stat().st_size}
 wall=session['wall_s'];ct=inv['run/standing/contacts.jsonl']['size_bytes'];npz=sum(v['size_bytes']for k,v in inv.items()if '/substeps_'in k)
 result={'schema':'canonical_performance_local_evidence_v1','native_executed':False,'bindings':bindings,'source_payloads_verified':checked,
 'actual32':{'invocation':audit['expected_invocation'],'envs':32,'source':'standing003','controlled_substeps':session['steps'],'controls':session['controls'],
 'session_wall_s':wall,'whole_native_wall_s':audit['native_state']['wall_s'],'global_controls_per_wall_s':session['controls']/wall,'environment_control_transitions_per_wall_s':32*session['controls']/wall,'global_substeps_per_wall_s':session['steps']/wall,'simulated_seconds_per_wall_s':session['steps']*.0025/wall,
 'raw_files':len(inv),'raw_bytes':audit['raw_total_bytes'],'contacts_bytes':ct,'contacts_fraction_raw_bytes':ct/audit['raw_total_bytes'],'substep_npz_bytes':npz,'control_npz_bytes':inv['run/standing/control_trace.npz']['size_bytes'],'combined_physical_quiet_passes':sum(x['pass']for x in audit['standing_report']['replicas']),'all_pass':audit['standing_report']['all_pass'],
 'interpretation':'Acquisition completed; combined physical/quiet admission rejected and host600s timeout distinct. Session wall includes collection, raw writing/close; not a native physics kernel timer.'},
 'current004_identical_to_actual003_modules':same,'existing_cpu_profile':{'receipt_scope':'Prior Mac local1-env and replicated32 strata; weighted estimates, not Spark component attribution. Already-adopted optimization, not an additional future speedup.',
 'candidate_weighted32_classifier_s':profile['candidate_weighted_32_classifier_s'],'clearance_weighted32_s':profile['baseline_weighted_32_cpu_components_s']['clearance'],'json_weighted32_s':profile['baseline_weighted_32_cpu_components_s']['json'],'all_three_weighted32_s':profile['candidate_plus_unchanged_clearance_json_s'],'old_one_env_rows':profile['parity']['actual_rows_bitexact_candidate_vs_parent'],'machine':profile['distribution']['machine'],'python':profile['distribution']['python'],'numpy':profile['distribution']['numpy']},
 'scope_limits':['No Spark component timers or synchronization attribution available.','No measurement of GPU vectorization,1024 SDF scenes or actual PPO training throughput.','No entire32 contactstream downloaded/read/reconstructed.','Current PPO002 source binding is standing003, not a standing004 admission. Any final accepted successor needs explicit reviewed binding.']}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'verified_sources':checked,'actual32':result['actual32'],'modules_identical':same},indent=2))
if __name__=='__main__':main()
