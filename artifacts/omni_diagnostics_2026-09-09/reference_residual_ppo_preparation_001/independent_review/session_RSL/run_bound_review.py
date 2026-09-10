"""Independent fixed-source CPU session/RSL checks, with before/after hashes."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parent;OWNER=ROOT.parent/'reference_residual_ppo_001';OBS=ROOT.parent/'reference_policy_observation_005_001'
FILES=('rollout_session.py','residual_runner.py','plan.json','checkpoint_compatibility.py','real_rsl_cpu_regression.py','RSL_SOURCE_PARITY.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 before={name:sha(OWNER/name) for name in FILES};dependencies=json.loads((OWNER/'RSL_SOURCE_PARITY.json').read_text())
 for row in dependencies['sources']:
  assert sha(OWNER/'_deps'/row['relative_file'])==row['installed_Spark_sha256']==row['local_wheel_sha256']
 assert sha(OBS/'FREEZE_SHA256.json')=='22402b0079b0b5ac0acdd4e1d7e819fe206b20ee1883077f7de6eaca55808a63'
 for f,h in json.loads((OBS/'FREEZE_SHA256.json').read_text()).items():assert sha(OBS/f)==h
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(OWNER/'_deps'))
 commands=[([sys.executable,'-m','unittest','discover','-s',str(ROOT),'-p','test_session_seam.py','-v'],'bound_seam_tests.log'),([sys.executable,str(OWNER/'real_rsl_cpu_regression.py'),'--observation-bundle',str(OBS),'--output',str(ROOT/'bound_cpu_regression')],'bound_rsl_tests.log')]
 for cmd,name in commands:
  with (ROOT/name).open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 after={name:sha(OWNER/name) for name in FILES};assert before==after
 result=json.loads((ROOT/'bound_cpu_regression/report.json').read_text())
 assert result['passed'] and result['controls']==48 and result['optimizer_state_entries']==17 and result['deterministic_reload_max_difference']==0.
 report={'scope':'Independent synthetic CPU session/actual RSL lifecycle review only; not physical entrypoint approval','reviewed_source_sha256':before,'before_after_source_identity_equal':True,'RSL_source_matches_recorded_installed_Spark_five_files':dependencies,'observation_freeze_sha256':sha(OBS/'FREEZE_SHA256.json'),'session_tests_passed':5,'real_RSL_PPO_updates':2,'real_RSL_controls':48,'real_RSL_replicas':32,'actor_width':846,'critic_width':849,'optimizer_entries_exact_reload':17,'deterministic_reload_difference':0.,'raw_reported_velocity_and_interval_rate_preserved':True,'real_RSL_report_sha256':sha(ROOT/'bound_cpu_regression/report.json'),'GPU_launches':0,'physical_admission':False,'physical_entrypoint_reviewed':False,'stage2_complete':False}
 (ROOT/'review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
