"""Read-only frozen consumer/fixture and reviewed-host binding verification."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OWNER=ROOT/'tmp/reference_residual_ppo_source_001'
HOST=ROOT/'tmp/reference_residual_ppo_launch_001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=OWNER/'FREEZE_SHA256.json'
assert sha(manifest)=='258c304f409373e49667bc7fb653026fcb0754c2c92d6dfd70eb04769c3e754b'
expected=json.loads(manifest.read_text())
actual={str(p.relative_to(OWNER)):sha(p) for p in OWNER.rglob('*') if p.is_file() and p!=manifest}
assert expected==actual
assert not any(p.is_symlink() for p in OWNER.rglob('*'))
provenance=json.loads((OWNER/'inputs/PROVENANCE.json').read_text())
fixtures={}
for phase in ('standing','wave'):
 rawpath=ROOT/'tmp/reference_physics_results_009/run'/phase/'trace.npz'
 fixturepath=OWNER/'inputs'/f'{phase}_scoring_fixture.npz'
 assert sha(rawpath)==provenance[phase]['source_sha256']
 assert sha(fixturepath)==provenance[phase]['output_sha256']
 with np.load(rawpath) as raw,np.load(fixturepath) as fixture:
  for key in fixture.files:np.testing.assert_array_equal(raw[key],fixture[key],err_msg=phase+'/'+key)
  fixtures[phase]={'array_equal_keys':fixture.files,'control_rows':len(fixture['time_s']),'raw_sha256':sha(rawpath),'fixture_sha256':sha(fixturepath)}
seam=json.loads((ROOT/'tmp/reference_residual_ppo_independent_review_001/report.json').read_text()) if (ROOT/'tmp/reference_residual_ppo_independent_review_001/report.json').exists() else None
result={'passed':True,'scope':'Independent read-only CPU physical-consumer and host review; no simulator/GPU admission',
 'consumer_manifest_sha256':sha(manifest),'consumer_payload_count':len(actual),'consumer_payloads':actual,
 'host_files':{p.name:sha(p) for p in (HOST/'launch_residual_ppo_spark.py',HOST/'test_host.py')},
 'exact_actual009_fixture_comparison':fixtures,'tests':{'frozen_consumer_physical_score_and_admission':4,'actual_finalizer_fault_cases':5,'host_orchestration':10},
 'previous_session_rsl_receipt_sha256':sha(ROOT/'tmp/reference_residual_ppo_independent_review_001/FREEZE_SHA256.json'),
 'limitations':['No new CUDA execution or physical exploration admission.','Host review binds the named code/test hashes; its final freeze and outer guard remain root-owned.','Final metadata-only plan rerun is owner evidence; independent prior real-RSL regression remains separately frozen.','Raw SDK rates and interval angle evidence remain separate; native velocity fidelity is not qualified.','Two-update stand-only integration is not a walking PPO or Stage2 completion.']}
print(json.dumps(result,indent=2,allow_nan=False))
