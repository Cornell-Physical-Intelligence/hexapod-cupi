"""Read-only world-frame covariance and synthetic directional metric review."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];owner=ROOT/'tmp/reference_directional_adapter_001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(owner/'FREEZE_SHA256.json')=='1ffc294c64d3e3eb47a13710115ae94c9033da398e015854c9d6645138387032'
m=json.loads((owner/'FREEZE_SHA256.json').read_text())
for name,digest in m.items():assert sha(owner/name)==digest,name
sys.path.insert(0,str(owner));from test_directional import fixture
from directional_metrics import motion_evidence
from directional_contract import CASES
results=[]
for case in CASES:
 original,_=fixture(case);a=motion_evidence(original,CASES[case])
 for angle in (.73,1.5707963267948966,-2.1):
  c,s=np.cos(angle),np.sin(angle);R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
  rotated={k:v.copy() for k,v in original.items()}
  rotated['rotation_world_from_body']=R@rotated['rotation_world_from_body']
  rotated['position_world_m']=rotated['position_world_m']@R.T+np.array([3.,-5.,0.])
  rotated['velocity_world_mps']=rotated['velocity_world_mps']@R.T
  b=motion_evidence(rotated,CASES[case])
  for key in ('actual_heading_change_rad','max_planar_excursion_from_start_m','measured_command_axis_displacement_m','integrated_world_velocity_command_axis_m','mean_body_command_axis_velocity_mps'):
   if key in a:np.testing.assert_allclose(a[key],b[key],atol=1e-12,rtol=1e-12,err_msg=case+'/'+key)
  assert b['signed_translation_pass'] and b['signed_yaw_pass']
  results.append({'case':case,'world_yaw_offset_rad':angle,'same_motion_verdict':True})
print(json.dumps({'passed':True,'owner_freeze_sha256':sha(owner/'FREEZE_SHA256.json'),'owner_payload_count':len(m),
 'reviewed_sources':{k:sha(owner/k) for k in ['directional_metrics.py','directional_contract.py','run_directional_physics.py']},
 'synthetic_frame_covariance_cases':results,'no_GPU_or_physical_claim':True,'limits':['No new raw runtime audit.','World/root-link displacement and velocity remain separate from reported COM body twist.','Body gyro Z is not a general exact Euler heading derivative under tilt; actual heading and reported yaw are both separately required and discrepancy remains diagnostic.','Four limited cases do not establish omni or PPO qualification.']},indent=2))
