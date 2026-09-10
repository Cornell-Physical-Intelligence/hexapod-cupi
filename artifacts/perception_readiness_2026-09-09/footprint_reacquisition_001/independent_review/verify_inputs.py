"""Read-only independent exact-frame/time/contact lineage review."""
from pathlib import Path
import json,hashlib,numpy as np
root=Path(__file__).resolve().parents[1];p=root/'sequential_footprint_reacquisition_001';raw=root/'reference_physics_results_004/run/wave'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
m=json.loads((p/'FREEZE_SHA256.json').read_text());m=m.get('files',m)
for f,h in m.items():assert sha(p/f)==h,f
assert sha(p/'FREEZE_SHA256.json')=='39b4669d05de39e62fb67dfd66adf0e23c56f6a3f5209c16583301627abc6c73'
timing=json.loads((p/'inputs/reference_timing_source004.json').read_text());refs=json.loads((raw/'reference_states.json').read_text())
with np.load(p/'inputs/motion_source004.npz') as d,np.load(raw/'trace.npz') as actual,np.load(p/'inputs/measured_contacts_source004.npz') as c:
 for k in ['time_s','distal_contact','reference_point_world_m']:assert np.array_equal(d[k],actual[k],equal_nan=True),k
 for k in c.files:assert np.array_equal(c[k],actual[k],equal_nan=True),k
 valid=[]
 for entry in refs:
  r=entry.get('result',entry.get('reset'))
  if not r['valid'][0]:continue
  matches=np.flatnonzero(np.isclose(d['time_s'][:,0],r['target_time_s'],rtol=0,atol=1e-9));assert len(matches)==1
  i=int(matches[0]);s=r['state'];active=int(d['active_swing_leg_index'][i])
  if active>=0:assert np.array_equal(d['planned_footprint_centres_world_m'][i,active],s['swing']['endpoint_world_m'])
  valid.append({'index':i,'time_s':r['target_time_s'],'mode':s['mode'],'current_leg':s['current_leg'],'swing':s['swing'] and {k:s['swing'][k] for k in ['start_s','duration_s','endpoint_world_m']},'landing':s['landing'] and {k:s['landing'][k] for k in ['start_s','duration_s','endpoint_world_m']}})
 assert valid==timing
 with np.load(p/'inputs/visibility.npz') as v:
  for i in v['indices']:assert abs(float(d['time_s'][i+2,0]-d['time_s'][i,0])-.04)<1e-9
  receipt_count=len(v['indices'])
result={'owner_freeze_sha256':sha(p/'FREEZE_SHA256.json'),'owner_files_verified':len(m),'reference_rows_exact':len(timing),'capture_receipt_pairs_exact40ms':receipt_count,'contacts_and_shared_motion_arrays_byte_equal_actual004':True,
 'reviewed_sources_sha256':{f:sha(p/f) for f in ['prepare_inputs.py','replay_checker.py','footprint_checker.py']},'scope':'Independent source/input frame-time-contact review; no sensor, terrain, physical abort or actor qualification'}
print(json.dumps(result,indent=2))
