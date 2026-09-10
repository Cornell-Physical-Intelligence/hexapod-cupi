"""Exact actual standing001 contact failure decomposition; CPU/read-only raw inputs."""
from pathlib import Path
import hashlib,importlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];RAW=ROOT/'tmp/canonical_native_standing_terminal_001';D=RAW/'run/standing';SOURCE=ROOT/'tmp/updated_native_standing_001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
audit=json.loads((RAW/'audit.json').read_text());assert audit['audit_verified']and audit['terminal_outcome']=='authentic_terminal_failure'
for rel,v in audit['raw_inventory'].items():assert sha(RAW/rel)==v['sha256'] and (RAW/rel).stat().st_size==v['size_bytes']
manifest=json.loads((SOURCE/'FREEZE_SHA256.json').read_text());assert len(manifest)==40 and all(sha(SOURCE/k)==v for k,v in manifest.items())
sys.path.insert(0,str(SOURCE));mod=importlib.import_module('standing_math')
with np.load(D/'failed_contact_buffer.npz')as f:z={k:f[k]for k in f.files}
cv=json.loads((D/'contact_view.json').read_text());session=json.loads((D/'session.json').read_text());partial=json.loads((D/'failed_partial_step.json').read_text());poses=np.array(partial['link_pose_xyzw']);names=session['body_names']
with np.load(SOURCE/'geometry/geometry_extrema.npz')as f:geo=mod.Geometry(json.loads((SOURCE/'geometry/geometry.json').read_text()),{k:f[k]for k in f.files},names)
sensor_map=[(0,p.rsplit('/',1)[1])for p in cv['sensor_paths']];data=[z[k]for k in ['force','point','normal','separation','counts','starts']]
try:mod.classify_contacts(data,sensor_map,poses,geo,1);raise AssertionError('Frozen failure did not reproduce')
except ValueError as e:assert str(e)=='Invalid patch normal'
rows=[];ranges=[]
for i,(count,start)in enumerate(zip(z['counts'][:,0],z['starts'][:,0])):
 if count:ranges.append({'sensor_path':cv['sensor_paths'][i],'start':int(start),'count':int(count)})
 for k in range(int(start),int(start+count)):
  f=float(z['force'][k,0]);point=z['point'][k].astype(float);normal=z['normal'][k].astype(float);sep=float(z['separation'][k,0]);body=sensor_map[i][1]
  norm=float(np.linalg.norm(normal));inactive=f==0 and np.array_equal(normal,np.zeros(3))and sep==0
  cap,local=geo.cap(body,point,poses[0,names.index(body)])
  rows.append({'sensor_path':cv['sensor_paths'][i],'index':k,'force_n':f,'normal_norm':norm,'normal':normal.tolist(),'point_world_m':point.tolist(),'separation_m':sep,'exact_zero_tuple':inactive,'cap_geometry':cap,'shape_point_m':local.tolist()})
invalid=[r for r in rows if abs(r['normal_norm']-1)>1e-3];nonzero=[r for r in rows if r['force_n']!=0];zero_valid=[r for r in rows if r['force_n']==0 and abs(r['normal_norm']-1)<=1e-3]
assert len(rows)==153 and len({r['index']for r in rows})==153 and len(invalid)==127 and all(r['exact_zero_tuple']for r in invalid)
force_by_body={}
for r in rows:force_by_body.setdefault(r['sensor_path'],np.zeros(3));force_by_body[r['sensor_path']]+=r['force_n']*np.asarray(r['normal'])
with np.load(D/'substeps_000.npz')as f:prior={k:f[k]for k in f.files}
result={'scope':'Actual failed acquisition analysis; no native rerun, threshold change or physical standing pass','source_freeze_sha256':sha(SOURCE/'FREEZE_SHA256.json'),'audit_sha256':sha(RAW/'audit.json'),'raw_payloads_verified':len(audit['raw_inventory']),'failed_buffer_sha256':sha(D/'failed_contact_buffer.npz'),'native_api_sha256':'b913fdb1a3c0cf04b62aa01d6ceaf7d96cfe577a510d0e02b71485018661c903','old_classifier_failure_reproduced':'Invalid patch normal','actual_physics_steps':session['steps'],'complete_captured_steps':session['captured_steps'],'complete_controls':session['controls'],'failed_explicit_counter':partial['actual_explicit_counter'],'counted_slots':len(rows),'capacity':cv['capacity'],'disjoint_ranges':ranges,'invalid_normals':len(invalid),'invalid_first_last_indices':[invalid[0]['index'],invalid[-1]['index']],'all_invalid_exact_zero_force_normal_separation':True,'invalid_all_cap_geometry':all(r['cap_geometry']for r in invalid),'invalid_unique_points':len({tuple(r['point_world_m'])for r in invalid}),'nonzero_force_slots':len(nonzero),'zero_force_valid_normal_slots':len(zero_valid),'nonzero_normal_max_abs_unit_error':max(abs(r['normal_norm']-1)for r in nonzero),'nonzero_invalid_normals':0,'all_used_finite':all(np.isfinite([r['force_n'],*r['normal'],*r['point_world_m'],r['separation_m']]).all()for r in rows),'force_sum_world_by_tibia_n':{k:v.tolist()for k,v in force_by_body.items()},'failed_root_plate_height_m':float(np.asarray(partial['root_pose_xyzw'])[0,2]),'failed_max_abs_q_rad':float(abs(np.asarray(partial['joint_position_rad'])).max()),'failed_max_abs_sdk_dq_rad_s':float(abs(np.asarray(partial['joint_velocity_rad_s'])).max()),'prior_requested_peak_nm':float(abs(prior['computed_torque_nm']).max()),'prior_applied_peak_nm':float(abs(prior['applied_torque_nm']).max()),'proposal':'Only an exact finite tuple force=0, normal=(0,0,0), separation=0 may be recorded as inactive zero-normal evidence; retain indices/point/body/category, add no support, preserve all raw patches and reject every nonzero-force invalid normal or other malformed tuple at existing tolerance. No native cause for the zero tuples is established.'}
print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
