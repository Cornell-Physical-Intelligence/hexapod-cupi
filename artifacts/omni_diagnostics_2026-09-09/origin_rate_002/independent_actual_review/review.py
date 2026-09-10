"""Read-only independent recomputation of exact origin002 actual evidence."""
from pathlib import Path
import ast,hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parent;TMP=ROOT.parent;INPUT=TMP/'reference_origin_results_002';RUN=INPUT/'run'
SOURCE=TMP/'reference_origin_adapter_002/source_origin_002'
CASES={'origin_a':(0.,0.),'near':(3.,-5.),'far':(30.,-50.),'near_opposite':(-3.,5.),'origin_repeat':(0.,0.)}
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as d:return {k:d[k] for k in d.files}
def equal(a,b):
 a=np.asarray(a);b=np.asarray(b)
 return bool(np.array_equal(a,b,equal_nan=True)) if a.dtype.kind in 'fc' else bool(np.array_equal(a,b))
def physical(t):
 post=slice(200,None)
 return dict(controls=len(t['time_s']),replicas=t['time_s'].shape[1],terminations=int(t['terminated'].sum()),truncations=int(t['truncated'].sum()),min_support=int(t['distal_contact'][post].sum(-1).min()),nonfoot_env_steps=int(np.logical_or.reduce([t[k][post].any(-1) if t[k].ndim==3 else t[k][post] for k in ('shaft_contact','coxa_contact','femur_contact','base_contact')]).sum()),requested_peak_nm=float(np.abs(t['computed_torque_nm'][post]).max()),applied_peak_nm=float(np.abs(t['applied_torque_nm'][post]).max()),max_joint_saturation_fraction=float((np.abs(t['computed_torque_nm'][post])>1.6).mean(0).max()),target_lag_rad=float(np.abs(t['reference_to_executable_lag_rad']).max()),zero_residual_actions=bool(not t['raw_residual_action'].any()),zero_requested_twist=bool(not t['requested_command'].any()))
def quiet_fn():
 # Re-execute the unchanged scoring function/constant without Isaac imports.
 tree=ast.parse((SOURCE/'tools/omni_quiet_review.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='quiet_metrics' or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='QUIET_GATES' for t in n.targets)]
 ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned unchanged quiet scorer>','exec'),ns);return ns['quiet_metrics'],ns['QUIET_GATES']
def main():
 audit=read(INPUT/'remote_audit.json');assert len(audit['raw_payloads'])==96
 for path,value in audit['raw_payloads'].items():assert sha(INPUT/path)==value,path
 manifest=read(SOURCE/'campaign_source_hashes.json');assert len(manifest)==933
 assert sha(SOURCE/'campaign_source_hashes.json')==audit['source_manifest_sha256']
 for path,value in manifest.items():assert sha(SOURCE/path)==value,path
 campaign=read(RUN/'campaign.json');assert campaign['completed_cases']==list(CASES) and campaign['status']=='completed'
 scorer,bounds=quiet_fn();standing=load(RUN/'standing/trace.npz');stand=read(RUN/'standing/state.json');sg=stand['gate'];sp=physical(standing)
 scores=[scorer(standing,i,200,standing['joint_names'].tolist(),.02) for i in range(32)]
 assert all(s['pass'] and s['window_duration_s']>=10 for s in scores) and sg['passed']
 assert sp['min_support']==6 and sp['nonfoot_env_steps']==sp['terminations']==sp['truncations']==0 and sp['max_joint_saturation_fraction']==0
 initial0=read(RUN/'origin_a/initial_readback.json');cold=read(SOURCE/'tools/origin_initial_state.json');rows=[];arrays={};ratearrays={}
 for name,xy in CASES.items():
  p=RUN/name;t=load(p/'trace.npz');d=load(p/'physics_substeps.npz');arrays[name]=t;ratearrays[name]=d
  state=read(p/'state.json');r=read(p/'initial_readback.json');sync=r['origin_storage_sync'];reported=read(p/'rate_discrepancy.json');gate=state['unchanged_physical_gate'];quiet=state['unchanged_quiet_gate']
  assert state['status']=='completed' and state['control_steps']==1000 and state['failure'] is None
  assert r['passed'] and all(r['checks'].values()) and sync['passed'] and all(sync['checks'].values())
  assert sync['terrain_and_scene_same_view_storage'] is False
  for field in ('terrain_after_m','scene_after_m'):assert equal(sync[field],[[*xy,0.]])
  for field in ('terrain','scene'):assert equal(np.asarray(sync[field+'_before_m'])[:,2],np.asarray(sync[field+'_after_m'])[:,2])
  assert equal(np.asarray(sync['default_pose_before'])[:,2:],np.asarray(sync['default_pose_after'])[:,2:])
  assert r['reset_injections']==2 and r['body_pose_written_after_first_step'] is False
  for key in r['actual']:assert equal(r['actual'][key],r['expected'][key]),(name,key)
  assert r['expected']['names']==cold['joint_names_runtime']
  for key,rawkey in [('q','joint_position_rad'),('v','joint_velocity_rad_s'),('target','joint_target_rad')]:assert equal(r['actual'][key],[cold['initial'][rawkey]])
  cp=np.r_[cold['initial']['root_link_position_relative_m'],cold['initial']['root_link_quaternion_world_xyzw']].astype(np.float32);cp[:2]+=np.asarray(xy,np.float32)
  assert equal(r['actual']['root_pose'],cp[None])
  for key in ('q','v','target','root_com_velocity'):assert equal(r['actual'][key],initial0['actual'][key]),(name,key)
  assert equal(np.asarray(r['actual']['root_pose'])[:,2:],np.asarray(initial0['actual']['root_pose'])[:,2:])
  assert equal(np.asarray(r['actual']['root_pose'])[:,:2],np.asarray([xy],np.float32))
  for tk,rk in [('joint_position_rad','q'),('joint_velocity_rad_s','v'),('root_link_position_world_m','root_pose')]:
   expected=np.asarray(r['actual'][rk]);expected=expected[:,:3] if rk=='root_pose' else expected
   assert equal(d[tk][0],expected),(name,tk)
  assert equal(d['root_link_quaternion_world_xyzw'][0],np.asarray(r['actual']['root_pose'])[:,3:])
  ground=read(p/'ground_readback.json');assert ground['passed'] and len(ground['external_colliders'])==1
  plane=ground['external_colliders'][0];assert plane['type']=='Plane' and plane['collision_enabled'] and equal(plane['world_normal'],[0,0,1]) and plane['world_origin_m'][2]==0
  assert d['joint_position_rad'].shape==(8001,1,18) and t['joint_position_rad'].shape==(1000,1,18)
  np.testing.assert_array_equal(d['relative_physics_index'],np.arange(8001));np.testing.assert_array_equal(d['control_index'],np.r_[-1,np.repeat(np.arange(1000),8)]);np.testing.assert_array_equal(d['substep_index'],np.r_[0,np.tile(np.arange(1,9),1000)])
  np.testing.assert_allclose(d['time_s'],np.arange(8001)*.0025,rtol=0,atol=1e-12)
  for key in ('joint_position_rad','joint_velocity_rad_s','computed_torque_nm','applied_torque_nm'):assert equal(d[key][8::8],t[key]),(name,key)
  assert equal(d['root_link_position_world_m'][8::8],t['position_world_m'])
  assert equal(d['root_link_quaternion_world_xyzw'][8::8],t['quaternion_world_xyzw'])
  assert equal(t['quaternion_world_xyzw'][...,[3,0,1,2]],t['quaternion_world_wxyz'])
  ph=physical(t);assert ph['min_support']==6 and ph['nonfoot_env_steps']==ph['terminations']==ph['truncations']==0 and ph['max_joint_saturation_fraction']==0
  assert ph['requested_peak_nm']==gate['post_settle_max_requested_torque_nm'] and gate['passed'] and state['all_existing_bounds_met']
  qscore=scorer(t,0,200,t['joint_names'].tolist(),.02)
  for key,value in qscore.items():assert value==quiet[key],(name,key)
  assert qscore['pass'] and qscore['window_duration_s']==16
  # Independent quadrature over true400Hz boundary samples4.00...20.00s.
  dt=np.diff(d['time_s'][1600:]);q=d['joint_position_rad'][1600:,0].astype(float);v=d['joint_velocity_rad_s'][1600:,0].astype(float);delta=q[-1]-q[0]
  integrals={'left':np.sum(v[:-1]*dt[:,None],axis=0),'right':np.sum(v[1:]*dt[:,None],axis=0),'trapezoid':np.sum((v[:-1]+v[1:])*.5*dt[:,None],axis=0)};err=integrals['trapezoid']-delta;fd=np.diff(q,axis=0)/dt[:,None]
  jointrows=[]
  for j,joint in enumerate(d['joint_names'].tolist()):
   old=reported['joints'][j];assert old['joint']==joint
   for method,data in integrals.items():np.testing.assert_allclose(data[j],old['reported_rate_integral_rad'][method],rtol=0,atol=2e-13)
   np.testing.assert_allclose(delta[j],old['angle_delta_rad'],rtol=0,atol=0)
   jointrows.append({'joint':joint,'delta_rad':float(delta[j]),'range_rad':float(np.ptp(q[:,j])),'integral_rad':{k:float(x[j]) for k,x in integrals.items()},'trapezoid_error_rad':float(err[j]),'reported_rms_rad_s':float(np.sqrt(np.mean(v[1:,j]**2))),'angle_interval_rms_rad_s':float(np.sqrt(np.mean(fd[:,j]**2)))})
  frames={}
  for frame in ('link','com'):
   pos=d[f'root_{frame}_position_world_m'][1600:,0].astype(float);vel=d[f'root_{frame}_velocity_world_mps'][1600:,0].astype(float);displacement=pos[-1]-pos[0];iv=np.sum((vel[:-1]+vel[1:])*.5*dt[:,None],axis=0);error=float(np.linalg.norm(iv-displacement))
   np.testing.assert_allclose(error,reported['root_frames'][frame]['displacement_difference_norm_m']['trapezoid'][0],rtol=0,atol=2e-14)
   frames[frame]={'displacement_m':displacement.tolist(),'trapezoid_integral_m':iv.tolist(),'error_norm_m':error,'range_m':np.ptp(pos,axis=0).tolist()}
  quat=d['root_link_quaternion_world_xyzw'][1600:,0].astype(float);rot=Rotation.from_quat(quat);angle=(rot[1:]*rot[:-1].inv()).as_rotvec().sum(0);omega=d['root_angular_velocity_world_rad_s'][1600:,0].astype(float);angle_iv=np.sum((omega[:-1]+omega[1:])*.5*dt[:,None],axis=0)
  np.testing.assert_allclose(angle,reported['world_angular']['sum_actual_increment_rotation_vectors_rad'][0],rtol=0,atol=2e-14)
  derived=np.asarray(r['body_positions_world_m'])-np.array([*xy,0.]);bodydiff=float(np.abs(derived-np.asarray(initial0['body_positions_world_m'])).max())
  worst=int(np.argmax(abs(err)))
  rows.append(dict(case=name,xy_m=xy,initial_generalized_nonXY_and_target_exact=True,origin_storage_checks_passed=True,separate_scene_terrain_storage=True,derived19body_local_position_max_difference_m=bodydiff,physical=ph,quiet={k:v for k,v in qscore.items() if k!='joints'},postsettle400Hz_requested_peak_nm=float(np.abs(d['computed_torque_nm'][1601:]).max()),postsettle400Hz_applied_peak_nm=float(np.abs(d['applied_torque_nm'][1601:]).max()),worst_joint=jointrows[worst],joints=jointrows,root_frames=frames,world_angular_increment_sum_rad=angle.tolist(),world_angular_rate_integral_rad=angle_iv.tolist(),initial_substep_readback_exact=True,control_substep_endpoints_exact=True,all_physical_quiet_bounds_met=True))
 repeat={'control_fields':{k:equal(v,arrays['origin_repeat'][k]) for k,v in arrays['origin_a'].items()},'substep_fields':{k:equal(v,ratearrays['origin_repeat'][k]) for k,v in ratearrays['origin_a'].items()}}
 for name in CASES:
  for key in ('joint_target_rad','reference_position_rad','raw_residual_action','requested_command'):assert equal(arrays[name][key],arrays['origin_a'][key]),(name,key)
 base=np.array([j['trapezoid_error_rad'] for j in rows[0]['joints']]);effects=[]
 for r in rows:
  e=np.array([j['trapezoid_error_rad'] for j in r['joints']]);effects.append({'case':r['case'],'max_abs_paired_change_rad':float(abs(e-base).max()),'max_error_rad':float(abs(e).max()),'max_error_ratio_to_origin':float(abs(e).max()/abs(base).max())})
 result={'scope':'Independent actual five-case origin002 review; no new threshold/native mechanism/velocity admission','input_raw_payloads_verified':96,'local_source_files_verified':933,'all_five_executed_joint_targets_exact':True,'source_manifest_sha256':audit['source_manifest_sha256'],'remote_audit_sha256':sha(INPUT/'remote_audit.json'),'root_reported_assets_verified':audit['admitted_asset_files'],'standing':{'physical':sp,'all32_quiet_recomputed_pass':True,'quiet_bounds':bounds,'max_reported_joint_rms_rad_s':max(s['max_joint_velocity_rms_rad_s'] for s in scores)},'cases':rows,'origin_repeat_exact_array_equality':repeat,'paired_effects':effects,'native_cause_established':False,'velocity_fidelity_qualified':False,'GPU_launches':0}
 (ROOT/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'raw':96,'source':933,'all32_standing_quiet':True,'repeat_control_exact':all(repeat['control_fields'].values()),'repeat_substep_exact':all(repeat['substep_fields'].values()),'paired_effects':effects},indent=2))
 for r in rows:print(r['case'],'tau',r['physical']['requested_peak_nm'],'rms',r['quiet']['max_joint_velocity_rms_rad_s'],'delta',r['worst_joint']['delta_rad'],'int',r['worst_joint']['integral_rad']['trapezoid'],'err',r['worst_joint']['trapezoid_error_rad'],'bodyquant',r['derived19body_local_position_max_difference_m'])
if __name__=='__main__':main()
