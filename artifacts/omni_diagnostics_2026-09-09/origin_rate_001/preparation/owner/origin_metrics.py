"""All named angle/rate and link/COM discrepancy evidence; no replacement gate."""
import numpy as np

def compare_measurements(d,case_xy,start=200,end=1000):
 lo,hi=start*8,end*8
 if len(d['time_s'])!=end*8+1:raise ValueError('Complete1000-control400Hz acquisition required')
 np.testing.assert_array_equal(d['relative_physics_index'],np.arange(end*8+1))
 np.testing.assert_array_equal(d['control_index'],np.r_[-1,np.repeat(np.arange(end),8)])
 np.testing.assert_array_equal(d['substep_index'],np.r_[0,np.tile(np.arange(1,9),end)])
 dt=np.diff(d['time_s'][lo:hi+1]);np.testing.assert_allclose(dt,.0025,rtol=0,atol=1e-12)
 q=d['joint_position_rad'][lo:hi+1].astype(float);v=d['joint_velocity_rad_s'][lo:hi+1].astype(float);delta=q[-1]-q[0]
 fd=np.diff(q,axis=0)/dt[:,None,None]
 integ=lambda v:{'left':np.einsum('t,tej->ej',dt,v[:-1]),'right':np.einsum('t,tej->ej',dt,v[1:]),'trapezoid':np.einsum('t,tej->ej',dt,.5*(v[:-1]+v[1:]))}
 ints=integ(v);rows=[]
 for j,name in enumerate(d['joint_names'].tolist()):
  rows.append(dict(joint=name,angle_delta_rad=float(delta[0,j]),angle_range_rad=float(np.ptp(q[:,0,j])),reported_rate_integral_rad={k:float(x[0,j]) for k,x in ints.items()},reported_integral_minus_delta_rad={k:float(x[0,j]-delta[0,j]) for k,x in ints.items()},reported_rate_rms_rad_s=float(np.sqrt(np.mean(v[1:,0,j]**2))),angle_increment_interval_rate_rms_rad_s=float(np.sqrt(np.mean(fd[:,0,j]**2))),interval_disagreement_rms_rad_s=float(np.sqrt(np.mean((fd[:,0,j]-.5*(v[1:,0,j]+v[:-1,0,j]))**2)))))
 frames={}
 for frame in ('link','com'):
  raw=d[f'root_{frame}_position_world_m'][lo:hi+1];p=raw.astype(float)-np.array([*case_xy,0.])[None,None,:]
  vel=d[f'root_{frame}_velocity_world_mps'][lo:hi+1].astype(float);distance=p[-1]-p[0];iv=integ(vel)
  frames[frame]={'local_displacement_m':distance.tolist(),'position_range_m':np.ptp(p,axis=0).tolist(),
   'reported_velocity_integrals_m':{k:x.tolist() for k,x in iv.items()},'displacement_difference_norm_m':{k:np.linalg.norm(x-distance,axis=-1).tolist() for k,x in iv.items()},
   'raw_position_dtype':str(raw.dtype),'max_abs_raw_world_position_m':np.max(np.abs(raw),axis=0).tolist(),
   'max_abs_float32_spacing_m':np.max(np.abs(np.spacing(raw.astype(np.float32))),axis=0).tolist(),
   'offset_subtraction':'float64 offline after retaining original float32 samples; does not recover lost precision'}
 # World angular increment: q_new * conjugate(q_old), using actual SDKXYZW.
 quat=d['root_link_quaternion_world_xyzw'][lo:hi+1].astype(float);norm=np.linalg.norm(quat,axis=-1,keepdims=True);quat/=norm
 a,b=quat[1:],quat[:-1]
 vec=a[...,3,None]*(-b[...,:3])+b[...,3,None]*a[...,:3]+np.cross(a[...,:3],-b[...,:3])
 w=a[...,3]*b[...,3]+np.sum(a[...,:3]*b[...,:3],axis=-1)
 sign=np.where(w<0,-1.,1.);vec*=sign[...,None];w*=sign
 length=np.linalg.norm(vec,axis=-1);angle=2*np.arctan2(length,w)
 rotvec=vec*np.divide(angle,length,out=np.zeros_like(angle),where=length>1e-15)[...,None]
 angular=d['root_angular_velocity_world_rad_s'][lo:hi+1].astype(float)
 angular_fd=rotvec/dt[:,None,None]
 return {'control_boundaries':[start,end],'duration_s':float(dt.sum()),'joint_names_runtime':d['joint_names'].tolist(),'joints':rows,'root_frames':frames,
  'world_angular':{'quaternion_norm_max_error':float(np.abs(norm-1).max()),'sum_actual_increment_rotation_vectors_rad':rotvec.sum(0).tolist(),
   'reported_angular_velocity_integrals_rad':{k:x.tolist() for k,x in integ(angular).items()},'angle_increment_interval_rate_rms_rad_s':np.sqrt(np.mean(angular_fd**2,axis=0)).tolist(),
   'reported_rate_rms_rad_s':np.sqrt(np.mean(angular[1:]**2,axis=0)).tolist(),'interval_disagreement_rms_rad_s':np.sqrt(np.mean((angular_fd-.5*(angular[1:]+angular[:-1]))**2,axis=0)).tolist(),
   'semantics':'World-frame local quaternion increments; their vector sum is a diagnostic, not a finite-rotation composition substitute'},
  'velocity_measurement_qualified':False,'new_acceptance_threshold':None,'native_cause_established':False}
