"""Explicit native XYZW/world-to-body conversion, then proper forward-left-up map."""
import numpy as np
MAP=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])

def rotations_xyzw(quaternion):
 q=np.asarray(quaternion,dtype=np.float64)
 if q.ndim!=2 or q.shape[1]!=4 or not np.isfinite(q).all()or np.any(abs((q*q).sum(1)-1)>2e-5):raise ValueError('Expected unit native XYZW quaternions')
 x,y,z,w=q.T
 return np.stack([np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],1),
  np.stack([2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],1),
  np.stack([2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],1)],1)

def world_to_native_body(vectors,quaternion):
 v=np.asarray(vectors,dtype=np.float64);r=rotations_xyzw(quaternion)
 if v.shape!=(len(r),3)or not np.isfinite(v).all():raise ValueError('Wrong world vectors')
 return np.einsum('nji,nj->ni',r,v)

def native_to_navigation(vectors):
 v=np.asarray(vectors,dtype=np.float64)
 if v.shape[-1]!=3 or not np.isfinite(v).all():raise ValueError('Wrong native body vectors')
 return v@MAP.T

def navigation_to_native(vectors):
 v=np.asarray(vectors,dtype=np.float64)
 if v.shape[-1]!=3 or not np.isfinite(v).all():raise ValueError('Wrong navigation vectors')
 return v@MAP

def body_velocity_at_root_origin(root_com_velocity_world,root_pose_xyzw,root_com_local):
 """Native root velocity is COM linear+world angular; move its origin explicitly."""
 velocity=np.asarray(root_com_velocity_world,dtype=np.float64);pose=np.asarray(root_pose_xyzw,dtype=np.float64)
 if velocity.shape!=(len(pose),6)or pose.shape[1:]!=(7,):raise ValueError('Wrong root velocity/pose')
 r=rotations_xyzw(pose[:,3:]);local=np.broadcast_to(np.asarray(root_com_local,dtype=np.float64),(len(pose),3))
 lever=np.einsum('nij,nj->ni',r,local)
 root_velocity=velocity[:,:3]-np.cross(velocity[:,3:],lever)
 return world_to_native_body(root_velocity,pose[:,3:]),world_to_native_body(velocity[:,3:],pose[:,3:])
