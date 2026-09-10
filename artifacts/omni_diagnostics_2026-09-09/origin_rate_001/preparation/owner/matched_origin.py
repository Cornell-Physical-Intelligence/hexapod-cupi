"""Install a common cold initial state inside reset only; no step-time pose control."""
from contextlib import contextmanager
import numpy as np
from origin_contract import CASES
from physics_telemetry import array


def expected_initial(payload,case):
 if case not in CASES:raise ValueError('Unknown matched-origin case')
 initial=payload['initial'];names=tuple(payload['joint_names_runtime'])
 if len(names)!=18 or len(set(names))!=18:raise ValueError('18 unique named joints required')
 q=np.asarray(initial['joint_position_rad'],dtype=np.float32)[None,:]
 v=np.asarray(initial['joint_velocity_rad_s'],dtype=np.float32)[None,:]
 target=np.asarray(initial['joint_target_rad'],dtype=np.float32)[None,:]
 pose=np.r_[initial['root_link_position_relative_m'],initial['root_link_quaternion_world_xyzw']].astype(np.float32)[None,:]
 pose[0,:2]+=np.asarray(CASES[case],dtype=np.float32)
 velocity=np.r_[initial['root_com_velocity_world_mps'],initial['root_angular_velocity_world_rad_s']].astype(np.float32)[None,:]
 if q.shape!=(1,18) or v.shape!=(1,18) or target.shape!=(1,18) or pose.shape!=(1,7) or velocity.shape!=(1,6):raise ValueError('Malformed matched reset arrays')
 if any(not np.isfinite(x).all() for x in (q,v,target,pose,velocity)):raise ValueError('Nonfinite matched reset')
 if not np.array_equal(q,target) or np.any(v) or np.any(velocity):raise ValueError('Selected009 cold reset requires exact q=target and zero velocities')
 if not np.array_equal(pose[0,3:],np.array([0,0,0,1],dtype=np.float32)):raise ValueError('Selected009 uprightXYZW convention changed')
 return dict(names=names,q=q,v=v,target=target,root_pose=pose,root_com_velocity=velocity)


def matched_class(base,payload,case):
 expected=expected_initial(payload,case)
 class MatchedOriginEnvironment(base):
  def _reset_idx(self,env_ids):
   super()._reset_idx(env_ids)
   if not hasattr(self,'reference_residual_controller'):return
   if self.num_envs!=1 or tuple(self._robot.joint_names)!=expected['names']:raise RuntimeError('Exact single-replica runtime joint order required')
   if int(self._sim_step_counter)!=0:
    self.matched_origin_unexpected_reset=True
    return  # Parent handles terminal reset; never inject another diagnostic pose.
   import torch
   ids=torch.arange(1,device=self.device)
   tensor=lambda a:torch.as_tensor(a,device=self.device,dtype=torch.float32)
   q=tensor(expected['q']);v=tensor(expected['v']);pose=tensor(expected['root_pose']);velocity=tensor(expected['root_com_velocity'])
   # All writes occur inside the cold reset, before the normal reset forward/readback.
   self._terrain.env_origins[ids,:2]=tensor(np.asarray(CASES[case])[None,:])
   self._robot.write_root_pose_to_sim_index(root_pose=pose,env_ids=ids)
   self._robot.write_root_velocity_to_sim_index(root_velocity=velocity,env_ids=ids)
   self._robot.write_joint_position_to_sim_index(position=q,env_ids=ids)
   self._robot.write_joint_velocity_to_sim_index(velocity=v,env_ids=ids)
   self._robot.set_joint_position_target_index(target=q,env_ids=ids)
   self._robot.set_joint_velocity_target_index(target=torch.zeros_like(q),env_ids=ids)
   self.reference_residual_controller.reset(q,q,env_ids=ids)
   for name in ('_previous_processed_joint_target','_processed_actions','omni_filtered_target'):
    getattr(self,name)[ids]=q
   self._has_previous_processed_joint_target[ids]=True
   self._reference_fresh[ids]=False
   self.reference_residual_target=None;self.reference_residual_sample=None
   self.matched_origin_reset_injections=getattr(self,'matched_origin_reset_injections',0)+1
 return MatchedOriginEnvironment


@contextmanager
def patched_factory(payload,case):
 import reference_residual_env as module
 original=module.ReferenceResidualPhysicsEnv
 module.ReferenceResidualPhysicsEnv=matched_class(original,payload,case)
 try:yield
 finally:module.ReferenceResidualPhysicsEnv=original


def initial_readback(env,payload,case):
 expected=expected_initial(payload,case);d=env._robot.data
 actual={'q':array(d.joint_pos),'v':array(d.joint_vel),
  'root_pose':np.concatenate([array(d.root_link_pos_w),array(d.root_link_quat_w)],axis=-1),
  'root_com_velocity':np.concatenate([array(d.root_com_lin_vel_w),array(d.root_link_ang_vel_w)],axis=-1),
  'target':array(env.reference_residual_controller.reference_position).astype(np.float32)}
 checks={key:bool(np.array_equal(value,expected[key])) for key,value in actual.items()}
 checks['articulation_position_target']=np.array_equal(array(d.joint_pos_target),expected['target'])
 checks['articulation_velocity_target_zero']=not array(d.joint_vel_target).any()
 checks['runtime_names']=tuple(env._robot.joint_names)==expected['names']
 checks['before_first_physics_step']=int(env._sim_step_counter)==0
 checks['scene_origin_alias']=env.scene.env_origins is env._terrain.env_origins
 checks['scene_origins']=np.array_equal(array(env.scene.env_origins),np.array([[*CASES[case],0]],dtype=np.float32))
 checks['commands_zero']=not array(env._commands).any()
 checks['reference_velocity_zero']=not array(env.reference_residual_controller.reference_velocity).any()
 checks['residual_position_zero']=not array(env.reference_residual_controller.residual_position).any()
 checks['residual_velocity_zero']=not array(env.reference_residual_controller.residual_velocity).any()
 body_positions=array(d.body_link_pos_w)
 body_quaternions=array(d.body_link_quat_w)
 checks['body_geometry_finite']=bool(body_positions.shape==(1,19,3) and body_quaternions.shape==(1,19,4) and np.isfinite(body_positions).all() and np.isfinite(body_quaternions).all())
 result={'case':case,'checks':checks,'passed':all(checks.values()),'actual':{k:v.tolist() for k,v in actual.items()},
  'expected':{k:(list(v) if k=='names' else v.tolist()) for k,v in expected.items()},'actual_scene_origins_m':array(env.scene.env_origins).tolist(),
  'body_names':list(env._robot.body_names),'body_positions_world_m':body_positions.tolist(),'body_quaternion_world_xyzw':body_quaternions.tolist(),
  'reset_injections':getattr(env,'matched_origin_reset_injections',0),'body_pose_written_after_first_step':False}
 return result


def ground_readback(env):
 """Require exactly one collision Plane with world normal+Z and height0.

 Installed GroundPlane spawner explicitly locates type Plane for physics material;
 the visible grid mesh has separate scale and is not the collision surface.
 """
 from pxr import Usd,UsdGeom,UsdPhysics,Gf
 stage=env.sim.stage;rows=[]
 for prim in stage.Traverse():
  if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
  path=str(prim.GetPath())
  if '/Robot/' in path:continue
  mat=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
  axis=prim.GetAttribute('axis').Get();basis={'X':Gf.Vec3d(1,0,0),'Y':Gf.Vec3d(0,1,0),'Z':Gf.Vec3d(0,0,1)}.get(str(axis))
  normal=None if basis is None else np.asarray(mat.TransformDir(basis),dtype=float)
  if normal is not None:normal/=np.linalg.norm(normal)
  origin=np.asarray(mat.Transform(Gf.Vec3d(0,0,0)),dtype=float)
  rows.append({'path':path,'type':prim.GetTypeName(),'axis':str(axis),'world_origin_m':origin.tolist(),
   'world_normal':None if normal is None else normal.tolist(),'collision_enabled':UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()})
 passed=validate_ground_rows(rows,env._terrain.cfg.terrain_type)
 return {'passed':bool(passed),'terrain_type':env._terrain.cfg.terrain_type,'external_colliders':rows,
  'translation_equivalence':'Infinite plane normal+Z atZ0 is invariant under all declaredXY translations; finite visible grid is not support geometry'}

def validate_ground_rows(rows,terrain_type):
 return (terrain_type=='plane' and len(rows)==1 and rows[0]['type']=='Plane'
  and rows[0]['collision_enabled'] is True and rows[0]['world_normal'] is not None
  and np.allclose(rows[0]['world_normal'],[0,0,1],rtol=0,atol=1e-12) and abs(rows[0]['world_origin_m'][2])<1e-12)
