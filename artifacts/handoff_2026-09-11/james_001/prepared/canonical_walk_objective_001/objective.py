"""Proposed reward mathematics only. Native safety/quiet gates are independent."""
from dataclasses import dataclass
import numpy as np

SCHEMA='canonical_omni_reward_proposal_v1'


@dataclass(frozen=True)
class Weights:
 planar_tracking:float=5.
 yaw_tracking:float=2.
 tilt:float=.2
 roll_pitch_rate:float=.05
 vertical_rate:float=.05
 applied_effort:float=.02
 requested_excess:float=.1
 target_delta:float=.02
 target_second_difference:float=.01
 requested_target_excess:float=.05
 contact_slip:float=.05
 zero_command_angle_rate:float=.05
 zero_command_target_delta:float=.05
 terminal:float=20.


def bounded_square(x):
 """Smooth bounded cost with no hard reward clipping and nonzero finite tails."""
 with np.errstate(over='ignore'):return 1.-1./(1.+np.asarray(x,float)**2)


def terminal_margin(weights=Weights(),dt=.02,gamma=.99):
 costs=sum(v for k,v in vars(weights).items()if k not in ['planar_tracking','yaw_tracking','terminal'])
 # Worst infinite discounted negative cost plus largest terminal-step positive credit.
 required=dt*costs/(1-gamma)+dt*(weights.planar_tracking+weights.yaw_tracking)
 return {'maximum_cost_rate':costs,'minimum_discounted_continuing_return':-dt*costs/(1-gamma),
         'strict_terminal_penalty_lower_bound':required,'declared_terminal_penalty':weights.terminal,'sufficient':weights.terminal>required}


def reward(command,linear_body,angular_body,gravity_body,computed_torque,applied_torque,target_delta,target_previous_delta,requested_minus_executed,
           interval_angle_rate,contact_tangent_speed,contact_active,valid_interval,terminated,truncated,*,dt=.02,weights=Weights()):
 """All velocities alreadyforward/left/up; command is operator/requested, never actor-reduced."""
 c=np.asarray(command,float)
 if any(not np.isfinite(v)or v<0 for v in vars(weights).values()):raise ValueError('Reward cost/benefit weights must be finite nonnegative')
 if c.ndim!=2 or c.shape[1]!=3 or not np.isfinite(c).all():raise ValueError('Malformed requested command')
 n=len(c)
 def array(x,shape,name):
  a=np.asarray(x,float)
  if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Malformed/nonfinite '+name)
  return a
 v=array(linear_body,(n,8,3),'eight body-root linear velocities');w=array(angular_body,(n,8,3),'eight body angular velocities');g=array(gravity_body,(n,8,3),'eight projected gravities')
 if not np.allclose(np.linalg.norm(g,axis=2),1.,atol=1e-6,rtol=0):raise ValueError('Projected gravity must be normalized')
 raw=array(computed_torque,(n,8,18),'all eight computed torque samples');applied=array(applied_torque,(n,8,18),'all eight applied torque samples')
 delta=array(target_delta,(n,18),'actual target delta');previous=array(target_previous_delta,(n,18),'previous actual delta');dq=array(interval_angle_rate,(n,18),'actual interval angle rate')
 excess=array(requested_minus_executed,(n,18),'requested minus executed target')
 slip=array(contact_tangent_speed,(n,8,6),'all eight material point slip samples')
 if np.any(slip<0):raise ValueError('Slip speed cannot be negative')
 active=np.asarray(contact_active);valid=np.asarray(valid_interval);term=np.asarray(terminated);timeout=np.asarray(truncated)
 for a,shape in [(active,(n,8,6)),(valid,(n,)),(term,(n,)),(timeout,(n,))]:
  if a.shape!=shape or a.dtype!=bool:raise ValueError('Explicit boolean mask required')
 if not valid.all():raise ValueError('Invalid interval cannot be replaced by zero-rate reward')
 if dt!=.02:raise ValueError('This proposal is explicitly50Hz; reversion required for another clock')
 if not terminal_margin(weights,dt)['sufficient']:raise ValueError('Terminal penalty does not dominate bounded negative continuation')
 sigma_v=np.maximum(.02,.5*np.linalg.norm(c[:,:2],axis=1));sigma_w=np.maximum(.05,.5*abs(c[:,2]))
 zero=np.all(c==0,axis=1)
 components={
  'planar_tracking':weights.planar_tracking*np.exp(-np.mean(np.sum((v[:,:,:2]-c[:,None,:2])**2,axis=2),axis=1)/sigma_v**2),
  'yaw_tracking':weights.yaw_tracking*np.exp(-np.mean((w[:,:,2]-c[:,None,2])**2,axis=1)/sigma_w**2),
  # g_xy^2 alone aliases an upside-down body to upright. This is |g-[0,0,-1]|^2.
  'tilt':-weights.tilt*np.mean(bounded_square(np.sqrt(np.maximum(2*(1+g[:,:,2]),0))/.15),axis=1),
  'roll_pitch_rate':-weights.roll_pitch_rate*np.mean(bounded_square(w[:,:,:2]/.25),axis=(1,2)),
  'vertical_rate':-weights.vertical_rate*np.mean(bounded_square(v[:,:,2]/.1),axis=1),
  'applied_effort':-weights.applied_effort*np.mean(bounded_square(applied/1.6),axis=(1,2)),
  'requested_excess':-weights.requested_excess*np.mean(bounded_square(np.maximum(abs(raw)-1.6,0)/1.6),axis=(1,2)),
  'target_delta':-weights.target_delta*np.mean(bounded_square(delta/.04),axis=1),
  'target_second_difference':-weights.target_second_difference*np.mean(bounded_square((delta-previous)/.04),axis=1),
  'requested_target_excess':-weights.requested_target_excess*np.mean(bounded_square(excess/.04),axis=1),
  'contact_slip':-weights.contact_slip*np.sum(bounded_square(slip/.05)*active,axis=(1,2))/(8*6),
  'zero_command_angle_rate':-weights.zero_command_angle_rate*np.mean(bounded_square(dq/.05),axis=1)*zero,
  'zero_command_target_delta':-weights.zero_command_target_delta*np.mean(bounded_square(delta/.002),axis=1)*zero}
 rates=sum(components.values());value=dt*rates-weights.terminal*term
 if not np.isfinite(value).all():raise ValueError('Nonfinite proposed reward')
 return {'reward':value,'component_rates':components,'termination_penalty':-weights.terminal*term,
  'bootstrap_allowed':~term,'timeout':timeout.copy(),'requested_command':c.copy(),'schema':SCHEMA,
  'physics_admission':False,'quality_admission':False}


def material_point_tangent_velocity(point_world,normal_world,previous_pose,current_pose,dt):
 """Finite interval velocity of the currentcontact MATERIAL point, not patch-location drift.

 Poses (...,4,4) map native linklocal coordinates to world. Normals areunitworldvectors.
 This finite interval is a diagnostic approximation, not instantaneous frictional power.
 """
 p=np.asarray(point_world,float);normal=np.asarray(normal_world,float);old=np.asarray(previous_pose,float);new=np.asarray(current_pose,float)
 if p.shape!=normal.shape or p.shape[-1]!=3 or old.shape!=p.shape[:-1]+(4,4)or new.shape!=old.shape:raise ValueError('Invalid material point frames')
 if not all(np.isfinite(a).all()for a in [p,normal,old,new])or not np.isfinite(dt)or dt<=0:raise ValueError('Invalid material point values/clock')
 if not np.allclose(np.linalg.norm(normal,axis=-1),1.,atol=1e-6,rtol=0):raise ValueError('Contact normal is not active/unit')
 for pose in [old,new]:
  rot=pose[...,:3,:3]
  if not np.allclose(np.swapaxes(rot,-1,-2)@rot,np.eye(3),atol=1e-6,rtol=0)or not np.allclose(np.linalg.det(rot),1.,atol=1e-6,rtol=0)or not np.allclose(pose[...,3,:],[0,0,0,1],atol=1e-12,rtol=0):raise ValueError('Pose is not a rigid transform')
 local=np.einsum('...ji,...j->...i',new[...,:3,:3],p-new[...,:3,3])
 old_world=np.einsum('...ij,...j->...i',old[...,:3,:3],local)+old[...,:3,3]
 velocity=(p-old_world)/dt
 return velocity-np.sum(velocity*normal,axis=-1,keepdims=True)*normal
