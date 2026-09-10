"""Explicit selected-row joint initialization; never callable as step-time pose control."""
from contextlib import contextmanager
from types import MethodType
import torch

def tensor(x):return x.torch if hasattr(x,'torch') else x

def arm_targets(plan, arm, names, nominal):
    names=tuple(names)
    if arm not in ('recorded_failed_pose','canonical_pose') or len(set(names))!=18 or set(names)!=set(plan['joint_names_runtime']):
        raise ValueError('Exact named18-joint A/B arm required')
    if nominal.shape!=(32,18) or not torch.isfinite(nominal).all():raise ValueError('Finite imported32-row default required')
    if arm=='canonical_pose':return nominal.to(torch.float32).clone()
    by_name=dict(zip(plan['joint_names_runtime'],plan['recorded_failure']['reset_q_rad']))
    return nominal.new_tensor([by_name[n] for n in names],dtype=torch.float32).expand(32,-1).clone()

class ResetPoseOverride:
    """Wrapper installed after env construction; caller explicitly arms each reset.

    The unchanged original reset runs once and consumes its normal RNG draws.
    Only afterward are selected q and position-target caches replaced. Native
    root/contact/sensor reset behavior is retained. This helper does not admit
    recovery, refresh sensors, schedule resets, or run any simulation step.
    """
    def __init__(self, env, plan, arm):
        if env.num_envs!=32 or env.step_dt!=.02:raise ValueError('Declared32-row50Hz experiment only')
        self.env=env;self.plan=plan;self.arm=arm;self.records=[];self.next_epoch=0;self.pending=None
        self.q=arm_targets(plan,arm,env._robot.joint_names,tensor(env._robot.data.default_joint_pos))
        lim=tensor(env._robot.data.soft_joint_pos_limits)
        if ((self.q<lim[...,0]+.02)|(self.q>lim[...,1]-.02)).any():raise ValueError('Selected reset loses unchanged joint reserve')
        self.original=env._reset_idx

    def arm_next(self):
        if self.pending is not None or self.next_epoch>=len(self.plan['epochs']):raise RuntimeError('Only one next declared reset can be armed')
        epoch=self.plan['epochs'][self.next_epoch]
        if int(self.env._sim_step_counter)!=epoch['at_control']*8:raise RuntimeError('Reset counter differs from declared boundary')
        self.pending=epoch
        return torch.tensor(epoch['rows'],device=self.env.device,dtype=torch.int32)

    def _snapshot(self):
        d=self.env._robot.data;c=self.env.reference_residual_controller
        result={k:tensor(getattr(d,k)).clone() for k in ('joint_pos','joint_vel','joint_pos_target','joint_vel_target','root_link_pos_w','root_link_quat_w','root_com_lin_vel_w','root_link_ang_vel_w')}
        result.update({k:getattr(c,k).clone() for k in ('reference_position','reference_velocity','residual_position','residual_velocity')})
        result.update(previous=self.env._previous_processed_joint_target.clone(),processed=self.env._processed_actions.clone(),actions=self.env._actions.clone())
        return result

    def _reset(self, instance, env_ids):
        if instance is not self.env or self.pending is None:raise RuntimeError('Unarmed reset rejected before original state write')
        ids=torch.arange(32,device=instance.device,dtype=torch.int32) if env_ids is None else torch.as_tensor(env_ids,device=instance.device)
        epoch=self.pending
        if ids.dtype not in (torch.int32,torch.int64) or ids.ndim!=1 or ids.tolist()!=epoch['rows']:
            raise RuntimeError('Selected reset rows/order differs from declared epoch')
        if int(instance._sim_step_counter)!=epoch['at_control']*8:raise RuntimeError('Physics advanced after reset was armed')
        self.pending=None
        record=dict(arm=self.arm,epoch=epoch['id'],counter_before=int(instance._sim_step_counter),rows=ids.tolist(),status='original_reset_started')
        self.records.append(record)
        before_original=self._snapshot()
        with torch.inference_mode():self.original(env_ids)
        before=self._snapshot();record['original_reset_q']=before['joint_pos'][ids].cpu().tolist()
        unselected=torch.ones(32,device=instance.device,dtype=torch.bool);unselected[ids]=False
        original_checks={k:bool(torch.equal(before[k][unselected],before_original[k][unselected])) for k in before}
        record['original_reset_unselected_checks']=original_checks
        if not all(original_checks.values()):raise RuntimeError('Original reset changed unselected state')
        if int(instance._sim_step_counter)!=record['counter_before']:raise RuntimeError('Original reset unexpectedly advanced physics')
        if before['joint_vel'][ids].any() or before['actions'][ids].any():raise RuntimeError('Original reset did not produce zero joint velocity/action')
        q=self.q[ids].clone();robot=instance._robot
        with torch.inference_mode():
            robot.write_joint_position_to_sim_index(position=q,env_ids=ids)
            robot.set_joint_position_target_index(target=q,env_ids=ids)
            instance.reference_residual_controller.reset(q.double(),q.double(),env_ids=ids)
            instance._previous_processed_joint_target[ids]=q
            instance._processed_actions[ids]=q
            instance._has_previous_processed_joint_target[ids]=True
            instance._reference_fresh[ids]=False
        after=self._snapshot();other=torch.ones(32,device=instance.device,dtype=torch.bool);other[ids]=False
        checks={k:bool(torch.equal(after[k][other],before[k][other])) for k in after}
        for k in ('root_link_pos_w','root_link_quat_w','root_com_lin_vel_w','root_link_ang_vel_w','joint_vel','joint_vel_target'):
            checks['entire_'+k+'_unchanged']=bool(torch.equal(after[k],before[k]))
        for k in ('joint_pos','joint_pos_target','reference_position','previous','processed'):
            checks['selected_'+k+'_exact']=bool(torch.equal(after[k][ids].float(),q))
        for k in ('reference_velocity','residual_position','residual_velocity','actions'):
            checks['selected_'+k+'_zero']=not bool(after[k][ids].any())
        checks['counter_unchanged']=int(instance._sim_step_counter)==record['counter_before']
        record.update(status='readback_completed',checks=checks,passed=all(checks.values()),q=q.cpu().tolist(),root_after=after['root_link_pos_w'].cpu().tolist(),quaternion_after_xyzw=after['root_link_quat_w'].cpu().tolist())
        if not record['passed']:raise RuntimeError('Selected reset initialization readback rejected')
        self.next_epoch+=1

    @contextmanager
    def installed(self):
        if self.env._reset_idx!=self.original:raise RuntimeError('Reset owner changed before installation')
        self.env._reset_idx=MethodType(self._reset,self.env)
        try:yield self
        finally:self.env._reset_idx=self.original
