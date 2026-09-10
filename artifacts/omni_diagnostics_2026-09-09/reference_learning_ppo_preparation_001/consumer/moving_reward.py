"""CPU proposal only: explicit motion/finite-stop/quiet reward scope.

No source009 environment, observation identity, checkpoint or physical gate is
modified. Inputs must be the current action/reference command and pre-reset
measurement, never a queued command for the next packet.
"""
from dataclasses import dataclass
import math
import torch
SCHEMA='wave005_requested_goal_moving_reward_v2'
WAVE_FREEZE='5f46b6f99c172bb037004e758f4f3715d41da42725abd10bfd313ee2b84fe286'
PHYSICAL_SOURCE='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
QUIET_KEYS=('stand_joint_velocity','stand_target_velocity','stand_posture','stand_raw_action')
RAW_KEYS=(*QUIET_KEYS,'airtime')

@dataclass(frozen=True)
class Normalization:
    translation_mps:float=.005
    yaw_rad_s:float=.015
    quiet_translation_sigma_mps:float=.04
    quiet_yaw_sigma_rad_s:float=.10
    def __post_init__(self):
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in self.__dict__.values()):
            raise ValueError('Positive finite reward normalization required')


def _field(value,shape,dtype,device,name,finite=True):
    if not isinstance(value,torch.Tensor) or value.shape!=shape or value.dtype!=dtype or value.device!=device:
        raise ValueError('Malformed reward field: '+name)
    if finite and not torch.isfinite(value).all():raise ValueError('Nonfinite reward field: '+name)
    return value


def phase(requested,state,measured_support,measurement_valid,normalization=Normalization()):
    if not isinstance(requested,torch.Tensor) or requested.ndim!=2 or requested.shape[1]!=3 or requested.shape[0]<1 or requested.dtype!=torch.float64:
        raise ValueError('Complete float64 forward/left/yaw batch required')
    n=requested.shape[0];device=requested.device
    _field(requested,(n,3),requested.dtype,device,'requested')
    for key in ('ready','quiet_valid','sw_active','land_active'):
        _field(state[key],(n,),torch.bool,device,key)
    for key in ('failure','mode','current_leg'):_field(state[key],(n,),torch.int64,device,key)
    for key in ('requested','command','command_target','rate'):_field(state[key],(n,3),torch.float64,device,key)
    for key in ('time','quiet_time','factor'):_field(state[key],(n,),torch.float64,device,key)
    support=_field(measured_support,(n,6),torch.bool,device,'measured_support')
    valid=_field(measurement_valid,(n,),torch.bool,device,'measurement_valid')
    if not torch.equal(requested,state['requested']):raise ValueError('Reward command differs from current reference; queued commands are unsupported')
    if not (state['ready'] & (state['failure']==0) & valid).all():raise ValueError('Failed/stale/uninitialized row cannot be rewarded')
    if ((state['mode']<0)|(state['mode']>8)|(state['current_leg'] < -1)|(state['current_leg']>5)).any():raise ValueError('Unknown reference state')
    if (requested[:,:2].norm(dim=-1)>normalization.translation_mps+1e-12).any() or (requested[:,2].abs()>normalization.yaw_rad_s+1e-12).any() :
        raise ValueError('Requested command exceeds the declared mathematical envelope; directional admission is separate')
    if ((state['factor']<0)|(state['factor']>1)).any() or not torch.allclose(state['command_target'],state['factor'][:,None]*requested,atol=1e-12,rtol=0):
        raise ValueError('Inconsistent governed reference target/factor')
    if (state['time']<0).any() or (state['quiet_valid']&(state['quiet_time']<0)).any():raise ValueError('Negative reference clock')
    moving=(requested!=0).any(-1)
    quiet=(~moving)&state['quiet_valid']&(state['mode']==5)&(state['current_leg']==-1)&~state['sw_active']&~state['land_active']
    quiet &= (state['command']==0).all(-1)&(state['rate']==0).all(-1)&(state['quiet_time']<=state['time'])
    if (state['quiet_valid']&~quiet).any():raise ValueError('Inconsistent finite quiet state')
    required=torch.where(moving | (state['current_leg']>=0),5,6)
    if (support.sum(-1)<required).any():raise ValueError('Existing required measured support is missing')
    stopping=(~moving)&~quiet
    target=torch.where(moving[:,None],requested,state['command']).clone()  # User goal persists through contact-governor pauses; only finite stop uses filtered motion.
    if (target[:,:2].norm(dim=-1)>normalization.translation_mps+1e-12).any() or (target[:,2].abs()>normalization.yaw_rad_s+1e-12).any():raise ValueError('Filtered reference exceeds proposal envelope')
    return {'commanded_motion':moving,'finite_supported_stop':stopping,'settled_quiet':quiet,
            'tracking_target':target,'reference_governor_factor':state['factor'].clone(),
            'reference_command_target':state['command_target'].clone(),'schema':SCHEMA,'physical_admission':False,'policy_training_allowed':False}


def tracking(velocity,yaw_rate,scope,normalization=Normalization()):
    target=scope['tracking_target'];n=len(target);device=target.device
    _field(velocity,(n,2),torch.float64,device,'velocity_in_command_frame')
    _field(yaw_rate,(n,),torch.float64,device,'yaw_rate_in_command_frame')
    # Preserve the exact original zero-target tracking bandwidth. Smoothly
    # normalize at the proposed admitted-speed scale, without dividing by a
    # tiny requested speed or declaring that these bandwidths are optimal.
    r=(target[:,:2].norm(dim=-1)/normalization.translation_mps).clamp(0,1)
    y=(target[:,2].abs()/normalization.yaw_rad_s).clamp(0,1)
    blend=r.square()*(3-2*r);yb=y.square()*(3-2*y)
    sigma=normalization.quiet_translation_sigma_mps*(1-blend)+normalization.translation_mps*blend
    ysigma=normalization.quiet_yaw_sigma_rad_s*(1-yb)+normalization.yaw_rad_s*yb
    linear=torch.exp(-(velocity-target[:,:2]).square().sum(-1)/sigma.square())
    yaw=torch.exp(-(yaw_rate-target[:,2]).square()/ysigma.square())
    moving=scope['commanded_motion']
    # Fixed envelope denominators give a finite, direction-invariant limit at
    # zero. Progress is absent during a requested stop, including its tail.
    progress=((velocity*target[:,:2]).sum(-1)/normalization.translation_mps**2).clamp(-1,1)*moving
    yaw_progress=(yaw_rate*target[:,2]/normalization.yaw_rad_s**2).clamp(-1,1)*moving
    return {'linear_tracking':linear,'yaw_tracking':yaw,'linear_progress':progress,'yaw_progress':yaw_progress}


def scoped_terms(velocity,yaw_rate,scope,unmasked,normalization=Normalization()):
    if set(unmasked)!=set(RAW_KEYS):raise ValueError('Supply all five original unmasked components exactly')
    n=len(scope['tracking_target']);device=scope['tracking_target'].device
    for key,value in unmasked.items():_field(value,(n,),torch.float64,device,key)
    if any((unmasked[k]<0).any() for k in QUIET_KEYS):raise ValueError('Squared quiet components cannot be negative')
    result=tracking(velocity,yaw_rate,scope,normalization)
    for key in QUIET_KEYS:result[key]=unmasked[key]*scope['settled_quiet']
    result['airtime']=unmasked['airtime']*scope['commanded_motion']
    return result
