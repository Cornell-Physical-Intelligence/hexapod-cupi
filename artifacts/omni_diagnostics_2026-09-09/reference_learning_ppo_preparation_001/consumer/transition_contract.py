"""CPU-only learning-mask/return prototype; no simulator or RSL replacement."""
import torch

def _shape(name,x,shape,dtype):
    if not isinstance(x,torch.Tensor) or x.shape!=shape or x.dtype!=dtype:
        raise ValueError('Bad transition field: '+name)
    if not torch.isfinite(x).all():raise ValueError('Nonfinite transition field: '+name)

def learning_masks(active_before,terminated,truncated,fatal):
    shape=active_before.shape
    if len(shape)!=1:raise ValueError('Per-environment masks required')
    for name,x in [('active',active_before),('terminated',terminated),('truncated',truncated),('fatal',fatal)]:_shape(name,x,shape,torch.bool)
    if fatal.any():raise RuntimeError('Campaign-level data/actuator/integrity failure; do not train this rollout')
    if (terminated&truncated).any():raise ValueError('Termination and time-limit truncation must be distinct')
    if ((terminated|truncated)&~active_before).any():raise RuntimeError('Startup/reset recovery failed; requires review')
    return {'learnable':active_before.clone(),'done':terminated|truncated,
            'terminal_bootstrap_zero':terminated.clone(),'requires_pre_reset_value':active_before&~terminated}

def masked_gae(reward,value,next_value,learnable,terminated,truncated,episode,bootstrap_episode,*,gamma=.995,lam=.99):
    """No advantage chain crosses a disabled row or episode boundary.

    next_value is explicitly V(final pre-reset next observation) for a truncation,
    or V(next same-episode observation) for a normal step. True terminal events
    bootstrap zero. Warm-up rows are absent from actor/value/normalizer batches.
    """
    shape=reward.shape
    if len(shape)!=2 or 0 in shape:raise ValueError('Nonempty [control,environment] arrays required')
    for name,x in [('reward',reward),('value',value),('next_value',next_value)]:_shape(name,x,shape,torch.float64)
    for name,x in [('learnable',learnable),('terminated',terminated),('truncated',truncated)]:_shape(name,x,shape,torch.bool)
    for name,x in [('episode',episode),('bootstrap_episode',bootstrap_episode)]:_shape(name,x,shape,torch.int64)
    if not (0<gamma<=1 and 0<lam<=1):raise ValueError('Invalid discount parameters')
    if (terminated&truncated).any() or ((terminated|truncated)&~learnable).any():raise ValueError('Malformed episode outcome')
    if (learnable&~terminated&(bootstrap_episode!=episode)).any():raise ValueError('Bootstrap value came from a reset/different episode')
    out=torch.zeros_like(reward);carry=torch.zeros(shape[1],dtype=reward.dtype,device=reward.device)
    for t in reversed(range(shape[0])):
        if t==shape[0]-1:chain=torch.zeros(shape[1],dtype=torch.bool,device=reward.device)
        else:chain=learnable[t+1]&(episode[t+1]==episode[t])
        chain &= learnable[t]&~terminated[t]&~truncated[t]
        delta=reward[t]+gamma*torch.where(terminated[t],0.,next_value[t])-value[t]
        carry=torch.where(learnable[t],delta+gamma*lam*chain*carry,0.)
        out[t]=carry
    returns=torch.where(learnable,out+value,0.)
    return {'advantage':out,'returns':returns,'mini_batch_flat_indices':torch.nonzero(learnable.flatten()).flatten(),
            'policy_training_allowed':False,'actual_RSL_integration_tested':False}
