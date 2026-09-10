"""CAPS-style policy-mean regularization; preserves the 315/318 model interface.

Only auxiliary rollout storage gains keys. Command/action history is never
perturbed. Temporal pairs exclude resets and command changes. Normalizers are
not updated by synthetic neighbors. This is a proposal, not admitted physics.
"""
import math
import torch
from tensordict import TensorDict
WIDTH=315

def validated_options(o):
 if not isinstance(o,dict) or set(o)!={'temporal_weight','spatial_weight','noise_scale','noise_seed'}:raise ValueError('Exact CAPS options required')
 for k in ['temporal_weight','spatial_weight','noise_scale']:
  if isinstance(o[k],bool) or not isinstance(o[k],(float,int)) or not math.isfinite(o[k]) or not 0<=o[k]<=1:raise ValueError('Invalid CAPS coefficient')
 if not isinstance(o['noise_seed'],int) or isinstance(o['noise_seed'],bool):raise ValueError('Integer independent noise seed required')
 return dict(o)

def noise_scales(device,dtype):
 s=torch.zeros(5,63,device=device,dtype=dtype)
 s[:,:3]=.00375;s[:,3:6]=.01;s[:,9:27]=.005;s[:,27:45]=.0025
 return s.flatten()

def latest_commands(obs):return obs[:,4*63+6:4*63+9]

class PairState:
 def __init__(self,policy):
  self._check(policy);self.current=policy.detach().clone();self.previous=torch.zeros_like(policy);self.valid=torch.zeros(len(policy),1,device=policy.device,dtype=torch.bool)
 @staticmethod
 def _check(p):
  if p.ndim!=2 or p.shape[1]!=WIDTH or not torch.isfinite(p).all():raise ValueError('Finite Nx315 actor observations required')
 def advance(self,next_policy,dones):
  self._check(next_policy)
  if next_policy.shape!=self.current.shape or dones.shape!=(len(next_policy),):raise ValueError('Pair shape mismatch')
  unchanged=(latest_commands(next_policy)==latest_commands(self.current)).all(-1)
  self.valid=(~dones.bool() & unchanged)[:,None]
  self.previous=torch.where(self.valid,self.current,0.).detach().clone()
  self.current=next_policy.detach().clone()
 def decorate(self,obs):
  if not torch.equal(obs['policy'],self.current):raise ValueError('Cached observation changed without a step')
  out=obs.clone();out['caps_previous_policy']=self.previous.clone();out['caps_pair_valid']=self.valid.clone();return out

class CapsPairWrapper:
 def __init__(self,env):
  self.env=env;self.cached=env.get_observations().clone();self.pairs=PairState(self.cached['policy']);self.failure=None
 def __getattr__(self,key):return getattr(self.env,key)
 @property
 def episode_length_buf(self):return self.env.episode_length_buf
 @episode_length_buf.setter
 def episode_length_buf(self,value):self.env.episode_length_buf=value
 def get_observations(self):
  if self.failure is not None:raise RuntimeError(self.failure)
  return self.pairs.decorate(self.cached)
 def step(self,action):
  if self.failure is not None:raise RuntimeError(self.failure)
  try:
   obs,reward,done,extras=self.env.step(action)
   self.pairs.advance(obs['policy'],done);self.cached=obs.clone()
   return self.get_observations(),reward,done,extras
  except BaseException as exc:self.failure=repr(exc);raise

class MeanRegularizer:
 def __init__(self,options,device):
  self.options=validated_options(options);self.generator=torch.Generator(device=device);self.generator.manual_seed(options['noise_seed']);self.stats=[]
 def __call__(self,actor,obs):
  o=self.options
  if actor.obs_groups!=['policy'] or actor.is_recurrent:raise ValueError('Exact feedforward policy-only315 actor required')
  if not o['temporal_weight'] and not o['spatial_weight']:
   # No actor calls or random draws: the zero branch is an exact PPO ablation.
   return obs['policy'].new_zeros(())
  p=obs['policy'];previous=obs['caps_previous_policy'];valid=obs['caps_pair_valid']
  PairState._check(p);PairState._check(previous)
  if previous.shape!=p.shape or valid.shape!=(len(p),1) or not torch.isfinite(valid).all() or not ((valid==0)|(valid==1)).all():raise ValueError('Malformed temporal pairing evidence')
  # RSL5.0.1 RolloutStorage stores observation keys as float tensors.
  # Require exact binary values before interpreting this auxiliary mask.
  valid=valid.bool()
  mu=actor(obs)
  prev=obs.clone();prev['policy']=previous
  temporal=((mu-actor(prev)).square().mean(-1)*valid[:,0]).sum()/valid.sum().clamp_min(1)
  neighbor=obs.clone();noise=torch.randn(p.shape,device=p.device,dtype=p.dtype,generator=self.generator).clamp(-3,3)*noise_scales(p.device,p.dtype)*o['noise_scale'];neighbor['policy']=p+noise
  spatial=(mu-actor(neighbor)).square().mean()
  loss=o['temporal_weight']*temporal+o['spatial_weight']*spatial
  if not torch.isfinite(loss):raise ValueError('Nonfinite regularization loss')
  self.stats.append({'temporal':float(temporal.detach()),'spatial':float(spatial.detach()),'weighted':float(loss.detach()),'valid_pair_fraction':float(valid.float().mean())})
  return loss
