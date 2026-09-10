"""Explicit 25% quiet rows; others alternate8s motion /8s stop targets.

No new observation field or hidden motor state: targets ramp through the exact
old slew_commands function into the command already observed by the actor.
"""
import math
import torch
class StandStopSchedule:
 def __init__(self,count,device,seed=157,dt=.02):
  if count<4 or dt!=.02:raise ValueError('At least4 rows at exact50Hz required')
  self.n=count;self.device=device;self.dt=dt
  self.generator=torch.Generator(device=device);self.generator.manual_seed(seed)
  ids=torch.arange(count,device=device)
  self.quiet=ids%4==0;self.offset=(ids%4)*200
  self.age=torch.zeros(count,device=device,dtype=torch.long);self.episode=torch.zeros_like(self.age)
  self.motion=torch.zeros(count,3,device=device);self.reset(ids)
 def reset(self,ids):
  ids=ids.long();u=torch.rand(len(ids),4,device=self.device,generator=self.generator)
  a=2*math.pi*u[:,0];speed=.06+.14*u[:,1];yaw=(.12+.28*u[:,2])*torch.where(u[:,3]<.5,-1.,1.)
  # A separate draw prevents correlation of yaw sign and motion category.
  category=torch.rand(len(ids),device=self.device,generator=self.generator)
  translating=category>=.25;turning=(category<.25)|(category>=.75)
  self.motion[ids]=torch.stack((speed*a.cos()*translating,speed*a.sin()*translating,yaw*turning),-1)
  self.age[ids]=0;self.episode[ids]+=1
 def advance(self):
  self.age+=1;moving=((self.age+self.offset)%800<400)&~self.quiet
  return torch.where(moving[:,None],self.motion,0.)
 def state(self):return {'age_controls':self.age.clone(),'episode':self.episode.clone(),'quiet_rows':self.quiet.clone(),'motion_twist':self.motion.clone(),'phase_offset_controls':self.offset.clone()}

def make_environment(parent):
 class BalancedOmniEnv(parent):
  def __init__(self,cfg,render_mode=None,*,evaluation=False,**kw):
   self.direct_schedule=None
   # The parent's evaluation flag only disables random command sampling;
   # its physical stepping, reward timing and command slew remain exact.
   super().__init__(cfg,render_mode,evaluation=True,**kw)
   if not evaluation:self.direct_schedule=StandStopSchedule(self.num_envs,self.device,cfg.seed,self.step_dt)
  def _sample_commands(self,ids):
   self._commands[ids]=0
   if hasattr(self,'omni_targets'):self.omni_targets[ids]=0
   if self.direct_schedule is not None:self.direct_schedule.reset(ids)
  def _get_rewards(self):
   if self.direct_schedule is not None:self.omni_targets.copy_(self.direct_schedule.advance())
   return super()._get_rewards()
 return BalancedOmniEnv
