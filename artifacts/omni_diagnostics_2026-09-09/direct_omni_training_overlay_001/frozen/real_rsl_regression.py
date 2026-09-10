"""Actual RSL5.0.1 on CPU, synthetic physics; not a motor admission."""
import argparse,copy,hashlib,json
from pathlib import Path
import torch,yaml
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
from caps import CapsPairWrapper
from checkpoint_load import load_repair_checkpoint
HERE=Path(__file__).parent
CP_SHA='1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8'
class Synthetic:
 def __init__(self):
  self.num_envs=8;self.num_actions=18;self.device='cpu';self.cfg={};self.episode_length_buf=torch.zeros(8,dtype=torch.long);self.max_episode_length=100;self.t=0;self.hist=torch.zeros(8,5,63);self.last=torch.zeros(8,18)
 def get_observations(self):
  p=self.hist.flatten(1).clone();return TensorDict({'policy':p,'critic':torch.cat((p,torch.zeros(8,3)),1)},batch_size=[8])
 def step(self,action):
  self.t+=1;self.episode_length_buf+=1;self.hist[:,:-1]=self.hist[:,1:].clone()
  f=torch.zeros(8,63);f[:,5]=-1;f[:,9:27]=.01*torch.tanh(action);f[:,27:45]=(action-self.last)*.001;f[:,45:]=action
  f[:,6]=.5 if self.t<12 or self.t>=30 else 0;self.hist[:,-1]=f;self.last=action.clone()
  done=torch.zeros(8,dtype=torch.long)
  if self.t%17==0:done[0]=1;self.episode_length_buf[0]=0;self.hist[0]=f[0,None]
  reward=1.-.01*action.square().mean(-1)
  return self.get_observations(),reward,done,{'time_outs':torch.zeros(8,dtype=torch.bool)}
def cfg(kind):
 c=yaml.safe_load((HERE/'inputs/agent.yaml').read_text());c.update(device='cpu',num_steps_per_env=24,save_interval=10000,logger='tensorboard')
 c['algorithm']['entropy_coef']=0.;c['algorithm']['learning_rate']=5e-5
 if kind!='ordinary':
  c['algorithm']['class_name']='caps_ppo:CapsPPO';c['algorithm']['caps_options']={'temporal_weight':.1 if kind=='caps' else 0.,'spatial_weight':.1 if kind=='caps' else 0.,'noise_scale':1.,'noise_seed':1157}
 return c
def compare(a,b):
 return set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)
def main():
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True);torch.set_num_threads(2)
 report={'scope':'Actual RSL/real315checkpoint with synthetic CPU environment; no Isaac/actuator qualification','original_checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'branches':{}}
 final={}
 for kind in ['ordinary','zero','caps']:
  torch.manual_seed(157);e=CapsPairWrapper(Synthetic());r=OnPolicyRunner(e,cfg(kind),log_dir=str(a.output/kind),device='cpu')
  options={'checkpoint_sha256':CP_SHA,'exploration_std':.1,'entropy_coef':0.,'optimizer':'reset','learning_rate':5e-5}
  init=load_repair_checkpoint(r,a.checkpoint,options)
  initial={k:v.clone() for k,v in r.alg.actor.state_dict().items()}
  r.learn(2,init_at_random_ep_len=False)
  out=a.output/kind/'final.pt'
  if out.exists():raise FileExistsError(out)
  r.save(str(out));before={k:v.clone() for k,v in r.alg.actor.state_dict().items()};beforecritic={k:v.clone() for k,v in r.alg.critic.state_dict().items()};obs=e.get_observations();action=r.alg.actor(obs).detach().clone()
  torch.manual_seed(157);e2=CapsPairWrapper(Synthetic());r2=OnPolicyRunner(e2,cfg(kind),device='cpu');r2.load(str(out),map_location='cpu');assert compare(before,r2.alg.actor.state_dict()) and compare(beforecritic,r2.alg.critic.state_dict());assert torch.equal(action,r2.alg.actor(obs))
  report['branches'][kind]={'controls':e.t,'updates':2,'initialization':init,'checkpoint_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'exact_actor_critic_and_action_reload':True,'optimizer_entries':len(r.alg.optimizer.state),'mean_actor_weights_changed':not torch.equal(initial['mlp.0.weight'],before['mlp.0.weight']),'caps_last_minibatch_metrics':r.alg.regularizer.stats[-1] if hasattr(r.alg,'regularizer') and r.alg.regularizer.stats else None}
  assert e.t==48 and len(r.alg.optimizer.state)>0
  final[kind]=(before,beforecritic)
  if getattr(r.logger,'writer',None):r.logger.writer.flush();r.logger.writer.close()
 assert all(compare(final['ordinary'][i],final['zero'][i]) for i in [0,1]);assert not compare(final['zero'][0],final['caps'][0])
 report['zero_weight_PPO_actor_critic_bit_exact_after_two_updates']=True;report['positive_caps_changes_optimizer_result']=True
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('REAL_RSL_CAPS_CPU_PASS')
if __name__=='__main__':main()
