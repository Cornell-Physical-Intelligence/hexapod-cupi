import ast,copy,inspect,unittest
from pathlib import Path
import torch
from tensordict import TensorDict
from caps import PairState,MeanRegularizer,noise_scales,CapsPairWrapper
from curriculum import StandStopSchedule,make_environment
from caps_ppo import CapsPPO
from rsl_rl.algorithms import PPO
class Actor(torch.nn.Module):
 obs_groups=['policy'];is_recurrent=False
 def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(315,18)/315);self.inputs=[]
 def forward(self,obs):self.inputs.append(obs['policy'].detach().clone());return obs['policy']@self.weight
class Tests(unittest.TestCase):
 def test_exact_PPO_update_delta(self):
  a=ast.parse(inspect.getsource(PPO.update).lstrip());b=ast.parse(inspect.getsource(CapsPPO._caps_update).lstrip());b.body[0].name='update'
  loop=next(n for n in ast.walk(b) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='batch')
  inserted=[n for n in loop.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.BinOp) and isinstance(n.value.right,ast.Call) and isinstance(n.value.right.func,ast.Attribute) and n.value.right.func.attr=='regularizer']
  self.assertEqual(len(inserted),1);loop.body.remove(inserted[0]);self.assertEqual(ast.dump(a),ast.dump(b))
 def test_reset_and_command_change_mask(self):
  p=torch.zeros(3,315);x=PairState(p);p.fill_(99);self.assertTrue((x.current==0).all())
  n=torch.ones(3,315);n[:,258:261]=0;n[2,258]=.5;x.advance(n,torch.tensor([False,True,False]));self.assertEqual(x.valid[:,0].tolist(),[True,False,False]);self.assertTrue((x.previous==0).all())
  obs=TensorDict({'policy':n,'critic':torch.zeros(3,318)},batch_size=[3]);packet=x.decorate(obs);packet['policy'].zero_();self.assertTrue(torch.equal(x.current,n))
 def test_spatial_command_action_history_unchanged(self):
  a=Actor();p=torch.ones(4,315);obs=TensorDict({'policy':p,'caps_previous_policy':torch.zeros_like(p),'caps_pair_valid':torch.ones(4,1,dtype=torch.bool)},batch_size=[4]);r=MeanRegularizer(dict(temporal_weight=.1,spatial_weight=.1,noise_scale=1.,noise_seed=991),'cpu');loss=r(a,obs);loss.backward();self.assertTrue(torch.isfinite(a.weight.grad).all())
  delta=(a.inputs[-1]-p).reshape(4,5,63);self.assertTrue((delta[:,:,6:9]==0).all());self.assertTrue((delta[:,:,45:]==0).all());self.assertGreater(float(delta.abs().sum()),0)
 def test_zero_ablation_no_random_or_actor_calls(self):
  a=Actor();r=MeanRegularizer(dict(temporal_weight=0.,spatial_weight=0.,noise_scale=1.,noise_seed=991),'cpu');obs={'policy':torch.zeros(2,315)};before=torch.get_rng_state().clone();g=r.generator.get_state().clone();self.assertEqual(float(r(a,obs).detach()),0);self.assertEqual(len(a.inputs),0);self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertTrue(torch.equal(g,r.generator.get_state()))
 def test_temporal_reset_samples_do_not_contribute(self):
  a=Actor();p=torch.ones(2,315);obs=TensorDict({'policy':p,'caps_previous_policy':torch.zeros_like(p),'caps_pair_valid':torch.zeros(2,1,dtype=torch.bool)},batch_size=[2]);r=MeanRegularizer(dict(temporal_weight=1.,spatial_weight=0.,noise_scale=0.,noise_seed=991),'cpu');self.assertEqual(float(r(a,obs).detach()),0)
 def test_storage_binary_float_mask_and_corruption(self):
  a=Actor();p=torch.ones(2,315);obs=TensorDict({'policy':p,'caps_previous_policy':torch.zeros_like(p),'caps_pair_valid':torch.tensor([[1.],[0.]])},batch_size=[2]);r=MeanRegularizer(dict(temporal_weight=.1,spatial_weight=.1,noise_scale=1.,noise_seed=991),'cpu');self.assertTrue(torch.isfinite(r(a,obs)))
  obs['caps_pair_valid'][0]=.5
  with self.assertRaises(ValueError):r(a,obs)
 def test_curriculum_full_cycle_modes_and_bounds(self):
  s=StandStopSchedule(128,'cpu');rows=torch.stack([s.advance() for _ in range(800)]);moving=(rows!=0).any(-1)
  self.assertTrue((moving[:,s.quiet]==False).all());self.assertTrue((moving[:,~s.quiet].sum(0)==400).all());self.assertLessEqual(float(rows[:,:,:2].norm(dim=-1).max()),.200001);self.assertLessEqual(float(rows[:,:,2].abs().max()),.400001)
  self.assertTrue((rows[:,:,0]<0).any() and (rows[:,:,0]>0).any() and (rows[:,:,1]<0).any() and (rows[:,:,1]>0).any() and (rows[:,:,2]<0).any() and (rows[:,:,2]>0).any())
  old=s.state();s.reset(torch.tensor([2],dtype=torch.int32));self.assertEqual(int(s.age[2]),0);self.assertTrue(torch.equal(s.motion[3:],old['motion_twist'][3:]))
 def test_command_changes_after_current_reward_not_before(self):
  class Parent:
   def __init__(self,cfg,render_mode,evaluation,**kw):self.num_envs=4;self.device='cpu';self.step_dt=.02;self._commands=torch.zeros(4,3);self.omni_targets=torch.zeros(4,3);self.scored=[]
   def _get_rewards(self):self.scored.append(self._commands.clone());self._commands.copy_(self.omni_targets);return self._commands.new_zeros(4)
  cls=make_environment(Parent);e=cls(type('Cfg',(),{'seed':157})());old=e._commands.clone();e._get_rewards();self.assertTrue(torch.equal(e.scored[-1],old));self.assertTrue((e._commands[1:]!=0).any());e._sample_commands(torch.tensor([1],dtype=torch.int32));self.assertTrue((e._commands[1]==0).all())
if __name__=='__main__':unittest.main()
