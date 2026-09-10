"""Execute the real archived reward method with a small CPU state fixture."""
import ast,copy,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from caps import MeanRegularizer
from curriculum import make_environment
HERE=Path(__file__).parent
class Wrapped:
 def __init__(self,v):self.torch=v
class Tests(unittest.TestCase):
 def test_evaluation_flag_reads_are_only_command_paths(self):
  t=ast.parse((HERE/'inputs/old_omni_env.py').read_text());cls=next(n for n in t.body if isinstance(n,ast.ClassDef));owners=[]
  for method in cls.body:
   if isinstance(method,ast.FunctionDef):
    for n in ast.walk(method):
     if isinstance(n,ast.Attribute) and n.attr=='omni_evaluation' and isinstance(n.ctx,ast.Load):owners.append(method.name)
  self.assertCountEqual(owners,['_sample_commands','set_evaluation_targets','_get_rewards'])
  r=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_get_rewards')
  block=next(n for n in ast.walk(r) if isinstance(n,ast.If) and any(isinstance(x,ast.Attribute) and x.attr=='omni_evaluation' for x in ast.walk(n.test)))
  calls=[n for n in ast.walk(block) if isinstance(n,ast.Call)]
  self.assertFalse(any(isinstance(n.func,ast.Attribute) and 'reward' in n.func.attr for n in calls))
  self.assertTrue(any(isinstance(n.func,ast.Name) and n.func.id=='sample_commands' for n in calls))
 def test_mean_forward_and_normalizers_not_changed(self):
  obs=TensorDict({'policy':torch.randn(8,315),'caps_previous_policy':torch.randn(8,315),'caps_pair_valid':torch.ones(8,1,dtype=torch.bool)},batch_size=[8])
  actor=MLPModel(obs,{'actor':['policy']},'actor',18,hidden_dims=[32],obs_normalization=True,distribution_cfg={'class_name':'GaussianDistribution','init_std':.1,'std_type':'scalar'})
  actor.update_normalization(obs);before={k:v.clone() for k,v in actor.state_dict().items() if 'obs_normalizer' in k}
  rng=torch.get_rng_state().clone();a=actor(obs);b=actor(obs);self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
  reg=MeanRegularizer(dict(temporal_weight=.1,spatial_weight=.1,noise_scale=1.,noise_seed=991),'cpu');reg(actor,obs).backward()
  self.assertTrue(all(torch.equal(v,actor.state_dict()[k]) for k,v in before.items()));self.assertIsNone(actor.distribution.std_param.grad)
 def test_actual_reward_scores_old_command_then_slews(self):
  import sys
  # The exact old method imports its raw-action helper; it is byte-identical
  # to the copied checkpoint_load module, with zero extra reward weight.
  import checkpoint_load;sys.modules['omni_repair_training']=checkpoint_load
  text=(HERE/'inputs/old_omni_env.py').read_text();tree=ast.parse(text);cl=next(n for n in tree.body if isinstance(n,ast.ClassDef));method=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='_get_rewards')
  # Load exact old math definitions without Isaac imports.
  mathfile=HERE/'inputs/old_omni_math.py'
  ns={};exec(compile(mathfile.read_text(),str(mathfile),'exec'),ns)
  quiet=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='quiet_stand_terms');exec(compile(ast.Module(body=[quiet],type_ignores=[]),'quiet','exec'),ns);exec(compile(ast.Module(body=[method],type_ignores=[]),'archived_reward','exec'),ns)
  n=4;zz=lambda *shape:torch.zeros(*shape);d=NS()
  values={'root_lin_vel_b':zz(n,3),'root_ang_vel_b':zz(n,3),'computed_torque':zz(n,18),'applied_torque':zz(n,18),'soft_joint_pos_limits':torch.stack((-torch.ones(n,18)*3,torch.ones(n,18)*3),-1),'joint_pos':zz(n,18),'default_joint_pos':zz(n,18),'joint_vel':zz(n,18),'root_lin_vel_w':zz(n,3),'projected_gravity_b':torch.tensor([0.,0.,-1.]).expand(n,-1),'root_pos_w':torch.tensor([0.,0.,.13]).expand(n,-1)}
  for k,v in values.items():setattr(d,k,Wrapped(v))
  sensors=[NS(data=NS(net_forces_w_history=Wrapped(zz(n,1,1,3)),last_air_time=Wrapped(zz(n,1))),compute_first_contact=lambda dt:Wrapped(zz(n,1))) for _ in range(6)]
  class Parent:
   def __init__(self,cfg,render_mode,evaluation,**kw):
    self.num_envs=n;self.device='cpu';self.step_dt=.02;self.omni_evaluation=evaluation;self.cfg=NS(seed=157,nominal_height_m=.13,omni_reward_weights={'linear_tracking':1.});self._robot=NS(data=d);self._commands=zz(n,3);self.omni_targets=zz(n,3);self._actions=zz(n,18);self.omni_previous_action=zz(n,18);self.omni_previous_velocity=zz(n,18);self._processed_actions=zz(n,18);self.omni_previous_target=zz(n,18);self.omni_raw_policy_action=zz(n,18);self.reset_terminated=zz(n).bool();self._episode_elapsed_s=zz(n);self.omni_sums={};self._coxa_contact_sensor=NS(data=NS(net_forces_w_history=Wrapped(zz(n,1,6,3))));self._femur_contact_sensors=sensors;self._feet_contact_sensors=sensors
   _get_rewards=ns['_get_rewards']
   def _vector_in_command_frame(self,x):return x
   def _get_foot_contact_state(self):return torch.ones(n,6,dtype=torch.bool),zz(n,6).bool(),zz(n,6)
  e=make_environment(Parent)(NS(seed=157));reward=e._get_rewards();self.assertTrue(torch.equal(reward,torch.full((n,),.02)));self.assertTrue(torch.equal(e.omni_sums['linear_tracking'],reward));self.assertTrue((e._commands[1:]!=0).any());self.assertTrue((e._commands[0]==0).all())
if __name__=='__main__':unittest.main()
