"""Objective scaling and actual RSL instrumentation equivalence, CPU only."""
import copy,hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.runners import OnPolicyRunner
from caps import MeanRegularizer,CapsPairWrapper
from gradient_diagnostics import gradient_decomposition
from direct_training import equal_tree
H=Path(__file__).resolve().parent

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
oldcaps=module('native003_caps_oracle',H.parent/'native_003/caps.py')
sys.path.append(str(H.parent/'training'))
prior=module('native004_synthetic_fixture',H.parent/'training/real_rsl_regression.py')

class QuietPriority(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(2)
 def fixture(self):
  torch.manual_seed(318)
  p=torch.randn(6,315)*.1;previous=torch.randn_like(p)*.1
  p[:,258:261]=0;previous[:,258:261]=0
  p[1,258]=previous[1,258]=.5
  p[2,260]=previous[2,260]=-.5
  previous[3,258]=.001 # Changing command even though current is zero.
  valid=torch.tensor([[1],[1],[1],[0],[0],[1]],dtype=torch.bool)
  obs=TensorDict({'policy':p,'caps_previous_policy':previous,'caps_pair_valid':valid},batch_size=[6])
  a=MLPModel(obs,{'actor':['policy']},'actor',18,hidden_dims=[32],obs_normalization=True,distribution_cfg={'class_name':'GaussianDistribution','init_std':.1,'std_type':'scalar'})
  a.update_normalization(obs)
  return a,obs
 def test_equal_quiet_coefficient_exact_parent_loss_grad_rng_and_normalizers(self):
  actor,obs=self.fixture();o={'temporal_weight':.1,'spatial_weight':.1,'noise_scale':1.,'noise_seed':991}
  parent=oldcaps.MeanRegularizer(o,'cpu');new=MeanRegularizer({**o,'quiet_temporal_weight':.1},'cpu')
  before=copy.deepcopy(actor.state_dict());rng=torch.get_rng_state().clone()
  x=parent(actor,obs);gx=torch.autograd.grad(x,tuple(actor.parameters()),allow_unused=True)
  y=new(actor,obs);gy=torch.autograd.grad(y,tuple(actor.parameters()),allow_unused=True)
  self.assertTrue(torch.equal(x,y));self.assertTrue(all(a is None and b is None or a is not None and b is not None and torch.equal(a,b) for a,b in zip(gx,gy)))
  self.assertTrue(equal_tree(before,actor.state_dict()));self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(torch.equal(parent.generator.get_state(),new.generator.get_state()))
 def test_quiet_increment_uses_all_valid_denominator_and_both_commands(self):
  actor,obs=self.fixture();o={'temporal_weight':.1,'spatial_weight':.1,'noise_scale':1.,'noise_seed':991}
  old=MeanRegularizer({**o,'quiet_temporal_weight':.1},'cpu');new=MeanRegularizer({**o,'quiet_temporal_weight':1.},'cpu')
  baseline=old(actor,obs);changed=new(actor,obs)
  prev=obs.clone();prev['policy']=obs['caps_previous_policy'];d=(actor(obs)-actor(prev)).square().mean(-1)
  expected=.9*(d[0]+d[5])/4
  self.assertTrue(torch.allclose(changed-baseline,expected,atol=1e-7,rtol=1e-6))
  self.assertEqual(new.stats[-1]['valid_pairs'],4);self.assertEqual(new.stats[-1]['quiet_pairs'],2);self.assertEqual(new.stats[-1]['moving_pairs'],2)
  self.assertEqual(old.stats[-1]['spatial'],new.stats[-1]['spatial'])
  # No valid quiet pair: higher quiet weight has no effect on objective/gradients.
  obs['caps_pair_valid'][0]=False;obs['caps_pair_valid'][5]=False
  a=MeanRegularizer({**o,'quiet_temporal_weight':.1},'cpu');b=MeanRegularizer({**o,'quiet_temporal_weight':1.},'cpu')
  self.assertTrue(torch.equal(a(actor,obs),b(actor,obs)))
 def test_grad_measurement_preserves_existing_grad_buffers_rng_and_model(self):
  actor,obs=self.fixture();reg=MeanRegularizer({'temporal_weight':.1,'quiet_temporal_weight':1.,'spatial_weight':.1,'noise_scale':1.,'noise_seed':1},'cpu');reg(actor,obs)
  ppo=actor(obs).square().mean();params=tuple(actor.parameters())
  for i,p in enumerate(params):p.grad=torch.full_like(p,float(i+1))
  grads=[p.grad.clone() for p in params];state=copy.deepcopy(actor.state_dict());rng=torch.get_rng_state().clone();gen=reg.generator.get_state().clone()
  result=gradient_decomposition(ppo,reg.last_terms,params)
  self.assertTrue(all(torch.equal(x,p.grad) for x,p in zip(grads,params)));self.assertTrue(equal_tree(state,actor.state_dict()));self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(torch.equal(gen,reg.generator.get_state()))
  self.assertEqual(set(result['norms']),{'ppo_actor','quiet_temporal','moving_temporal','spatial'})
 def test_real_rsl_two_updates_diagnostics_on_off_exact_and_sparse_labels(self):
  outcomes={};logs={}
  for enabled in [False,True]:
   torch.manual_seed(157);env=CapsPairWrapper(prior.Synthetic());cfg=prior.cfg('caps');cfg['algorithm']['caps_options']['quiet_temporal_weight']=1.;cfg['algorithm']['gradient_diagnostics']=enabled
   with tempfile.TemporaryDirectory() as d:
    runner=OnPolicyRunner(env,cfg,log_dir=d,device='cpu')
    from checkpoint_load import load_repair_checkpoint
    cp=H.parents[1]/'ppo_repair_003_preparation/original/policy.pt'
    load_repair_checkpoint(runner,cp,{'checkpoint_sha256':prior.CP_SHA,'exploration_std':.1,'entropy_coef':0.,'optimizer':'reset','learning_rate':5e-5})
    observed=[];original=runner.alg.update
    def update():
     result=original();observed.append(copy.deepcopy(runner.alg.last_update_diagnostics));return result
    runner.alg.update=update
    runner.learn(2,init_at_random_ep_len=False)
    outcomes[enabled]={'save':copy.deepcopy(runner.alg.save()),'rng':torch.get_rng_state().clone(),'noise_rng':runner.alg.regularizer.generator.get_state().clone(),'obs':env.get_observations().clone(),'history':env.env.hist.clone(),'lr':runner.alg.learning_rate,'controls':env.env.t};logs[enabled]=observed
    if getattr(runner.logger,'writer',None):runner.logger.writer.flush();runner.logger.writer.close()
  for k in ['save','rng','noise_rng','history','lr','controls']:self.assertTrue(equal_tree(outcomes[False][k],outcomes[True][k]),k)
  self.assertTrue(all(torch.equal(outcomes[False]['obs'][k],outcomes[True]['obs'][k]) for k in outcomes[False]['obs'].keys()))
  self.assertEqual(logs[False],[[],[]]);self.assertEqual([len(x) for x in logs[True]],[20,20]);self.assertEqual(outcomes[True]['controls'],48)
  self.assertEqual(sum(r['gradient'] is not None for u in logs[True] for r in u),2)
  for u,rows in enumerate(logs[True],1):
   self.assertEqual([r['minibatch'] for r in rows],list(range(1,21)));self.assertTrue(all(r['update']==u and r['kl_mean'] is not None for r in rows))
  self.assertIsNotNone(logs[True][0][0]['gradient']);self.assertEqual(logs[True][0][0]['pair_counts']['valid_pairs'],logs[True][0][0]['pair_counts']['quiet_pairs']+logs[True][0][0]['pair_counts']['moving_pairs'])
  from direct_contract import validate_optimizer_diagnostics
  r={'optimizer_diagnostics_schema':'direct315_actor_gradients_v1','optimizer_updates':[{'completed_update':u,'learning_rate':rows[-1]['learning_rate_after'],'minibatches':rows} for u,rows in enumerate(logs[True],1)]}
  self.assertEqual(validate_optimizer_diagnostics(r,{'updates':2,'replicas':8,'branch':'quiet_priority'})['minibatches'],40)
  # No guarantee of bitwise CUDA kernels is inferred from this CPU replay.

if __name__=='__main__':unittest.main()
