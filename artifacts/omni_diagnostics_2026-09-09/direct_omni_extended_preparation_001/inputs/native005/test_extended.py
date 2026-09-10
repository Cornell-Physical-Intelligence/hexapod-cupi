"""Fixed-budget successor tests. Synthetic CPU evidence, no simulator allocation."""
import ast, copy, importlib.util, json, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch, yaml
from direct_config import selection, configure, decision_updates, gradient_updates, extended_protocol
from direct_contract import validate_extended_checkpoints, validate_optimizer_diagnostics, validate_result, runtime_arguments, sha, URDF, CHECKPOINT
from direct_training import learn_and_verify, equal_tree
from test_optimizer_contract import diagnostic_receipt
H=Path(__file__).resolve().parent;P=H.parent/'native_004'

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
parent_config=module('parent004_config',P/'direct_config.py')

def extended_diagnostics():
 r=diagnostic_receipt(500,1024)
 gradient=copy.deepcopy(r['optimizer_updates'][0]['minibatches'][0]['gradient'])
 for update in (100,250,500):
  for mb in (1,20):r['optimizer_updates'][update-1]['minibatches'][mb-1]['gradient']=copy.deepcopy(gradient)
 return r

def fake_evidence(folder,allocation='extended',branch='caps',fail_at=None):
 """Calls the actual training wrapper using RSL's inspected save/update ordering."""
 p=Path(folder);(p/'policy').mkdir();sel=selection('train',allocation,branch,None,None)
 cfg=configure(yaml.safe_load((H.parent/'training/inputs/agent.yaml').read_text()),sel)
 audit={'controls':sel['updates']*24,'replicas':sel['replicas'],'terminations_per_row':[0]*sel['replicas'],'truncations_per_row':[0]*sel['replicas'],'requested_torque_max_per_row_nm':[7.]*sel['replicas'],'applied_torque_max_per_row_nm':[1.6]*sel['replicas'],'requested_saturation_fraction_per_row':[.1]*sel['replicas']}
 for name,key in [('training_trace.npz','trace_sha256'),('training_joint_trace.npz','joint_trace_sha256'),('training_events.json','event_ledger_sha256')]:
  (p/name).write_bytes(b'synthetic raw fixture');audit[key]=sha(p/name)
 env=NS(direct_audit_rows=[],device='cpu');saved=[];alg=NS(learning_rate=1e-5,last_update_diagnostics=[],gradient_update_milestones=gradient_updates(sel),actor=NS(distribution=NS(std_param=torch.ones(18)*.1)))
 d=extended_diagnostics() if allocation=='extended' else diagnostic_receipt(sel['updates'],sel['replicas'])
 cursor=0
 def update():
  nonlocal cursor
  if fail_at==cursor+1:raise RuntimeError('synthetic failed update')
  alg.last_update_diagnostics=copy.deepcopy(d['optimizer_updates'][cursor]['minibatches']);cursor+=1
  return {'surrogate':.1}
 alg.update=update
 alg.storage=NS(num_transitions_per_env=24,num_envs=sel['replicas'])
 runner=NS(current_learning_iteration=1847,cfg=cfg,alg=alg)
 def save(path):
  saved.append(Path(path).name);Path(path).write_bytes(('synthetic-state-iteration-'+str(runner.current_learning_iteration)).encode())
 def learn(num_learning_iterations,init_at_random_ep_len):
  assert init_at_random_ep_len is True
  for it in range(1847,1847+num_learning_iterations):
   env.direct_audit_rows.extend([None]*24);runner.alg.update();runner.current_learning_iteration=it
   if it%cfg['save_interval']==0:runner.save(p/'policy'/('model_'+str(it)+'.pt'))
  runner.save(p/'policy'/('model_'+str(runner.current_learning_iteration)+'.pt'))
 runner.save=save;runner.learn=learn
 def reload(*args):return {'passed':True,'exact_actor_critic_normalizer_optimizer':True,'exact_deterministic_action':True,'optimizer_entries':17,'checkpoint_sha256':sha(p/'policy/final.pt')}
 with patch('direct_training.verify_reload',side_effect=reload),patch('direct_training.export_audit',return_value=audit),patch('direct_training.torch.cuda.is_available',return_value=False):
  if fail_at is not None:
   try:learn_and_verify(env,None,runner,sel,p)
   except RuntimeError:pass
   else:raise AssertionError('failure not propagated')
   assert alg.update is update
   return json.loads((p/'training_receipt.json').read_text()),saved
  receipt=learn_and_verify(env,None,runner,sel,p)
 assert alg.update is update
 identity={'selection':sel,'plan_sha256':'p'*64,'source_manifest_sha256':'s'*64}
 state={'status':'completed','mode':'train','variant':'f050_t060','urdf_sha256':URDF,'plan_sha256':identity['plan_sha256'],'stance_index':0,'direct_selection':sel,'iterations':sel['updates'],'checkpoint_sha256':receipt['final_checkpoint_sha256'],'training_receipt':receipt}
 for name,value in [('state.json',state),('repair_initialization.json',{'checkpoint_sha256':CHECKPOINT,'actor_and_critic_preserved_except_std':True,'observation_normalizers_preserved':True,'optimizer_state_entries':0})]:
  (p/name).write_text(json.dumps(value))
 return receipt,saved,identity,state

class Extended(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(2)
 def test_old_allocation_full_config_and_behavior_sources_exact(self):
  cfg=yaml.safe_load((H.parent/'training/inputs/agent.yaml').read_text())
  for allocation in ('smoke','pilot'):
   for branch in ('curriculum','caps','quiet_priority'):
    before=parent_config.selection('train',allocation,branch,None,None);after=selection('train',allocation,branch,None,None)
    self.assertEqual({**before,'schema':after['schema']},after)
    self.assertEqual(parent_config.configure(cfg,before),configure(cfg,after))
    self.assertEqual(decision_updates(after),[2] if allocation=='smoke' else [10,25,50])
  for name in ('caps.py','curriculum.py','gradient_diagnostics.py','direct_stop_evaluation.py','direct_quiet_metrics.py'):
   self.assertEqual((P/name).read_bytes(),(H/name).read_bytes(),name)
  # Actual PPO update body changes only where sparse update membership is read.
  a=(P/'caps_ppo.py').read_text();b=(H/'caps_ppo.py').read_text().replace('self.current_update in self.gradient_update_milestones','self.current_update in (1,10,25,50)')
  funcs=lambda s:{n.name:ast.dump(n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
  self.assertEqual(funcs(a),funcs(b))
 def test_fixed_extended_capacity_and_explicit_branch(self):
  cfg=yaml.safe_load((H.parent/'training/inputs/agent.yaml').read_text())
  for branch in ('caps','quiet_priority'):
   sel=selection('train','extended',branch,None,500);r=configure(cfg,sel)
   self.assertEqual((sel['replicas'],r['num_steps_per_env'],r['max_iterations'],r['save_interval']),(1024,24,500,1))
   self.assertEqual(r['algorithm']['gradient_update_milestones'],[1,10,25,50,100,250,500])
   self.assertEqual(sel['replicas']*sel['controls_per_update']*sel['updates'],12288000)
   self.assertEqual(runtime_arguments('train','extended',branch)[-1],'500')
  for n in (50,499,501,True):
   with self.assertRaises(ValueError):selection('train','extended','caps',None,n)
  self.assertEqual(extended_protocol()['train_deadline_seconds'],1800)
 def test_500_actual_wrapper_relative_saves_decisions_and_receipt(self):
  with tempfile.TemporaryDirectory() as d:
   r,saves,identity,_=fake_evidence(d)
   self.assertEqual(len(set(x for x in saves if x.startswith('model_'))),500)
   self.assertEqual(saves[-3:],['model_2346.pt','model_2346.pt','final.pt'])
   self.assertEqual(len(saves),502);self.assertEqual(r['updates_completed'],500)
   self.assertEqual(list(r['decision_checkpoints']),['1','10','25','50','100','250','500'])
   self.assertEqual(r['decision_checkpoints']['50']['native_iteration'],1896)
   self.assertEqual(r['decision_checkpoints']['500']['native_iteration'],2346)
   self.assertEqual(validate_result(d,'train',identity)['optimizer_diagnostics']['sparse_actor_gradient_rows'],14)
   self.assertEqual(r['extended_readback']['ordinary_save_interval'],1)
 def test_parent50_wrapper_save_calls_unchanged(self):
  with tempfile.TemporaryDirectory() as d:
   r,saves,identity,_=fake_evidence(d,'pilot')
   self.assertNotIn('extended_readback',r);self.assertEqual(len(saves),52)
   self.assertEqual(saves[:2],['model_1847.pt','model_1848.pt']);self.assertEqual(saves[-3:],['model_1896.pt','model_1896.pt','final.pt'])
   self.assertEqual(list(r['decision_checkpoints']),['10','25','50']);self.assertTrue(validate_result(d,'train',identity)['complete'])
 def test_missing_stale_mislabelled_decision_and_failed_state_reject(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);r,_,identity,state=fake_evidence(p)
   for change in ({'status':'failed'},{'iterations':50},{'training_receipt':None}):
    (p/'state.json').write_text(json.dumps({**state,**change}))
    with self.assertRaises((ValueError,TypeError)):validate_result(p,'train',identity)
   (p/'state.json').write_text(json.dumps(state))
   q=p/'policy/model_1946.pt';original=q.read_bytes();q.unlink()
   with self.assertRaises(ValueError):validate_result(p,'train',identity)
   q.write_bytes(original)
   for key,value in [('native_iteration',1945),('completed_updates',99)]:
    changed=copy.deepcopy(r);changed['decision_checkpoints']['100'][key]=value
    with self.assertRaises(ValueError):validate_extended_checkpoints(p,changed)
   q.write_bytes(b'stale checkpoint')
   with self.assertRaises(ValueError):validate_result(p,'train',identity)
   q.write_bytes(original);(p/'policy/decision_500.pt').write_bytes(b'wrong final decision')
   with self.assertRaises(ValueError):validate_result(p,'train',identity)
 def test_failed_collection_keeps_prefix_and_cannot_claim_completion(self):
  with tempfile.TemporaryDirectory() as d:
   receipt,saves=fake_evidence(d,fail_at=101)
   self.assertFalse(receipt['complete']);self.assertEqual(receipt['updates_completed'],100)
   self.assertIn('synthetic failed update',receipt['error']);self.assertEqual(len(saves),100)
   self.assertFalse((Path(d)/'policy/final.pt').exists())
   with self.assertRaises(ValueError):validate_extended_checkpoints(d,receipt)
 def test_changed_extended_runtime_storage_or_cadence_fails_before_collection(self):
  sel=selection('train','extended','caps',None,None)
  cfg=configure(yaml.safe_load((H.parent/'training/inputs/agent.yaml').read_text()),sel)
  for kind in ('storage_controls','storage_rows','save_interval','max_iterations','gradient_cadence'):
   with tempfile.TemporaryDirectory() as d:
    current=copy.deepcopy(cfg);storage=NS(num_transitions_per_env=24,num_envs=1024)
    milestones=gradient_updates(sel)
    if kind=='storage_controls':storage.num_transitions_per_env=500
    if kind=='storage_rows':storage.num_envs=128
    if kind=='save_interval':current['save_interval']=50
    if kind=='max_iterations':current['max_iterations']=50
    if kind=='gradient_cadence':milestones=(1,10,25,50)
    def untouched():raise AssertionError('optimizer was reached')
    alg=NS(update=untouched,storage=storage,gradient_update_milestones=milestones)
    runner=NS(current_learning_iteration=1847,cfg=current,alg=alg)
    with patch('direct_training.export_audit',return_value={'controls':0}),patch('direct_training.torch.cuda.is_available',return_value=False):
     with self.assertRaisesRegex(RuntimeError,'allocation/save/diagnostic'):learn_and_verify(NS(),None,runner,sel,d)
    receipt=json.loads((Path(d)/'training_receipt.json').read_text())
    self.assertFalse(receipt['complete']);self.assertEqual(receipt['updates_completed'],0);self.assertIs(alg.update,untouched)
 def test_sparse_masks_complete_10000_minibatches_and_missing_extra_reject(self):
  selected=selection('train','extended','caps',None,None);r=extended_diagnostics()
  result=validate_optimizer_diagnostics(r,selected);self.assertEqual((result['minibatches'],result['sparse_actor_gradient_rows']),(10000,14))
  for u,m in ((100,1),(250,20),(500,20)):
   old=r['optimizer_updates'][u-1]['minibatches'][m-1]['gradient'];r['optimizer_updates'][u-1]['minibatches'][m-1]['gradient']=None
   with self.assertRaises(ValueError):validate_optimizer_diagnostics(r,selected)
   r['optimizer_updates'][u-1]['minibatches'][m-1]['gradient']=old
  r['optimizer_updates'][98]['minibatches'][0]['gradient']=old
  with self.assertRaises(ValueError):validate_optimizer_diagnostics(r,selected)
 def test_parent_and_new_actual_one_update_loss_rng_and_state_exact(self):
  # Small synthetic real-RSL replay of old50 configuration; no 50/500 training allocation.
  sys.path.append(str(H.parent/'training'))
  prior=module('extended_fixture',H.parent/'training/real_rsl_regression.py')
  parent=module('extended_parent_ppo',P/'caps_ppo.py')
  from rsl_rl.runners import OnPolicyRunner
  from caps import CapsPairWrapper
  from caps_ppo import CapsPPO
  from checkpoint_load import load_repair_checkpoint
  outputs=[]
  for cls in (parent.CapsPPO,CapsPPO):
   torch.manual_seed(157);env=CapsPairWrapper(prior.Synthetic());cfg=prior.cfg('caps');cfg['max_iterations']=50
   with tempfile.TemporaryDirectory() as d,patch('caps_ppo.CapsPPO',cls):
    runner=OnPolicyRunner(env,cfg,log_dir=d,device='cpu')
    load_repair_checkpoint(runner,H.parents[1]/'ppo_repair_003_preparation/original/policy.pt',{'checkpoint_sha256':prior.CP_SHA,'exploration_std':.1,'entropy_coef':0.,'optimizer':'reset','learning_rate':5e-5})
    losses=[];old_update=runner.alg.update
    def update():
     result=old_update();losses.append(copy.deepcopy(result));return result
    runner.alg.update=update;runner.learn(1,init_at_random_ep_len=False)
    outputs.append({'state':copy.deepcopy(runner.alg.save()),'losses':losses,'rng':torch.get_rng_state().clone(),'noise_rng':runner.alg.regularizer.generator.get_state().clone(),'history':env.env.hist.clone(),'lr':runner.alg.learning_rate,'diagnostics':copy.deepcopy(runner.alg.last_update_diagnostics)})
    if getattr(runner.logger,'writer',None):runner.logger.writer.close()
  self.assertTrue(equal_tree(outputs[0],outputs[1]))

if __name__=='__main__':unittest.main()
