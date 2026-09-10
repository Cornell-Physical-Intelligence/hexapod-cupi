"""Real RSL normalizer-update/load regression; no optimizer or physics step."""
import copy, hashlib, importlib.util, json, sys, tempfile, unittest
from pathlib import Path
import torch
from direct_training import verify_reload, equal_tree

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'training'

def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module

# Pin successor imports before loading the historical synthetic fixture.
# Otherwise that fixture can cache training/caps.py for later discovery tests.
import caps as current_caps
assert Path(current_caps.__file__).resolve()==HERE/'caps.py'
sys.path.insert(0,str(PRIOR))
try: prior=load_file('_prior_reload_fixture',PRIOR/'real_rsl_regression.py')
finally:sys.path.pop(0)
old=load_file('_native002_reload',HERE.parent/'native_002/direct_training.py')

class InferenceReload(unittest.TestCase):
    def runner(self):
        from rsl_rl.runners import OnPolicyRunner
        from caps import CapsPairWrapper
        torch.set_num_threads(2)
        env=CapsPairWrapper(prior.Synthetic())
        runner=OnPolicyRunner(env,prior.cfg('caps'),device='cpu')
        runner.load(str(PRIOR/'cpu_smoke_002/caps/final.pt'),map_location='cpu')
        return runner,env
    def dirty_and_save(self,runner,env,path):
        # Exact RSL PPO.process_env_step normalization calls under runner.learn's
        # inference context. No random samples, learner update or simulated step.
        with torch.inference_mode():
            obs=env.get_observations()
            runner.alg.actor.update_normalization(obs)
            runner.alg.critic.update_normalization(obs)
        for model in [runner.alg.actor,runner.alg.critic]:
            self.assertTrue(torch.is_inference(model.obs_normalizer._std))
        state=runner.alg.save();state.update(iter=runner.current_learning_iteration,infos=None)
        torch.save(state,path)
    def test_actual_update_reproduces_old_failure_and_finalizer_restores_exactly(self):
        runner,env=self.runner()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'final.pt';self.dirty_and_save(runner,env,path)
            expected=copy.deepcopy(runner.alg.save());checkpoint_hash=hashlib.sha256(path.read_bytes()).hexdigest()
            parameters=[id(p) for m in [runner.alg.actor,runner.alg.critic] for p in m.parameters()]
            optimizer=id(runner.alg.optimizer);rng=torch.random.get_rng_state().clone()
            with self.assertRaisesRegex(RuntimeError,'Inplace update to inference tensor outside InferenceMode'):
                old.verify_reload(runner,env,path)
            result=verify_reload(runner,env,path)
            self.assertTrue(result['passed']);self.assertEqual(result['optimizer_entries'],17)
            self.assertEqual(result['inference_buffers_cloned_after_learning'],['actor.obs_normalizer._std','critic.obs_normalizer._std'])
            self.assertTrue(equal_tree(expected,runner.alg.save()))
            self.assertEqual(parameters,[id(p) for m in [runner.alg.actor,runner.alg.critic] for p in m.parameters()])
            self.assertEqual(optimizer,id(runner.alg.optimizer));self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            self.assertEqual(checkpoint_hash,hashlib.sha256(path.read_bytes()).hexdigest());self.assertEqual(env.t,0)
    def test_ordinary_buffers_are_not_replaced(self):
        runner,env=self.runner();before={k:id(v) for k,v in runner.alg.actor.named_buffers()}
        result=verify_reload(runner,env,PRIOR/'cpu_smoke_002/caps/final.pt')
        self.assertEqual(result['inference_buffers_cloned_after_learning'],[])
        self.assertEqual(before,{k:id(v) for k,v in runner.alg.actor.named_buffers()})
    def test_changed_optimizer_checkpoint_remains_rejected(self):
        runner,env=self.runner()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'final.pt';self.dirty_and_save(runner,env,path)
            state=torch.load(path,weights_only=False);key=next(iter(state['optimizer_state_dict']['state']))
            state['optimizer_state_dict']['state'][key]['exp_avg'].add_(.1);torch.save(state,path)
            expected=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError,'Strict actor/critic/normalizer/optimizer reload differs'):
                verify_reload(runner,env,path)
            self.assertEqual(expected,hashlib.sha256(path.read_bytes()).hexdigest())

if __name__=='__main__':unittest.main()
