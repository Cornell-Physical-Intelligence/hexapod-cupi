"""Real installed-version PPO; synthetic episodes exercise reset and bootstrap seams."""
import copy
import tempfile
import unittest
from pathlib import Path
import torch
from tensordict import TensorDict
from masked_runner import MaskedRunner
from moving_runner import (runner_config, initialize_scratch, contract, BINDING_KEYS,
                           save_checkpoint, reload_checkpoint)


class Episodes:
    def __init__(self, n=4):
        self.num_envs=n; self.num_actions=18; self.device='cpu'
        self.max_episode_length=2200; self.episode_length_buf=torch.zeros(n, dtype=torch.long)
        self.cfg={'provenance':'synthetic episode/mask fixture; no physical admission'}
        self.episodes=torch.zeros(n,dtype=torch.long); self.warm=torch.zeros(n,dtype=torch.long)
        self.control=0; self.valid_total=0; self.events=[]

    def get_observations(self):
        valid=self.warm==0
        x=torch.full((self.num_envs,846), float(self.control%7)*.001)
        x[:,0]=self.episodes.float(); x[:,1]=torch.arange(self.num_envs)*.1
        # Deliberately huge finite disabled data proves normalization exclusion.
        x[~valid]=1e6
        return TensorDict({'policy':x,'critic':torch.cat([x,torch.zeros(self.num_envs,3)],-1),
            'learning_valid':valid[:,None], 'episode_id':self.episodes[:,None].clone()},
            batch_size=[self.num_envs], device='cpu')

    def step(self, actions):
        assert torch.equal(actions[self.warm>0], torch.zeros_like(actions[self.warm>0]))
        active=self.warm==0; previous=self.episodes.clone()
        self.control+=1; self.episode_length_buf+=active
        term=torch.zeros(self.num_envs,dtype=torch.bool); trunc=term.clone()
        if self.control==17: term[0]=True
        if self.control==29: trunc[1]=True
        if self.control==300: term[2]=True
        self.valid_total+=int(active.sum())
        final=self.get_observations().clone()
        final['policy'][:,2]=.25; final['critic'][:,2]=.25
        reward=-.01*actions.square().sum(-1)+actions[:,0]*.05
        reward[term]=-3.; reward[trunc]=1.; reward[~active]=123456.
        self.warm=torch.clamp(self.warm-1,min=0)
        self.warm[term|trunc]=20
        self.episodes[term|trunc]+=1; self.episode_length_buf[term|trunc]=0
        extra={'learnable':active,'terminated':term,'truncated':trunc,'fatal':torch.zeros_like(term),
            'episode_id':previous,'final_observation':final,'final_observation_valid':active.clone(),
            'final_episode_id':previous.clone()}
        self.events.append(copy.deepcopy(extra))
        return self.get_observations(),reward,term|trunc,{'masked_transition':extra}


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def runner(self):
        torch.manual_seed(157)
        env=Episodes(); logs=tempfile.TemporaryDirectory(); self.addCleanup(logs.cleanup)
        runner=MaskedRunner(env,runner_config(),log_dir=logs.name,device='cpu')
        initialize_scratch(runner)
        return env,runner

    def test_actual_two_updates_and_exact_reload(self):
        env,runner=self.runner()
        identity=contract('synthetic_masked_CPU_integration',{k:'0'*64 for k in BINDING_KEYS})
        audits=[]
        with tempfile.TemporaryDirectory() as d:
            save_checkpoint(runner,Path(d)/'initial.pt',identity)
            reports=runner.collect_updates(2,maximum_updates=2,
                after_update=lambda _:audits.append(copy.deepcopy(runner.alg.last_return_audit)))
            self.assertEqual(env.control,512); self.assertEqual(runner.current_learning_iteration,2)
            self.assertEqual(runner.alg.normalizer_rows,env.valid_total)
            self.assertEqual(env.valid_total,4*512-60)
            self.assertEqual(sum(r['finite_terminal_transitions'] for r in reports),2)
            self.assertEqual(sum(r['time_limit_transitions'] for r in reports),1)
            self.assertEqual(float(audits[0]['returns'][16,0,0]),-3.)
            self.assertAlmostEqual(float(audits[0]['returns'][28,1,0]),
                1.+.99*float(audits[0]['next_value'][28,1]),places=6)
            self.assertTrue((audits[0]['returns'][17:37,0]==0).all())
            mask=audits[1]['learnable'].flatten()
            for ids in runner.alg.storage.last_batch_indices:self.assertTrue(mask[ids].all())
            for name,value in runner.alg.actor.state_dict().items():
                if 'mean' in name:self.assertLess(float(value.abs().max()),100.)
            self.assertEqual(len(runner.alg.optimizer.state),17)
            save_checkpoint(runner,Path(d)/'final.pt',identity)
            runner.alg.eval_mode()
            with torch.inference_mode():fixed=env.get_observations();before=runner.alg.actor(fixed).clone()
            with torch.no_grad():next(runner.alg.actor.parameters()).add_(1.)
            next(iter(runner.alg.optimizer.state.values()))['exp_avg'].add_(1.)
            result=reload_checkpoint(runner,Path(d)/'final.pt',identity)
            with torch.inference_mode():after=runner.alg.actor(fixed)
            self.assertTrue(torch.equal(before,after));self.assertTrue(result['actor_critic_normalizer_and_optimizer_exact'])
            with self.assertRaises(FileExistsError):save_checkpoint(runner,Path(d)/'final.pt',identity)
            bad=copy.deepcopy(identity);bad['lineage']='old_stand_smoke'
            with self.assertRaises(ValueError):reload_checkpoint(runner,Path(d)/'final.pt',bad)
            with self.assertRaises(ValueError):runner.collect_updates(1,maximum_updates=2)

    def test_wrong_reset_bootstrap_and_fatal_rejected(self):
        for bad in ('episode','valid','fatal','mask'):
            env,runner=self.runner();obs=env.get_observations()
            with torch.inference_mode():
                action=runner.alg.act(obs); nxt,r,d,extras=env.step(action)
                x=extras['masked_transition']
                if bad=='episode':x['final_episode_id'][0]+=1
                if bad=='valid':x['final_observation_valid'][0]=False
                if bad=='fatal':x['fatal'][0]=True
                if bad=='mask':x['learnable'][0]=False
                with self.assertRaises((ValueError,RuntimeError)):runner.alg.process_env_step(nxt,r,d,extras)
            self.assertEqual(runner.alg.storage.step,0)
            self.assertFalse(runner.alg.optimizer.state)


if __name__=='__main__':unittest.main()
