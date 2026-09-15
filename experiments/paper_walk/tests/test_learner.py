"""CPU algorithm tests. Synthetic fixtures are not robot or walking evidence."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FROZEN_LEARNER = (ROOT.parents[1] / 'artifacts/restart_2026-09-14/paper_walk_execution_001'
                  / 'source_006/learner.py')
FROZEN_SHA256 = 'ebfc8a691be0c4b750e2174ab983cd36c636b54138780e7babac8d1f7cf032a2'
sys.path.insert(0,str(ROOT))
from learner import (Config, PPOLearner, MotionPrior, generalized_advantage,
                     exact_equal, OBS_WIDTH, CRITIC_WIDTH)


class SyntheticEnv:
    """Two rows with visibly distinct terminal/reset AMP states."""
    def __init__(self, n=2):
        self.n=n
        self.counter=torch.zeros(n,dtype=torch.int64)
        self.obs=torch.zeros(n,OBS_WIDTH)
        self.amp=torch.zeros(n,61)
        self.reset_calls=[]

    def packet(self):
        return {"obs":self.obs.clone(),"critic":torch.cat((self.obs,torch.zeros(self.n,3)),-1),"amp":self.amp.clone()}

    def reset(self, indices=None):
        indices=torch.arange(self.n) if indices is None else indices.cpu()
        self.reset_calls.append(indices.tolist())
        self.counter[indices]=0
        self.obs[indices]=0
        self.amp[indices]=-10
        return self.packet()

    def step(self, action):
        self.counter+=1
        self.obs[:,-18:]=action.cpu().clamp(-1,1)
        self.amp=self.counter[:,None].float().expand(-1,61).clone()
        terminal=torch.tensor([True,False]) & (self.counter==2)
        truncated=torch.tensor([False,True]) & (self.counter==3)
        return {**self.packet(),"reward":1-action.cpu().square().mean(-1),
                "terminated":terminal,"truncated":truncated}


def tiny():
    return Config(num_envs=2,rollout_steps=4,epochs=2,minibatches=2,
                  actor_hidden=(16,),memory_hidden=(16,),estimator_hidden=(8,),
                  critic_hidden=(16,),discriminator_hidden=(16,8))


class LearnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def prior(self, directory):
        path=Path(directory)/"prior.npz"
        rng=np.random.default_rng(1)
        states=rng.normal(size=(24,61)).astype(np.float32)
        np.savez(path,states=states,next_states=states+.01)
        return path

    def assert_bytes_equal(self, first, second):
        """Also catch signed-zero differences that torch.equal would overlook."""
        if isinstance(first, torch.Tensor):
            self.assertIsInstance(second, torch.Tensor)
            self.assertEqual((first.dtype, first.shape), (second.dtype, second.shape))
            self.assertEqual(first.detach().cpu().contiguous().numpy().tobytes(),
                             second.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(first, np.ndarray):
            self.assertEqual((first.dtype, first.shape), (second.dtype, second.shape))
            self.assertEqual(first.tobytes(), second.tobytes())
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_bytes_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertIs(type(first), type(second))
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_bytes_equal(a, b)
        else:
            self.assertEqual(first, second)

    def test_timeout_bootstrap_fall_no_bootstrap_and_no_gae_leak(self):
        rewards=torch.tensor([[1.,1.],[7.,9.]])
        values=torch.tensor([[2.,2.],[3.,4.]])
        next_values=torch.tensor([[5.,5.],[8.,8.]])
        terminated=torch.tensor([[True,False],[False,False]])
        truncated=torch.tensor([[False,True],[False,False]])
        advantage,returns=generalized_advantage(rewards,values,next_values,terminated,truncated,.9,.95)
        self.assertAlmostEqual(float(advantage[0,0]),-1.)
        self.assertAlmostEqual(float(advantage[0,1]),3.5)
        self.assertAlmostEqual(float(returns[0,1]),5.5)

    def test_amp_style_endpoints_detached_and_input_gradient_penalty(self):
        amp=MotionPrior(tiny())
        first=torch.randn(3,61,requires_grad=True);second=torch.randn(3,61)
        for parameter in amp.discriminator.parameters():
            torch.nn.init.zeros_(parameter)
        amp.discriminator[-1].bias.data.fill_(1.)
        reward=amp.style_reward(first,second)
        self.assertTrue(torch.equal(reward,torch.ones(3)))
        self.assertFalse(reward.requires_grad)
        amp.discriminator[-1].bias.data.fill_(-1.)
        self.assertTrue(torch.equal(amp.style_reward(first,second),torch.zeros(3)))
        fresh=MotionPrior(tiny())
        loss,metrics=fresh.loss(first,second,first.detach(),second,10.)
        loss.backward()
        self.assertIsNone(first.grad)
        self.assertGreater(float(metrics['gradient_penalty']),0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in fresh.discriminator.parameters()))

    def test_real_updates_both_networks_reset_pairing_and_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);prior=self.prior(root)
            learner=PPOLearner(SyntheticEnv(),prior,root/'first',tiny(),device='cpu')
            initial_actor={k:v.clone() for k,v in learner.model.state_dict().items()}
            initial_d={k:v.clone() for k,v in learner.amp.discriminator.state_dict().items()}
            rollout=learner.collect()
            # terminal state 2 is recorded, never replaced by reset state -10.
            self.assertEqual(float(rollout['amp_next'][1,0,0]),2.)
            self.assertEqual(float(rollout['amp'][2,0,0]),-10.)
            self.assertEqual(float(rollout['amp_next'][2,1,0]),3.)
            self.assertEqual(float(rollout['amp'][3,1,0]),-10.)
            learner.update(rollout)
            self.assertFalse(exact_equal(initial_actor,learner.model.state_dict()))
            self.assertFalse(exact_equal(initial_d,learner.amp.discriminator.state_dict()))
            self.assertEqual((learner.optimizer_steps,learner.discriminator_steps),(4,4))
            probe=torch.randn(2,231)
            expected=learner.act(probe).clone()
            path=root/'checkpoint.pt';receipt=learner.save(path)
            self.assertTrue(receipt['strict_serialization_readback'])
            expected_random=torch.rand(5)
            restored=PPOLearner(SyntheticEnv(),prior,root/'second',tiny(),device='cpu')
            restored.load(path)
            self.assertTrue(torch.equal(expected,restored.act(probe)))
            self.assertTrue(torch.equal(expected_random,torch.rand(5)))
            self.assertTrue(exact_equal(learner.optimizer.state_dict(),restored.optimizer.state_dict()))
            self.assertTrue(exact_equal(learner.discriminator_optimizer.state_dict(),restored.discriminator_optimizer.state_dict()))
            self.assertEqual(restored.updates,1)

    def test_selective_reset_history_corruption_is_rejected(self):
        class BadEnv(SyntheticEnv):
            def reset(self,indices=None):
                output=super().reset(indices)
                if indices is not None:
                    self.obs[:]=99
                    output=self.packet()
                return output
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            learner=PPOLearner(BadEnv(),self.prior(root),root/'out',tiny(),device='cpu')
            with self.assertRaisesRegex(ValueError,'unaffected environment'):
                learner.collect()

    def test_training_loop_saves_first_and_final_real_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            learner=PPOLearner(SyntheticEnv(),self.prior(root),root/'out',tiny(),device='cpu')
            result=learner.train(2,checkpoint_interval=25)
            self.assertTrue((root/'out/checkpoint_000001.pt').is_file())
            self.assertTrue((root/'out/checkpoint_000002.pt').is_file())
            self.assertEqual(result['update'],2)
            self.assertEqual(learner.transitions,16)

    def test_reporting_successor_preserves_frozen_source006_updates_and_rng_bytes(self):
        # A pinned, read-only reference is used only in this regression test;
        # production must never import artifact source or relax checkpoint IDs.
        if not FROZEN_LEARNER.is_file():
            self.skipTest('Local immutable source006 regression fixture is unavailable')
        self.assertEqual(hashlib.sha256(FROZEN_LEARNER.read_bytes()).hexdigest(), FROZEN_SHA256)
        name = '_paper_walk_frozen_source006_reporting_reference'
        spec = importlib.util.spec_from_file_location(name, FROZEN_LEARNER)
        frozen = importlib.util.module_from_spec(spec)
        sys.modules[name] = frozen
        try:
            spec.loader.exec_module(frozen)
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                prior = self.prior(root)
                for seed in (20260914, 73):
                    with self.subTest(seed=seed):
                        config = tiny()
                        config.seed = seed
                        old = frozen.PPOLearner(SyntheticEnv(), prior, root/f'old_{seed}',
                                                frozen.Config(**{k:v for k,v in asdict(config).items()
                                                                 if k in frozen.Config.__dataclass_fields__}), device='cpu')
                        new = PPOLearner(SyntheticEnv(), prior, root/f'new_{seed}', config, device='cpu')
                        rng = PPOLearner._rng()
                        for _ in range(2):
                            PPOLearner._restore_rng(rng)
                            old_rollout = old.collect()
                            old_metrics = old.update(old_rollout)
                            old_rng = PPOLearner._rng()
                            PPOLearner._restore_rng(rng)
                            new_rollout = new.collect()
                            new_metrics = new.update(new_rollout)
                            rng = PPOLearner._rng()
                            self.assert_bytes_equal(old_rollout, new_rollout)
                            self.assert_bytes_equal(old_rng, rng)
                            for key in ('model', 'amp', 'optimizer', 'discriminator_optimizer'):
                                self.assert_bytes_equal(getattr(old, key).state_dict(), getattr(new, key).state_dict())
                            for key in ('updates', 'transitions', 'optimizer_steps', 'discriminator_steps', 'episodes', 'bc_steps'):
                                self.assertEqual(getattr(old, key), getattr(new, key))
                            renamed = {'policy_loss': 'discriminator_policy_loss',
                                       'expert_loss': 'discriminator_expert_loss',
                                       'actor_grad_norm': 'model_grad_norm'}
                            for key, value in old_metrics.items():
                                self.assertEqual(value, new_metrics[renamed.get(key, key)])
                        old_checkpoint = root/f'old_{seed}.pt'
                        old.save(old_checkpoint)
                        with self.assertRaisesRegex(ValueError, 'Checkpoint model/config/prior identity differs'):
                            new.load(old_checkpoint)
        finally:
            sys.modules.pop(name, None)

    def test_reported_norms_are_preclip_and_ppo_loss_is_not_discriminator_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = tiny()
            config.epochs = config.minibatches = 1
            learner = PPOLearner(SyntheticEnv(), self.prior(root), root/'out', config, device='cpu')
            rollout = learner.collect()
            # Deliberately large value error makes the difference between preclip
            # and postclip norms observable, while actor gradients remain active.
            rollout['returns'] = torch.arange(8, dtype=torch.float32).reshape(4, 2) + 100.
            rollout['value'] = torch.arange(8, dtype=torch.float32).reshape(4, 2) * .5
            with torch.no_grad():
                distribution, _ = learner.model.distribution(rollout['obs'].reshape(-1, OBS_WIDTH))
                ratio = (distribution.log_prob(rollout['actions'].reshape(-1, 18)).sum(-1)
                         - rollout['log_prob'].flatten()).exp()
                advantage = rollout['advantage'].flatten()
                advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
                expected_policy_loss = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean()
            measured = {}
            original_clip = torch.nn.utils.clip_grad_norm_
            model_ids = {id(p) for p in learner.model.parameters()}

            def inspect_clip(parameters, *args, **kwargs):
                parameters = list(parameters)
                if {id(p) for p in parameters} == model_ids:
                    sums = {'actor': 0., 'critic': 0.}
                    for name, parameter in learner.model.named_parameters():
                        if parameter.grad is not None:
                            group = 'critic' if name.startswith('critic.') else 'actor'
                            sums[group] += float(parameter.grad.detach().double().square().sum())
                    measured.update({key: value**.5 for key, value in sums.items()})
                    measured['model'] = sum(sums.values())**.5
                return original_clip(parameters, *args, **kwargs)

            with patch('torch.nn.utils.clip_grad_norm_', side_effect=inspect_clip):
                metrics = learner.update(rollout)
            for group in ('actor', 'critic', 'model'):
                self.assertAlmostEqual(metrics[group+'_grad_norm'], measured[group], delta=measured[group]*2e-6)
            self.assertGreater(metrics['model_grad_norm'], config.max_grad_norm)
            self.assertGreater(metrics['actor_grad_norm'], 0.)
            self.assertGreater(metrics['critic_grad_norm'], 0.)
            self.assertAlmostEqual(metrics['policy_loss'], float(expected_policy_loss), delta=1e-6)
            self.assertNotAlmostEqual(metrics['policy_loss'], metrics['discriminator_policy_loss'])
            self.assertNotIn('expert_loss', metrics)
            self.assertEqual(metrics['return_mean'], 103.5)
            self.assertAlmostEqual(metrics['return_std'], np.sqrt(5.25), places=6)
            self.assertAlmostEqual(metrics['explained_variance'], .75, places=6)

    def test_explained_variance_is_null_for_constant_returns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            learner = PPOLearner(SyntheticEnv(), self.prior(root), root/'out', tiny(), device='cpu')
            rollout = learner.collect()
            rollout['returns'] = torch.full_like(rollout['returns'], 7.)
            metrics = learner.update(rollout)
            self.assertEqual(metrics['return_mean'], 7.)
            self.assertEqual(metrics['return_std'], 0.)
            self.assertIsNone(metrics['explained_variance'])
            json.dumps(metrics, allow_nan=False)


if __name__=='__main__':
    unittest.main()
