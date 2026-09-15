"""CPU policy-update boundaries; synthetic data never establishes robot behavior."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.distributions import Independent, Normal, kl_divergence

from experiments.paper_walk.tests.test_learner import SyntheticEnv, tiny, FROZEN_LEARNER, FROZEN_SHA256
from learner import PPOLearner, diagonal_gaussian_kl, exact_equal, SCHEMA


class StabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def data(self, root, *, velocity=True, states=True):
        rng = np.random.default_rng(831)
        values = rng.normal(size=(24,61)).astype(np.float32)
        arrays = dict(states=values, next_states=values+.01,
                      observations=rng.normal(size=(24,231)).astype(np.float32),
                      actions=rng.normal(size=(24,18)).astype(np.float32)*.1)
        if velocity:
            arrays['velocity_targets_navigation_mps'] = np.stack((-values[:,37],values[:,36],values[:,38]),-1)
        if not states:
            del arrays['states']
        path = Path(root)/'data.npz'
        np.savez(path, **arrays)
        return path

    def learner(self, root, config=None, name='run'):
        return PPOLearner(SyntheticEnv(), self.data(root), Path(root)/name,
                          tiny() if config is None else config, device='cpu')

    @staticmethod
    def flat(rollout):
        return {k:v.reshape(-1,*v.shape[2:]) for k,v in rollout.items()}

    def test_analytic_kl_direction_dimension_reduction_and_invalid_values(self):
        old_mean = torch.zeros(3,18,dtype=torch.float64)
        old_std = torch.full_like(old_mean,.1)
        self.assertTrue(torch.equal(diagonal_gaussian_kl(old_mean,old_std,old_mean,old_std),torch.zeros(3,dtype=torch.float64)))
        shift = old_mean + .02
        actual = diagonal_gaussian_kl(old_mean,old_std,shift,old_std)
        torch.testing.assert_close(actual,torch.full((3,),18*.02**2/(2*.1**2),dtype=torch.float64))
        new_std=old_std*2
        expected=kl_divergence(Independent(Normal(old_mean,old_std),1),Independent(Normal(shift,new_std),1))
        actual=diagonal_gaussian_kl(old_mean,old_std,shift,new_std)
        torch.testing.assert_close(actual,expected,rtol=0,atol=1e-14)
        self.assertFalse(torch.equal(actual,diagonal_gaussian_kl(shift,new_std,old_mean,old_std)))
        for bad in (torch.zeros_like(old_std),old_std*float('nan'),old_std*float('inf'),-old_std):
            with self.assertRaises(ValueError):diagonal_gaussian_kl(old_mean,bad,shift,new_std)
        with self.assertRaises(ValueError):diagonal_gaussian_kl(old_mean[:,:17],old_std,shift,new_std)

    def test_configuration_rejects_invalid_kl_bc_rates_and_velocity_options(self):
        for field in ('target_kl','bc_learning_rate'):
            for value in (0.,-1.,float('nan'),float('inf'),True):
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                    replace(tiny(),**{field:value}).validate()
        for changes in ({'bc_optimizer':'unknown'},{'bc_optimizer':'separate'},
                        {'bc_learning_rate':1e-5},{'bc_velocity_coefficient':-.1},
                        {'bc_velocity_coefficient':float('nan')},{'bc_velocity_coefficient':True},
                        {'bc_detach_velocity':1}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):replace(tiny(),**changes).validate()
        replace(tiny(),target_kl=.02,learning_rate=1e-4,bc_optimizer='separate',
                bc_learning_rate=3e-4,bc_velocity_coefficient=1.,bc_detach_velocity=True).validate()

    def test_guard_is_rng_stat_invariant_and_sees_estimator_changes(self):
        with TemporaryDirectory() as root:
            learner=self.learner(root,replace(tiny(),target_kl=.02))
            rollout=learner.collect();data=self.flat(rollout)
            state=deepcopy(learner.model.state_dict());rng=learner._rng()
            self.assertLess(learner.policy_kl(data),1e-12)
            learner.act(data['obs'])
            self.assertTrue(exact_equal(state,learner.model.state_dict()))
            self.assertTrue(exact_equal(rng,learner._rng()))
            with torch.no_grad():learner.model.estimator[-1].bias.add_(1.)
            self.assertGreater(learner.policy_kl(data),0.)
            before=deepcopy(learner.model.state_dict());rng=learner._rng()
            with self.assertRaisesRegex(ValueError,'Stale collection-time'):learner.update(rollout)
            self.assertEqual((learner.optimizer_steps,learner.discriminator_steps),(0,0))
            self.assertTrue(exact_equal(before,learner.model.state_dict()))
            self.assertTrue(exact_equal(rng,learner._rng()))

    def test_crossing_step_retained_model_stops_discriminator_schedule_and_rng_continue(self):
        with TemporaryDirectory() as root:
            learner=self.learner(root,replace(tiny(),target_kl=.02))
            rollout=learner.collect();rng=learner._rng()
            baseline=self.learner(root,tiny(),'baseline')
            for name in ('model','amp','optimizer','discriminator_optimizer'):
                getattr(baseline,name).load_state_dict(deepcopy(getattr(learner,name).state_dict()))
            captured={};original=learner.optimizer.step
            def crossing(*args,**kwargs):
                result=original(*args,**kwargs)
                with torch.no_grad():learner.model.policy[-1].bias.add_(1.)
                captured['model']=deepcopy(learner.model.state_dict())
                captured['optimizer']=deepcopy(learner.optimizer.state_dict())
                return result
            learner._restore_rng(rng)
            with patch.object(learner.optimizer,'step',side_effect=crossing) as model_steps:
                metrics=learner.update(rollout)
            after_rng=learner._rng()
            self.assertEqual(model_steps.call_count,1)
            self.assertTrue(metrics['early_stopped']);self.assertGreater(metrics['kl_final'],.02)
            self.assertEqual((metrics['ppo_batches'],metrics['discriminator_batches']),(1,4))
            self.assertEqual((learner.optimizer_steps,learner.discriminator_steps),(1,4))
            self.assertTrue(exact_equal(captured['model'],learner.model.state_dict()))
            self.assertTrue(exact_equal(captured['optimizer'],learner.optimizer.state_dict()))
            baseline._restore_rng(rng);baseline.update(rollout)
            self.assertTrue(exact_equal(after_rng,baseline._rng()))
            self.assertTrue(exact_equal(learner.amp.state_dict(),baseline.amp.state_dict()))
            self.assertTrue(exact_equal(learner.discriminator_optimizer.state_dict(),baseline.discriminator_optimizer.state_dict()))
            self.assertEqual(len(metrics['kl_after_model_steps']),1)
            with torch.no_grad():
                data=self.flat(rollout);distribution,_=learner.model.distribution(data['obs'])
                log_ratio=distribution.log_prob(data['actions']).sum(-1)-data['log_prob']
                expected_k3=float(((log_ratio.exp()-1)-log_ratio).mean())
            self.assertEqual(metrics['final_full_rollout_k3'],expected_k3)
            self.assertNotEqual(metrics['final_full_rollout_k3'],metrics['approx_kl'])
            json.dumps(metrics,allow_nan=False)

    def test_missing_stale_or_nonfinite_reference_cannot_mutate_optimizers(self):
        for mode in ('missing','log_prob','nonfinite'):
            with self.subTest(mode=mode),TemporaryDirectory() as root:
                learner=self.learner(root,replace(tiny(),target_kl=.02));rollout=learner.collect()
                if mode=='missing':del rollout['old_action_mean']
                if mode=='log_prob':rollout['log_prob']+=1
                if mode=='nonfinite':rollout['old_action_std'][0,0,0]=float('nan')
                state=deepcopy(learner.model.state_dict());rng=learner._rng()
                with self.assertRaises(ValueError):learner.update(rollout)
                self.assertTrue(exact_equal(state,learner.model.state_dict()))
                self.assertTrue(exact_equal(rng,learner._rng()))
                self.assertEqual((learner.optimizer_steps,learner.discriminator_steps,learner.updates),(0,0,0))

    def test_normalizer_drift_uses_same_current_states_and_weights_without_gating(self):
        with TemporaryDirectory() as root:
            learner=self.learner(root,replace(tiny(),target_kl=.02))
            learner.collect()
            probe=learner.current['obs'].clone()
            parameters={k:v.detach().clone() for k,v in learner.model.named_parameters()}
            old_count=learner.model.obs_normalizer.count.clone()
            with torch.no_grad():
                before,_=learner.model.distribution(probe)
                old_mean,old_std=before.mean.clone(),before.stddev.clone()
            learner._normalizers()
            with torch.no_grad():
                after,_=learner.model.distribution(probe)
                expected=float(diagonal_gaussian_kl(old_mean,old_std,after.mean,after.stddev).mean())
            self.assertGreater(learner.model.obs_normalizer.count,old_count)
            self.assertGreater(expected,0)
            self.assertEqual(learner.normalizer_policy_kl,expected)
            self.assertTrue(torch.equal(probe,learner.current['obs']))
            self.assertTrue(exact_equal(parameters,{k:v.detach() for k,v in learner.model.named_parameters()}))
            self.assertEqual((learner.optimizer_steps,learner.discriminator_steps),(0,0))

    def test_bc_detach_only_blocks_action_gradient_into_estimator(self):
        with TemporaryDirectory() as root:
            learner=self.learner(root);obs=torch.randn(8,231)
            mean,velocity=learner.model.actor(obs,detach_velocity=True)
            original,_=learner.model.actor(obs)
            self.assertTrue(torch.equal(mean,original))
            mean.square().mean().backward()
            self.assertTrue(all(p.grad is None for p in learner.model.estimator.parameters()))
            self.assertTrue(any(p.grad is not None and bool(p.grad.any()) for p in learner.model.policy.parameters()))
            learner.optimizer.zero_grad(set_to_none=True)
            _,velocity=learner.model.actor(obs,detach_velocity=True)
            (velocity-1).square().mean().backward()
            self.assertTrue(any(p.grad is not None and bool(p.grad.any()) for p in learner.model.estimator.parameters()))
            self.assertTrue(all(p.grad is None for p in learner.model.policy.parameters()))
            learner.optimizer.zero_grad(set_to_none=True)
            distribution,_=learner.model.distribution(obs)
            distribution.mean.square().mean().backward()
            self.assertTrue(any(p.grad is not None and bool(p.grad.any()) for p in learner.model.estimator.parameters()))

    def test_supervised_separate_bc_keeps_ppo_empty_logs_components_and_restores_exactly(self):
        with TemporaryDirectory() as root:
            cfg=replace(tiny(),target_kl=.02,learning_rate=1e-4,bc_optimizer='separate',bc_learning_rate=3e-4,
                        bc_velocity_coefficient=1.,bc_detach_velocity=True)
            learner=self.learner(root,cfg);prior=Path(root)/'data.npz'
            initial_std=learner.model.log_std.detach().clone();critic=deepcopy(learner.model.critic.state_dict())
            learner.pretrain_bc(prior,51,batch_size=8)
            self.assertEqual(learner.optimizer.state,{})
            self.assertEqual(learner.optimizer.param_groups[0]['lr'],1e-4)
            self.assertEqual(learner.bc_steps,51)
            self.assertTrue(torch.equal(initial_std,learner.model.log_std))
            self.assertTrue(exact_equal(critic,learner.model.critic.state_dict()))
            logs=[json.loads(x) for x in (learner.output/'bc_metrics.jsonl').read_text().splitlines()]
            self.assertEqual([r['bc_step'] for r in logs],[50,51])
            for row in logs:
                self.assertAlmostEqual(row['total_loss'],row['action_mse']+row['velocity_mse_m2_s2'],delta=1e-6)
                self.assertGreater(row['estimator_grad_norm_preclip'],0)
                self.assertGreater(row['model_grad_norm_preclip'],0)
                self.assertGreater(row['gradient_clip_coefficient'],0)
                self.assertLessEqual(row['gradient_clip_coefficient'],1)
                self.assertAlmostEqual(row['gradient_clip_coefficient'],
                    min(1.,cfg.max_grad_norm/(row['model_grad_norm_preclip']+1e-6)),delta=1e-6)
            receipt=json.loads((learner.output/'behavior_cloning.json').read_text())
            self.assertTrue(receipt['bc_optimizer_discarded']);self.assertFalse(receipt['privileged_velocity_is_actor_input'])
            checkpoint=Path(root)/'bc.pt';learner.save(checkpoint)
            restored=self.learner(root,cfg,'restored');restored.load(checkpoint)
            self.assertTrue(exact_equal(learner.model.state_dict(),restored.model.state_dict()))
            self.assertEqual(restored.optimizer.state,{})
            output=restored.update(restored.collect())
            self.assertEqual(output['optimizer_steps'],output['ppo_batches'])
            for state in restored.optimizer.state.values():self.assertEqual(int(state['step']),output['ppo_batches'])
            with self.assertRaisesRegex(ValueError,'only before PPO'):restored.pretrain_bc(prior,1)
            for key,value in (('schema','canonical_paper_walk_ppo_amp_v1'),('learner_sha256','0'*64)):
                payload=torch.load(checkpoint,weights_only=False);payload[key]=value
                invalid=Path(root)/(key+'.pt');torch.save(payload,invalid)
                with self.assertRaises(ValueError):learner.load(invalid)
            wrong=self.learner(root,replace(cfg,target_kl=.01),'wrong')
            with self.assertRaisesRegex(ValueError,'model/config/prior'):wrong.load(checkpoint)
            self.assertEqual(torch.load(checkpoint,weights_only=False)['schema'],SCHEMA)

    def test_velocity_targets_must_be_present_finite_aligned_and_pre_hold(self):
        for mode in ('missing','shape','nan','wrong_frame','missing_states','states_shape','states_nan'):
            with self.subTest(mode=mode),TemporaryDirectory() as root:
                learner=self.learner(root,replace(tiny(),bc_velocity_coefficient=1.))
                path=Path(root)/'data.npz'
                with np.load(path) as raw:arrays={k:raw[k].copy() for k in raw.files}
                if mode=='missing':del arrays['velocity_targets_navigation_mps']
                if mode=='shape':arrays['velocity_targets_navigation_mps']=arrays['velocity_targets_navigation_mps'][:2]
                if mode=='nan':arrays['velocity_targets_navigation_mps'][0,0]=np.nan
                if mode=='wrong_frame':arrays['velocity_targets_navigation_mps'][0,0]+=1
                if mode=='missing_states':del arrays['states']
                if mode=='states_shape':arrays['states']=arrays['states'][:2]
                if mode=='states_nan':arrays['states'][0,0]=np.nan
                np.savez(path,**arrays)
                before=deepcopy(learner.model.state_dict());rng=learner._rng()
                optimizer=deepcopy(learner.optimizer.state_dict())
                discriminator=deepcopy(learner.discriminator_optimizer.state_dict())
                with self.assertRaises(ValueError):learner.pretrain_bc(path,1)
                self.assertTrue(exact_equal(before,learner.model.state_dict()))
                self.assertTrue(exact_equal(rng,learner._rng()))
                self.assertTrue(exact_equal(optimizer,learner.optimizer.state_dict()))
                self.assertTrue(exact_equal(discriminator,learner.discriminator_optimizer.state_dict()))
                self.assertFalse((learner.output/'behavior_cloning.json').exists())
                self.assertFalse((learner.output/'bc_metrics.jsonl').exists())
                self.assertEqual(learner.bc_steps,0)

    def test_unsupervised_bc_defaults_and_separate_optimizer_preserve_legacy_weights_rng(self):
        if not FROZEN_LEARNER.is_file():self.skipTest('Immutable local regression fixture unavailable')
        self.assertEqual(hashlib.sha256(FROZEN_LEARNER.read_bytes()).hexdigest(),FROZEN_SHA256)
        name='_legacy_bc_stability_fixture';spec=importlib.util.spec_from_file_location(name,FROZEN_LEARNER)
        legacy=importlib.util.module_from_spec(spec);sys.modules[name]=legacy;spec.loader.exec_module(legacy)
        try:
            for optimizer in ('shared','separate'):
                with self.subTest(optimizer=optimizer),TemporaryDirectory() as root:
                    cfg=replace(tiny(),bc_optimizer=optimizer,bc_learning_rate=3e-4 if optimizer=='separate' else None,
                                learning_rate=1e-4 if optimizer=='separate' else 3e-4)
                    current=self.learner(root,cfg)
                    old_config=asdict(tiny())
                    old_config={k:v for k,v in old_config.items() if k in legacy.Config.__dataclass_fields__}
                    old=legacy.PPOLearner(SyntheticEnv(),Path(root)/'data.npz',Path(root)/'old',legacy.Config(**old_config),device='cpu')
                    rng=current._rng();old.pretrain_bc(Path(root)/'data.npz',51,batch_size=8);old_rng=old._rng()
                    current._restore_rng(rng);current.pretrain_bc(Path(root)/'data.npz',51,batch_size=8)
                    self.assertTrue(exact_equal(old_rng,current._rng()))
                    self.assertTrue(exact_equal(old.model.state_dict(),current.model.state_dict()))
                    self.assertTrue(exact_equal(old.amp.state_dict(),current.amp.state_dict()))
                    if optimizer=='shared':self.assertTrue(exact_equal(old.optimizer.state_dict(),current.optimizer.state_dict()))
                    else:self.assertEqual(current.optimizer.state,{})
        finally:sys.modules.pop(name,None)


if __name__=='__main__':unittest.main()
