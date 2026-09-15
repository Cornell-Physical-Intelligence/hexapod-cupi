"""CPU rollback proofs; synthetic policies are not physical walking evidence."""
from copy import deepcopy
from dataclasses import asdict, fields, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from experiments.paper_walk.tests import test_learner as _base_tests
from experiments.paper_walk.tests import test_learner_stability as _stability_tests
from experiments.paper_walk.tests.test_learner import SyntheticEnv, tiny, ROOT
from learner import exact_equal


class RollbackTests(unittest.TestCase):
    data = _stability_tests.StabilityTests.data
    learner = _stability_tests.StabilityTests.learner
    flat = staticmethod(_stability_tests.StabilityTests.flat)
    assert_bytes_equal = _base_tests.LearnerTests.assert_bytes_equal

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, **changes):
        return replace(tiny(), target_kl=.02, rollback_kl_steps=True,
                       freeze_actor_obs_normalizer=True, **changes)

    def test_invalid_rollback_configuration(self):
        for changes in ({'target_kl':None,'rollback_kl_steps':True},
                        {'rollback_kl_steps':1}, {'freeze_actor_obs_normalizer':1},
                        {'kl_backtrack_halvings':True}, {'kl_backtrack_halvings':-1},
                        {'kl_backtrack_factor':0}, {'kl_backtrack_factor':1},
                        {'kl_backtrack_factor':float('nan')}, {'kl_backtrack_factor':True},
                        {'zero_accepted_update_limit':0}, {'zero_accepted_update_limit':True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(tiny(), **changes).validate()

    def test_rejected_trials_restore_every_model_adam_byte_and_keep_discriminator_rng_schedule(self):
        with TemporaryDirectory() as root:
            cfg=self.config(epochs=5,minibatches=4)
            learner=self.learner(root,cfg)
            rollout=learner.collect();rng=learner._rng()
            baseline=self.learner(root,replace(cfg,rollback_kl_steps=False),'baseline')
            for name in ('model','amp','optimizer','discriminator_optimizer'):
                getattr(baseline,name).load_state_dict(deepcopy(getattr(learner,name).state_dict()))
            model=deepcopy(learner.model.state_dict());optimizer=deepcopy(learner.optimizer.state_dict())
            original=learner.optimizer.step
            def crossing():
                original()
                with torch.no_grad():learner.model.policy[-1].bias.add_(1.)
            learner._restore_rng(rng)
            with patch.object(learner.optimizer,'step',side_effect=crossing) as steps:
                metrics=learner.update(rollout)
            after_rng=learner._rng()
            self.assertEqual(steps.call_count,4)
            self.assertEqual((metrics['ppo_batches'],metrics['discriminator_batches']),(0,20))
            self.assertEqual(metrics['rejected_model_trials'],4)
            self.assertEqual(metrics['stop_reason'],'no_accepted_step')
            self.assertEqual(metrics['kl_final'],0.)
            for key in ('policy_loss','value_loss','entropy','estimator_loss','approx_kl',
                        'actor_grad_norm','critic_grad_norm','model_grad_norm','clip_fraction'):
                self.assertIsNone(metrics[key])
            self.assert_bytes_equal(model,learner.model.state_dict())
            self.assert_bytes_equal(optimizer,learner.optimizer.state_dict())
            baseline._restore_rng(rng);baseline.update(rollout)
            self.assert_bytes_equal(after_rng,baseline._rng())
            self.assert_bytes_equal(learner.amp.state_dict(),baseline.amp.state_dict())
            self.assert_bytes_equal(learner.discriminator_optimizer.state_dict(),baseline.discriminator_optimizer.state_dict())
            json.dumps(metrics,allow_nan=False)

    def test_retry_is_exactly_one_clean_adam_step_and_cumulative_reference_is_retained(self):
        with TemporaryDirectory() as root:
            learner=self.learner(root,self.config());rollout=learner.collect();data=self.flat(rollout)
            for p in learner.model.parameters():p.grad=torch.full_like(p,.01)
            # Populate Adam moments before the snapshot, without changing the policy reference.
            learner.optimizer.step()
            distribution,_=learner.model.distribution(data['obs'])
            data['old_action_mean']=distribution.mean.detach().clone()
            data['old_action_std']=distribution.stddev.detach().clone()
            model=deepcopy(learner.model.state_dict());optimizer=deepcopy(learner.optimizer.state_dict())
            gradients=[p.grad.clone() for p in learner.model.parameters()];rng=learner._rng()
            original=learner.optimizer.step
            def sensitive():
                original()
                with torch.no_grad():learner.model.policy[-1].bias.add_(200*learner.optimizer.param_groups[0]['lr'])
            with patch.object(learner.optimizer,'step',side_effect=sensitive):
                accepted,kl,attempts=learner._bounded_model_step(data)
            self.assertTrue(accepted);self.assertLessEqual(kl,.02)
            self.assertGreater(len(attempts),1)
            self.assertEqual([x['learning_rate'] for x in attempts],
                             [learner.config.learning_rate*.5**i for i in range(len(attempts))])
            result=deepcopy(learner.model.state_dict());adam=deepcopy(learner.optimizer.state_dict())
            self.assert_bytes_equal(rng,learner._rng())
            learner.model.load_state_dict(model);learner.optimizer.load_state_dict(deepcopy(optimizer))
            for p,g in zip(learner.model.parameters(),gradients):p.grad=g.clone()
            learner.optimizer.param_groups[0]['lr']=attempts[-1]['learning_rate']
            sensitive()
            self.assert_bytes_equal(result,learner.model.state_dict())
            self.assert_bytes_equal(adam,learner.optimizer.state_dict())
            # Second proposal still compares to the original collected distribution,
            # and restarts its LR ladder at the configured base, not the last retry.
            with patch.object(learner.optimizer,'step',side_effect=sensitive):
                accepted2,kl2,attempts2=learner._bounded_model_step(data)
            self.assertEqual(attempts2[0]['learning_rate'],learner.config.learning_rate)
            self.assertLessEqual(kl2,.02)
            self.assertTrue(accepted2 or exact_equal(result,learner.model.state_dict()))

    def test_nonfinite_trial_restores_primed_optimizer_and_rng_before_error(self):
        with TemporaryDirectory() as root:
            learner=self.learner(root,self.config());data=self.flat(learner.collect())
            for p in learner.model.parameters():p.grad=torch.ones_like(p)
            learner.optimizer.step()
            model=deepcopy(learner.model.state_dict());adam=deepcopy(learner.optimizer.state_dict());rng=learner._rng()
            with patch.object(learner,'policy_kl',return_value=float('nan')):
                with self.assertRaisesRegex(ValueError,'Nonfinite trial'):learner._bounded_model_step(data)
            self.assert_bytes_equal(model,learner.model.state_dict())
            self.assert_bytes_equal(adam,learner.optimizer.state_dict())
            self.assert_bytes_equal(rng,learner._rng())

    def test_freeze_preserves_loaded_actor_stats_while_other_statistics_advance(self):
        with TemporaryDirectory() as root:
            learner=self.learner(root,self.config())
            learner.model.obs_normalizer.update(torch.randn(7,231))
            saved=deepcopy(learner.model.obs_normalizer.state_dict())
            learner.collect()
            critic_count=learner.model.critic_normalizer.count.clone()
            amp_count=learner.amp.normalizer.count.clone()
            learner.collect()
            self.assert_bytes_equal(saved,learner.model.obs_normalizer.state_dict())
            self.assertGreater(learner.model.critic_normalizer.count,critic_count)
            self.assertGreater(learner.amp.normalizer.count,amp_count)
            self.assertEqual(learner.normalizer_policy_kl,0.)

    def test_three_zero_updates_stop_save_strict_counter_and_delta_throughput(self):
        with TemporaryDirectory() as root:
            learner=self.learner(root,self.config());learner.transitions=100
            original=learner.optimizer.step
            def crossing():
                original()
                with torch.no_grad():learner.model.policy[-1].bias.add_(1.)
            with patch.object(learner.optimizer,'step',side_effect=crossing):
                receipt=learner.train(5,callback=lambda m:False if m['update']==3 else True)
            self.assertEqual((learner.updates,learner.optimizer_steps,learner.discriminator_steps),(3,0,12))
            self.assertEqual(receipt['training_stop']['count'],3)
            self.assertTrue((learner.output/'training_stop.json').is_file())
            rows=[json.loads(x) for x in (learner.output/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),3);self.assertEqual(rows[-1]['allocation_new_transitions'],24)
            self.assertEqual(rows[-1]['allocation_new_transitions_per_s'],24/rows[-1]['elapsed_s'])
            restored=self.learner(root,self.config(),'restored');restored.load(receipt['path'])
            self.assertEqual(restored.consecutive_zero_accepted_updates,3)
            payload=torch.load(receipt['path'],weights_only=False)
            del payload['counters']['consecutive_zero_accepted_updates']
            bad=Path(root)/'missing_counter.pt';torch.save(payload,bad)
            with self.assertRaisesRegex(ValueError,'counters'):restored.load(bad)

    def test_disabled_flags_match_frozen_source017_model_optimizers_metrics_and_rng_bytes(self):
        frozen=ROOT.parents[1]/'artifacts/restart_2026-09-14/paper_walk_execution_001/source_017/learner.py'
        if not frozen.is_file():self.skipTest('Immutable local source017 oracle unavailable')
        self.assertEqual(hashlib.sha256(frozen.read_bytes()).hexdigest(),
                         'b0a3cf503a0c710239d0245ef24fb961d0d05f06feaf37fffee7b2c802caf77a')
        name='_source017_disabled_rollback_oracle';spec=importlib.util.spec_from_file_location(name,frozen)
        legacy=importlib.util.module_from_spec(spec);sys.modules[name]=legacy;spec.loader.exec_module(legacy)
        try:
            for target in (None,.02):
                with self.subTest(target=target),TemporaryDirectory() as root:
                    cfg=replace(tiny(),target_kl=target);current=self.learner(root,cfg)
                    old_config=legacy.Config(**{f.name:asdict(cfg)[f.name] for f in fields(legacy.Config)})
                    old=legacy.PPOLearner(SyntheticEnv(),Path(root)/'data.npz',Path(root)/'old',old_config,device='cpu')
                    for _ in range(2):
                        rng=current._rng();new_metrics=current.update(current.collect());after=current._rng()
                        old._restore_rng(rng);old_metrics=old.update(old.collect())
                        for key in ('model','amp','optimizer','discriminator_optimizer'):
                            self.assert_bytes_equal(getattr(current,key).state_dict(),getattr(old,key).state_dict())
                        self.assert_bytes_equal(after,old._rng())
                        self.assertEqual(new_metrics,old_metrics)
        finally:
            sys.modules.pop(name,None)


if __name__=='__main__':unittest.main()
