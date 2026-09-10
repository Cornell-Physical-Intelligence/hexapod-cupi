import unittest
import torch
from reference_residual import ResidualConfig, ReferenceResidualTarget, reject_old_checkpoint


NAMES = tuple('joint_'+str(i) for i in range(18))


def make(profile='formal_004', dtype=torch.float64, count=2):
    cfg = ResidualConfig(profile, .02, .25, 2., 8.)
    controller = ReferenceResidualTarget(NAMES, {n:-1. for n in NAMES}, {n:1. for n in NAMES}, count, cfg, dtype=dtype)
    zero = torch.zeros(count,18,dtype=dtype)
    controller.reset(zero,zero)
    return controller,zero


class ResidualTests(unittest.TestCase):
    def step(self, controller, reference, action):
        return controller.step(reference, action, reference_valid=torch.ones(len(reference),dtype=torch.bool))

    def test_constant_bias_is_finite_position_offset_not_velocity_windup(self):
        controller,q = make()
        action = torch.ones_like(q)*.7
        for _ in range(3000):
            result = self.step(controller,q,action)
        expected = .02*torch.tanh(action)
        torch.testing.assert_close(result['target_position_rad'],expected,atol=1e-12,rtol=0)
        self.assertLess(float(result['target_velocity_rad_s'].abs().max()),1e-10)

    def test_zero_residual_emits_valid_reference_exactly(self):
        controller,q = make()
        previous = q.clone()
        for step in range(250):
            # Starts at zero velocity/acceleration; endpoint samples remain in budget.
            t = (step+1)*.02
            reference = q+.05*(1-torch.cos(torch.tensor(t,dtype=q.dtype)))
            result = self.step(controller,reference,q)
            self.assertTrue(torch.equal(result['target_position_rad'],reference))
            torch.testing.assert_close((reference-previous)/.02,result['target_velocity_rad_s'])
            previous = reference.clone()

    def test_reference_over_budget_or_invalid_fails_without_mutation(self):
        controller,q = make()
        before = controller.observable_state(q).clone()
        with self.assertRaises(ValueError):
            self.step(controller,q+.01,q)
        self.assertTrue(torch.equal(before,controller.observable_state(q)))
        with self.assertRaises(ValueError):
            controller.step(q,q,reference_valid=torch.tensor([True,False]))
        self.assertTrue(torch.equal(before,controller.observable_state(q)))

    def test_residual_and_total_rates_hold_during_noise_and_reversals(self):
        controller,q = make(dtype=torch.float32)
        generator = torch.Generator().manual_seed(24)
        previous=q.clone();previous_v=q.clone()
        for step in range(4000):
            action = torch.randn(q.shape,generator=generator,dtype=q.dtype)*2
            result = self.step(controller,q,action)
            target=result['target_position_rad'];velocity=result['target_velocity_rad_s']
            self.assertLessEqual(float(target.abs().max()),.020001)
            self.assertLessEqual(float(velocity.abs().max()),.250001)
            self.assertLessEqual(float(result['target_acceleration_rad_s2'].abs().max()),2.00001)
            torch.testing.assert_close((target-previous)/.02,velocity,atol=1e-6,rtol=0)
            torch.testing.assert_close((velocity-previous_v)/.02,result['target_acceleration_rad_s2'],atol=1e-5,rtol=0)
            previous=target;previous_v=velocity

    def test_zero_goal_does_not_disable_return_feedback(self):
        controller,q = make()
        for _ in range(100):self.step(controller,q,torch.ones_like(q))
        self.assertGreater(float(controller.residual_position.abs().max()),.01)
        result=self.step(controller,q,q)
        self.assertGreater(float(result['target_velocity_rad_s'].abs().max()),0.)
        for _ in range(500):result=self.step(controller,q,q)
        self.assertLess(float(result['target_position_rad'].abs().max()),1e-12)

    def test_profiles_remain_distinct_with_reference_reserve(self):
        formal,_ = make('formal_004');diagnostic,_ = make('diagnostic_003')
        self.assertEqual(formal.config.reference_velocity_rad_s,1.75)
        self.assertEqual(diagnostic.config.reference_velocity_rad_s,1.25)
        self.assertNotEqual(formal.config.contract(),diagnostic.config.contract())

    def test_named_order_reset_and_inference_storage(self):
        names=tuple(reversed(NAMES))
        controller=ReferenceResidualTarget(names,{n:-1. for n in names},{n:1. for n in names},2,ResidualConfig('formal_004',.02,.25,2.,8.))
        q=torch.zeros(2,18,dtype=torch.float64);controller.reset(q,q)
        with torch.inference_mode():self.step(controller,q,torch.ones_like(q))
        untouched=controller.observable_state(q)[1].clone()
        controller.reset(q[:1],q[:1],env_ids=[0])
        self.assertTrue(torch.equal(untouched,controller.observable_state(q)[1]))
        self.assertTrue(torch.equal(controller.residual_position[0],q[0]))
        with self.assertRaises(ValueError):controller.reset(q[:1],q[:1]+.001,env_ids=[0])

    def test_observation_recovers_entire_persistent_state(self):
        controller,q=make()
        result=self.step(controller,q,torch.ones_like(q))
        state=controller.observable_state(q)
        self.assertEqual(state.shape,(2,72))
        r=state[:,36:54]*controller.config.residual_radius_rad
        rv=state[:,54:]*controller.config.residual_velocity_rad_s
        torch.testing.assert_close(state[:,:18]+q-r,controller.reference_position)
        torch.testing.assert_close(state[:,18:36]*controller.config.total_velocity_rad_s-rv,controller.reference_velocity)
        torch.testing.assert_close(r,controller.residual_position)

    def test_reference_joint_margin_and_bad_inputs_rejected(self):
        controller,q=make()
        with self.assertRaises(ValueError):controller.reset(q+.99,q+.99)
        with self.assertRaises(ValueError):self.step(controller,q,torch.full_like(q,float('nan')))
        with self.assertRaises(ValueError):controller.step(q,q,reference_valid=[1,1])
        with self.assertRaises(ValueError):ResidualConfig('formal_004',.02,2.,2.,8.)
        with self.assertRaises(ValueError):reject_old_checkpoint({'actor_width':495})


if __name__=='__main__':unittest.main()
