"""Preparation tests: synthetic objectives only, never optimizer/actor fitting."""
from pathlib import Path
import importlib.util
import unittest
import numpy as np
import torch

spec=importlib.util.spec_from_file_location('refit_preparation_helper',Path(__file__).with_name('refit_bc.py'))
helper=importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class PreparationTests(unittest.TestCase):
    def data(self):
        data={'control_index':np.zeros(3760,np.int64),'source_kind':np.zeros(3760,np.int8),
              'commands':np.zeros((3760,3),np.float32)}
        data['control_index'][:23]=200
        data['source_kind'][:23]=1
        data['commands'][:23,0]=.05
        return data

    def test_exact_selection_and_full_dataset_weight_normalization(self):
        data=self.data()
        mask,weighted=helper.weights_for(data,'onset_weight20')
        _,uniform=helper.weights_for(data,'uniform')
        self.assertEqual(np.flatnonzero(mask).tolist(),list(range(23)))
        self.assertEqual(float(weighted.sum()),4197)
        self.assertEqual(float(uniform.sum()),3760)
        self.assertTrue(np.array_equal(weighted[~mask],uniform[~mask]))
        self.assertEqual(int((weighted==20).sum()),23)

    def test_wrong_selection_and_unknown_arm_rejected(self):
        data=self.data()
        data['commands'][0]=0
        with self.assertRaisesRegex(ValueError,'Onset selection'):
            helper.weights_for(data,'uniform')
        with self.assertRaisesRegex(ValueError,'Unknown'):
            helper.weights_for(self.data(),'weighted_sampling')

    def test_uniform_original_loss_and_gradients_bitwise_equal(self):
        mean=torch.arange(54,dtype=torch.float32).reshape(3,18).div(80).requires_grad_()
        target=torch.ones_like(mean)*.1
        velocity=torch.arange(9,dtype=torch.float32).reshape(3,3).div(40).requires_grad_()
        truth=torch.zeros_like(velocity)
        original=(mean-target).square().mean()+(velocity-truth).square().mean()
        actual,_,_=helper.objective(mean,target,velocity,truth,torch.ones(3),1.,'uniform')
        self.assertTrue(torch.equal(actual,original))
        a=torch.autograd.grad(actual,(mean,velocity),retain_graph=True)
        b=torch.autograd.grad(original,(mean,velocity))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)))

    def test_weighting_changes_action_gradient_only_and_fixed_denominator(self):
        mean=torch.ones((2,18),requires_grad=True)
        target=torch.zeros_like(mean)
        velocity=torch.ones((2,3),requires_grad=True)
        truth=torch.zeros_like(velocity)
        weighted,_,velocity_loss=helper.objective(mean,target,velocity,truth,torch.tensor([20.,1.]),4197/3760,'onset_weight20')
        uniform,_,velocity_original=helper.objective(mean,target,velocity,truth,torch.ones(2),1.,'uniform')
        a=torch.autograd.grad(weighted,(mean,velocity),retain_graph=True)
        b=torch.autograd.grad(uniform,(mean,velocity))
        self.assertTrue(torch.equal(velocity_loss,velocity_original))
        self.assertTrue(torch.equal(a[1],b[1]))
        self.assertAlmostEqual(float(a[0][0,0]/a[0][1,0]),20.,places=5)
        self.assertAlmostEqual(float(a[0][1,0]/b[0][1,0]),3760/4197,places=6)
        # A batch denominator would produce a different scalar; this remains
        # the declared fixed empirical-objective estimator for uniform draws.
        self.assertAlmostEqual(float(weighted.detach()),10.5/(4197/3760)+1,places=5)

    def test_weighting_does_not_consume_sampling_rng(self):
        before=torch.get_rng_state().clone()
        data=self.data()
        for arm in helper.ARMS:
            helper.weights_for(data,arm)
            helper.objective(torch.zeros(2,18),torch.ones(2,18),torch.zeros(2,3),
                torch.ones(2,3),torch.ones(2),1.,arm)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))


if __name__=='__main__':
    unittest.main()
