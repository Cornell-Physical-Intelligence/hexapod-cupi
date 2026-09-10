import json
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from canonical_stance_startup import CanonicalStanceStartup

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tmp/omni_reference_residual_002'))
from reference_residual import ReferenceResidualTarget,ResidualConfig


def fixture():
    plan=json.loads((ROOT/'tmp/reference_physics_adapter_001/source_001/robot/hexapod_mkii_length_study/training_plan.json').read_text())
    named=plan['variants']['f050_t060']['stances'][0]['joint_positions_rad']
    names=tuple(reversed(named))
    nominal=np.broadcast_to(np.asarray([named[n] for n in names],dtype=np.float32).astype(float),(32,18)).copy()
    start=nominal+np.random.default_rng(27103).uniform(-.03,.03,nominal.shape)
    lower=nominal-1.;upper=nominal+1.
    return names,start,nominal,lower,upper


class StartupTests(unittest.TestCase):
    def test_exact_random_start_canonical_end_and_no_input_mutation(self):
        _,start,nominal,lower,upper=fixture()
        original=start.copy();original_nominal=nominal.copy()
        s=CanonicalStanceStartup(start,nominal,lower,upper)
        np.testing.assert_array_equal(s.sample(0)['q_ref'],start)
        np.testing.assert_array_equal(s.sample(100)['q_ref'],nominal)
        np.testing.assert_array_equal(s.sample(200)['q_ref'],nominal)
        np.testing.assert_array_equal(start,original)
        np.testing.assert_array_equal(nominal,original_nominal)
        start[:]=999.
        np.testing.assert_array_equal(s.sample(0)['q_ref'],original)

    def test_analytic_C2_endpoints_and_discrete_first_hold_knots(self):
        _,start,nominal,lower,upper=fixture()
        s=CanonicalStanceStartup(start,nominal,lower,upper)
        for i in (0,100,101,200):
            row=s.sample(i)
            self.assertEqual(np.abs(row['analytic_velocity_rad_s']).max(),0.)
            self.assertEqual(np.abs(row['analytic_acceleration_rad_s2']).max(),0.)
        q=np.stack([s.sample(i)['q_ref'] for i in range(203)])
        v=np.diff(q,axis=0)/.02
        a=np.diff(np.concatenate([np.zeros_like(v[:1]),v]),axis=0)/.02
        for i in range(1,203):
            np.testing.assert_allclose(s.sample(i)['v_ref'],v[i-1],atol=0,rtol=0)
            np.testing.assert_allclose(s.sample(i)['a_ref'],a[i-1],atol=0,rtol=0)
        self.assertLess(np.abs(v).max(),.0282)
        self.assertLess(np.abs(a).max(),.0434)
        self.assertGreater(np.abs(s.sample(101)['a_ref']).max(),0.)
        self.assertEqual(np.abs(s.sample(102)['a_ref']).max(),0.)
        self.assertLess(np.abs(s.sample(1)['q_ref']-start).max(),3e-7)

    def test_full_controller_uses_identical_discrete_knots_in_reverse_named_order(self):
        names,start,nominal,lower,upper=fixture()
        s=CanonicalStanceStartup(start,nominal,lower,upper)
        c=ReferenceResidualTarget(names,dict(zip(names,lower[0])),dict(zip(names,upper[0])),32,
            ResidualConfig('formal_004',.02,.25,2.,8.))
        c.reset(torch.tensor(start),torch.tensor(start))
        for i in range(1,203):
            row=s.sample(i)
            with torch.inference_mode():
                target=c.step(row['q_ref'],torch.zeros((32,18)),reference_valid=row['valid'],
                    analytic_reference_velocity=row['analytic_velocity_rad_s'],
                    analytic_reference_acceleration=row['analytic_acceleration_rad_s2'])
            np.testing.assert_array_equal(target['target_position_rad'].numpy(),row['q_ref'])
            np.testing.assert_array_equal(target['reference_velocity_rad_s'].numpy(),row['v_ref'])
            np.testing.assert_array_equal(target['reference_acceleration_rad_s2'].numpy(),row['a_ref'])
            self.assertEqual(target['residual_position_rad'].abs().max().item(),0.)
        np.testing.assert_array_equal(c.reference_position.numpy(),nominal)
        self.assertEqual(c.reference_velocity.abs().max().item(),0.)
        c.reset(torch.tensor(start),torch.tensor(start))

    def test_limits_rates_and_bad_inputs_rejected_without_retiming(self):
        _,start,nominal,lower,upper=fixture()
        with self.assertRaisesRegex(ValueError,'budget'):
            CanonicalStanceStartup(start,nominal,lower,upper,duration_s=.02)
        with self.assertRaisesRegex(ValueError,'margin'):
            CanonicalStanceStartup(start,nominal,lower,nominal+.01)
        with self.assertRaises(ValueError):
            CanonicalStanceStartup(start[:,:17],nominal,lower,upper)
        with self.assertRaises(ValueError):
            CanonicalStanceStartup(start,nominal,lower,upper,duration_s=2.001)
        s=CanonicalStanceStartup(start,nominal,lower,upper)
        for index in (-1,.5,True):
            with self.assertRaises(ValueError):s.sample(index)


if __name__=='__main__':unittest.main()
