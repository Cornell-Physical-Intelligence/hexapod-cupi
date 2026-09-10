import unittest
from time_governor import *

class TimeGovernorTests(unittest.TestCase):
    def test_global_clock_preserves_exact_stance_twist_relation(self):
        r=TwistReference(config=Config(stance_travel_m=.06,lift_m=.01))
        c=tensor([[.10,.03,.2]]);p=tensor([.10]);a=.15
        state=r.state(p,c,torch.zeros_like(c))
        xy=state['feet'][...,:2]
        admitted=a*c
        actual=a*state['foot_velocity'][...,:2]
        expected=-nav_to_body(admitted)[:,None,:]-admitted[:,None,2,None]*torch.stack((-xy[...,1],xy[...,0]),-1)
        self.assertLess((actual-expected).abs().max(),1e-12)

    def test_clock_scaling_does_not_change_curve_or_silently_track_original_request(self):
        r=TwistReference(config=Config(stance_travel_m=.06,lift_m=.01))
        a=.2;physical=FilterState(r);virtual=FilterState(r)
        for _ in range(30):
            result=advance_governed(physical,[[.10,0,.2]],.02,a)
            other=virtual.step([[.10,0,.2]],a*.02)
        self.assertLess((result['q_checked']-other['q_checked']).abs().max(),1e-12)
        self.assertLess((result['admitted_command_target']-tensor([[.02,0,.04]])).abs().max(),1e-12)
        self.assertLess((result['physical_q_velocity']-a*other['q_velocity']).abs().max(),1e-12)
        self.assertLess((result['admitted_acceleration']-a*a*virtual.rate).abs().max(),1e-12)

    def test_rate_budget_reserves_feedback_and_rejects_invalid_inputs(self):
        a=time_scale_from_rate_bound(10.)
        self.assertAlmostEqual(a*10,1.125)
        self.assertLess(a*10+.25,1.5)
        for bound in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):time_scale_from_rate_bound(bound)

if __name__=='__main__':unittest.main()
