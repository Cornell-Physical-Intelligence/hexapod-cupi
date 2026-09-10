import math
import unittest
import torch
from velocity_action import *

NAMES=tuple(f'joint_{i}' for i in range(18))
LOW={n:-1. for n in NAMES};HIGH={n:1. for n in NAMES}

def make(profile='formal_004',count=2,dtype=torch.float64,names=NAMES):
    c=JointTargetVelocity(names,LOW,HIGH,count,VelocityActionConfig(profile),dtype=dtype)
    c.reset(torch.zeros(count,18,dtype=dtype));return c

class VelocityTests(unittest.TestCase):
    def test_profiles_separate_and_exact_budget(self):
        for profile,delta in [('diagnostic_003',.03),('formal_004',.04)]:
            c=make(profile);self.assertEqual(c.config.max_velocity_rad_s,delta/.02)
            last=c.position.clone();lastv=c.velocity.clone()
            for _ in range(500):
                x=c.step(torch.ones_like(c.position));self.assertLessEqual(float((x.target_position_rad-last).abs().max()),delta+1e-12)
                self.assertLessEqual(float((x.target_velocity_rad_s-lastv).abs().max()),.16+1e-12)
                last=x.target_position_rad;lastv=x.target_velocity_rad_s
            self.assertTrue(torch.allclose(c.position,torch.ones_like(c.position),atol=1e-10))
            self.assertLess(float(c.velocity.abs().max()),1e-10)
    def test_reversal_acceleration_and_no_teleport(self):
        c=make();last=c.position.clone();lastv=c.velocity.clone()
        for i in range(100):
            x=c.step(torch.full_like(c.position,1. if i<15 else -1.))
            self.assertTrue(torch.allclose((x.target_position_rad-last)/.02,x.target_velocity_rad_s,atol=1e-12,rtol=0))
            self.assertLessEqual(float((x.target_velocity_rad_s-lastv).abs().max()),.16+1e-12)
            self.assertTrue((c.position<=1).all() and (c.position>=-1).all())
            last=c.position.clone();lastv=c.velocity.clone()
    def test_zero_action_settles_then_holds_and_feedback_stays_active(self):
        c=make();c.step(torch.ones_like(c.position)*.3)
        for _ in range(15):c.step(torch.zeros_like(c.position))
        q=c.position.clone()
        for _ in range(100):c.step(torch.zeros_like(c.position))
        self.assertTrue(torch.equal(c.position,q));self.assertTrue(torch.equal(c.velocity,torch.zeros_like(c.velocity)))
        c.step(torch.ones_like(c.position)*.1)
        self.assertTrue((c.position>q).all())  # no zero-body-command mask exists
    def test_partial_reset_and_observed_state(self):
        c=make();c.step(torch.ones_like(c.position));old=c.position[1].clone();oldv=c.velocity[1].clone()
        c.reset(torch.full((1,18),.2),[0]);self.assertTrue(torch.equal(c.position[1],old));self.assertTrue(torch.equal(c.velocity[1],oldv))
        self.assertTrue(torch.equal(c.velocity[0],torch.zeros(18,dtype=c.velocity.dtype)))
        frame=append_executable_state(torch.zeros(2,63),c,torch.zeros(2,18))
        self.assertEqual(frame.shape,(2,99));self.assertTrue(torch.equal(frame[:,63:81],c.position))
        self.assertTrue(torch.equal(frame[:,81:],c.velocity/2.))
        before=c.position.clone();frame.zero_();self.assertTrue(torch.equal(c.position,before))
    def test_named_order_and_batching(self):
        a=make(count=1);b=make(count=1,names=NAMES[::-1]);action=torch.arange(18,dtype=torch.float64)[None]/18
        x=a.step(action);y=b.step(action.flip(-1));self.assertTrue(torch.allclose(x.target_position_rad,y.target_position_rad.flip(-1)))
        with self.assertRaises(ValueError):JointTargetVelocity(NAMES[:-1]+(NAMES[0],),LOW,HIGH,1,VelocityActionConfig('formal_004'))
    def test_invalid_inputs_do_not_commit_state(self):
        c=make();old=c.position.clone()
        for action in (torch.full((2,18),float('nan')),torch.zeros(2,17)):
            with self.assertRaises(ValueError):c.step(action)
            self.assertTrue(torch.equal(c.position,old))
        for dt in (0,.021,float('inf'),True):
            with self.assertRaises(ValueError):c.step(torch.zeros(2,18),dt)
        with self.assertRaises(ValueError):c.reset(torch.full((1,18),2.),[0])
        with self.assertRaises(ValueError):c.reset(torch.full((1,18),.999),[0],target_velocity=torch.ones(1,18))
        self.assertTrue(torch.equal(c.position,old))
    def test_float32_random_noise_and_anti_windup(self):
        g=torch.Generator().manual_seed(57);c=make(count=8,dtype=torch.float32);last=c.position.clone();lastv=c.velocity.clone()
        for i in range(4000):
            action=torch.randn(8,18,generator=g)*(.1 if i<2000 else 2.)
            x=c.step(action)
            self.assertTrue(torch.isfinite(x.target_position_rad).all())
            self.assertLessEqual(float(c.position.abs().max()),1.+1e-6)
            self.assertLessEqual(float((c.position-last).abs().max()),.04+1e-6)
            self.assertLessEqual(float((c.velocity-lastv).abs().max()),.16+1e-5)
            last=c.position.clone();lastv=c.velocity.clone()
    def test_bound_outward_hold_and_inward_release(self):
        c=make();c.reset(torch.ones(2,18))
        for _ in range(50):x=c.step(torch.ones(2,18))
        self.assertTrue(torch.equal(c.position,torch.ones_like(c.position)));self.assertTrue((x.target_velocity_rad_s==0).all())
        x=c.step(-torch.ones(2,18));self.assertTrue((x.target_position_rad<1).all());self.assertTrue((x.target_acceleration_rad_s2>=-8.-1e-10).all())
    def test_old_and_other_profile_checkpoints_rejected(self):
        cfg=VelocityActionConfig('formal_004');meta={'lineage':LINEAGE,'profile':'formal_004','actor_width':495,'critic_width':498,'action_semantics':'normalized_joint_target_velocity','integration':'semi_implicit_discrete','max_acceleration_rad_s2':8.}
        self.assertTrue(verify_new_lineage_checkpoint(meta,cfg,'a'*64,'a'*64))
        for change in ({'actor_width':315},{'profile':'diagnostic_003'},{'max_acceleration_rad_s2':10},{'action_semantics':'position_offset'}):
            with self.assertRaises(ValueError):verify_new_lineage_checkpoint({**meta,**change},cfg,'a'*64,'a'*64)
        with self.assertRaises(ValueError):verify_new_lineage_checkpoint(meta,cfg,'a'*64,'b'*64)
    def test_substep_duration_obeys_same_rates(self):
        for dt in (.0025,.005,.01,.02):
            c=make();x=c.step(torch.ones(2,18),dt)
            self.assertTrue(torch.allclose(x.target_velocity_rad_s,torch.full_like(x.target_velocity_rad_s,8.*dt)))
            self.assertTrue(torch.allclose(x.target_position_rad,8.*torch.full_like(x.target_position_rad,dt*dt)))

if __name__=='__main__':unittest.main()
