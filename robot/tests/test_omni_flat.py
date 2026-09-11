"""Failure-oriented checks for frame independence, command coverage and history."""
import math
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
from experiments.c_length_study.tools.omni_flat_math import sample_commands, slew_commands, tracking_terms, evaluation_scenarios, transition_sequence, ObservationHistory, scenario_gate, integrate_body_twist, trajectory_command
from experiments.c_length_study.tools.omni_path_demo import path_follower, path_reference, demo_specs


class OmniContracts(unittest.TestCase):
    def test_bearing_invariant_tracking_and_requested_turns(self):
        c=torch.tensor([[.12, .05, .3],[-.12,-.05,-.3]])
        good=tracking_terms(c[:,:2],c[:,2],c)
        wrong=tracking_terms(c[:,:2],-c[:,2],c)
        idle=tracking_terms(torch.zeros(2,2),torch.zeros(2),c)
        self.assertTrue(torch.allclose(good['linear_tracking'],torch.ones(2)))
        self.assertTrue(torch.all(good['yaw_tracking']>wrong['yaw_tracking']))
        self.assertTrue(torch.all(good['linear_tracking']>idle['linear_tracking']))
        a=.73;rot=torch.tensor([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
        v=torch.tensor([[.08,.02],[-.10,-.04]])
        before=tracking_terms(v,c[:,2],c)
        after=tracking_terms(v@rot.T,c[:,2],torch.cat((c[:,:2]@rot.T,c[:,2:]),1))
        for k in before:self.assertTrue(torch.allclose(before[k],after[k],atol=1e-6),k)
        overspeed=tracking_terms(c[:,:2]*2,c[:,2]*2,c)
        self.assertTrue(torch.all(overspeed['linear_tracking']<good['linear_tracking']))

    def test_sampler_spans_circle_turn_signs_and_standing(self):
        torch.manual_seed(41);c=sample_commands(40000,'cpu')
        speed=c[:,:2].norm(dim=1);yaw=c[:,2]
        self.assertLessEqual(float(speed.max()),.200001);self.assertLessEqual(float(yaw.abs().max()),.400001)
        moving=speed>.01;bins=torch.floor((torch.atan2(c[moving,1],c[moving,0])+math.pi)/(2*math.pi)*16).long().clamp(max=15)
        counts=torch.bincount(bins,minlength=16)
        self.assertLess(float(counts.max()/counts.min()),1.3)
        self.assertGreater(int(((speed==0)&(yaw==0)).sum()),3000)
        self.assertGreater(int(((speed==0)&(yaw>.1)).sum()),3000)
        self.assertGreater(int(((speed==0)&(yaw<-.1)).sum()),3000)
        self.assertGreater(int((moving&(yaw>.1)).sum()),3000)
        self.assertGreater(int((moving&(yaw<-.1)).sum()),3000)
        self.assertGreater(int((moving&(yaw.abs()>.005)&(yaw.abs()<.10)).sum()),1500)

    def test_slew_reversal_and_stop_respect_vector_limit(self):
        now=torch.tensor([[.2,0,.4],[-.1,.1,-.4]])
        target=-now
        for _ in range(400):
            nxt=slew_commands(now,target,.02)
            self.assertTrue(torch.all((nxt[:,:2]-now[:,:2]).norm(dim=1)<=.005001))
            self.assertTrue(torch.all((nxt[:,2]-now[:,2]).abs()<=.016001))
            now=nxt
        self.assertTrue(torch.allclose(now,target))
        for _ in range(200):now=slew_commands(now,torch.zeros_like(now),.02)
        self.assertTrue(torch.allclose(now,torch.zeros_like(now)))

    def test_history_is_idempotent_and_reset_does_not_leak(self):
        h=ObservationHistory(2,3,1,'cpu')
        h.observe(torch.tensor([[1.],[10.]]),0)
        h.observe(torch.tensor([[2.],[20.]]),1)
        result=h.observe(torch.tensor([[2.],[20.]]),1)
        self.assertEqual(result.tolist(),[[1.,1.,2.],[10.,10.,20.]])
        h.reset(torch.tensor([1]));result=h.observe(torch.tensor([[3.],[99.]]),2)
        self.assertEqual(result.tolist(),[[1.,2.,3.],[99.,99.,99.]])

    def test_evaluation_cannot_hide_a_bad_direction_in_mean(self):
        self.assertEqual(len(evaluation_scenarios()),77)
        row={'command':[.1,0,0],'terminations':0,'truncations':0,'nonfoot_fraction':0,
             'torque_saturation_fraction':0,'planar_error_mps':.01,'yaw_error_rad_s':.01,
             'tilt_rms_deg':1,'vertical_velocity_rms_mps':.01}
        self.assertTrue(scenario_gate(row))
        for key,value in [('terminations',1),('truncations',1),('planar_error_mps',.08),('yaw_error_rad_s',.15),('torque_saturation_fraction',.02)]:
            self.assertFalse(scenario_gate({**row,key:value}),key)
        self.assertEqual(transition_sequence()[-1][0],'stop')

    def test_arcs_strafe_and_turn_in_place_share_one_twist_contract(self):
        pose=torch.zeros(4,3,dtype=torch.float64)
        command=torch.tensor([[1.,0.,1.],[0.,1.,1.],[1.,0.,0.],[0.,0.,1.]],dtype=torch.float64)
        result=integrate_body_twist(pose,command,math.pi/2)
        expected=torch.tensor([[1.,1.,math.pi/2],[-1.,1.,math.pi/2],[math.pi/2,0.,0.],[0.,0.,math.pi/2]],dtype=torch.float64)
        self.assertTrue(torch.allclose(result,expected,atol=1e-9))
        near_zero=integrate_body_twist(pose[:1],torch.tensor([[1.,0.,1e-12]],dtype=torch.float64),1.)
        self.assertTrue(torch.allclose(near_zero[0,:2],torch.tensor([1.,0.],dtype=torch.float64),atol=1e-9))
        rotated=pose[:1].clone();rotated[:,2]=math.pi/2
        self.assertTrue(torch.allclose(integrate_body_twist(rotated,command[2:3],1.)[0,:2],torch.tensor([0.,1.],dtype=torch.float64),atol=1e-9))

    def test_curved_path_need_not_rotate_the_body(self):
        start=trajectory_command('fixed_heading_bend',0,8,[0,0,0])
        end=trajectory_command('fixed_heading_bend',8,8,[0,0,0])
        self.assertAlmostEqual(start[0],.1);self.assertAlmostEqual(end[1],.1)
        self.assertEqual(start[2],0);self.assertEqual(end[2],0)
        self.assertGreater(trajectory_command('s_curve',2.5,10,[0,0,0])[2],0)
        self.assertLess(trajectory_command('s_curve',7.5,10,[0,0,0])[2],0)

    def test_path_follower_uses_actual_heading_and_respects_envelope(self):
        reference=torch.tensor([[0.,0.,0.]])
        actual=torch.tensor([[0.,0.,math.pi/2]])
        request=path_follower(actual,reference,torch.tensor([[.1,0.,0.]]))
        self.assertAlmostEqual(float(request[0,0]),0,places=6)
        self.assertAlmostEqual(float(request[0,1]),-.1,places=6)
        self.assertAlmostEqual(float(request[0,2]),-.4,places=6)
        actual=torch.tensor([[-10.,-10.,-math.pi]])
        request=path_follower(actual,reference,torch.zeros(1,3))
        self.assertLessEqual(float(request[:,:2].norm()),.200001)
        self.assertLessEqual(float(request[:,2].abs()),.400001)

    def test_reference_paths_are_finite_and_end_at_stand(self):
        for name,_,duration in demo_specs():
            poses,commands=path_reference(name,duration,.02,torch.zeros(1,3))
            self.assertTrue(torch.isfinite(poses).all())
            self.assertEqual(tuple(poses.shape),tuple(commands.shape))
            self.assertTrue(torch.allclose(commands[-1],torch.zeros(3),atol=1e-6))
            if name=='turn':self.assertTrue(torch.allclose(poses[:,:2],torch.zeros_like(poses[:,:2])))
            else:self.assertGreater(float(poses[-1,:2].norm()),.5)


if __name__=='__main__':unittest.main()
