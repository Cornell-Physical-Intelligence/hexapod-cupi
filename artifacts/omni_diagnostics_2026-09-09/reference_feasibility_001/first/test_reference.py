import math
import unittest
import torch
from reference import *

class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = SerialGeometry()
        cls.reference = TwistReference(cls.geometry)

    def test_named_frozen_identity_and_neutral_inverse(self):
        self.assertEqual(self.geometry.urdf_sha, 'e916b1ebbb2720c90f23974bbffa5da120b32d2e5409fe419fb1492b8556746c')
        out = self.geometry.ik(self.geometry.feet0[None])
        self.assertTrue(out['valid'].all())
        self.assertLess((out['q_checked'][0]-self.geometry.q0).abs().max(), 1e-5)

    def test_stance_exact_no_slip_for_constant_combined_twists(self):
        # Phase .1 puts all six legs in stance for the declared .65 duty.
        c = tensor([[.1,0,0],[0,.1,.2],[-.1,.07,-.4],[0,0,.4]])
        p = torch.full((len(c),),.1,dtype=DTYPE)
        out = self.reference.state(p,c,torch.zeros_like(c))
        xy=out['feet'][...,:2]
        Jp=torch.stack((-xy[...,1],xy[...,0]),-1)
        expected=-nav_to_body(c)[:,None,:]-c[:,None,2,None]*Jp
        self.assertLess((out['foot_velocity'][...,:2]-expected).abs().max(),1e-10)
        self.assertTrue(torch.isfinite(out['q_velocity']).all())

    def test_se2_flow_against_independent_matrix_exponential(self):
        from scipy.linalg import expm
        for c in ([.1,.07,0],[.1,.07,1e-12],[.1,.07,-.4]):
            for time in (-.8,0,.8):
                actual=inverse_twist_flow(self.geometry.feet0,tensor([c]),tensor([[time]*6]))[0]
                vx,vy=c[1],-c[0];w=c[2]
                A=np.array([[0,-w,vx],[w,0,vy],[0,0,0]])
                points=np.c_[self.geometry.feet0[:,:2],np.ones(6)]
                expected=(expm(-A*time)@points.T).T[:,:2]
                self.assertLess(np.abs(actual.numpy()-expected).max(),1e-12)

    def test_original_forward_reference_shape_is_recovered(self):
        q=tensor(self.geometry.reference['positions_rad']).reshape(-1,6,3)
        original,_,_=self.geometry.fk(q)
        phase=(torch.arange(len(q),dtype=DTYPE)+.5)/len(q)
        command=tensor([[.2,0,0]]).expand(len(q),3)
        feet=self.reference.feet(phase,command,frequency_override=1.3)
        self.assertLess((feet-original).norm(dim=-1).max(),2e-5)

    def test_periodic_position_velocity_acceleration_continuity(self):
        c=tensor([[.10,.03,.2]])
        def derivatives(phase):
            p=tensor([phase]).requires_grad_()
            value=self.reference.feet(p,c)
            v=torch.stack([torch.autograd.grad(value[0,i,j],p,create_graph=True,retain_graph=True)[0][0]
                           for i in range(6) for j in range(3)])
            a=torch.stack([torch.autograd.grad(x,p,retain_graph=True)[0][0] for x in v])
            return value.detach().flatten(),v.detach(),a.detach()
        for boundary in (0.,.15,.5,.65,1.):
            left=derivatives(boundary-1e-8);right=derivatives(boundary+1e-8)
            for a,b,tol in zip(left,right,(1e-7,1e-6,1e-4)):
                self.assertLess((a-b).abs().max(),tol)

    def test_zero_reference_does_not_disable_feedback(self):
        c=torch.zeros(8,3,dtype=DTYPE);p=torch.linspace(0,1,8,dtype=DTYPE)
        out=self.reference.state(p,c,torch.zeros_like(c))
        self.assertLess((out['feet']-self.geometry.feet0).abs().max(),1e-12)
        self.assertLess(out['q_velocity'].abs().max(),1e-12)
        residual=torch.zeros_like(out['q_checked']);residual[:,:,0]=.2
        target,limited=compose_joint_feedback(out['q_checked'],residual,self.geometry,reference_valid=out['valid'])
        self.assertLess((target-out['q_checked']-residual*.12).abs().max(),1e-12)
        self.assertFalse(limited.any())

    def test_filter_reversal_continuity_and_settling(self):
        f=FilterState(self.reference)
        for _ in range(100):f.step([[.1,.05,.2]],.02)
        before=f.command.clone();rate=f.rate.clone();phase=f.phase.clone()
        feet_before=self.reference.feet(phase,before)
        out=f.step([[-.1,-.05,-.2]],1e-7)
        self.assertLess((f.command-before-rate*1e-7).abs().max(),1e-12)
        self.assertLess((out['feet']-feet_before).abs().max(),1e-6)
        for _ in range(600):out=f.step([[0,0,0]],.02)
        self.assertLess(f.command.abs().max(),1e-15)
        self.assertLess((out['feet']-self.geometry.feet0).abs().max(),1e-12)
        self.assertLess(out['q_velocity'].abs().max(),1e-12)

    def test_unreachable_joint_limited_and_nonfinite_requests_rejected(self):
        targets=self.geometry.feet0[None].repeat(3,1,1)
        targets[0,:,0]+=2.
        targets[1,:,2]=.05
        targets[2,:,1]=float('nan')
        out=self.geometry.ik(targets)
        self.assertFalse(out['valid'].any())
        with self.assertRaises(ValueError):
            compose_joint_feedback(out['q_checked'],torch.zeros_like(out['q_checked']),self.geometry,reference_valid=out['valid'])

    def test_fk_ik_roundtrip_sampled_serial_workspace(self):
        generator=torch.Generator().manual_seed(57)
        q=self.geometry.q0+(.2*torch.rand((128,6,3),generator=generator,dtype=DTYPE)-.1)
        feet,_,_=self.geometry.fk(q)
        out=self.geometry.ik(feet)
        self.assertTrue(out['valid'].all())
        self.assertLess(out['error_m'].max(),2e-5)
        self.assertLess((out['q_checked']-q).abs().max(),1e-5)

    def test_batching_equals_independent_commands(self):
        c=tensor([[.1,0,0],[0,.1,.2],[0,0,-.3]]);p=tensor([.1,.3,.8])
        many=self.reference.feet(p,c)
        for i in range(3):
            self.assertTrue(torch.allclose(many[i],self.reference.feet(p[i:i+1],c[i:i+1])[0],atol=1e-12,rtol=0))

if __name__=='__main__':unittest.main()
