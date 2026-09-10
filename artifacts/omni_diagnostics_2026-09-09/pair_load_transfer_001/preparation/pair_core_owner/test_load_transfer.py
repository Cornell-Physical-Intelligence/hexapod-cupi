"""Synthetic kinematics/contact controls only, with exact residual core checks."""
import unittest
from dataclasses import replace
import numpy as np
import torch
from load_transfer import PairLoadTransfer, TransferConfig, PAIR, CORNERS, Quintic
from synthetic_fixture import Fixture
from reference_residual import ReferenceResidualTarget, ResidualConfig


class TransferTests(unittest.TestCase):
    def test_named_order_preload_and_first_target_exact(self):
        f=Fixture(preload=.01)
        f=Fixture(tuple(reversed(f.names)),preload=.01)
        c=PairLoadTransfer(f.names); sample=f.snapshot(); out=c.reset(sample)
        np.testing.assert_array_equal(out['q_ref'],sample['joint_target_rad'])
        self.assertGreater(abs(c.base.preload_q).max(),.009)
        self.assertFalse(out['body_pose_prescribed'])
        q=c.step(sample)['q_ref'];np.testing.assert_array_equal(q,sample['joint_target_rad'])

    def test_noncanonical_initial_target_rejected_without_teleport(self):
        f=Fixture();snapshot=f.snapshot();snapshot['joint_target_rad'][0,2]+=.001
        with self.assertRaisesRegex(ValueError,'canonical C target'):
            PairLoadTransfer(f.names).reset(snapshot)

    def test_full_sequence_passes_exact_residual_core_and_returns_quiet(self):
        f=Fixture();c=PairLoadTransfer(f.names);initial=f.snapshot();c.reset(initial)
        limits=initial['soft_joint_pos_limits_rad'][0]
        core=ReferenceResidualTarget(f.names,dict(zip(f.names,limits[:,0])),dict(zip(f.names,limits[:,1])),1,
            ResidualConfig('formal_004',.02,.25,2.,8.))
        core.reset(initial['joint_target_rad'],initial['joint_target_rad'])
        vmax=amax=0
        for k in range(1100):
            out=c.step(f.snapshot());self.assertTrue(out['valid'][0],out['failure_reason'])
            result=core.step(out['q_ref'],np.zeros((1,18)),reference_valid=out['valid'])
            np.testing.assert_array_equal(result['target_position_rad'].numpy(),out['q_ref'])
            qleg=c.base._leg(out['q_ref'][0]);q0=c.base._leg(c.q0)
            np.testing.assert_array_equal(qleg[list(CORNERS)],q0[list(CORNERS)])
            vmax=max(vmax,abs(out['v_ref']).max());amax=max(amax,abs(out['a_ref']).max())
            f.advance(out)
        self.assertEqual(out['state']['mode'],'reference_quiet_hold')
        self.assertAlmostEqual(out['state']['reference_quiet_time_s'],10.)
        self.assertGreaterEqual(c.maximum_unloaded_s,1.)
        self.assertTrue(c.flight_seen.all());self.assertTrue((c.peak_lift>=.002).all())
        np.testing.assert_array_equal(out['q_ref'][0],c.q0)
        np.testing.assert_array_equal(out['v_ref'],np.zeros((1,18)))
        self.assertLess(vmax,1.75);self.assertLess(amax,6.)

    def test_early_stop_has_bounded_finite_continuous_return(self):
        for stop_at in (0,125,200,325):
            f=Fixture();c=PairLoadTransfer(f.names);c.reset(f.snapshot())
            for k in range(stop_at+780):
                out=c.step(f.snapshot(),stop=k>=stop_at)
                self.assertTrue(out['valid'][0],(stop_at,k,out['failure_reason']))
                f.advance(out)
            self.assertAlmostEqual(c.reference_quiet_time,stop_at*.02+3.)
            np.testing.assert_array_equal(c.q,c.q0)

    def test_no_unloading_cannot_become_physical_success(self):
        f=Fixture();f.force_never_flight=True;c=PairLoadTransfer(f.names);c.reset(f.snapshot())
        for _ in range(550):
            out=c.step(f.snapshot());self.assertTrue(out['valid'][0],out['failure_reason']);f.advance(out)
        self.assertEqual(c.maximum_unloaded_s,0.)
        self.assertFalse(c.flight_seen.any())
        self.assertEqual(out['diagnostic_outcome'],'not_yet_physically_scored')

    def test_required_contact_torque_pose_and_timing_reject_without_target(self):
        for kind in ('corner','shaft','torque','frame','target','time'):
            f=Fixture();c=PairLoadTransfer(f.names);c.reset(f.snapshot());m=f.snapshot()
            if kind=='corner':m['distal_contact'][0,0]=False
            if kind=='shaft':m['shaft_contact'][0,1]=True
            if kind=='torque':m['computed_torque_nm'][0,4]=1.6001
            if kind=='frame':m['quaternion_world_xyzw'][0]=[1,0,0,0]
            if kind=='target':m['joint_target_rad'][0,3]+=.001
            if kind=='time':m['time_s'][0]+=.02
            out=c.step(m);self.assertFalse(out['valid'][0],kind);self.assertIsNone(out['q_ref'])
            self.assertFalse(c.step(f.snapshot())['valid'][0])

    def test_missing_six_support_return_rejects(self):
        f=Fixture();c=PairLoadTransfer(f.names);c.reset(f.snapshot())
        for k in range(560):
            m=f.snapshot()
            if k>=480:
                m['distal_contact'][0,1]=False;m['contact_point_valid'][0,1]=False;m['contact_point_world_m'][0,1]=np.nan
            out=c.step(m)
            if not out['valid'][0]:break
            f.advance(out)
        self.assertIn('Six-support measured return',out['failure_reason'])

    def test_nonfinite_config_or_curve_cannot_enter_reference(self):
        for field in ('lift_m','ramp_s','maximum_body_displacement_m'):
            with self.assertRaises(ValueError):replace(TransferConfig(),**{field:float('nan')})
        for duration in (0.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):Quintic(0.,duration,np.zeros(18),np.zeros(18),np.zeros(18),np.zeros(18))
        bad=np.zeros(18);bad[4]=float('nan')
        with self.assertRaises(ValueError):Quintic(0.,3.,bad,np.zeros(18),np.zeros(18),np.zeros(18))

    def test_quintic_exact_endpoint_position_velocity_acceleration(self):
        q=np.linspace(-.3,.3,18);v=np.full(18,.1);a=np.full(18,-.05);end=np.zeros(18)
        curve=Quintic(2.,3.,q,v,a,end)
        for actual,wanted in zip(curve.sample(2.),(q,v,a)):np.testing.assert_allclose(actual,wanted,atol=1e-15)
        for actual,wanted in zip(curve.sample(5.),(end,end,end)):np.testing.assert_array_equal(actual,wanted)


if __name__=='__main__':unittest.main()
