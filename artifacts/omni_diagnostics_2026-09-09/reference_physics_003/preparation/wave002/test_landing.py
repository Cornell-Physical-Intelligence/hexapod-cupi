"""Actual frozen prefix replay plus synthetic landed continuation; no physics claim."""
from pathlib import Path
import json,sys,unittest
import numpy as np
from scipy.spatial.transform import Rotation
from wave_reference import WaveContactReference,SerialGeometry,tensor
ROOT=Path(__file__).resolve().parents[2]
TRACE=Path(__file__).resolve().parent/'inputs/actual_wave002_trace.npz'
REF=Path(__file__).resolve().parent/'inputs/actual_wave002_reference_states.json'

class ActualPrefix:
    def __init__(self):
        self.d=np.load(TRACE)
        self.g=SerialGeometry();self.names=list(self.d['joint_names'])
        self.r=WaveContactReference(self.names)
        self.r.reset(self.snapshot(199))
    def snapshot(self,k):
        return {name:self.d[name][k].copy() for name in self.d.files if self.d[name].shape[:1]==(274,)}
    def replay(self,*,touch_command=(.005,0,0)):
        for k in range(199,274):
            self.out=self.r.step(self.snapshot(k),touch_command if k==273 else [.005,0,0])
            if not self.out['valid'][0]:raise AssertionError((k,self.out['failure_reason']))
        return self.out

class LandedFixture:
    """Prescribed synthetic measured support; only tests controller contracts."""
    def __init__(self,prefix):
        self.r=prefix.r;self.out=prefix.out;self.snap=prefix.snapshot(273)
        self.points=self.snap['reference_point_world_m'][0].copy()
    def snapshot(self,contact=True):
        o=self.out;s={k:np.array(v,copy=True) for k,v in self.snap.items()}
        state=o['state'];R=np.asarray(state['desired_rotation_world_from_body']);p=np.asarray(state['desired_position_world_m'])
        s['time_s']=np.array([o['target_time_s']]);s['position_world_m']=p[None];s['rotation_world_from_body']=R[None]
        s['quaternion_world_xyzw']=Rotation.from_matrix(R).as_quat()[None]
        ik=self.r.g.ik(tensor(((R.T@(self.points-p).T).T)[None]));assert ik['valid'].all()
        s['joint_position_rad']=self.r._runtime(ik['q_checked'][0].numpy())[None]
        s['joint_target_rad']=o['q_ref'];s['executable_target_velocity_rad_s']=o['v_ref']
        s['reference_point_world_m']=self.points[None];s['reference_point_velocity_world_mps']=np.zeros((1,6,3))
        s['distal_contact']=np.ones((1,6),bool);s['distal_contact'][0,0]=contact
        s['contact_point_valid']=s['distal_contact'].copy();s['contact_point_world_m']=self.points[None].copy()
        s['contact_point_world_m'][~s['distal_contact']]=np.nan
        s['velocity_body_mps']=np.array([[o['admitted_command'][1],-o['admitted_command'][0],0.]])
        s['gyro_body_rad_s']=np.array([[0.,0.,o['admitted_command'][2]]])
        return s
    def step(self,contact=True,requested=(0,0,0)):
        self.out=self.r.step(self.snapshot(contact),requested);return self.out

class LandingTests(unittest.TestCase):
    def test_actual_prefix_targets_unchanged_until_returned_contact(self):
        p=ActualPrefix();saved=json.loads(REF.read_text());old={r['physical_step']:r['result'] for r in saved if 'result'in r}
        for k in range(199,273):
            out=p.r.step(p.snapshot(k),[.005,0,0]);self.assertTrue(out['valid'][0],out['failure_reason'])
            np.testing.assert_allclose(out['q_ref'],old[k+1]['q_ref'],atol=2e-12,rtol=0)
        original=p.r.swing;time=p.r.time
        before=original.sample(time)
        out=p.r.step(p.snapshot(273),[.005,0,0]);self.assertTrue(out['valid'][0],out['failure_reason'])
        self.assertEqual(out['state']['mode'],'landing_blend');self.assertEqual(p.r.touchdowns,0)
        for left,right in zip(before,p.r.landing.sample(time)):np.testing.assert_allclose(left,right,atol=1e-14)
        self.assertIs(p.r.swing,original)
        self.assertLessEqual(p.r.landing_original_contact_error_m,.012)
        self.assertGreater(out['state']['measured_flight_lift_m'],.002)
        end=p.r.landing.sample(p.r.landing.end_time)
        np.testing.assert_allclose(end[0],p.r.landing.end,atol=1e-14)
        np.testing.assert_allclose(end[1],0,atol=1e-13);np.testing.assert_allclose(end[2],0,atol=1e-12)
        # Exact actual-prefix curve has a small bounded planar excursion/return.
        points=np.array([p.r.landing.sample(t)[0] for t in np.linspace(time,p.r.landing.end_time,501)])
        excursion=np.linalg.norm(points[:,:2]-points[0,:2],axis=1).max()
        self.assertGreater(excursion,.001);self.assertLess(excursion,.0025)

    def test_stop_during_landing_completes_support_then_finite_quiet_without_new_liftoff(self):
        p=ActualPrefix();p.replay(touch_command=[0,0,0]);f=LandedFixture(p)
        for _ in range(600):
            out=f.step();self.assertTrue(out['valid'][0],out['failure_reason'])
            self.assertLessEqual(abs(out['v_ref']).max(),1.75+1e-5);self.assertLessEqual(abs(out['a_ref']).max(),6+1e-5)
        self.assertEqual(p.r.liftoffs,1);self.assertEqual(p.r.touchdowns,1)
        self.assertEqual(out['state']['mode'],'reference_quiet_hold')
        np.testing.assert_array_equal(out['admitted_command'],[0,0,0])

    def test_single_contact_then_loss_never_confirms_touchdown(self):
        p=ActualPrefix();p.replay();f=LandedFixture(p)
        for _ in range(100):
            out=f.step(contact=False)
            if not out['valid'][0]:break
        self.assertFalse(out['valid'][0]);self.assertIn('contact loss',out['failure_reason']);self.assertEqual(p.r.touchdowns,0)

    def test_three_stable_samples_required_after_blend_endpoint(self):
        p=ActualPrefix();p.replay();f=LandedFixture(p)
        counts=[]
        for _ in range(40):
            out=f.step();self.assertTrue(out['valid'][0],out['failure_reason'])
            counts.append((p.r.contact_count,p.r.touchdowns))
            if p.r.touchdowns:break
        self.assertIn((1,0),counts);self.assertIn((2,0),counts)
        self.assertEqual(counts[-1],(3,1))

    def test_three_dimensional_landing_consistency_is_checked(self):
        p=ActualPrefix();p.replay();f=LandedFixture(p)
        # Currentcontact-region guard is independently conservative in3D.
        f.points[0,0]+=.013
        out=f.step();self.assertFalse(out['valid'][0]);self.assertIn('contact region',out['failure_reason'])

    def test_contact_before_apex_or_without_clearance_is_rejected(self):
        for mode in ('before_apex','insufficient_clearance'):
            p=ActualPrefix()
            for k in range(199,273):
                out=p.r.step(p.snapshot(k),[.005,0,0]);self.assertTrue(out['valid'][0])
                if mode=='before_apex' and k==240:break
            snap=p.snapshot(241 if mode=='before_apex' else 273)
            snap['distal_contact'][0,0]=True;snap['contact_point_valid'][0,0]=True
            snap['contact_point_world_m'][0,0]=snap['reference_point_world_m'][0,0]
            if mode=='insufficient_clearance':p.r.flight_peak_z=p.r.flight_baseline_z+.0015
            out=p.r.step(snap,[.005,0,0]);self.assertFalse(out['valid'][0]);self.assertIn('measured2mm',out['failure_reason'])

    def test_large_original_endpoint_error_is_not_hidden_by_replacement(self):
        p=ActualPrefix()
        for k in range(199,273):p.r.step(p.snapshot(k),[.005,0,0])
        snap=p.snapshot(273);snap['reference_point_world_m'][0,0,0]+=.02
        out=p.r.step(snap,[.005,0,0]);self.assertFalse(out['valid'][0]);self.assertIn('correction bound',out['failure_reason'])

if __name__=='__main__':unittest.main()
