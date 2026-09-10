"""Scalar-oracle parity; synthetic contacts do not qualify actual walking."""
from copy import deepcopy
import ast
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from batch_wave import BatchWave004, MODES, SNAP_BOOL, SNAP_FLOAT

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'oracle'))
from wave_reference import WaveContactReference, AdvancedHorizontalSwing
from test_wave_reference import Fixture
torch.set_num_threads(1)


def pack(snapshots):
    keys=(*SNAP_FLOAT,*SNAP_BOOL,'executable_target_velocity_rad_s','soft_joint_pos_limits_rad')
    result={key:torch.as_tensor(np.concatenate([s[key] for s in snapshots],axis=0),
              dtype=torch.bool if key in SNAP_BOOL else torch.float64) for key in keys if key!='position_is_float32'}
    result['position_is_float32']=torch.tensor([s['position_world_m'].dtype==np.float32 for s in snapshots])
    return result


def commanded(values):return torch.tensor(values,dtype=torch.float64)


def assert_row(test,out,i,scalar,atol=3e-10):
    test.assertEqual(bool(out['valid'][i]),bool(scalar['valid'][0]),
                     (i,int(out['failure_code'][i]),scalar['failure_reason']))
    if not scalar['valid'][0]:
        test.assertTrue(torch.isnan(out['q_ref'][i]).all());return
    for key in ('q_ref','v_ref','a_ref'):
        np.testing.assert_allclose(out[key][i].numpy(),scalar[key][0],atol=atol,rtol=0,err_msg=key)
    test.assertEqual(MODES[int(out['state']['mode'][i])],scalar['state']['mode'])
    for b,s in [('flight_seen','flight_seen'),('flight_count','flight_count'),('contact_count','contact_count'),
                ('liftoffs','liftoffs'),('touchdowns','confirmed_touchdowns'),('landing_gap','landing_contact_gap_steps'),
                ('descent_seen','actual_descent_seen'),('order','next_wave_order_index')]:
        test.assertEqual(out['state'][b][i].item(),scalar['state'][s],(b,s))
    for b,s in [('position','desired_position_world_m'),('command','command_filter_velocity'),('rate','command_filter_rate'),
                ('anchors','reference_anchors_world_m'),('measured_anchors','measured_anchors_world_m'),
                ('preload_world','reference_minus_measured_preload_world_m')]:
        np.testing.assert_allclose(out['state'][b][i].numpy(),scalar['state'][s],atol=2e-12,rtol=0,err_msg=b)


class WaveBatchTests(unittest.TestCase):
    def new(self,n=1,names=None):
        fs=[Fixture(names) for _ in range(n)]
        refs=[WaveContactReference(f.names) for f in fs]
        b=BatchWave004(fs[0].names,n)
        scalar=[r.reset(f.snapshot()) for r,f in zip(refs,fs)]
        out=b.reset(pack([f.snapshot() for f in fs]),torch.ones(n,dtype=torch.bool),torch.zeros(n,dtype=torch.int64))
        for i,value in enumerate(scalar):assert_row(self,out,i,value)
        return b,refs,fs

    def test_asynchronous_swing_landing_stop_and_partial_reset_match_scalar(self):
        b,refs,fs=self.new(6);phases=set();mixed=False;landings=0
        for k in range(790):
            commands=[[.005,0,0] if k<250 else [0,0,0],
                      [0,.005,0] if 45<=k<350 else [0,0,0],
                      [.003,0,.005] if 80<=k<300 else [0,0,0],
                      [0,0,0],[-.005,0,0] if 120<=k<370 else [0,0,0],
                      [0,0,.015] if 10<=k<320 else [0,0,0]]
            if k==200:
                fs[2]=Fixture(fs[2].names);refs[2]=WaveContactReference(fs[2].names);refs[2].reset(fs[2].snapshot())
                old={key:value.clone() for key,value in b.s.items()}
                selected=torch.tensor([False,False,True,False,False,False])
                b.reset(pack([f.snapshot() for f in fs]),selected,torch.tensor([0,0,1,0,0,0]))
                for key in old:torch.testing.assert_close(b.s[key][~selected],old[key][~selected],atol=0,rtol=0)
            snaps=[f.snapshot() for f in fs]
            scalar=[r.step(s,c) for r,s,c in zip(refs,snaps,commands)]
            out=b.step(pack(snaps),commanded(commands))
            for i,value in enumerate(scalar):
                assert_row(self,out,i,value)
                self.assertTrue(value['valid'][0],value['failure_reason']);fs[i].advance(value)
                phases.add(value['state']['mode'])
                landings=max(landings,value['state']['confirmed_touchdowns'])
            mixed |= len({value['state']['mode'] for value in scalar})>=3
        self.assertTrue(mixed);self.assertGreater(landings,1)
        self.assertTrue({'swing','landing_blend','reference_quiet_hold','stopping_reference_motion'}<=phases)
        self.assertTrue((b.s['command']==0).all())

    def test_actual004_prefix_preserves_rejection_and_not_physical_pass(self):
        data=np.load(HERE/'inputs/actual004_trace.npz');names=tuple(data['joint_names'])
        def snapshot(k):return {key:data[key][k].copy() for key in data.files if data[key].shape[:1]==(625,)}
        scalar=WaveContactReference(names);initial=snapshot(199);scalar.reset(initial)
        batch=BatchWave004(names,1);batch.reset(pack([initial]),torch.tensor([True]),torch.tensor([0]))
        rejected=None
        for k in range(199,625):
            m=snapshot(k);old=scalar.step(m,[.005,0,0]);new=batch.step(pack([m]),commanded([[.005,0,0]]))
            assert_row(self,new,0,old,atol=5e-10)
            if not old['valid'][0]:
                rejected=(k,old['failure_reason'],int(new['failure_code'][0]));break
        self.assertIsNotNone(rejected)
        self.assertEqual(rejected[0],624);self.assertIn('12mm',rejected[1]);self.assertEqual(rejected[2],10)
        self.assertFalse(new['physical_admission']);self.assertFalse(new['policy_training_allowed'])

    def test_hull_margin_equals_scipy_with_mixed_support_exclusions(self):
        b,refs,fs=self.new(6)
        for i,f in enumerate(fs):
            f.R=np.array([[np.cos(.05*i),-np.sin(.05*i),0],[np.sin(.05*i),np.cos(.05*i),0],[0,0,1.]])
            f.p+=np.array([.02*i,-.03*i,0])
        snaps=[f.snapshot() for f in fs];m=pack(snaps);contact,_,_=b._read(m)
        excluded=torch.tensor([-1,0,1,2,3,4])
        margin,count=b._support(m,contact,excluded)
        for i,(r,s) in enumerate(zip(refs,snaps)):
            expected=r._support(r._read(s),None if i==0 else i-1)
            self.assertAlmostEqual(float(margin[i]),expected,places=12)
        self.assertEqual(count.tolist(),[6,5,5,5,5,5])
        broken={k:v.clone() for k,v in m.items()};broken['contact_point_world_m'][:,:,:2]=0
        bad,_=b._support(broken,contact,excluded);self.assertTrue(torch.isneginf(bad).all())

    def test_batched_prediction_matches198_scalar_filter_updates(self):
        b,refs,fs=self.new(4)
        for i,r in enumerate(refs):
            r.command=np.array([.001*i,.001*(i-1),.002*i]);r.command_rate=np.array([.001,-.002,.0005])
            r.yaw=.4*i;r.position=np.array([.1*i,-.05*i,.13])
            for key,value in [('command',r.command),('rate',r.command_rate),('yaw',r.yaw),('position',r.position)]:
                b.s[key][i]=torch.tensor(value,dtype=torch.float64)
        targets=commanded([[.005,0,0],[0,.005,.01],[-.003,-.003,-.01],[0,0,0]])
        p,R=b._predict(targets)
        for i,r in enumerate(refs):
            expected=r._predict(targets[i].numpy(),7.9)
            np.testing.assert_allclose(p[i].numpy(),expected[0],atol=2e-13,rtol=0)
            np.testing.assert_allclose(R[i].numpy(),expected[1],atol=2e-13,rtol=0)

    def test_first_liftoff_keeps_original_and_horizontal_clocks_and_coefficients(self):
        b,refs,fs=self.new(1);m=fs[0].snapshot();old=refs[0].step(m,[.005,0,0])
        out=b.step(pack([m]),commanded([[.005,0,0]]));assert_row(self,out,0,old)
        swing=refs[0].swing
        np.testing.assert_allclose(b.s['sw_coeff'][0],swing.coeff,atol=2e-13,rtol=0)
        np.testing.assert_allclose(b.s['sw_hcoeff'][0],swing.horizontal.coeff,atol=2e-13,rtol=0)
        for t in [0.,.03,.5,1.59999,1.6,1.60001,1.99,2.1]:
            pva=b._sample('swing',torch.tensor([t],dtype=torch.float64))
            for actual,expected in zip(pva,swing.sample(t)):
                np.testing.assert_allclose(actual[0].numpy(),expected,atol=2e-12,rtol=0)

    def test_failure_isolation_latch_stale_reset_and_nonfinite_inputs(self):
        b,refs,fs=self.new(4);snap=[f.snapshot() for f in fs]
        snap[1]['distal_contact'][0,:2]=False
        snap[2]['position_world_m'][0,0]=np.nan
        snap[3]['time_s'][0]+=.1
        old_q=b.s['q'].clone();out=b.step(pack(snap),commanded([[0,0,0]]*4))
        self.assertEqual(out['valid'].tolist(),[True,False,False,False])
        self.assertEqual(out['failure_code'].tolist(),[0,4,2,20])
        torch.testing.assert_close(b.s['q'][1:],old_q[1:],atol=0,rtol=0)
        clean=[f.snapshot() for f in fs];clean[0]['time_s'][0]=.02
        again=b.step(pack(clean),commanded([[0,0,0]]*4));self.assertEqual(again['valid'].tolist(),[True,False,False,False])
        chosen=torch.tensor([False,True,False,False]);out=b.reset(pack(clean),chosen,torch.tensor([0,0,0,0]))
        self.assertEqual(int(out['failure_code'][1]),24)
        out=b.reset(pack(clean),chosen,torch.tensor([0,1,0,0]));self.assertTrue(out['valid'][1]);self.assertFalse(out['valid'][2])

    def test_reset_named_order_preload_and_inference_mode_lifecycle(self):
        f=Fixture();names=tuple(reversed(f.names));f=Fixture(names,preload=.01)
        scalar=WaveContactReference(names);old=scalar.reset(f.snapshot())
        with torch.inference_mode():
            b=BatchWave004(names,1);out=b.reset(pack([f.snapshot()]),torch.tensor([True]),torch.tensor([0]))
        assert_row(self,out,0,old)
        for _ in range(3):
            m=f.snapshot();old=scalar.step(m,[0,0,0]);out=b.step(pack([m]),commanded([[0,0,0]]));assert_row(self,out,0,old)
            # Keep the declared measured preload, rather than the ideal fixture
            # snapping actual q to target and unintentionally losing support.
            f.time=old['target_time_s'];f.v=old['v_ref'][0].copy()
        np.testing.assert_array_equal(out['q_ref'].numpy(),old['q_ref'])
        self.assertGreater(float(b.s['preload_q'].abs().max()),.009)

    def test_landing_contact_gap_confirmation_and_stop_are_independent_per_row(self):
        b,refs,fs=self.new(3)
        for k in range(130):
            snaps=[f.snapshot() for f in fs]
            old=[r.step(m,[.005,0,0]) for r,m in zip(refs,snaps)]
            out=b.step(pack(snaps),commanded([[.005,0,0]]*3))
            for i,value in enumerate(old):assert_row(self,out,i,value);fs[i].advance(value)
            if refs[0].mode=='landing_blend':break
        else:self.fail('Fixture never reached provisional landing')
        self.assertTrue(b.s['land_active'].all());self.assertTrue((b.s['touchdowns']==0).all())
        baseline_liftoffs=b.s['liftoffs'].clone()
        for k in range(50):
            snaps=[f.snapshot() for f in fs]
            for i,m in enumerate(snaps):
                active=refs[i].current_leg
                if active is None:continue
                touch=(i==0 or (i==2 and (k>=8 or k%2==1)))
                m['distal_contact'][0,active]=touch;m['contact_point_valid'][0,active]=touch
                m['contact_point_world_m'][0,active]=m['reference_point_world_m'][0,active] if touch else np.nan
            old=[r.step(m,[0,0,0]) for r,m in zip(refs,snaps)]
            out=b.step(pack(snaps),commanded([[0,0,0]]*3))
            for i,value in enumerate(old):
                assert_row(self,out,i,value)
                if value['valid'][0]:fs[i].advance(value)
            if k<5:self.assertTrue(out['valid'].all())
            else:self.assertEqual(int(out['failure_code'][1]),14)
            if k<2:self.assertEqual(int(out['state']['touchdowns'][0]),0)
        self.assertEqual(b.s['touchdowns'].tolist(),[1,0,1])
        torch.testing.assert_close(b.s['liftoffs'],baseline_liftoffs,atol=0,rtol=0)
        self.assertTrue(b.s['stop_valid'][[0,2]].all())

    def test_mixed_reset_position_precision_matches_scalar_and_is_observable(self):
        b,refs,fs=self.new(2)
        fs[1].p=fs[1].p.astype(np.float32)
        refs[1]=WaveContactReference(fs[1].names);refs[1].reset(fs[1].snapshot())
        b.reset(pack([f.snapshot() for f in fs]),torch.tensor([False,True]),torch.tensor([0,1]))
        self.assertEqual(b.s['position_float32'].tolist(),[False,True])
        for _ in range(15):
            snaps=[f.snapshot() for f in fs];old=[r.step(m,[.004,.001,.005]) for r,m in zip(refs,snaps)]
            out=b.step(pack(snaps),commanded([[.004,.001,.005]]*2))
            for i,value in enumerate(old):assert_row(self,out,i,value);fs[i].advance(value)
        bad=pack([f.snapshot() for f in fs]);bad['position_is_float32'][1]=False
        out=b.step(bad,commanded([[0,0,0]]*2));self.assertEqual(int(out['failure_code'][1]),2)

    def test_dynamic_source_has_no_host_transfer_or_replica_loop(self):
        tree=ast.parse((HERE/'batch_wave.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotIn(node.func.attr,('cpu','numpy','item','tolist'))
            if isinstance(node,ast.For) and isinstance(node.iter,ast.Call) and isinstance(node.iter.func,ast.Name):
                if node.iter.func.id=='range':self.assertEqual([a.value for a in node.iter.args],[198])


if __name__=='__main__':unittest.main()
