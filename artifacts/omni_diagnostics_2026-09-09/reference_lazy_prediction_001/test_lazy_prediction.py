"""Exact full-state parity on actual recordings and asynchronous synthetic rows."""
from pathlib import Path
from copy import deepcopy
from types import MethodType
import unittest,sys
import numpy as np
import torch
from lazy_prediction import build_classes
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'tmp/reference_tensor_wave_005_001').is_dir())
PARENT=ROOT/'tmp/reference_tensor_wave_005_001'
Original,Lazy,module,OLD,NEW=build_classes(PARENT)
sys.path.insert(0,str(PARENT));from test_batch_wave import pack,Fixture,WaveContactReference

def same(test,a,b):
    test.assertEqual(set(a.s),set(b.s))
    for key in a.s:torch.testing.assert_close(a.s[key],b.s[key],atol=0,rtol=0,equal_nan=True,msg=key)
    for key in ('q_ref','v_ref','a_ref','valid','failure_code'):
        torch.testing.assert_close(a.output()[key],b.output()[key],atol=0,rtol=0,equal_nan=True,msg=key)

def count_predict(obj):
    original=obj._predict;obj.predict_calls=0
    def call(self,target):self.predict_calls+=1;return original(target)
    obj._predict=MethodType(call,obj)

class LazyTests(unittest.TestCase):
    def new(self,n=1,names=None):
        fs=[Fixture(names) for _ in range(n)];names=fs[0].names
        a,b=Original(names,n),Lazy(names,n);m=pack([f.snapshot() for f in fs]);ids=torch.zeros(n,dtype=torch.long)
        for obj in (a,b):obj.reset(m,torch.ones(n,dtype=torch.bool),ids);count_predict(obj)
        return a,b,fs
    def test_quiet_has_no_prediction_and_exact_float32_reset_stop_state(self):
        a,b,fs=self.new(32)
        for k in range(80):
            m=pack([f.snapshot() for f in fs])
            if k==0:
                for f in fs:f.p=f.p.astype(np.float32)
                m=pack([f.snapshot() for f in fs]);ids=torch.ones(32,dtype=torch.long)
                for obj in (a,b):obj.reset(m,torch.ones(32,dtype=torch.bool),ids)
            for obj in (a,b):obj.step(m,torch.zeros(32,3,dtype=torch.float64))
            same(self,a,b)
            for f in fs:f.time+=.02
        self.assertEqual(a.predict_calls,80);self.assertEqual(b.predict_calls,0)
        self.assertNotEqual(a.identity['implementation'],b.identity['implementation'])
        self.assertFalse(b.output()['policy_training_allowed'])
    def test_mixed_asynchronous_stance_swing_landing_stop_and_reset(self):
        a,b,fs=self.new(6);scalar=[WaveContactReference(f.names) for f in fs]
        for s,f in zip(scalar,fs):s.reset(f.snapshot())
        modes=set();launch_steps=0
        for k in range(790):
            command=[[.005,0,0] if k<250 else [0,0,0],[0,.005,0] if 45<=k<350 else [0,0,0],
                [.003,0,.005] if 80<=k<300 else [0,0,0],[0,0,0],[-.005,0,0] if 120<=k<370 else [0,0,0],
                [0,0,.015] if 10<=k<320 else [0,0,0]]
            if k==200:
                fs[2]=Fixture(fs[2].names);scalar[2]=WaveContactReference(fs[2].names);scalar[2].reset(fs[2].snapshot())
                select=torch.tensor([False,False,True,False,False,False]);ids=torch.tensor([0,0,1,0,0,0]);m=pack([f.snapshot() for f in fs])
                for obj in (a,b):obj.reset(m,select,ids)
                same(self,a,b)
            snaps=[f.snapshot() for f in fs];m=pack(snaps);cmd=torch.tensor(command,dtype=torch.float64);oldcount=b.predict_calls
            for obj in (a,b):obj.step(m,cmd)
            same(self,a,b);launch_steps+=b.predict_calls>oldcount;modes.update(b.s['mode'].tolist())
            for f,s,snap,c in zip(fs,scalar,snaps,command):
                out=s.step(snap,c);self.assertTrue(out['valid'][0]);f.advance(out)
        self.assertTrue({1,5,6,8}<=modes);self.assertLess(launch_steps,60);self.assertEqual(a.predict_calls,790)
    def test_full_actual009_contact_unqualified_rebound_and_stop(self):
        z=np.load(PARENT/'inputs/actual009_trace.npz');arrays={k:z[k] for k in z.files};names=tuple(arrays['joint_names'])
        rows=[{k:v[i] for k,v in arrays.items() if v.shape[:1]==(2400,)} for i in range(2400)]
        a,b=Original(names,1),Lazy(names,1)
        for obj in (a,b):obj.reset(pack([rows[199]]),torch.tensor([True]),torch.tensor([0]));count_predict(obj)
        rebound=False
        for i in range(199,2399):
            cmd=torch.tensor([[.005,0,0] if i<1399 else [0,0,0]],dtype=torch.float64)
            for obj in (a,b):obj.step(pack([rows[i]]),cmd)
            same(self,a,b);self.assertTrue(b.active.all())
            rebound |= bool(b.s['unqualified_contact_returns'][0]>0)
        self.assertTrue(rebound);self.assertEqual(int(b.s['touchdowns'][0]),11);self.assertEqual(int(b.s['mode'][0]),5)
        self.assertEqual(a.predict_calls,2200);self.assertEqual(b.predict_calls,11)
    def test_invalid_rows_failure_latch_and_reversed_named_order(self):
        a,b,fs=self.new(4,tuple(reversed(Fixture().names)))
        rows=[f.snapshot() for f in fs];rows[1]['position_world_m'][0,0]=np.nan;rows[2]['time_s'][0]=.1;rows[3]['distal_contact'][0,:2]=False
        for obj in (a,b):obj.step(pack(rows),torch.zeros(4,3,dtype=torch.float64))
        same(self,a,b);self.assertEqual(b.s['failure'].tolist(),[0,2,20,4]);self.assertEqual(b.predict_calls,0)
        rows=[f.snapshot() for f in fs];rows[0]['time_s'][0]=.02
        for obj in (a,b):obj.step(pack(rows),torch.zeros(4,3,dtype=torch.float64))
        same(self,a,b)
        with self.assertRaises(ValueError):b.step(pack(rows),torch.zeros(4,2,dtype=torch.float64))
if __name__=='__main__':unittest.main()
