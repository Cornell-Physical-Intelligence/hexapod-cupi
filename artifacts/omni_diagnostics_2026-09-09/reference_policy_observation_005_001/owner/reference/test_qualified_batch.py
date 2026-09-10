"""Successor qualification semantics including actual complete009 replay."""
from copy import deepcopy
from pathlib import Path
import json
import unittest
import numpy as np
import torch
from batch_wave import BatchWave005,MODES
from test_batch_wave import pack,commanded,assert_row,Fixture,WaveContactReference

HERE=Path(__file__).parent
ACTUAL_REPORT=None

def snapshot_reader(path):
    z=np.load(path);count=len(z['time_s']);arrays={k:z[k] for k in z.files if z[k].shape[:1]==(count,)}
    return tuple(z['joint_names']),count,lambda k:{name:value[k].copy() for name,value in arrays.items()}

class QualifiedBatchTests(unittest.TestCase):
    def test_actual009_full_recorded_contacts_and_stop_match_scalar_and_executed_reference(self):
        global ACTUAL_REPORT
        names,count,read=snapshot_reader(HERE/'inputs/actual009_trace.npz')
        recorded=json.loads((HERE/'inputs/actual009_reference_states.json').read_text())
        results={row['physical_step']:row['result'] for row in recorded if 'result' in row}
        scalar=WaveContactReference(names);scalar.reset(read(199))
        batch=BatchWave005(names,1);batch.reset(pack([read(199)]),torch.tensor([True]),torch.tensor([0]))
        maximum={k:0. for k in ('q_ref','v_ref','a_ref')};modes=set();rebound=None;last=None
        for index in range(199,count-1):
            step=index+1;command=[.005,0,0] if step<1400 else [0.,0.,0.]
            old=scalar.step(read(index),command);new=batch.step(pack([read(index)]),commanded([command]))
            assert_row(self,new,0,old,atol=5e-10);self.assertTrue(new['valid'][0],(index,int(new['failure_code'][0])))
            saved=results[step]
            for key in maximum:
                err=float(np.max(np.abs(new[key].numpy()-np.asarray(saved[key]))));maximum[key]=max(maximum[key],err)
                np.testing.assert_allclose(new[key].numpy(),saved[key],atol=5e-10,rtol=0,err_msg=f'{step}/{key}')
            for b,s in [('unqualified_return_time','last_unqualified_return_time_s'),('unqualified_lift','last_unqualified_lift_m')]:
                self.assertEqual(bool(new['state']['unqualified_return_valid'][0]),saved['state'][s] is not None)
                if saved['state'][s] is not None:self.assertAlmostEqual(float(new['state'][b][0]),saved['state'][s],places=12)
            self.assertEqual(MODES[int(new['state']['mode'][0])],saved['state']['mode'] if 'state' in saved else old['state']['mode'])
            modes.add(old['state']['mode'])
            if step==1037:
                rebound=dict(control_step=step,qualified=bool(new['state']['flight_seen'][0]),raw_samples=int(new['state']['raw_force_free_samples'][0]),returns=int(new['state']['unqualified_contact_returns'][0]),mode=old['state']['mode'])
            last=new
        self.assertEqual(int(last['state']['touchdowns'][0]),11);self.assertEqual(MODES[int(last['state']['mode'][0])],'reference_quiet_hold')
        self.assertFalse(rebound['qualified']);self.assertEqual(rebound['returns'],1);self.assertEqual(rebound['raw_samples'],2)
        self.assertTrue({'unloading','swing','landing_blend','reference_quiet_hold'}<=modes)
        ACTUAL_REPORT=dict(controls_replayed=count-200,all_controls_valid=True,confirmed_steps=11,final_mode='reference_quiet_hold',
            rebound_retained=rebound,max_abs_difference_vs_recorded_executable_reference=maximum,modes=sorted(modes),
            physical_result_source='actual009 external frozen screen; CPU replay does not add a new physics admission',tensor_GPU_verified=False)
    def test_actual008_rebound_and_old003_apex_rejection_remain_distinct(self):
        outcomes={}
        for number in ('003','008'):
            names,count,read=snapshot_reader(HERE/f'oracle/inputs/actual_wave{number}_trace.npz')
            scalar=WaveContactReference(names);scalar.reset(read(199))
            batch=BatchWave005(names,1);batch.reset(pack([read(199)]),torch.tensor([True]),torch.tensor([0]))
            for index in range(199,count):
                old=scalar.step(read(index),[.005,0,0]);new=batch.step(pack([read(index)]),commanded([[.005,0,0]]))
                assert_row(self,new,0,old,atol=5e-10)
                if not new['valid'][0]:break
            outcomes[number]=(index,bool(new['valid'][0]),int(new['failure_code'][0]))
        self.assertEqual(outcomes['003'],(603,False,25));self.assertEqual(outcomes['008'],(1036,True,0))
    def test_mixed_rows_unqualified_blips_qualified_obstacle_and_deadline(self):
        fs=[Fixture() for _ in range(3)];refs=[WaveContactReference(f.names) for f in fs]
        b=BatchWave005(fs[0].names,3)
        for r,f in zip(refs,fs):r.reset(f.snapshot())
        b.reset(pack([f.snapshot() for f in fs]),torch.ones(3,dtype=torch.bool),torch.zeros(3,dtype=torch.int64))
        bases=None
        for k in range(55):
            snapshots=[f.snapshot() for f in fs]
            if bases is None:bases=[s['reference_point_world_m'][0,0,2] for s in snapshots]
            if k>0:
                for i,s in enumerate(snapshots):
                    touch=(k%3==0);lift=0. if touch else (.003 if i==1 else .000008)
                    s['distal_contact'][0,0]=touch;s['contact_point_valid'][0,0]=touch
                    s['reference_point_world_m'][0,0,2]=bases[i]+lift;s['reference_point_velocity_world_mps'][0,0]=0
                    s['contact_point_world_m'][0,0]=s['reference_point_world_m'][0,0] if touch else np.nan
            olds=[]
            for i,(r,s) in enumerate(zip(refs,snapshots)):
                olds.append(r.step(s,[.005,0,0]))
            new=b.step(pack(snapshots),commanded([[.005,0,0]]*3))
            for i,old in enumerate(olds):
                assert_row(self,new,i,old)
                if old['valid'][0]:fs[i].advance(old)
            if k==3:self.assertEqual(new['valid'].tolist(),[True,False,True]);self.assertEqual(int(new['failure_code'][1]),8)
        self.assertEqual(new['failure_code'].tolist(),[25,8,25]);self.assertTrue((new['state']['touchdowns']==0).all())

if __name__=='__main__':unittest.main()
