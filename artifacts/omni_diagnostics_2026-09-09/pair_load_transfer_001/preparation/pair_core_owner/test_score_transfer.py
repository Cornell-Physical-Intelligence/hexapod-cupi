import copy,unittest
import numpy as np
from synthetic_fixture import Fixture
from load_transfer import PairLoadTransfer
from score_transfer import score_transfer,check_substep_batch


def synthetic_run():
    f=Fixture();c=PairLoadTransfer(f.names);c.reset(f.snapshot());rows=[];refs=[]
    for k in range(1100):
        out=c.step(f.snapshot())
        assert out['valid'][0],out['failure_reason']
        f.advance(out);rows.append(f.snapshot());refs.append(out)
    data={key:np.stack([r[key] for r in rows]) for key in rows[0]};data['joint_names']=np.array(f.names)
    sub=synthetic_substeps(data)
    return data,refs,sub


def synthetic_substeps(data):
    n=len(data['time_s']);count=n*8+1
    sub={key:np.concatenate([data[key][:1],np.repeat(data[key],8,axis=0)]) for key in ('computed_torque_nm','applied_torque_nm')}
    sub.update(time_s=np.arange(count)*.0025,sdk_sim_timestamp_s=np.arange(count)*.0025+4.,
               control_index=np.r_[-1,np.repeat(np.arange(n),8)],substep_index=np.r_[0,np.tile(np.arange(1,9),n)],sim_step_counter=np.arange(count)+1600)
    return sub


class ScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data,cls.refs,cls.sub=synthetic_run()

    def test_online_substep_hook_stops_before_next_reference_on_interior_peak(self):
        rows=[dict(control_index=4,substep_index=i,sim_step_counter=1600+32+i,time_s=(32+i)*.0025,sdk_sim_timestamp_s=4+(32+i)*.0025,computed_torque_nm=np.ones((1,18)),applied_torque_nm=np.ones((1,18))) for i in range(1,9)]
        kwargs=dict(initial_counter=1600,control_row=rows[-1])
        self.assertTrue(check_substep_batch(rows,4,**kwargs)['complete'])
        rows[3]['computed_torque_nm'][0,7]=1.61
        with self.assertRaisesRegex(ValueError,'Stop before next target'):check_substep_batch(rows,4,**kwargs)
        with self.assertRaisesRegex(ValueError,'Eight'):check_substep_batch(rows[:-1],4,**kwargs)

    def test_synthetic_fixture_satisfies_proposed_checks_but_never_qualifies_physics(self):
        r=score_transfer(self.data,self.refs,substeps=self.sub)
        self.assertTrue(r['proposed_criteria_met'],r['failed_proposed_criteria'])
        self.assertFalse(r['physics_qualification']);self.assertFalse(r['external_source_asset_provenance_verified'])
        self.assertGreaterEqual(r['longest_measured_unloaded_hold_s'],1.)
        self.assertLess(r['force_review']['relative_vertical_balance_error'],1e-10)

    def test_missing_or_overlimit_substeps_fail_despite_good_control_samples(self):
        r=score_transfer(self.data,self.refs);self.assertFalse(r['proposed_criteria_met'])
        sub={k:v.copy() for k,v in self.sub.items()};sub['computed_torque_nm'][1311,0,4]=1.601
        r=score_transfer(self.data,self.refs,substeps=sub)
        self.assertFalse(r['proposed_criteria_met']);self.assertIn('Substep',str(r['failed_proposed_criteria']))

    def test_reordered_short_or_misaligned_substep_evidence_rejected(self):
        for kind in ('short_applied','cadence','counter','reorder','endpoint'):
            sub={k:v.copy() for k,v in self.sub.items()}
            if kind=='short_applied':sub['applied_torque_nm']=sub['applied_torque_nm'][:1]
            if kind=='cadence':sub['sdk_sim_timestamp_s'][200]+=.0025
            if kind=='counter':sub['sim_step_counter'][200]+=1
            if kind=='reorder':sub['substep_index'][200:202]=sub['substep_index'][200:202][::-1]
            if kind=='endpoint':sub['applied_torque_nm'][200,0,2]+=.01
            with self.assertRaises(ValueError,msg=kind):score_transfer(self.data,self.refs,substeps=sub)

    def test_reference_time_and_actual_emitted_target_alignment_required(self):
        for kind in ('time','nan_time','target'):
            refs=copy.deepcopy(self.refs)
            if kind=='time':refs[240]['target_time_s']+=.02
            if kind=='nan_time':refs[240]['target_time_s']=float('nan')
            if kind=='target':refs[240]['q_ref'][0,4]+=.001
            with self.assertRaises(ValueError,msg=kind):score_transfer(self.data,refs,substeps=self.sub)

    def test_false_clock_liftoff_without_measured_flight_fails(self):
        d={k:v.copy() for k,v in self.data.items()};d['distal_contact'][:]=True;d['contact_point_valid'][:]=True
        d['contact_point_world_m']=d['reference_point_world_m'].copy()
        r=score_transfer(d,self.refs,substeps=self.sub)
        self.assertFalse(r['proposed_criteria_met']);self.assertEqual(r['longest_measured_unloaded_hold_s'],0.)

    def test_measured_quiet_and_contiguous_time_cannot_be_spoofed_by_quiet_label(self):
        d={k:v.copy() for k,v in self.data.items()};d['joint_velocity_rad_s'][-300:,0,2]=.04
        r=score_transfer(d,self.refs,substeps=self.sub);self.assertFalse(r['proposed_criteria_met'])
        d={k:v.copy() for k,v in self.data.items()};d['time_s'][300,0]+=.01
        with self.assertRaises(ValueError):score_transfer(d,self.refs,substeps=self.sub)


if __name__=='__main__':unittest.main()
