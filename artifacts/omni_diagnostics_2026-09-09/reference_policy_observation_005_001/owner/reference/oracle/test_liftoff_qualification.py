"""Measured-state contract cases and actual-prefix regressions; not physics."""
import unittest
import numpy as np
from wave_reference import WaveContactReference
from test_wave_reference import Fixture
from replay_histories import replay

class QualificationTests(unittest.TestCase):
    def new(self):
        f=Fixture();r=WaveContactReference(f.names);r.reset(f.snapshot())
        out=r.step(f.snapshot(),[.005,0,0]);self.assertEqual(out['state']['mode'],'unloading');f.advance(out)
        return f,r,float(r.flight_baseline_z)
    def observed(self,f,r,z0,touch,dz):
        s=f.snapshot();s['reference_point_world_m'][0,0,2]=z0+dz
        s['reference_point_velocity_world_mps'][0,0]=0.
        s['distal_contact'][0,0]=touch;s['contact_point_valid'][0,0]=touch
        s['contact_point_world_m'][0,0]=s['reference_point_world_m'][0,0] if touch else np.nan
        o=r.step(s,[.005,0,0])
        if o['valid'][0]:f.advance(o)
        return o
    def test_two_micrometre_force_free_samples_return_to_unloading(self):
        f,r,z=self.new()
        for dz in [4e-6,8e-6]:
            out=self.observed(f,r,z,False,dz);self.assertTrue(out['valid'][0]);self.assertFalse(r.flight_seen)
        out=self.observed(f,r,z,True,0.)
        self.assertTrue(out['valid'][0]);self.assertEqual(r.mode,'unloading');self.assertEqual(r.flight_count,0)
        self.assertEqual(r.raw_force_free_samples,2);self.assertEqual(r.raw_force_free_runs,1)
        self.assertEqual(r.unqualified_contact_returns,1);self.assertEqual(r.last_unqualified_run_samples,2)
        self.assertAlmostEqual(r.last_unqualified_lift_m,8e-6,places=12)
        self.assertEqual(r.flight_peak_z,r.flight_baseline_z);self.assertEqual(r.touchdowns,0);self.assertIsNone(r.landing)
    def test_separate_blips_cannot_accumulate_count_or_height(self):
        f,r,z=self.new()
        for _ in range(3):
            out=self.observed(f,r,z,False,.003)
            self.assertTrue(out['valid'][0]);self.assertFalse(r.flight_seen);self.assertEqual(r.flight_count,1)
            out=self.observed(f,r,z,True,0.)
            self.assertTrue(out['valid'][0]);self.assertFalse(r.flight_seen);self.assertEqual(r.flight_count,0)
        self.assertEqual(r.raw_force_free_samples,3);self.assertEqual(r.raw_force_free_runs,3);self.assertEqual(r.unqualified_contact_returns,3)
        out=self.observed(f,r,z,False,1e-5);out=self.observed(f,r,z,False,2e-5)
        self.assertFalse(r.flight_seen);self.assertEqual(r.touchdowns,0)
    def test_real_qualification_does_not_ignore_early_obstacle_return(self):
        f,r,z=self.new()
        for dz in [.0021,.0025]:out=self.observed(f,r,z,False,dz)
        self.assertTrue(out['valid'][0]);self.assertTrue(r.flight_seen);self.assertEqual(r.mode,'swing')
        out=self.observed(f,r,z,True,.002)
        self.assertFalse(out['valid'][0]);self.assertIn('passed apex',out['failure_reason']);self.assertEqual(r.touchdowns,0)
    def test_repeated_unqualified_rebounds_have_finite_apex_deadline(self):
        f,r,z=self.new()
        for k in range(100):
            out=self.observed(f,r,z,k%3==2,0. if k%3==2 else 1e-5)
            if not out['valid'][0]:break
        self.assertFalse(out['valid'][0]);self.assertIn('by swing apex',out['failure_reason'])
        self.assertLessEqual(r.time-r.swing.t0,1.02+1e-9);self.assertEqual(r.touchdowns,0)
    def test_stop_during_unloading_completes_current_swing_without_new_liftoff(self):
        f,r,z=self.new();initial=r.liftoffs
        for _ in range(650):
            out=r.step(f.snapshot(),[0.,0.,0.]);self.assertTrue(out['valid'][0],out['failure_reason']);f.advance(out)
        self.assertEqual(r.liftoffs,initial);self.assertEqual(r.touchdowns,1);self.assertEqual(r.mode,'reference_quiet_hold')
        np.testing.assert_array_equal(r.command,[0,0,0])
    def test_actual008_final_unload_rebound_is_not_a_flight_or_landing(self):
        r=replay('008');o=r['outcomes']
        self.assertTrue(o['parent']['rejected']);self.assertEqual(o['parent']['sample'],1036)
        self.assertFalse(o['successor']['rejected']);self.assertEqual(o['successor']['confirmed_touchdowns'],7)
        self.assertFalse(o['successor']['last_flight_seen']);self.assertEqual(o['successor']['last_flight_count'],0)
        self.assertEqual(o['successor']['last_mode'],'unloading');self.assertEqual(o['successor']['unqualified_contact_returns'],1)
        self.assertEqual(o['successor']['last_unqualified_run_samples'],2)
        for difference in r['prefix_target_max_parent_difference'].values():self.assertEqual(difference,0.)
    def test_old_insufficient_clearance_and_endpoint_failures_remain_rejected(self):
        a=replay('003')['outcomes'];b=replay('004')['outcomes']
        self.assertTrue(a['parent']['rejected']);self.assertTrue(a['successor']['rejected'])
        self.assertIn('by swing apex',a['successor']['reason']);self.assertLess(a['successor']['sample'],a['parent']['sample'])
        self.assertTrue(b['parent']['rejected']);self.assertTrue(b['successor']['rejected'])
        self.assertEqual(b['successor']['sample'],624);self.assertIn('12mm',b['successor']['reason'])
    def test_old002_previously_reviewed_provisional_landing_is_unchanged(self):
        r=replay('002');o=r['outcomes'];self.assertFalse(o['parent']['rejected']);self.assertFalse(o['successor']['rejected'])
        self.assertEqual(o['successor']['last_mode'],'landing_blend');self.assertEqual(o['successor']['confirmed_touchdowns'],0)
        for difference in r['prefix_target_max_parent_difference'].values():self.assertEqual(difference,0.)

if __name__=='__main__':unittest.main()
