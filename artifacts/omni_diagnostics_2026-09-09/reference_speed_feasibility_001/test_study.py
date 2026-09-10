"""Meaningful bounds/lineage checks for the synthetic feasibility screen."""
from dataclasses import asdict
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from study import *
class SpeedStudyTests(unittest.TestCase):
    def test_only_declared_timing_and_translation_cap_change(self):
        old=asdict(WaveConfig());new=asdict(configuration(.75,.02))
        self.assertEqual({k for k in old if old[k]!=new[k]},{'swing_s','max_translation_mps'})
        self.assertEqual(new['lift_m'],.007);self.assertEqual(new['reference_velocity_rad_s'],1.75);self.assertEqual(new['reference_acceleration_rad_s2'],6.)
        with self.assertRaises(ValueError):configuration(.4,.02)
        verify()
    def test_actual_soft_limits_are_named_and_more_restrictive_than_URDF(self):
        f=IdealFixture();m=f.snapshot();r=WaveContactReference(f.names);out=r.reset(m)
        self.assertTrue(out['valid'][0]);np.testing.assert_array_equal(out['q_ref'],m['joint_target_rad'])
        urdf=ET.parse(ROOT/'oracle/geometry/f050_t060.urdf').getroot()
        limits={j.get('name'):[float(j.find('limit').get(k)) for k in ('lower','upper')] for j in urdf.findall('joint')}
        hard=np.array([limits[n] for n in f.names])
        self.assertTrue((f.actual_soft_limits[:,0]>=hard[:,0]).all());self.assertTrue((f.actual_soft_limits[:,1]<=hard[:,1]).all())
        self.assertTrue(np.any(f.actual_soft_limits!=hard))
        np.testing.assert_array_equal(r.lower,np.maximum(f.g.lower.numpy(),r._leg(f.actual_soft_limits[:,0])))
        np.testing.assert_array_equal(r.upper,np.minimum(f.g.upper.numpy(),r._leg(f.actual_soft_limits[:,1])))
    def test_audit_hook_keeps_parent_targets_and_verdicts(self):
        f=IdealFixture();a=AuditReference(f.names,configuration(1.,.01));b=WaveContactReference(f.names,configuration(1.,.01));a.reset(f.snapshot());b.reset(f.snapshot())
        for k in range(100):
            snap=f.snapshot();x=a.step(snap,[.01,0,0]);y=b.step(snap,[.01,0,0])
            self.assertEqual(x['valid'][0],y['valid'][0]);self.assertEqual(x['failure_reason'],y['failure_reason'])
            if not x['valid'][0]:break
            for key in ('q_ref','v_ref','a_ref'):np.testing.assert_array_equal(x[key],y[key])
            f.advance(x)
    def test_overbudget_attempt_is_saved_but_never_emitted_or_clipped(self):
        f=IdealFixture();r=AuditReference(f.names,configuration(.5,.04));r.reset(f.snapshot())
        for _ in range(300):
            out=r.step(f.snapshot(),[.04,0,0])
            if not out['valid'][0]:break
            f.advance(out)
        self.assertFalse(out['valid'][0]);self.assertIsNone(out['q_ref']);self.assertTrue(r.bound_attempts)
        last=r.bound_attempts[-1]
        self.assertTrue(last['maximum_velocity_rad_s']>1.75 or last['maximum_acceleration_rad_s2']>6. or last['minimum_joint_margin_rad']<.02)
        again=r.step(f.snapshot(),[0,0,0]);self.assertFalse(again['valid'][0]);self.assertIsNone(again['q_ref'])
    def test_never_airborne_synthetic_contact_cannot_complete_step(self):
        f=IdealFixture();f.force_never_flight=True;r=AuditReference(f.names,configuration(1.5,.01));r.reset(f.snapshot())
        for _ in range(60):
            out=r.step(f.snapshot(),[.01,0,0])
            if not out['valid'][0]:break
            f.advance(out)
        self.assertFalse(out['valid'][0]);self.assertIn('apex',out['failure_reason']);self.assertEqual(r.touchdowns,0)
if __name__=='__main__':unittest.main()
