import json,unittest
from dataclasses import replace
import numpy as np
from footprint_checker import *

class CheckerTests(unittest.TestCase):
    def ledger(self):return FootprintEvidenceLedger(origin_xy=(0.,0.),size_xy=(.12,.12),resolution_m=.02,world_frame='odom',clock_id='capture',sensors=[SensorSchedule('cam','nominal_unqualified',1.,.1,.04)])
    def frame(self,ledger,*,capture=1.,receive=1.04,sequence=0,visible=True,height=0.,sigma=.001):
        ix,iy=np.indices(ledger.map.shape);points=np.column_stack(((ix.ravel()+.5)*.02,(iy.ravel()+.5)*.02,np.full(ix.size,height)))
        mask=np.full(ledger.map.shape,visible,bool)
        points=points[mask.ravel()]
        cloud=WorldPoints(points,np.full(len(points),sigma**2),np.full(len(points),capture),'odom','cam','nominal_unqualified',{'synthetic_only':True})
        return FrameReceipt('cam',sequence,capture,receive,'capture',cloud,np.ones_like(mask),mask,np.zeros_like(mask),'exact_mesh_fixture')
    def geometry(self,ledger,now,*,hazard=False,inside=True):
        shape=ledger.map.shape;one=np.ones(shape,bool);pit=np.full(shape,hazard,bool)
        return SupportGeometry(one,np.full(shape,inside,bool),pit,~pit,'odom','capture',now,'Explicit synthetic flat/pit support fixture; never observation','teacher_fixture_only')
    def query(self,ledger,now=1.04,use=None,uncertainty=None,**kwargs):
        proposal=FootstepProposal('new_lf','lf',(.05,.05,0.),.006,1.,now if use is None else use,'odom','capture')
        return ledger.query(proposal,self.geometry(ledger,now),uncertainty or DriftUncertainty(0.,0.,0.,0.),now_s=now,**kwargs)
    def test_missing_latest_frame_retains_only_unexpired_evidence(self):
        l=self.ledger();l.ingest(self.frame(l),now_s=1.04)
        r=self.query(l,1.14,use=1.20);self.assertTrue(r['sensor_health']['cam']['missing_latest_scheduled_frame']);self.assertEqual(r['decision'],'lease_covers_requested_use_time')
        r=self.query(l,1.26);self.assertEqual(r['decision'],'blocked');self.assertGreater(r['reason_cell_counts']['stale_now'],0)
        self.assertTrue(l.map.observed.all());np.testing.assert_array_equal(l.map.capture,1.)
    def test_occluded_landing_is_unknown_and_does_not_get_zero_height(self):
        l=self.ledger();l.ingest(self.frame(l,visible=False),now_s=1.04)
        r=self.query(l);self.assertEqual(r['decision'],'blocked');self.assertGreater(r['reason_cell_counts']['unobserved'],0)
        self.assertTrue(np.isnan(l.map.height).all());self.assertFalse(l.map.observed.any())
    def test_new_occlusion_does_not_refresh_old_capture(self):
        l=self.ledger();l.ingest(self.frame(l),now_s=1.04);l.ingest(self.frame(l,capture=1.1,receive=1.14,sequence=1,visible=False),now_s=1.14)
        np.testing.assert_array_equal(l.map.capture,1.);self.assertEqual(self.query(l,1.14)['decision'],'lease_covers_requested_use_time')
        self.assertEqual(self.query(l,1.26)['decision'],'blocked')
    def test_drift_and_uncertainty_fail_closed(self):
        l=self.ledger();l.ingest(self.frame(l),now_s=1.04)
        r=self.query(l,1.20,uncertainty=DriftUncertainty(0.,.1,0.,0.));self.assertEqual(r['decision'],'blocked');self.assertGreater(r['reason_cell_counts']['uncertain_now'],0)
        r=self.query(l,1.21,uncertainty=DriftUncertainty(0.,0.,.1,0.));self.assertTrue(r['outside_map_now']);self.assertEqual(r['decision'],'blocked')
        with self.assertRaises(ValueError):self.query(l,1.22,uncertainty=DriftUncertainty(float('nan'),0.,0.,0.))
    def test_future_capture_receipt_and_per_point_times_rejected_atomically(self):
        l=self.ledger()
        for frame in [self.frame(l,capture=1.1,receive=1.2),self.frame(l,capture=1.1,receive=1.0)]:
            with self.assertRaises(ValueError):l.ingest(frame,now_s=1.04)
        f=self.frame(l);f.cloud.capture_time_s[:]=1.05
        with self.assertRaises(ValueError):l.ingest(f,now_s=1.04)
        self.assertFalse(l.map.observed.any());self.assertFalse(l.receipts)
    def test_cannot_query_past_after_new_data_or_anticipate_new_proposal(self):
        l=self.ledger();l.ingest(self.frame(l),now_s=1.04)
        with self.assertRaises(ValueError):self.query(l,1.03)
        p=FootstepProposal('future','lf',(.05,.05,0.),.006,1.2,1.3,'odom','capture')
        with self.assertRaises(ValueError):l.query(p,self.geometry(l,1.04),DriftUncertainty(0,0,0,0),now_s=1.04)
    def test_future_use_needs_existing_capture_lease_not_expected_frames(self):
        l=self.ledger();l.ingest(self.frame(l),now_s=1.04);r=self.query(l,use=3.04)
        self.assertEqual(r['decision'],'fresh_now_only');self.assertFalse(r['lease_covers_requested_use_time']);self.assertFalse(r['future_observations_assumed']);self.assertFalse(r['physical_stop_or_abort_proven'])
        self.assertGreater(r['reason_cell_counts']['expires_before_use'],0)
    def test_age_boundary_and_late_receipt(self):
        l=self.ledger();l.ingest(self.frame(l),now_s=1.04);self.assertEqual(self.query(l,1.25)['decision'],'lease_covers_requested_use_time');self.assertEqual(self.query(l,1.250001)['decision'],'blocked')
        l=self.ledger();l.ingest(self.frame(l,receive=1.4),now_s=1.4);self.assertFalse(l.map.observed.any());self.assertEqual(l.frames[0]['expired_points'],36)
    def test_planted_contact_never_fills_new_footprint(self):
        l=self.ledger();contact=PlantedContact('lf',1.04,(.05,.05,0.),True,True,'odom','capture')
        r=self.query(l,planted_contacts=[contact]);self.assertEqual(r['decision'],'blocked');self.assertTrue(r['measured_current_contact_support'][0]['measured_contact_support']);self.assertFalse(l.map.observed.any())
        with self.assertRaises(ValueError):self.query(l,1.05,planted_contacts=[contact])
    def test_float32_measured_contact_receipt_is_json_serializable(self):
        l=self.ledger();contact=PlantedContact('lf',1.04,tuple(np.array([.05,.05,0.],np.float32)),True,True,'odom','capture')
        result=self.query(l,planted_contacts=[contact]);result.pop('masks')
        encoded=json.dumps(result,allow_nan=False);self.assertIn('measured_contact_support',encoded)
    def test_observed_pit_is_not_support_and_height_is_preserved(self):
        l=self.ledger();l.ingest(self.frame(l,height=-.08),now_s=1.04)
        p=FootstepProposal('pit','lf',(.05,.05,-.08),.006,1.,1.04,'odom','capture')
        r=l.query(p,self.geometry(l,1.04,hazard=True),DriftUncertainty(0,0,0,0),now_s=1.04)
        self.assertGreater(r['usable_now_use_cells'],0);self.assertEqual(r['eligible_use_cells'],0);self.assertEqual(r['decision'],'blocked');np.testing.assert_array_equal(l.map.height,-.08)
    def test_wrong_frame_clock_geometry_and_occluded_cloud_rejected(self):
        l=self.ledger();f=self.frame(l);f.clock_id='other'
        with self.assertRaises(ValueError):l.ingest(f,now_s=1.04)
        f=self.frame(l);f.mesh_clear_mask[:]=False
        with self.assertRaises(ValueError):l.ingest(f,now_s=1.04)
        l.ingest(self.frame(l),now_s=1.04);p=FootstepProposal('wrong','lf',(.05,.05,0.),.006,1.,1.04,'map','capture')
        with self.assertRaises(ValueError):l.query(p,self.geometry(l,1.04),DriftUncertainty(0,0,0,0),now_s=1.04)
    def test_outside_course_and_reordered_frames_do_not_pass(self):
        l=self.ledger();l.ingest(self.frame(l),now_s=1.04)
        with self.assertRaises(ValueError):l.ingest(self.frame(l),now_s=1.05)
        f=self.frame(l,capture=1.1,receive=1.14,sequence=1);l.ingest(f,now_s=1.14)
        with self.assertRaises(ValueError):l.ingest(self.frame(l,capture=1.05,receive=1.15,sequence=2),now_s=1.15)
        p=FootstepProposal('boundary','lf',(.05,.05,0.),.006,1.,1.15,'odom','capture')
        r=l.query(p,self.geometry(l,1.15,inside=False),DriftUncertainty(0,0,0,0),now_s=1.15);self.assertEqual(r['decision'],'blocked');self.assertGreater(r['reason_cell_counts']['outside_course'],0)

if __name__=='__main__':unittest.main()
