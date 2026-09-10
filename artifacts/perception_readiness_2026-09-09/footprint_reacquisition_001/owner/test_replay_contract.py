"""Actual-evidence alignment and end-to-end lease assertions."""
import json,unittest
import numpy as np
from replay_checker import ROOT,verify,pose_matrix
class ReplayContractTests(unittest.TestCase):
    def test_hashed004_excerpts_match_time_and_named_original_endpoints(self):
        identity=verify();self.assertEqual(identity['valid_reference_rows_matched_to_actual_times'],426)
        d=np.load(ROOT/'inputs/motion_source004.npz');c=np.load(ROOT/'inputs/measured_contacts_source004.npz')
        np.testing.assert_array_equal(d['time_s'],c['time_s']);np.testing.assert_array_equal(d['distal_contact'],c['distal_contact'])
        refs=json.loads((ROOT/'inputs/reference_timing_source004.json').read_text());self.assertEqual(len({r['index'] for r in refs}),426)
        for r in refs:
            self.assertAlmostEqual(r['time_s'],d['time_s'][r['index'],0],places=9)
            active=int(d['active_swing_leg_index'][r['index']])
            if active>=0:np.testing.assert_array_equal(r['swing']['endpoint_world_m'],d['planned_footprint_centres_world_m'][r['index'],active])
            pose_matrix(d,r['index'])
    def test_report_uses_capture_leases_without_future_frame_inference(self):
        report=json.loads((ROOT/'report.json').read_text())
        self.assertFalse(report['future_observations_assumed']);self.assertFalse(report['actor_integrated']);self.assertEqual(report['max_capture_age_s'],.25)
        for case in report['cases']:
            for row in case['rows']:
                self.assertFalse(row['future_observations_assumed']);self.assertFalse(row['physical_stop_or_abort_proven'])
                if row['requested_use_time_s']-row['query_time_s']>.25:self.assertFalse(row['lease_covers_requested_use_time'])
                for contact in row['measured_current_contact_support']:
                    self.assertFalse(contact['updates_terrain_map']);self.assertFalse(contact['certifies_new_footprint'])
    def test_actual_missing_frame_is_detected_without_renewing_map(self):
        report=json.loads((ROOT/'report.json').read_text());cases={c['assumptions']['id']:c for c in report['cases']}
        missing=cases['one_missing_capture_5p5']['rows'];baseline=cases['continuous_six_camera']['rows']
        detected=[r for r in missing if any(s['missing_latest_scheduled_frame'] for s in r['sensor_health'].values())]
        self.assertEqual(len(detected),5);self.assertAlmostEqual(detected[0]['query_time_s'],5.54)
        self.assertFalse(detected[0]['acquisition_received_this_control'])
        outage=cases['outage_8p0_to_8p6']['rows'];stale=[r for r in outage if 8.16<=r['query_time_s']<=8.4]
        self.assertTrue(stale);self.assertTrue(all(r['retained_usable_fraction']==0 for r in stale));self.assertTrue(all(r['ever_observed_fraction']>0 for r in stale))
if __name__=='__main__':unittest.main()
