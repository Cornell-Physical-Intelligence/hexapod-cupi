import unittest
from dataclasses import replace
import numpy as np
from supervisor import MapSnapshot,map_eligibility,swept_foot_envelope,assess,teacher_eligibility


def fixture(mode='synthetic_interface'):
    shape=(60,60)
    return MapSnapshot(np.zeros(shape),np.full(shape,.002**2),np.ones(shape,bool),np.full(shape,1.),(-.6,-.6),.02,'odom','shared_acquisition_clock',1.05,'synthetic_known_geometry',mode)

def envelope(m,poses=None,times=None,feet=None):
    return swept_foot_envelope(m,poses_xy_yaw=[[0,0,0],[.01,0,0]] if poses is None else poses,
        relative_times_s=[0,.10] if times is None else times,
        foot_centers_body_xy=[[.25,0]] if feet is None else feet,pad_radius_m=.01,uncertainty_radius_m=.005,
        world_frame=m.world_frame,clock_id=m.clock_id,controller_id='synthetic_supplied_pose_path_not_braking_model')

def geometry(m,inside=None):
    return map_eligibility(m,reference_surface_height_m=0.,inside_course=np.ones(m.height_m.shape,bool) if inside is None else inside)

def check(m,g=None,e=None):
    return assess(m,geometry(m) if g is None else g,envelope(m) if e is None else e,requested_twist=[.005,0,0])

class SupervisorTests(unittest.TestCase):
    def test_known_flat_small_future_envelope_permitted(self):
        r=check(fixture());self.assertTrue(r['permit_candidate']);self.assertGreater(r['required_cells'],0)
        self.assertFalse(r['stop_feasibility_proven']);self.assertEqual(r['valid_until_s'],1.25)

    def test_observed_pit_is_usable_but_ineligible(self):
        m=fixture();xy=m.centers();pit=np.linalg.norm(xy-[.25,0],axis=-1)<.08;m.height_m[pit]=-.12
        r=check(m);self.assertFalse(r['permit_candidate']);self.assertIn('known_hazard',r['reason_codes'])
        self.assertEqual(r['usable_now_required_cells'],r['required_cells']);self.assertLess(r['eligible_required_cells'],r['required_cells'])
        self.assertEqual(m.height_m[pit].min(),-.12)

    def test_unknown_is_not_zero_height_support(self):
        m=fixture();mask=envelope(m).required;m.observed[mask]=False;m.height_m[mask]=np.nan
        r=check(m);self.assertFalse(r['permit_candidate']);self.assertIn('unobserved',r['reason_codes'])
        np.testing.assert_array_equal(r['admitted_twist'],[0,0,0]);self.assertFalse(r['stop_feasibility_proven'])

    def test_stale_uncertain_future_negative_variance_are_rejected(self):
        for fault in ('stale','uncertain','future','negative_variance'):
            m=fixture()
            if fault=='stale':m.capture_time_s[:]=.7
            if fault=='uncertain':m.variance_m2[:]=.02**2
            if fault=='future':m.capture_time_s[:]=1.06
            if fault=='negative_variance':m.variance_m2[:]=-.01
            r=check(m);self.assertFalse(r['permit_candidate'],fault);self.assertIn('unusable_now',r['reason_codes'])

    def test_current_usable_data_cannot_silently_cover_later_stopping_use(self):
        m=fixture();r=check(m,e=envelope(m,times=[0,.5]))
        self.assertEqual(r['usable_now_required_cells'],r['required_cells'])
        self.assertFalse(r['permit_candidate']);self.assertIn('expires_before_required_use',r['reason_codes'])

    def test_wrong_frame_or_clock_raises_before_decision(self):
        m=fixture();g=geometry(m)
        for bad in (replace(g,world_frame='map'),replace(g,clock_id='usb_clock')):
            with self.assertRaisesRegex(ValueError,'Wrong frame'):check(m,g=bad)
        with self.assertRaisesRegex(ValueError,'frame/clock'):
            swept_foot_envelope(m,poses_xy_yaw=[[0,0,0]],relative_times_s=[0],foot_centers_body_xy=[[.25,0]],pad_radius_m=.01,uncertainty_radius_m=0,world_frame='body',clock_id=m.clock_id,controller_id='synthetic')

    def test_outside_course_and_outside_raster_stay_invalid(self):
        m=fixture();inside=m.centers()[...,0]<.20
        r=check(m,g=geometry(m,inside));self.assertFalse(r['permit_candidate']);self.assertIn('outside_course',r['reason_codes'])
        r=check(m,e=envelope(m,feet=[[.59,0]]));self.assertFalse(r['permit_candidate']);self.assertIn('envelope_outside_map',r['reason_codes'])

    def test_swept_rotation_detects_pit_missed_by_endpoints(self):
        m=fixture();xy=m.centers();pit=np.linalg.norm(xy-[.177,.177],axis=-1)<.045;m.height_m[pit]=-.12
        g=geometry(m)
        start=envelope(m,poses=[[0,0,0]],times=[0]);end=envelope(m,poses=[[0,0,np.pi/2]],times=[0])
        self.assertTrue(check(m,g,start)['permit_candidate']);self.assertTrue(check(m,g,end)['permit_candidate'])
        swept=envelope(m,poses=[[0,0,0],[0,0,np.pi/2]])
        r=check(m,g,swept);self.assertFalse(r['permit_candidate']);self.assertIn('known_hazard',r['reason_codes'])
        self.assertGreater(swept.required.sum(),(start.required|end.required).sum())

    def test_empty_envelope_is_not_a_pass(self):
        m=fixture();e=envelope(m);e.required[:]=False
        self.assertIn('empty_required_envelope',check(m,e=e)['reason_codes'])

    def test_existing_local_map_adapter_keeps_unseen_cells_unknown(self):
        from perception_replay import LocalHeightMap,WorldPoints
        from supervisor import snapshot_from_local_map
        model=LocalHeightMap()
        model.integrate(WorldPoints(np.array([[.25,0,-.12]]),np.array([.002**2]),np.array([1.]),'odom','synthetic_camera','synthetic_known',{}))
        m=snapshot_from_local_map(model,query_time_s=1.05,clock_id='shared',calibration_id='synthetic_known',source_mode='synthetic_interface')
        self.assertEqual(m.observed.sum(),1);self.assertEqual(np.isfinite(m.height_m).sum(),1)
        r=check(m);self.assertFalse(r['permit_candidate']);self.assertIn('unobserved',r['reason_codes'])

    def test_teacher_truth_is_separate_and_keeps_pit_ineligible(self):
        import torch
        torch.set_num_threads(1)
        from terrain_readiness import terrain_mesh
        from hexapod_terrain.support_queries import TerrainSupportQueries
        v,f,meta=terrain_mesh('pit',1103);meta['id']='synthetic_pit'
        q=TerrainSupportQueries([(meta,None,v,f)])
        m=fixture('ideal_teacher');g=teacher_eligibility(m,q,0)
        center=(30,30);self.assertTrue(g.geometry_hit[center]);self.assertTrue(g.avoidance_hazard[center]);self.assertFalse(g.support_geometry[center])
        r=check(m,g,envelope(m,feet=[[0,0]]));self.assertFalse(r['permit_candidate']);self.assertIn('known_hazard',r['reason_codes'])
        m.source_mode='simulated_depth'
        with self.assertRaisesRegex(ValueError,'Teacher geometry'):teacher_eligibility(m,q,0)

if __name__=='__main__':unittest.main()
