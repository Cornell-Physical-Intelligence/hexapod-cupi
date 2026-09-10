import unittest,json
import numpy as np
from replay import ROOT,ground_grid,footprint_mask,pose_matrix,StudyRobot,screen,PoseTimeline,PoseSample,calibrated_point_cloud,LocalHeightMap
from sensor_mount_math import optical_visibility


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=dict(np.load(ROOT/'inputs/motion_source004.npz'))
        cls.mount=json.loads((ROOT/'inputs/mounts.json').read_text())

    def test_depth_range_and_stereo_overlap_precede_mesh_clearance(self):
        profile=self.mount['profile'];p=np.array([[0,0,.069],[0,0,.07],[0,0,.2],[0,0,.501],[0,0,-.2],[10,0,.2]])
        np.testing.assert_array_equal(optical_visibility(p,profile),[False,True,True,False,False,False])
        self.assertFalse(profile['calibrated']);self.assertEqual((profile['width'],profile['height']),(848,480))

    def test_actual_xyzw_is_checked_against_recorded_rotation(self):
        T=pose_matrix(self.data,400);self.assertTrue(np.isfinite(T).all())
        bad={k:v.copy() for k,v in self.data.items()};bad['quaternion_world_xyzw'][400,0]=bad['quaternion_world_xyzw'][400,0][[3,0,1,2]]
        with self.assertRaisesRegex(ValueError,'XYZW'):pose_matrix(bad,400)

    def test_exact_mesh_blocks_body_ray_and_ambiguous_origin_is_unknown(self):
        screen.ROOT=ROOT/'assets';robot=StudyRobot(screen.ROOT/'robot/hexapod_mkii_length_study/urdf/f050_t060.urdf')
        fk=robot.forward(dict(zip(self.data['joint_names'],self.data['joint_position_rad'][199,0])))
        exact=screen.ExactMeshScreen(robot)
        clear,blocked,_=exact.clearance(np.array([0.,0.,.3]),np.array([[0.,0.,-.2]]),fk)
        self.assertFalse(clear[0]);self.assertTrue(blocked[0])
        box=robot.boxes['body_mock'][0];centre=box.mean(0)
        clear,_,ambiguous=exact.clearance(centre,np.array([[0.,0.,-.2]]),fk)
        self.assertFalse(clear[0]);self.assertTrue(ambiguous[0])

    def test_map_never_converts_unknown_stale_future_or_masked_cells_to_support(self):
        timeline=PoseTimeline([PoseSample(4.,np.eye(4))],clock_id='x',world_frame='world')
        cloud=calibrated_point_cloud(np.array([[.01,.01,.2],[.03,.01,.2]]),capture_times_s=np.array([4.,4.]),
            receive_time_s=4.04,now_s=4.04,clock_id='x',timeline=timeline,body_from_sensor=np.eye(4),source_sensor='synthetic',
            calibration_id='synthetic_test',valid=np.ones(2,bool),robot_mask=np.array([False,True]),robot_mask_verified=True,
            isotropic_sigma_m=.003,min_range_m=.001)
        grid=LocalHeightMap(world_frame='world');self.assertFalse(grid.channels(4.04)[...,1].any())
        grid.integrate(cloud);self.assertEqual(grid.observed.sum(),1)
        self.assertEqual(grid.channels(4.25)[...,1].sum(),1)
        self.assertEqual(grid.channels(4.25001)[...,1].sum(),0)
        self.assertEqual(grid.channels(3.99)[...,1].sum(),0)
        self.assertEqual(grid.observed.sum(),1)

    def test_footprint_outside_map_fails_and_centres_are_explicit_world_coordinates(self):
        points=ground_grid();mask,sectors,inside=footprint_mask(points,np.array([[0,0,0],[.59,0,0]]))
        self.assertTrue(mask.any());self.assertFalse(inside)
        with self.assertRaises(ValueError):footprint_mask(points,np.array([[np.nan,0,0]]))


if __name__=='__main__':unittest.main()
