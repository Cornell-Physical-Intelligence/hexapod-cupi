"""Perception-frame, timing, map, uncertainty and negative-obstacle regression."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from experiments.terrain.tools.perception_replay import CameraCalibration, DepthFrame, LocalHeightMap, PoseSample, PoseTimeline, WorldPoints, calibrated_point_cloud, scene_height, synthetic_depth, transform_timed_points, unproject_depth, validate_transform
from experiments.terrain.tools.sensor_mount_math import transform


class PerceptionReplayTests(unittest.TestCase):
    def camera(self,extrinsic=None,**kwargs):
        return CameraCalibration('depth_optical','synthetic_test_calibration',3,3,2.,2.,1.,1.,.07,2.,
            np.eye(4) if extrinsic is None else extrinsic,calibrated=True,**kwargs)

    def timeline(self):
        return PoseTimeline([PoseSample(1.,np.eye(4)),PoseSample(1.1,np.eye(4))])

    def frame(self,depth,**kwargs):
        return DepthFrame('depth_optical','replay_clock',1.,1.02,np.asarray(depth),
            np.ones((3,3),dtype=bool),np.zeros((3,3),dtype=bool),robot_mask_verified=True,**kwargs)

    def cloud(self,points,times=1.,variance=1e-6):
        p=np.asarray(points,dtype=float)
        return WorldPoints(p,np.broadcast_to(variance,(len(p),)),np.broadcast_to(times,(len(p),)),
                           'odom','synthetic','test',{})

    def test_calibrated_optical_projection_and_extrinsic(self):
        ext=transform([.1,.2,.3],[0,0,np.pi/2])
        cloud=unproject_depth(self.frame(np.full((3,3),.5)),self.camera(ext),self.timeline(),now_s=1.05)
        np.testing.assert_allclose(cloud.points_m[4],[.1,.2,.8],atol=1e-12)
        # Right image ray becomes +bodyY after the known90degree transform.
        np.testing.assert_allclose(cloud.points_m[5],[.1,.45,.8],atol=1e-12)

    def test_clipping_masks_and_robot_depth_occlusion(self):
        depth=np.array([[0.,-.4,np.nan],[.05,.5,3.],[.5,.5,.5]])
        robot_depth=np.full((3,3),np.inf);robot_depth[1,1]=.3;robot_depth[2,0]=np.nan
        frame=self.frame(depth,robot_depth_m=robot_depth);frame.robot_mask[2,1]=True
        cloud=unproject_depth(frame,self.camera(),self.timeline(),now_s=1.05)
        self.assertEqual(len(cloud.points_m),1)
        np.testing.assert_allclose(cloud.points_m[0],[.25,.25,.5])

    def test_motion_compensation_uses_capture_not_receipt(self):
        timeline=PoseTimeline([PoseSample(1.,transform([0,0,0])),PoseSample(1.1,transform([.1,0,0]))])
        points=np.array([[.95,0,.5],[.925,0,.5]])
        times=np.array([1.05,1.075])
        cloud=transform_timed_points(points,np.repeat(np.eye(3)[None]*1e-6,2,axis=0),times,timeline,np.eye(4),
            now_s=1.1,receive_time_s=1.1,clock_id='replay_clock',source_sensor='cloud',calibration_id='synthetic')
        np.testing.assert_allclose(cloud.points_m,[[1,0,.5],[1,0,.5]],atol=1e-12)

    def test_pose_rotation_interpolation_and_no_extrapolation(self):
        timeline=PoseTimeline([PoseSample(1.,np.eye(4)),PoseSample(1.1,transform(rpy=[0,0,np.pi/2]))])
        pose=timeline.at(1.05)
        np.testing.assert_allclose(pose.world_from_body[:3,0],[2**-.5,2**-.5,0],atol=1e-12)
        with self.assertRaises(ValueError):timeline.at(1.11)
        long_gap=PoseTimeline([PoseSample(1.,np.eye(4)),PoseSample(2.,np.eye(4))])
        with self.assertRaises(ValueError):long_gap.at(1.5)

    def test_stale_future_and_wrong_clock_fail_closed(self):
        for now in (1.30,.99):
            frame=self.frame(np.full((3,3),.5))
            if now<frame.receive_time_s:
                with self.assertRaises(ValueError):unproject_depth(frame,self.camera(),self.timeline(),now_s=now)
            else:
                self.assertEqual(len(unproject_depth(frame,self.camera(),self.timeline(),now_s=now).points_m),0)
        frame=self.frame(np.full((3,3),.5));frame.capture_time_s=1.08
        self.assertEqual(len(unproject_depth(frame,self.camera(),self.timeline(),now_s=1.1).points_m),0)
        frame=self.frame(np.full((3,3),.5));frame.clock_id='unmapped_usb_clock'
        with self.assertRaises(ValueError):unproject_depth(frame,self.camera(),self.timeline(),now_s=1.05)

    def test_row_acquisition_times_and_missing_pose(self):
        frame=self.frame(np.full((3,3),.5),row_time_offsets_s=np.array([0,.05,.20]));frame.receive_time_s=1.22
        cloud=unproject_depth(frame,self.camera(),self.timeline(),now_s=1.22)
        self.assertEqual(len(cloud.points_m),6)
        self.assertEqual(cloud.diagnostics['no_pose_points'],3)
        self.assertEqual(sorted(set(cloud.capture_time_s)),[1.,1.05])

    def test_lidar_style_point_input_robot_mask_and_point_times(self):
        cloud=calibrated_point_cloud([[.2,0,0],[.3,0,0],[.05,0,0]],capture_times_s=[1.,1.,1.],
            receive_time_s=1.02,now_s=1.05,clock_id='replay_clock',timeline=self.timeline(),body_from_sensor=np.eye(4),
            source_sensor='lidar_frame',calibration_id='synthetic_extrinsic',valid=np.ones(3,dtype=bool),
            robot_mask=np.array([False,True,False]),robot_mask_verified=True)
        np.testing.assert_allclose(cloud.points_m,[[.2,0,0]])

    def test_moving_robot_mask_is_recomputed_for_each_frame(self):
        first=self.frame(np.full((3,3),.5));first.robot_mask[1,1]=True
        second=self.frame(np.full((3,3),.5));second.capture_time_s=1.1;second.receive_time_s=1.12
        second.robot_mask[1,2]=True
        a=unproject_depth(first,self.camera(),self.timeline(),now_s=1.05)
        b=unproject_depth(second,self.camera(),self.timeline(),now_s=1.15)
        self.assertFalse(np.any(np.all(np.isclose(a.points_m,[0,0,.5]),axis=1)))
        self.assertTrue(np.any(np.all(np.isclose(b.points_m,[0,0,.5]),axis=1)))
        self.assertFalse(np.any(np.all(np.isclose(b.points_m,[.25,0,.5]),axis=1)))

    def test_step_and_pit_are_preserved_and_never_interpolated(self):
        model=LocalHeightMap(origin_xy=(0,0),size_xy=(.1,.1),resolution_m=.02)
        model.integrate(self.cloud([[.01,.01,0],[.03,.01,.02],[.07,.01,-.08]]))
        fresh=model.channels(1.1)
        self.assertAlmostEqual(model.height[1,0],.02)
        self.assertAlmostEqual(model.height[3,0],-.08)
        self.assertEqual(fresh[2,0,1],0.)
        self.assertFalse(model.observed[2,0])
        self.assertEqual(model.channels(1.3)[...,1].sum(),0)
        # A later visible negative obstacle replaces previous floor evidence.
        model.integrate(self.cloud([[.01,.01,-.08]],times=1.1))
        self.assertAlmostEqual(model.height[0,0],-.08)
        model.integrate(self.cloud([[.01,.01,0]],times=1.))
        self.assertAlmostEqual(model.height[0,0],-.08)
        self.assertEqual(model.capture[0,0],1.1)

    def test_vertical_edges_and_uncertain_pose_are_not_confident_support(self):
        model=LocalHeightMap(origin_xy=(0,0),size_xy=(.1,.1),resolution_m=.02)
        model.integrate(self.cloud([[.01,.01,-.08],[.011,.01,0.]]))
        self.assertEqual(model.channels(1.1)[0,0,1],0.)
        timeline=PoseTimeline([PoseSample(1.,np.eye(4),.02,.01)])
        cloud=unproject_depth(self.frame(np.full((3,3),.5)),self.camera(),timeline,now_s=1.05)
        self.assertTrue((cloud.variance_z_m2>=.02**2).all())

    def test_simultaneous_camera_conflicts_remain_uncertain(self):
        model=LocalHeightMap(origin_xy=(0,0),size_xy=(.1,.1),resolution_m=.02)
        model.integrate(self.cloud([[.01,.01,-.08]]))
        model.integrate(self.cloud([[.01,.01,0.]]))
        self.assertEqual(model.channels(1.1)[0,0,1],0.)
        model.integrate(self.cloud([[.01,.01,0.]]))
        self.assertEqual(model.channels(1.1)[0,0,1],0.)

    def test_patch_axes_and_height_are_explicit(self):
        model=LocalHeightMap(origin_xy=(-.1,-.1),size_xy=(.2,.2),resolution_m=.02)
        # Bodyforward is-worldY, left is+worldX; patch row increases forward.
        patch=model.local_patch(transform([0,0,.12]),now_s=1.,extent_m=.08,resolution_m=.02)
        self.assertLess(patch['world_xy'][3,1,1],patch['world_xy'][0,1,1])
        self.assertGreater(patch['world_xy'][1,3,0],patch['world_xy'][1,0,0])
        self.assertFalse(patch['channels'][...,1].any())

    def test_synthetic_scene_has_true_negative_height(self):
        np.testing.assert_allclose(scene_height(np.array([.25,-.25,0]),np.zeros(3)),[.02,-.08,0])
        ext=transform([-.25,0,.4],[np.pi,0,0])
        calibration=self.camera(ext)
        depth=synthetic_depth(calibration,np.eye(4))
        self.assertAlmostEqual(depth[1,1],.48)

    def test_bad_calibration_and_missing_mask_are_rejected(self):
        with self.assertRaises(ValueError):validate_transform(np.diag([1,1,-1,1]))
        calibration=self.camera();frame=self.frame(np.full((3,3),.5));frame.robot_mask_verified=False
        with self.assertRaises(ValueError):unproject_depth(frame,calibration,self.timeline(),now_s=1.05)


if __name__=='__main__':
    unittest.main()
