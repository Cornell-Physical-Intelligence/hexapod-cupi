"""Geometry edge cases independent of CAD, CUDA and Isaac Sim."""
import math
import importlib.util
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from sensor_mount_math import (apply_transform, axis_rotation, look_outward,
                               nominal_profile, optical_visibility,
                               replay_visibility_contract, segments_hit_boxes, transform)


class SensorMountMathTests(unittest.TestCase):
    def test_parallel_segment_inside_and_beyond_target(self):
        box=np.array([[[1.,-1.,-1.],[2.,1.,1.]]])
        result=segments_hit_boxes(np.zeros(3),np.array([[3,0,0],[.5,0,0],[0,3,0],[3,4,0]]),box)
        np.testing.assert_array_equal(result,[True,False,False,False])
        self.assertTrue(segments_hit_boxes(np.array([1.5,0,0]),np.array([[1.5,3,0]]),box)[0])

    def test_optical_depth_not_radial_distance_and_stereo_band(self):
        p=nominal_profile('D455',640,360,.26)
        # Radial range exceeds min-Z, but optical depth does not.
        self.assertFalse(optical_visibility(np.array([[.15,0,.25]]),p)[0])
        z=.4
        left_x=-p['ppx']*z/p['fx']+.001
        self.assertFalse(optical_visibility(np.array([[left_x,0,z]]),p)[0])
        self.assertTrue(optical_visibility(np.array([[0,0,z]]),p)[0])
        self.assertFalse(optical_visibility(np.array([[0,0,-z]]),p)[0])

    def test_resolution_minimum_depth(self):
        point=np.array([[0,0,.30]])
        self.assertFalse(optical_visibility(point,nominal_profile('D455',1280,720,.52))[0])
        self.assertTrue(optical_visibility(point,nominal_profile('D455',640,360,.26))[0])

    def test_camera_and_robot_frame_transform(self):
        r=look_outward([0,-.2,.07],45)
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(r),1.)
        self.assertLess(r[1,2],0.)
        self.assertLess(r[2,2],0.)
        self.assertLess(r[0,0],0.) # image right is body right (-X) facing forward.
        t=transform([.2,-.1,.12],[.1,-.2,.3])
        p=np.array([[.2,.3,.4]])
        np.testing.assert_allclose((apply_transform(p,t)-t[:3,3])@t[:3,:3],p,atol=1e-12)

    def test_joint_axis_rotation(self):
        r=axis_rotation([0,0,-1],math.pi/2)
        np.testing.assert_allclose(apply_transform([1,0,0],r),[0,-1,0],atol=1e-12)

    def test_visibility_age_fixture_fails_closed(self):
        visible=np.ones(24,dtype=bool);visible[0]=False
        replay=replay_visibility_contract(visible)
        self.assertEqual(replay['fresh']['usable_count'],23)
        self.assertFalse(replay['fresh']['required_sector_all_samples_observed'][0])
        self.assertTrue(replay['fresh']['required_sector_all_samples_observed'][1])
        for name in ('all_streams_stale','no_observation','future_timestamp'):
            self.assertEqual(replay[name]['usable_count'],0)
            self.assertFalse(any(replay[name]['required_sector_all_samples_observed']))
        self.assertFalse(replay['fresh']['unseen_marked_usable'])

    @unittest.skipUnless(importlib.util.find_spec('trimesh') and importlib.util.find_spec('rtree'),'optional exact-ray dependencies')
    def test_exact_triangle_hit_and_segment_end(self):
        import trimesh
        from screen_sensor_mounts import ExactMeshScreen
        path=Path('test_box.stl')
        box=np.array([[[1.,-1.,-1.],[2.,1.,1.]]])
        robot=SimpleNamespace(boxes={'body':box},parts={'body':[(path,np.eye(4))]})
        exact=ExactMeshScreen(robot)
        mesh=trimesh.creation.box(extents=[1,2,2]);mesh.apply_translation([1.5,0,0])
        exact.meshes[path]=mesh
        clear,hit,ambiguous=exact.clearance(np.zeros(3),np.array([[3,0,0],[.5,0,0],[0,3,0]]),{'body':np.eye(4)})
        np.testing.assert_array_equal(clear,[False,True,True])
        np.testing.assert_array_equal(hit,[True,False,False])
        self.assertFalse(ambiguous.any())


if __name__=='__main__':
    unittest.main()
