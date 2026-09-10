from pathlib import Path
import math
import sys
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'isaaclab'), str(Path(__file__).parent)]
from course_queries import CourseQueries, UnresolvedRuntime
from observation_modes import TerrainObservationModes
from hexapod_terrain.support_queries import TerrainSupportQueries
from terrain_fixture_checks import load_catalog, vertical_surface_heights
from perception_replay import LocalHeightMap, WorldPoints


def pose(x=0., y=0., z=.1, yaw=math.pi/2):
    c,s=math.cos(yaw), math.sin(yaw)
    p=np.eye(4); p[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];p[:3,3]=[x,y,z]
    return p


def cloud(*, time=1., variance=1e-6, frame='odom', height=.2):
    return WorldPoints(np.array([[.01,.01,height]]),np.array([variance]),np.array([time]),
                       frame,'synthetic_fixture','explicit_test_calibration',{'synthetic':True})


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        ramp=load_catalog(ROOT/'artifacts/terrain_readiness_2026-09-09/mild_curriculum_002/terrain_catalog.json',
                          ['mild0_train_ramp_1103'])[0]
        pit=load_catalog(ROOT/'artifacts/terrain_readiness_2026-09-09/terrain_catalog.json',['train_pit_1103'])[0]
        # A deliberately missing upper triangle: inside AABB is not support.
        hole=({'id':'synthetic_half_square','family':'ramp'},Path('synthetic_not_admitted.usda'),
              np.array([[-1.,-1.,0.],[1.,-1.,0.],[-1.,1.,0.]]),np.array([[0,1,2]]))
        cls.records=[ramp,pit,hole]
        cls.oracle=TerrainSupportQueries(cls.records,device='cpu',dtype=torch.float64)

    def courses(self, fixtures=(0,0), origins=None, yaws=None, margin=.02):
        n=len(fixtures)
        return CourseQueries(self.oracle,fixture_indices=list(fixtures),
            origins_world=np.zeros((n,3)) if origins is None else origins,
            yaw_rad=np.zeros(n) if yaws is None else yaws,boundary_margin_m=margin,
            runtime=UnresolvedRuntime(None,None,None))

    def modes(self,courses=None):
        c=courses or self.courses();n=c.num_envs
        return TerrainObservationModes(c,map_origins_xy=[[-1.5,-1.5]]*n,map_sizes_xy=[[3,3]]*n,
            world_frame='odom',clock_id='sim_clock',reset_time_s=0.,registration='oracle_localization')

    def enqueue(self,m,row=0,c=None,receipt=1.04,**overrides):
        kw=dict(receive_time_s=receipt,clock_id='sim_clock',reset_epoch=m.epoch(row),provenance='synthetic_corrupted_map')
        kw.update(overrides);m.enqueue(row,c or cloud(),**kw)

    def test_translated_rotated_actual_ramp_against_independent_triangles(self):
        c=self.courses(origins=[[0,0,0],[7,-4,.8]],yaws=[0,1.1])
        local=np.array([[-1.,.1,.2],[.1,.2,.3],[.8,-.1,.4]])
        p=c.course_to_world([0,1],np.stack([local,local]));q=c.query_world([0,1],p)
        r=self.records[0];expected=vertical_surface_heights(r[2],r[3],local[:,:2])
        np.testing.assert_allclose(q.height_m[0],expected,atol=1e-10)
        np.testing.assert_allclose(q.height_m[1]-.8,expected,atol=1e-10)
        n=q.normal[0].numpy();cs,sn=math.cos(1.1),math.sin(1.1)
        np.testing.assert_allclose(q.normal[1],n@np.array([[cs,-sn,0],[sn,cs,0],[0,0,1]]).T,atol=1e-10)
        self.assertTrue(q.support_geometry.all())

    def test_missing_ineligible_and_course_edge_are_distinct(self):
        c=self.courses((1,2))
        q=c.query_world([0,1],[[[0,0,0],[2,0,0]],[[.75,.75,0],[-.5,-.5,0]]])
        self.assertTrue(q.geometry_hit[0,0]);self.assertAlmostEqual(float(q.height_m[0,0]),-.12)
        self.assertTrue(q.avoidance_hazard[0,0]);self.assertFalse(q.support_geometry[0,0])
        self.assertTrue(torch.isnan(q.height_m[0,1]));self.assertFalse(q.inside_course[0,1])
        self.assertTrue(q.inside_course[1,0]);self.assertFalse(q.geometry_hit[1,0])
        self.assertTrue(q.support_geometry[1,1])
        edge=self.courses((0,)).query_world([0],[[[-1.49,0,.1]]])
        self.assertTrue(edge.geometry_hit.item());self.assertFalse(edge.support_geometry.item())

    def test_height_clearance_preserve_invalid_required_support(self):
        c=self.courses((0,));points=np.array([[[-1.,0,.13],[-.9,.1,.13]]])
        q=c.relative_base_height([0],[[0,0,.13]],points,required_samples=[[True,True]])
        self.assertTrue(q.valid.item());self.assertAlmostEqual(q.height_m.item(),.13)
        q=c.relative_base_height([0],[[0,0,.13]],points,required_samples=[[True,True]],observation_usable=[[True,False]])
        self.assertFalse(q.valid.item());self.assertTrue(torch.isnan(q.height_m).item())
        no_required=c.relative_base_height([0],[[0,0,.13]],points,required_samples=[[False,False]])
        self.assertFalse(no_required.valid.item())
        clearance,_=c.point_clearances([0],[[[-1.,0,-.01]]]);self.assertAlmostEqual(clearance.item(),-.01)

    def test_course_reset_only_selected_rows_and_validates_before_mutation(self):
        c=self.courses();before=c.snapshot()
        c.reset_rows([1],fixture_indices=[1],origins_world=[[4,3,.2]],yaw_rad=[.5])
        after=c.snapshot()
        for k in before:self.assertTrue(torch.equal(before[k][0],after[k][0]))
        self.assertEqual(after['epochs'].tolist(),[0,1])
        with self.assertRaises(ValueError):c.reset_rows([1],fixture_indices=[200],origins_world=[[9,9,9]],yaw_rad=[0])
        for k in after:self.assertTrue(torch.equal(after[k],c.snapshot()[k]))
        # Caller/snapshot mutation cannot move a course behind the reset epoch.
        after['origins_world'].zero_();self.assertEqual(c.snapshot()['origins_world'][1,0],4.)

    def test_exact100_cell_centres_axes_rotation_and_plate_height(self):
        m=self.modes();a=m.packet(0,'ideal_teacher',pose(),now_s=0.)
        self.assertEqual(a.channels.shape,(100,100,4))
        np.testing.assert_allclose(a.sample_world_xy[0,0],[-.99,-.99],atol=1e-12)
        np.testing.assert_allclose(a.sample_world_xy[-1,-1],[.99,.99],atol=1e-12)
        np.testing.assert_allclose(a.sample_world_xy[51,50]-a.sample_world_xy[50,50],[.02,0],atol=1e-12)
        b=m.packet(0,'ideal_teacher',pose(x=2,y=-3,yaw=0),now_s=0.)
        np.testing.assert_allclose(b.sample_world_xy[51,50]-b.sample_world_xy[50,50],[0,-.02],atol=1e-12)
        np.testing.assert_allclose(b.sample_world_xy[50,51]-b.sample_world_xy[50,50],[.02,0],atol=1e-12)
        # First row of the ramp is flat and relative to the plate, not world Z.
        self.assertAlmostEqual(float(a.channels[0,50,0]),-.5,places=6)

    def test_timing_receipt_and_exact250ms_lease(self):
        m=self.modes();self.enqueue(m)
        self.assertEqual(m.packet(0,'corrupted_map',pose(),now_s=1.03).channels[...,1].sum(),0)
        a=m.packet(0,'corrupted_map',pose(),now_s=1.04)
        self.assertEqual(a.channels[...,1].sum(),1);self.assertAlmostEqual(float(a.channels[50,50,0]),.5)
        self.assertEqual(m.packet(0,'corrupted_map',pose(),now_s=1.25).channels[50,50,1],1)
        stale=m.packet(0,'corrupted_map',pose(),now_s=1.250001)
        np.testing.assert_array_equal(stale.channels[50,50],[0,0,1,1])
        self.assertIsNone(stale.geometry_truth)

    def test_map_matches_existing_oracle_and_queued_input_is_owned(self):
        m=self.modes();p=cloud();self.enqueue(m,c=p)
        oracle=LocalHeightMap(origin_xy=(-1.5,-1.5),size_xy=(3,3),resolution_m=.02,world_frame='odom');oracle.integrate(p)
        p.points_m[:]=999
        actual=m.packet(0,'corrupted_map',pose(),now_s=1.10)
        expected=oracle.local_patch(pose(),now_s=1.10,extent_m=2,resolution_m=.02)
        np.testing.assert_array_equal(actual.channels,expected['channels'])
        self.assertEqual(actual.provenance,('synthetic_corrupted_map',))

    def test_blind_no_truth_leak_and_pit_observed_is_not_eligible(self):
        m=self.modes(self.courses((1,)));self.enqueue(m)
        b=m.packet(0,'blind',pose(),now_s=1.04)
        self.assertIsNone(b.geometry_truth);self.assertEqual(b.provenance,())
        self.assertEqual(b.channels[...,:2].sum(),0);self.assertTrue((b.channels[...,2:]==1).all())
        ideal=m.packet(0,'ideal_teacher',pose(),now_s=1.04)
        self.assertEqual(ideal.channels[50,50,1],1)
        self.assertFalse(ideal.geometry_truth.support_geometry.reshape(100,100)[50,50])

    def test_selective_reset_isolates_delayed_packets_maps_and_clock(self):
        m=self.modes();self.enqueue(m,row=0);self.enqueue(m,row=1)
        m.packet(0,'corrupted_map',pose(),now_s=1.04);before=m.packet(1,'corrupted_map',pose(),now_s=1.04)
        old_epoch=m.epoch(0);self.enqueue(m,row=0,c=cloud(time=1.06),receipt=1.2)
        m.reset_rows([0],now_s=1.1,map_origins_xy=[[-1.5,-1.5]],map_sizes_xy=[[3,3]])
        with self.assertRaises(ValueError):self.enqueue(m,row=0,c=cloud(time=1.15),receipt=1.2,reset_epoch=old_epoch)
        with self.assertRaises(ValueError):self.enqueue(m,row=0,c=cloud(time=1.),receipt=1.2)
        self.assertEqual(m.packet(0,'corrupted_map',pose(),now_s=1.2).channels[...,1].sum(),0)
        after=m.packet(1,'corrupted_map',pose(),now_s=1.2)
        self.assertEqual(after.channels[50,50,1],1);self.assertEqual(after.reset_epoch,before.reset_epoch)
        self.assertAlmostEqual(float(after.channels[50,50,0]),float(before.channels[50,50,0]))

    def test_course_reset_cannot_reuse_prior_map_even_at_same_position(self):
        c=self.courses();m=self.modes(c)
        c.reset_rows([0],fixture_indices=[0],origins_world=[[0,0,0]],yaw_rad=[0])
        with self.assertRaisesRegex(ValueError,'Course reset'):m.packet(0,'ideal_teacher',pose(),now_s=1.)
        m.reset_rows([0],now_s=1.,map_origins_xy=[[-1.5,-1.5]],map_sizes_xy=[[3,3]])
        self.assertEqual(m.packet(0,'corrupted_map',pose(),now_s=1.).channels[...,1].sum(),0)
        self.assertEqual(m.packet(1,'blind',pose(),now_s=0.).reset_epoch,0)

    def test_rejects_future_uncertainty_wrong_frame_clock_and_clock_rewind(self):
        m=self.modes()
        for c in [cloud(time=2.),cloud(variance=-1),cloud(frame='map')]:
            with self.subTest(c=c),self.assertRaises(ValueError):self.enqueue(m,c=c)
        with self.assertRaises(ValueError):self.enqueue(m,clock_id='different')
        m.packet(0,'blind',pose(),now_s=2.)
        with self.assertRaises(ValueError):m.packet(0,'blind',pose(),now_s=1.)
        with self.assertRaises(ValueError):self.enqueue(m,receipt=1.5)
        uncertain=self.modes();self.enqueue(uncertain,c=cloud(variance=.016**2))
        self.assertEqual(uncertain.packet(0,'corrupted_map',pose(),now_s=1.04).channels[...,1].sum(),0)

    def test_explicit_unresolved_runtime_and_no_admission_from_hashes(self):
        self.assertFalse(self.courses().runtime.receipt()['runtime_admitted'])
        self.assertIsNone(self.courses().runtime.receipt()['actor_sha256'])
        with self.assertRaises(TypeError):UnresolvedRuntime()
        with self.assertRaises(ValueError):UnresolvedRuntime('guess',None,None)
        self.assertFalse(UnresolvedRuntime('a'*64,'b'*64,'c'*64).receipt()['runtime_admitted'])

    def test_old_capture_delivered_later_does_not_replace_newer_cell(self):
        m=self.modes()
        self.enqueue(m,c=cloud(time=1.05,height=.3),receipt=1.10)
        self.enqueue(m,c=cloud(time=1.,height=.9),receipt=1.15)
        a=m.packet(0,'corrupted_map',pose(),now_s=1.10)
        b=m.packet(0,'corrupted_map',pose(),now_s=1.15)
        self.assertEqual(a.channels[50,50,0],b.channels[50,50,0])
        self.assertAlmostEqual(float(b.channels[50,50,2]),.4,places=6)
        # Mutating an exported packet cannot alter its map's next output.
        b.channels[:]=0
        self.assertEqual(m.packet(0,'corrupted_map',pose(),now_s=1.15).channels[50,50,1],1)


if __name__=='__main__':unittest.main()
