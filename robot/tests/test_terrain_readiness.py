import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from experiments.terrain.tools.terrain_readiness import body_to_navigation, optical_rotation_body, terrain_channels, support_region_observed, terrain_mesh, validate_mesh, mesh_usda


class TerrainReadinessTests(unittest.TestCase):
    def test_frame_adapter_and_optical_axes(self):
        np.testing.assert_allclose(body_to_navigation([[0,-1,0],[1,0,0],[0,0,1]]),np.eye(3))
        for radial in ([0,-1],[1,0],[-1,1]):
            r=optical_rotation_body(radial,45)
            np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(r),1)
            self.assertLess(r[2,2],0)
        self.assertLess(optical_rotation_body([0,-1],45)[0,0],0)

    def test_unknown_stale_uncertain_and_future_cells_fail_closed(self):
        h=np.zeros((2,3));var=np.full((2,3),1e-6);seen=np.ones((2,3),dtype=bool)
        times=np.full((2,3),9.9)
        seen[0,1]=False;h[0,2]=np.nan;times[1,0]=9.;times[1,1]=11.;var[1,2]=1.
        out=terrain_channels(h,var,seen,times,now_s=10.)
        np.testing.assert_array_equal(out[...,1],[[1,0,0],[0,0,0]])
        self.assertTrue(np.isfinite(out).all())
        self.assertFalse(support_region_observed(out,np.ones((2,3),dtype=bool)))
        required=np.zeros((2,3),dtype=bool);required[0,0]=True
        self.assertTrue(support_region_observed(out,required))
        self.assertFalse(support_region_observed(out,np.zeros((2,3),dtype=bool)))

    def test_clock_and_mask_contract_rejected(self):
        a=np.zeros((2,2))
        with self.assertRaises(ValueError):terrain_channels(a,a,a,a,now_s=0)
        with self.assertRaises(ValueError):terrain_channels(a,a,a.astype(bool),a,now_s=float('nan'))

    def test_fixture_reproducibility_and_mesh_integrity(self):
        for family in ('smooth_rough','ramp','step','ridge','pit'):
            v,f,m=terrain_mesh(family,1103)
            v2,f2,m2=terrain_mesh(family,1103)
            np.testing.assert_array_equal(v,v2);np.testing.assert_array_equal(f,f2)
            self.assertEqual(m,m2)
            self.assertGreater(validate_mesh(v,f)['triangles'],0)
            text=mesh_usda(v,f)
            self.assertIn('physics:approximation = "none"',text)
            self.assertNotIn('RigidBodyAPI',text)

    def test_pit_is_not_filled_by_global_floor(self):
        v,f,m=terrain_mesh('pit',7103)
        # Independently intersect a vertical ray through the hole with all triangles.
        x,y=.013,.027;hits=[]
        for tri in v[f]:
            a,b,c=tri
            matrix=np.column_stack((b[:2]-a[:2],c[:2]-a[:2]))
            if abs(np.linalg.det(matrix))<1e-12:continue
            u,w=np.linalg.solve(matrix,np.array([x,y])-a[:2])
            if u>=-1e-9 and w>=-1e-9 and u+w<=1+1e-9:
                hits.append(a[2]+u*(b[2]-a[2])+w*(c[2]-a[2]))
        self.assertAlmostEqual(max(hits),-m['pit_depth_m'])
        self.assertEqual(m['curriculum'],'avoidance_only')

    def test_step_has_vertical_faces_and_flat_start(self):
        v,f,m=terrain_mesh('step',8209)
        tri=v[f];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        self.assertTrue((np.abs(normal[:,0])>0).any())
        self.assertAlmostEqual(v[v[:,0]<-.65,2].max(),0)
        self.assertAlmostEqual(v[:,2].max(),m['step_height_m'])


if __name__=='__main__':unittest.main()
