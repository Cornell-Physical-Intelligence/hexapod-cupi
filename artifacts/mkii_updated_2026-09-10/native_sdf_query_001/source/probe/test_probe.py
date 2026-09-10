import ast
import copy
import hashlib
import json
from pathlib import Path
import unittest
import tempfile
import numpy as np
import trimesh
from mesh_oracle import oracle,topology,ray_distance,frame_prediction
from native_probe import query_copy,query_pair
from score_probe import score,hypotheses

HERE=Path(__file__).resolve().parent


class OracleChecks(unittest.TestCase):
    def test_box_analytic_inside_outside_edge_and_scale(self):
        mesh=trimesh.creation.box(extents=[2,4,6])
        points=np.array([[1.2,.4,.3],[0,.2,.3],[-1.3,2.4,0],[.5,2.2,3.3],[.8,1.8,2.9]])
        for scale in (.001,1,1000):
            rows=oracle(points*scale,mesh.vertices*scale,mesh.faces)
            z=np.abs(points)-[1,2,3];expected=np.linalg.norm(np.maximum(z,0),axis=1)+np.minimum(np.max(z,axis=1),0)
            np.testing.assert_allclose([r['distance_m']for r in rows],expected*scale,atol=1e-9*scale)
            self.assertTrue(topology(mesh.vertices*scale,mesh.faces)['closed'])

    def test_rotated_translated_oracle_gradients(self):
        mesh=trimesh.creation.box(extents=[2,4,6]);a=.7
        R=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
        t=np.array([4.,-6.,2.]);p=np.array([[1.2,0,.2],[-1.2,0,.2],[0,2.2,.2]])
        original=oracle(p,mesh.vertices,mesh.faces);moved=oracle(p@R.T+t,mesh.vertices@R.T+t,mesh.faces)
        np.testing.assert_allclose([r['distance_m']for r in original],[r['distance_m']for r in moved],atol=1e-12)
        np.testing.assert_allclose(np.array([r['gradient']for r in original])@R.T,[r['gradient']for r in moved],atol=1e-12)
        T=np.eye(4);T[:3,:3]=R;T[:3,3]=t
        np.testing.assert_allclose(frame_prediction(p@R.T+t,mesh.vertices,mesh.faces,T),np.c_[[r['distance_m']for r in moved],[r['gradient']for r in moved]],atol=1e-12)

    def test_hole_is_not_filled_solid(self):
        mesh=trimesh.creation.annulus(r_min=.5,r_max=1,height=2,sections=128)
        info=topology(mesh.vertices,mesh.faces);self.assertTrue(info['closed']);self.assertTrue(info['consistently_oriented'])
        rows=oracle(np.array([[0.,0,0],[.75,0,.1]]),mesh.vertices,mesh.faces)
        self.assertGreater(rows[0]['distance_m'],.499);self.assertLess(rows[1]['distance_m'],-.249)
        self.assertAlmostEqual(ray_distance([0,0,0],[1,0,0],mesh.vertices,mesh.faces),.5,places=10)

    def test_open_or_inverted_topology_not_admitted(self):
        mesh=trimesh.creation.box()
        self.assertFalse(topology(mesh.vertices,mesh.faces[:-1])['closed'])
        self.assertLess(topology(mesh.vertices,mesh.faces[:,::-1])['signed_volume_m3'],0)

    def test_actual_tibia_closest_points_crosscheck(self):
        with np.load(HERE/'fixture/mesh.npz')as z:v=z['vertices'];f=z['faces']
        fixture=json.loads((HERE/'fixture/fixture.json').read_text());p=np.array(fixture['points'])
        own=oracle(p,v,f)
        # Trimesh's absolute tol.zero=1e-13 is also compared to fourth-order
        # edge-region determinants. Metre-sized sub-mm triangles fall below it.
        # Use float64 millimetres for this independent algorithm, then convert back.
        m=trimesh.Trimesh(v.astype(np.float64)*1000,f,process=False)
        _,dist,_=trimesh.proximity.closest_point_naive(m,p*1000)
        np.testing.assert_allclose(np.abs([r['distance_m']for r in own]),dist/1000,rtol=0,atol=2e-12)
        self.assertTrue(topology(v,f)['closed'])


class QueryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=json.loads((HERE/'fixture/fixture.json').read_text())
        cls.expected=np.c_[cls.fixture['distance_m'],cls.fixture['gradient']]
        cls.raw=np.tile(cls.expected[None],(6,1,1))
        cls.world=np.ones((6,len(cls.fixture['semantic_anchor_indices']),4))*.2
        with np.load(HERE/'fixture/mesh.npz')as z:v=z['vertices'];f=z['faces']
        points=np.asarray(cls.fixture['points'])[cls.fixture['semantic_anchor_indices']]
        cls.link=np.asarray([frame_prediction(points,v,f,row['shape_to_link'])for row in cls.fixture['shapes']])

    def evaluate(self,raw=None,fixture=None,repeat=None):
        raw=self.raw.copy()if raw is None else raw
        return score(raw,raw.copy()if repeat is None else repeat,self.fixture if fixture is None else fixture,self.world,self.link)

    def test_identity_frame_is_not_claimed_separable(self):
        anchors=self.fixture['semantic_anchor_indices']
        r=score(self.raw,self.raw,self.fixture,self.world,self.raw[:,anchors])
        self.assertFalse(r['declared_semantics_supported'])
        self.assertFalse(r['frame_comparison']['alternatives']['link']['identifiable_against_shape_frame'])

    def test_wrong_link_frame_output_cannot_pass_shape_semantics(self):
        raw=self.raw.copy();raw[:,self.fixture['semantic_anchor_indices']]=self.link
        self.assertFalse(self.evaluate(raw)['declared_semantics_supported'])

    def test_ideal_semantics_and_geometry(self):
        r=self.evaluate();self.assertTrue(r['declared_semantics_supported']);self.assertTrue(r['geometry_accuracy_within_proposed_bounds'])
        self.assertLess(r['anchor_normal_condition'],2)
        self.assertFalse(r['standing_admitted']);self.assertFalse(r['contact_admitted'])

    def test_channel_permutation_is_reported_not_adopted(self):
        r=self.evaluate(self.raw[:,:,[1,2,3,0]])
        self.assertFalse(r['declared_semantics_supported'])
        self.assertEqual(r['best_alternatives'][0]['distance_channel'],3)
        self.assertEqual(r['best_alternatives'][0]['gradient_channels'],[0,1,2])

    def test_sign_and_units_fail_declared_hypothesis(self):
        for raw in (-self.raw,self.raw*np.array([1000,1,1,1]),self.raw*np.array([.001,1,1,1])):
            self.assertFalse(self.evaluate(raw)['declared_semantics_supported'])

    def test_degenerate_anchor_normals_and_missing_inside_fail(self):
        for mode in ('normals','sign'):
            f=copy.deepcopy(self.fixture)
            if mode=='normals':f['gradient']=np.tile([1.,0,0],(len(f['points']),1)).tolist()
            else:f['distance_m']=np.abs(f['distance_m']).tolist()
            with self.assertRaises(ValueError):self.evaluate(fixture=f)

    def test_native_reused_buffer_is_copied(self):
        class View:
            def __init__(self):self.buffer=np.zeros((1,2,4))
            def get_sdf_and_gradients(self,data):self.buffer[:]=data;return self.buffer
        view=View();first=query_copy(view,1);second=query_copy(view,2)
        self.assertTrue(np.all(first==1));self.assertTrue(np.all(second==2));self.assertFalse(np.shares_memory(first,view.buffer))

    def test_second_query_failure_preserves_first_and_nonfinite_raw(self):
        class View:
            calls=0
            def get_sdf_and_gradients(self,data):
                self.calls+=1
                if self.calls==2:raise RuntimeError('injected second call failure')
                return np.ones((1,2,4))
        with tempfile.TemporaryDirectory()as d:
            path=Path(d)/'query.npz';view=View()
            with self.assertRaisesRegex(RuntimeError,'second call'):query_pair(view,np.zeros((2,3)),np.arange(2),lambda x:x,path)
            with np.load(path)as raw:self.assertTrue(np.all(raw['raw']==1));self.assertNotIn('repeat_raw',raw)
            self.assertEqual(json.loads(path.with_suffix('.json').read_text())['completed_calls'],1)
        class Nonfinite:
            def get_sdf_and_gradients(self,data):return np.full((1,2,4),np.nan)
        with tempfile.TemporaryDirectory()as d:
            path=Path(d)/'query.npz'
            with self.assertRaises(ValueError):query_pair(Nonfinite(),np.zeros((2,3)),np.arange(2),lambda x:x,path)
            with np.load(path)as raw:self.assertTrue(np.isnan(raw['raw']).all())

    def test_reordered_cached_return_and_nonfinite_rejected(self):
        repeat=self.raw[:,::-1].copy();self.assertFalse(self.evaluate(repeat=repeat)['semantics_checks']['repeat_identical'])
        raw=self.raw.copy();raw[0,0,0]=np.nan
        with self.assertRaises(ValueError):self.evaluate(raw)

    def test_cavity_fill_fails_geometry_without_relabeling_support(self):
        raw=self.raw.copy();indices=np.flatnonzero(np.array(self.fixture['regions'])=='void_between_surfaces');self.assertGreater(len(indices),0)
        raw[:,indices,0]=-.005
        r=self.evaluate(raw);self.assertFalse(r['geometry_accuracy_within_proposed_bounds']);self.assertGreater(r['sign_errors_beyond_3_nominal_cells'],0)

    def test_finite_difference_gradient_mismatch_is_separate(self):
        raw=self.raw.copy();i=self.fixture['derivative_stencils'][0]['pairs'][0][0];raw[:,i,0]+=.001
        r=self.evaluate(raw);self.assertFalse(r['semantics_checks']['gradient_distance_fd'])

    def test_addon_contains_no_simulation_or_state_mutation_calls(self):
        tree=ast.parse((HERE/'native_probe.py').read_text())
        calls=[n.func.attr for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)]
        self.assertFalse(set(calls)&{'step','reset','play','update','simulate','update_simulation','Set','AddReference','DefinePrim'})
        self.assertFalse(any(c.startswith(('set_joint','set_dof','write_joint','set_root','write_root'))for c in calls))


if __name__=='__main__':unittest.main()
