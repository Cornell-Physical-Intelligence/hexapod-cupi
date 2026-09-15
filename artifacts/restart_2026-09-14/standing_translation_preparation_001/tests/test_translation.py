"""Independent CPU fixtures; these records are not simulator evidence."""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'source'), str(ROOT/'analysis')]
import placement_contract as placement
import standing_contract as contract
from readout import detect


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


host = load('translation_host_fixture', ROOT/'host/launch_standing_spark.py')


def declared(label='xy14_4'):
    xy = [14.,4.] if label == 'xy14_4' else [0.,0.]
    return {'schema':'canonical_single_placement_diagnostic_v1','placement':label,
            'root':'/Robot','xy_m':xy,'reset_xyz_m':xy+[0.08161109101311194],
            'scope':'Single-robot placement diagnosis only; no batch or training admission'}


def authored(label='xy14_4'):
    xy = [14.,4.] if label == 'xy14_4' else [0.,0.]
    return {'base_matrix':[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],
                           [0.,0.,0.08161109101311204,1.]],
            'xform_op_order':['xformOp:transform','xformOp:translate:diagnosticPlacement'],
            'translation_m':xy+[0.],
            'computed_world_matrix':[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],
                                     xy+[0.08161109101311204,1.]]}


def reset(label='xy14_4'):
    xy = [14.,4.] if label == 'xy14_4' else [0.,0.]
    pose = [xy+[0.08161108940839767,0.,0.,0.,1.]]
    return {'placement':declared(label),'requested_root':pose,
            'placement_readback_source':'physics_articulation_view.get_root_transforms',
            'post_reset':{'root_pose_xyzw':copy.deepcopy(pose)}}


def write(root, name, value):
    (root/name).write_text(json.dumps(value, indent=2)+'\n')


class PlacementFixtures(unittest.TestCase):
    def test_independent_declaration_and_shared_reset(self):
        for label in ('origin','xy14_4'):
            self.assertEqual(placement.declaration(label), declared(label))
            placement.validate_authored(authored(label), declared(label))
            placement.validate_reset(reset(label), declared(label))
            self.assertEqual(placement.requested_root(label,0.08161109101311194),
                             reset(label)['requested_root'])
        with self.assertRaises(ValueError): placement.declaration('env23')

    def test_reset_mismatch_drift_and_usd_readback_rejected(self):
        bad = reset(); bad['requested_root'] = reset('origin')['requested_root']
        with self.assertRaises(ValueError): placement.validate_reset(bad,declared())
        bad = reset(); bad['post_reset']['root_pose_xyzw'][0][0] += .0001
        with self.assertRaises(ValueError): placement.validate_reset(bad,declared())
        bad = reset(); bad['placement_readback_source'] = 'USD attributes'
        with self.assertRaises(ValueError): placement.validate_reset(bad,declared())
        bad = authored(); bad['computed_world_matrix'][3][0] = 28.
        with self.assertRaises(ValueError): placement.validate_authored(bad,declared())

    def test_both_arms_same_transform_ops_and_only_xy_differs(self):
        origin = reset('origin'); translated = reset()
        self.assertEqual(origin['requested_root'][0][2:],translated['requested_root'][0][2:])
        self.assertEqual(authored('origin')['xform_op_order'],authored()['xform_op_order'])
        self.assertEqual(authored('origin')['base_matrix'],authored()['base_matrix'])

    def test_actual_translated_reset_writes_physics_pose_without_step(self):
        import numpy as np
        import test_standing as fixture
        from standing_session import NativeStandingSession
        old = sys.modules.get('warp')
        sys.modules['warp'] = SimpleNamespace(array=lambda x,**kw:fixture.A(x),
                                               uint32=np.uint32,float32=np.float32)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                view=fixture.View(); sim=fixture.Sim(view)
                native={'joint_names':fixture.NAMES,'body_names':fixture.BODIES}
                geometry=fixture.Geometry(fixture.META,fixture.CLOUDS,fixture.BODIES)
                session=NativeStandingSession(view,fixture.Contact(view),sim,fixture.MODEL,
                            native,geometry,temporary,fixture.save,'cpu',placement='xy14_4')
                session.reset_canonical()
                self.assertEqual(sim.counter,31)
                self.assertEqual(view.root.tolist(),[[14.,4.,0.08161108940839767,0.,0.,0.,1.]])
                self.assertTrue(np.array_equal(view.q,np.zeros((1,18),np.float32)))
                placement.validate_reset(json.loads((Path(temporary)/'initial_reset.json').read_text()),declared())
                session.close()
        finally:
            if old is None: sys.modules.pop('warp',None)
            else: sys.modules['warp']=old

    def test_event_detector_uses_all_steps_and_exact_leg(self):
        signs = {leg:{'patch_count':4,'inactive_count':0,'exact128_inactive_zero_tuples':False}
                 for leg in ('lf','lm','lr','rf','rm','rr')}
        clean = [{'sequence':i,'toe_force_norm_n':[12.]*6,
                  'toe_patch_signatures':copy.deepcopy(signs)} for i in range(8000)]
        self.assertEqual(detect(clean)['events'],[])
        changed = copy.deepcopy(clean)
        for sequence in (3,1710):
            changed[sequence]['toe_force_norm_n'][4] = 0.
            changed[sequence]['toe_patch_signatures']['rm'] = {
                'patch_count':128,'inactive_count':128,'exact128_inactive_zero_tuples':True}
        out = detect(changed)
        self.assertEqual([(x['sequence'],x['leg'],x['post_settle']) for x in out['events']],
                         [(3,'rm',False),(1710,'rm',True)])
        self.assertEqual(out['per_toe']['rm'],{'all_low_samples':2,'post_settle_low_samples':1,
                         'isolated_zero_losses':2,'isolated_zero_128_signature':2})
        self.assertEqual((contract.DT,contract.SETTLE,contract.CONTROLS),(.0025,200,1000))

    def test_fresh_host_binding_rejects_old_source_wrong_arm_and_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); path=root/'binding.json'
            args=SimpleNamespace(bindings=path,host_freeze_sha256='b'*64,
                                 placement='xy14_4',output=root/'fresh')
            good={'schema':'canonical_placement_host_binding_v1',
                  'source_freeze_sha256':host.SOURCE_FREEZE,'host_freeze_sha256':'b'*64,
                  'coordination_sha256':'c'*64,'placement':'xy14_4',
                  'output':str(root/'fresh'),'native_dispatch_authorized':True}
            write(root,'binding.json',good); self.assertEqual(host.verify_binding(args),good)
            for key,value in [('source_freeze_sha256','c87f2d340c935cd947c7aed3f3dcd7c9a9965b6e19d6726546abb560b27ab131'),
                              ('placement','origin'),('native_dispatch_authorized',False)]:
                bad=dict(good);bad[key]=value;write(root,'binding.json',bad)
                with self.assertRaises(ValueError): host.verify_binding(args)

    def test_parent_unchanged_and_no_physics_or_gate_delta(self):
        parent=ROOT/'parents/source005'
        frozen=json.loads((parent/'FREEZE_SHA256.json').read_text())
        for name,expected in frozen.items():
            self.assertEqual(hashlib.sha256((parent/name).read_bytes()).hexdigest(),expected,name)
        for name in ('standing_score.py','quiet_metrics.py','standing_math.py','servo_candidate.json',
                     'solver_recipe.py','solver_diagnostics.py','inspect_core.py',
                     'geometry/geometry.json','geometry/geometry_extrema.npz','ASSET_SHA256.json'):
            self.assertEqual((ROOT/'source'/name).read_bytes(),(parent/name).read_bytes(),name)
        old=load('historical_standing_contract_fixture',parent/'standing_contract.py')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            write(root,'state.json',{'schema':'canonical_single_placement_diagnostic_v1'})
            with self.assertRaisesRegex(ValueError,'Wrong native standing identity'):
                old.validate_result(root,{})

    def test_saved_aggregation_functions_remain_exact(self):
        repository=ROOT.parents[2]
        old=repository/'artifacts/mkii_updated_2026-09-10/standing32_channel_analysis_001/source/analyze_remote.py'
        new=ROOT/'analysis/saved_aggregation.py'
        def functions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
                    if isinstance(n,ast.FunctionDef)}
        original=functions(old)
        for name,node in functions(new).items(): self.assertEqual(node,original[name],name)

    def test_native_and_host_entries_compile_without_importing_isaac(self):
        for folder in ('source','host'):
            for path in (ROOT/folder).rglob('*.py'):
                compile(path.read_text(),str(path),'exec')
        self.assertNotIn('isaaclab',sys.modules)

    def test_standalone_host_contract_load_needs_no_source_pythonpath(self):
        script = ("import importlib.util,sys;from pathlib import Path;"
                  "p=Path(sys.argv[1]);s=importlib.util.spec_from_file_location('fixture_host',p);"
                  "h=importlib.util.module_from_spec(s);s.loader.exec_module(h);"
                  "c=h.load_module('fixture_contract',Path(sys.argv[2]));"
                  "assert c.SCHEMA=='canonical_single_placement_diagnostic_v1';"
                  "assert c.declaration('xy14_4')['xy_m']==[14.,4.];"
                  "assert not any(x in sys.modules for x in ('numpy','torch','isaaclab'))")
        with tempfile.TemporaryDirectory() as temporary:
            result=subprocess.run([sys.executable,'-B','-S','-c',script,
                                   str(ROOT/'host/launch_standing_spark.py'),
                                   str(ROOT/'source/standing_contract.py')],
                                  cwd=temporary,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__ == '__main__': unittest.main()
