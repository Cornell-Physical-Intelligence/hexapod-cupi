import ast,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import tempfile,unittest
from unittest.mock import Mock
import numpy as np
from recording_contract import *
from record_reference_video import parser_base,preflight
from native_recorder import NativeRecorder
from render_helpers import initial_rgb_frame,draw_live_command
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'tmp/reference_physics_adapter_009/source_009'
RUN=ROOT/'tmp/reference_physics_results_009/run'
class Array:
    def __init__(self,value):self.value=np.asarray(value)
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.value
class RecordingTests(unittest.TestCase):
    def args(self,output):return SimpleNamespace(source_root=SOURCE,package=SOURCE/'robot/hexapod_mkii_length_study',output=Path(output),campaign=RUN/'campaign.json',admission=RUN/'standing/admission.json',study_tree_receipt=RUN/'inputs/study_before.sha256.json',geometry_reference=SOURCE/'tools/geometry/candidate_c_reference.json',preflight_only=True)
    def test_actual009_readonly_preflight_exact_source_asset_identity(self):
        with tempfile.TemporaryDirectory() as p:
            args=self.args(Path(p)/'fresh');identity,geometry,prov=preflight(args)
            self.assertEqual(identity,prov['identity']);self.assertEqual(args.steps,2400);self.assertEqual(args.num_envs,1)
            self.assertFalse(args.output.exists());self.assertEqual(prov['asset_files'],550)
    def test_source_evidence_tampering_fails(self):
        with tempfile.TemporaryDirectory() as p:
            args=self.args(Path(p)/'out');fake=Path(p)/'campaign.json';fake.write_text('{}');args.campaign=fake
            with self.assertRaises(ValueError):verify_inputs(args)
    def test_output_inside_frozen_source_fails(self):
        with self.assertRaises(ValueError):preflight(self.args(SOURCE/'never_write_here'))
    def test_exact_frozen_main_and_serializable_ASTs_are_executed(self):
        namespace={'np':np};actual=load_frozen_functions(SOURCE,namespace)
        tree=ast.parse((SOURCE/'tools/run_reference_physics.py').read_text())
        expected={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('main','serializable')}
        self.assertEqual(actual,expected);self.assertEqual(namespace['serializable']({'x':np.array([1.,2.])}),{'x':[1.,2.]})
    def test_first_rgb_warmup_is_bounded_and_never_steps_physics(self):
        env=Mock();ready=np.arange(48,dtype=np.uint8).reshape(4,4,3);env.render.side_effect=[np.zeros((0,),np.uint8),ready]
        frame,report=initial_rgb_frame(env);self.assertTrue(report['ready']);self.assertEqual(report['physics_steps'],0);env.step.assert_not_called();np.testing.assert_array_equal(frame,ready)
        env=Mock();env.render.return_value=np.zeros((4,4,3),np.uint8)
        with self.assertRaises(RuntimeError):initial_rgb_frame(env,max_attempts=3)
        self.assertEqual(env.render.call_count,3);env.step.assert_not_called()
    def recorder(self,root):
        r=NativeRecorder(SimpleNamespace(output=Path(root)),{'source_manifest_sha256':SOURCE_MANIFEST_SHA256,'recording_source_hashes':{}},lambda:None)
        r.writer=Mock();r.last_raw=np.tile(np.arange(1280,dtype=np.uint8)[None,:,None],(720,1,3));r.drawing=Mock();r.camera=Mock();return r
    def row(self,step,terminal=False):return {'time_s':np.array([step*DT]),'position_world_m':np.array([[0.,-.005*step*DT,.13]]),'rotation_world_from_body':np.eye(3)[None],'quaternion_world_xyzw':np.array([[0.,0.,0.,1.]]),'quaternion_world_wxyz':np.array([[1.,0.,0.,0.]]),'terminated':np.array([terminal]),'truncated':np.array([False]),'distal_contact':np.ones((1,6),bool),'velocity_body_mps':np.array([[0.,-.005,0.]]),'raw_residual_action':np.zeros((1,18))}
    def test_native_frame_cadence_annotations_and_actual_trail(self):
        with tempfile.TemporaryDirectory() as p:
            r=self.recorder(p);env=Mock();env.render.return_value=r.last_raw
            for step in range(1,5):
                r.capture(self.row(step));r.after_step(env,(None,None,Array([False]),Array([False]),None))
            self.assertEqual(r.frames,2);self.assertEqual(env.render.call_count,2);self.assertEqual(len(r.trail),4)
            self.assertEqual(r.annotate(r.last_raw).shape,(720,1280,3));self.assertTrue((Path(p)/'first_frame.png').exists())
            np.testing.assert_array_equal(r.trail[-1],r.last_sample['position_world_m'][0,:2])
    def test_terminal_uses_last_valid_frame_without_post_reset_render(self):
        with tempfile.TemporaryDirectory() as p:
            r=self.recorder(p);env=Mock();r.capture(self.row(1,terminal=True));r.after_step(env,(None,None,Array([True]),Array([False]),None))
            env.render.assert_not_called();self.assertIsNotNone(r.terminal);self.assertEqual(r.frames,1)
    def test_ground_labels_preserve_five_mm_per_second_command(self):
        drawing=Mock();draw_live_command(drawing,[0.,0.,0.],[.005,0.,0.],'FORWARD')
        self.assertIn('FWD +0.005',drawing.text.call_args.args[1])
        self.assertIn('LEFT +0.000',drawing.text.call_args.args[1])
    def test_partial_failed_and_unverified_runs_never_claim_complete(self):
        with tempfile.TemporaryDirectory() as p:
            r=self.recorder(p);r.steps=2400;r.frames=1200
            save_json(Path(p)/'state.json',{'status':'completed','gate':{'passed':True}})
            report=r.finish(False);self.assertFalse(report['complete'])
            r.steps=123
            save_json(Path(p)/'state.json',{'status':'failed','gate':{'passed':False},'error':'measured gate'})
            report=r.finish(True);self.assertFalse(report['complete'])
            self.assertEqual(report['source_screen_failure'],'measured gate')
            self.assertEqual(report['recorded_control_steps'],123)
    def test_schedule_and_import_order_remain_bounded_and_actor_free(self):
        self.assertEqual(segment(0)[1],0);self.assertEqual(segment(200)[1],.005);self.assertEqual(segment(1399)[1],.005);self.assertEqual(segment(1400)[1],0)
        code=(Path(__file__).parent/'record_reference_video.py').read_text()
        self.assertLess(code.index('app=AppLauncher(args).app'),code.index('        import torch'))
        self.assertIn('threading.Timer(90,timeout)',code);self.assertIn('REFERENCE_SCREEN_APP_READY',code)
        self.assertNotIn('get_inference_policy',code);self.assertNotIn('write_root_pose',code)
        frame_code=(Path(__file__).parent/'native_recorder.py').read_text();self.assertNotIn('set_reference_targets',frame_code);self.assertNotIn('set_evaluation_targets',frame_code)

if __name__=='__main__':unittest.main()
