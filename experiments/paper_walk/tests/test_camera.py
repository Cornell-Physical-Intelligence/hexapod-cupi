"""Native camera invariants and actual RGB readiness without launching Isaac."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from experiments.paper_walk.camera import NativePolicyCamera, NATIVE_GETTERS, CAMERA_PRIM_PATH


class FakeNative:
    def __init__(self,output):
        self.output=Path(output);self.cfg=SimpleNamespace(render=True,physics_dt=.0025);self.num_envs=1
        self.counter=0;self.render_calls=[]
        self.values={name:np.zeros((1,18),np.float32) for name in NATIVE_GETTERS}
        self.values["get_root_transforms"]=np.array([[0.,0.,.1,0.,0.,0.,1.]],np.float32)
        self.sim=SimpleNamespace(get_physics_step_count=lambda:self.counter,render=self.render)

    def get(self,name):return self.values[name].copy()

    def render(self,**kwargs):self.render_calls.append(kwargs)


class Backend:
    def __init__(self,frames,mutation=None,initialize_mutation=None):
        self.frames=list(frames);self.steps=[];self.poses=[];self.timeline_time=2.5
        self.mutation=mutation;self.initialize_mutation=initialize_mutation

    def initialize(self,camera,width,height):
        self.env=camera.env
        original_render=self.env.sim.render
        def render(**kwargs):
            original_render(**kwargs)
            self.step(**kwargs)
        self.env.sim.render=render
        camera.viewport_manager=SimpleNamespace(set_camera_view=lambda path,**kw:self.poses.append((path,kw)))
        camera.timeline=SimpleNamespace(get_current_time=lambda:self.timeline_time)
        camera._initial_timeline_time=self.timeline_time
        camera.rgb=SimpleNamespace(get_data=self.data)
        camera.resolution=(width,height)
        if self.initialize_mutation:self.initialize_mutation(self)

    def step(self,**kwargs):
        self.steps.append(kwargs)
        if self.mutation:self.mutation(self)

    def data(self):
        return self.frames.pop(0) if len(self.frames)>1 else self.frames[0]


def pixels():
    return np.array([[[0,10,20,255],[200,100,0,255]],[[50,60,70,255],[220,230,240,255]]],np.uint8)


class NativeCameraTests(unittest.TestCase):
    def test_native_kit_capture_waits_for_real_rgb_with_progress(self):
        with TemporaryDirectory() as directory:
            env=FakeNative(directory);backend=Backend([np.empty(0),pixels()])
            with patch.object(NativePolicyCamera,"_initialize",lambda camera,w,h:backend.initialize(camera,w,h)):
                camera=NativePolicyCamera(env)
                rgb=camera(width=2,height=2)
                np.testing.assert_array_equal(rgb,pixels()[...,:3])
                self.assertEqual(backend.steps,[{}]*4)
                self.assertEqual(env.render_calls,[{}]*4)
                self.assertEqual(backend.poses[0][0],CAMERA_PRIM_PATH)
                self.assertEqual(env.counter,0)
                self.assertEqual(camera.receipt["successful_frames"],1)
                self.assertEqual(len(camera.receipt["last_frame"]["attempts"]),2)
                camera(width=2,height=2)
                self.assertEqual(len(backend.steps),6)
                self.assertEqual(camera.receipt["successful_frames"],2)
                progress=[json.loads(line) for line in (Path(directory)/"camera/progress.jsonl").read_text().splitlines()]
                self.assertTrue(any(row["stage"]=="render_1_1" and row["status"]=="started" for row in progress))
                self.assertTrue(any(row["stage"]=="read_cpu_rgb_1" and row["status"]=="completed" for row in progress))
                self.assertEqual(json.loads((Path(directory)/"camera/progress.json").read_text())["status"],"captured")

    def test_alpha_only_blank_frame_never_passes(self):
        blank=np.zeros((2,2,4),np.uint8);blank[:,:,3]=255
        with TemporaryDirectory() as directory:
            env=FakeNative(directory);backend=Backend([blank])
            with patch.object(NativePolicyCamera,"_initialize",lambda camera,w,h:backend.initialize(camera,w,h)):
                camera=NativePolicyCamera(env,max_capture_attempts=3)
                with self.assertRaisesRegex(RuntimeError,"bounded Kit"):
                    camera(width=2,height=2)
            self.assertEqual(len(backend.steps),6)
            failure=json.loads((Path(directory)/"camera/failure_00000.json").read_text())
            self.assertEqual(failure["attempts"][-1]["rgb_std"],0.)
            self.assertEqual(failure["attempts"][-1]["shape"],[2,2,4])

    def test_graph_creation_cannot_change_native_state(self):
        with TemporaryDirectory() as directory:
            env=FakeNative(directory)
            backend=Backend([pixels()],initialize_mutation=lambda b:b.env.values["get_dof_positions"].fill(.1))
            with patch.object(NativePolicyCamera,"_initialize",lambda camera,w,h:backend.initialize(camera,w,h)):
                camera=NativePolicyCamera(env)
                with self.assertRaisesRegex(RuntimeError,"graph construction"):
                    camera(width=2,height=2)
            self.assertEqual(backend.steps,[])
            self.assertTrue((Path(directory)/"camera/changed_native_state_00000.npz").is_file())

    def test_counter_and_native_velocity_each_fail_closed(self):
        mutations=(lambda b:setattr(b.env,"counter",1),
                   lambda b:b.env.values["get_dof_velocities"].fill(.1))
        for mutation in mutations:
            with self.subTest(mutation=mutation),TemporaryDirectory() as directory:
                env=FakeNative(directory);backend=Backend([pixels()],mutation=mutation)
                with patch.object(NativePolicyCamera,"_initialize",lambda camera,w,h:backend.initialize(camera,w,h)):
                    camera=NativePolicyCamera(env)
                    with self.assertRaises(RuntimeError):camera(width=2,height=2)
                self.assertEqual(camera.receipt["successful_frames"],0)
                self.assertIn(len(backend.steps),(1,2))

    def test_visual_clock_changes_are_diagnostic_and_physics_counter_sets_timestamp(self):
        with TemporaryDirectory() as directory:
            env=FakeNative(directory);env.counter=80
            backend=Backend([pixels()],mutation=lambda b:setattr(b,"timeline_time",b.timeline_time+.1),
                initialize_mutation=lambda b:setattr(b,"timeline_time",2.6))
            with patch.object(NativePolicyCamera,"_initialize",lambda camera,w,h:backend.initialize(camera,w,h)):
                camera=NativePolicyCamera(env)
                np.testing.assert_array_equal(camera(width=2,height=2),pixels()[...,:3])
            frame=camera.receipt["last_frame"]
            self.assertEqual(frame["physics_counter"],80)
            self.assertAlmostEqual(frame["physics_time_s"],.2)
            self.assertAlmostEqual(frame["physics_dt_s"],.0025)
            self.assertEqual(frame["visual_timeline"]["before_s"],2.5)
            self.assertAlmostEqual(frame["visual_timeline"]["after_s"],2.8)
            self.assertAlmostEqual(frame["visual_timeline"]["delta_s"],.3)
            self.assertTrue(frame["visual_timeline"]["diagnostic_only"])
            self.assertNotIn("timeline_time",frame)
            self.assertEqual(env.counter,80)
            self.assertEqual(camera.receipt["successful_frames"],1)
            saved=json.loads((Path(directory)/"camera/receipt.json").read_text())
            self.assertEqual(saved["last_frame"],frame)

    def test_cpu_annotator_dict_and_flat_buffer_are_native_formats(self):
        for value in ({"data":pixels()},pixels().reshape(-1)):
            with self.subTest(format=type(value).__name__),TemporaryDirectory() as directory:
                env=FakeNative(directory);backend=Backend([value])
                with patch.object(NativePolicyCamera,"_initialize",lambda camera,w,h:backend.initialize(camera,w,h)):
                    camera=NativePolicyCamera(env)
                    np.testing.assert_array_equal(camera(width=2,height=2),pixels()[...,:3])

    def test_capture_watchdog_terminates_only_its_stalled_child_and_preserves_stage(self):
        script='''
import sys,time
from experiments.paper_walk.camera import NativePolicyCamera
from experiments.paper_walk.tests.test_camera import FakeNative
def hang(camera,width,height):time.sleep(10)
NativePolicyCamera._initialize=hang
NativePolicyCamera(FakeNative(sys.argv[1]),capture_timeout_seconds=.15)(width=2,height=2)
'''
        with TemporaryDirectory() as directory:
            result=subprocess.run([sys.executable,"-c",script,directory],capture_output=True,text=True,timeout=5)
            self.assertEqual(result.returncode,1)
            progress=json.loads((Path(directory)/"camera/progress.json").read_text())
            self.assertEqual((progress["stage"],progress["status"]),("initialize_camera","started"))
            trace=(Path(directory)/"camera/timeout_tracebacks.log").read_text()
            self.assertIn("Timeout",trace)
            self.assertIn("hang",trace)
            self.assertEqual(json.loads((Path(directory)/"camera/receipt.json").read_text())["successful_frames"],0)

    def test_invalid_deadlines_rejected_before_output_creation(self):
        for value in (0,float("nan"),float("inf"),301,True):
            with self.subTest(value=value),TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError,"deadline"):
                    NativePolicyCamera(FakeNative(directory),capture_timeout_seconds=value)
                self.assertFalse((Path(directory)/"camera").exists())

    def test_disabled_visualizers_are_recorded_before_rejection_without_initializing(self):
        with TemporaryDirectory() as directory:
            env=FakeNative(directory)
            env.sim.visualizers=[]
            env.sim.cfg=SimpleNamespace(visualizer_cfgs=[SimpleNamespace(visualizer_type="kit")])
            env.sim.get_setting=lambda name: True if name=="/isaaclab/visualizer/disable_all" else None
            env.sim.initialize_visualizers=lambda: self.fail("Camera must not initialize visualizers")
            camera=NativePolicyCamera(env)
            with self.assertRaisesRegex(RuntimeError,"visualizer_inventory.json"):
                camera(width=2,height=2)
            facts=json.loads((Path(directory)/"camera/visualizer_inventory.json").read_text())
            self.assertEqual(facts["active"],[])
            self.assertEqual(facts["configured"][0]["visualizer_type"],"kit")
            self.assertTrue(facts["settings"]["/isaaclab/visualizer/disable_all"])
            self.assertFalse(facts["visualizer_initialization_requested_by_camera"])
            self.assertFalse(facts["physics_reset_requested_by_camera"])
            self.assertEqual(env.render_calls,[])
            self.assertEqual(env.counter,0)
            self.assertEqual(camera.receipt["visualizer_inventory"],facts)

    def test_visualizer_inventory_retains_actual_types_and_startup_errors(self):
        with TemporaryDirectory() as directory:
            env=FakeNative(directory)
            good=SimpleNamespace(cfg=SimpleNamespace(visualizer_type="kit"),_is_initialized=True,
                is_running=lambda:True,pumps_app_update=lambda:True)
            def unavailable():raise RuntimeError("no app")
            bad=SimpleNamespace(cfg=SimpleNamespace(visualizer_type="kit"),_is_initialized=False,
                is_running=unavailable,pumps_app_update=lambda:True)
            env.sim.visualizers=[good,bad]
            env.sim.cfg=SimpleNamespace(visualizer_cfgs=good.cfg)
            env.sim.get_setting=lambda name:None
            camera=NativePolicyCamera(env)
            with camera._deadline():
                with self.assertRaisesRegex(RuntimeError,"visualizer_inventory.json"):
                    camera._validated_kit_visualizers()
            facts=camera.receipt["visualizer_inventory"]["active"]
            self.assertEqual(facts[0]["type"],"SimpleNamespace")
            self.assertEqual(facts[0]["cfg_type"],"SimpleNamespace")
            self.assertTrue(facts[0]["initialized"])
            self.assertTrue(facts[0]["is_running"])
            self.assertFalse(facts[1]["initialized"])
            self.assertIn("no app",facts[1]["is_running"]["error"])
            env.sim.visualizers=[good]
            with camera._deadline():
                self.assertEqual(camera._validated_kit_visualizers(),[good])


if __name__=="__main__":unittest.main()
