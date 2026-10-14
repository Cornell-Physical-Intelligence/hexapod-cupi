"""Native Kit perspective RGB capture with audited physics and a hard deadline."""
from contextlib import contextmanager
from pathlib import Path
import faulthandler
import hashlib
import inspect
import json
import math
import os
import time
import numpy as np


CAMERA_PRIM_PATH="/OmniverseKit_Persp"
NATIVE_GETTERS=("get_dof_positions","get_dof_velocities","get_root_transforms",
                "get_root_velocities","get_link_transforms","get_dof_actuation_forces")


def _array(value):
    if hasattr(value,"detach"):
        value=value.detach().cpu().numpy()
    return np.array(value,copy=True)


def _save(path,value):
    temporary=path.with_suffix(path.suffix+".part")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+"\n")
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


class NativePolicyCamera:
    """Callable replacement for env.render; never steps or writes robot state."""
    def __init__(self,env,output=None,*,max_capture_attempts=16,capture_timeout_seconds=120.):
        if not env.cfg.render:
            raise ValueError("Native cameras must be enabled before app startup")
        if type(max_capture_attempts) is not int or not 1<=max_capture_attempts<=32:
            raise ValueError("Camera warmup attempts must be bounded1..32")
        if isinstance(capture_timeout_seconds,bool) or not isinstance(capture_timeout_seconds,(int,float)) or not math.isfinite(capture_timeout_seconds) or not .05<=capture_timeout_seconds<=300:
            raise ValueError("Camera capture deadline must be finite and bounded0.05..300 seconds")
        self.env=env
        self.output=Path(output) if output is not None else Path(env.output)/"camera"
        self.output.mkdir(parents=True,exist_ok=False)
        self.max_capture_attempts=max_capture_attempts
        self.capture_timeout_seconds=float(capture_timeout_seconds)
        self.rep=self.timeline=self.viewport_manager=self.product=self.rgb=None
        self.resolution=None;self.calls=0;self._capturing=False
        self.receipt={"schema":"canonical_native_policy_camera_v2",
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "camera_prim_path":CAMERA_PRIM_PATH,"max_capture_attempts":max_capture_attempts,
            "capture_timeout_seconds":self.capture_timeout_seconds,
            "timeout_action":"faulthandler dumps all Python stacks and exits this process; owned host supervisor performs cleanup",
            "native_getters_checked":list(NATIVE_GETTERS),"physics_state_comparison":"exact array equality",
            "robot_pose_writes":False,"simulation_steps":False,"successful_frames":0,
            "synchronization":"Regular SimulationContext.render drives initialized KitVisualizer; two render pumps before CPU RGB readback; no orchestrator.step or unguarded app.update",
            "capture_on_play":True,"annotator_device":"cpu",
            "frame_timestamp_source":"Native physics counter times configured physics_dt; visual timeline is diagnostic only",
            "native_step_clock":"Installed PhysxManager.step calls simulate(sim.cfg.dt, 0.0), independently of the Kit visual timeline",
            "api_reference":"https://github.com/isaac-sim/IsaacLab/blob/release/3.0.0-beta2/source/isaaclab_physx/isaaclab_physx/video_recording/isaacsim_kit_perspective_video.py"}
        _save(self.output/"receipt.json",self.receipt)

    def _progress(self,stage,status,**details):
        event={"call":self.calls,"stage":stage,"status":status,"wall_time_unix":time.time(),
            "elapsed_seconds":time.monotonic()-self._started,**details}
        _save(self.output/"progress.json",event)
        with (self.output/"progress.jsonl").open("a") as stream:
            stream.write(json.dumps(event,allow_nan=False)+"\n")
            stream.flush();os.fsync(stream.fileno())

    def _operation(self,stage,callback):
        self._progress(stage,"started")
        value=callback()
        self._progress(stage,"completed")
        return value

    @contextmanager
    def _deadline(self):
        if self._capturing:
            raise RuntimeError("Native camera captures must be serialized")
        self._capturing=True;self._started=time.monotonic()
        # A Python exception timer cannot interrupt a native extension that holds
        # the GIL. faulthandler's watchdog can still record stacks and terminate.
        with (self.output/"timeout_tracebacks.log").open("a") as trace:
            faulthandler.dump_traceback_later(self.capture_timeout_seconds,repeat=False,file=trace,exit=True)
            try:
                self._progress("capture","started",timeout_seconds=self.capture_timeout_seconds)
                yield
            finally:
                faulthandler.cancel_dump_traceback_later()
                self._capturing=False

    def _snapshot(self):
        state={name:_array(self.env.get(name)) for name in NATIVE_GETTERS}
        if not all(np.isfinite(value).all() for value in state.values()):
            raise RuntimeError("Nonfinite native state before/during RGB capture")
        return int(self.env.sim.get_physics_step_count()),state

    def _assert_unchanged(self,before,phase):
        counter,state=self._snapshot()
        changed=[name for name,value in before[1].items() if not np.array_equal(value,state[name])]
        if counter!=before[0] or changed:
            np.savez_compressed(self.output/f"changed_native_state_{self.calls:05d}.npz",
                **{"before__"+k:v for k,v in before[1].items()},**{"after__"+k:v for k,v in state.items()})
            raise RuntimeError(f"RGB capture changed native physics during {phase}: counter {before[0]}->{counter}; fields {changed}")

    def _validated_kit_visualizers(self):
        def observed(callback):
            try:return callback()
            except Exception as error:return {"error":repr(error)}
        visualizers=list(self.env.sim.visualizers)
        configured=getattr(self.env.sim.cfg,"visualizer_cfgs",None)
        configured=[] if configured is None else configured if isinstance(configured,list) else [configured]
        facts=[]
        for viz in visualizers:
            cfg=getattr(viz,"cfg",None)
            facts.append({"type":type(viz).__name__,"module":type(viz).__module__,
                "cfg_type":type(cfg).__name__,"visualizer_type":getattr(cfg,"visualizer_type",None),
                "initialized":observed(lambda:bool(viz._is_initialized)),
                "is_running":observed(lambda:bool(viz.is_running())),
                "pumps_app_update":observed(lambda:bool(viz.pumps_app_update()))})
        inventory={"configured":[{"cfg_type":type(cfg).__name__,"visualizer_type":getattr(cfg,"visualizer_type",None)} for cfg in configured],
            "active":facts,"settings":{name:observed(lambda name=name:self.env.sim.get_setting(name)) for name in
                ("/isaaclab/visualizer/disable_all","/isaaclab/visualizer/explicit","/isaaclab/visualizer/types",
                 "/isaaclab/has_gui","/isaaclab/render/offscreen","/isaaclab/cameras_enabled")},
            "visualizer_initialization_requested_by_camera":False,"physics_reset_requested_by_camera":False}
        self.receipt["visualizer_inventory"]=inventory
        _save(self.output/"visualizer_inventory.json",inventory)
        _save(self.output/"receipt.json",self.receipt)
        self._progress("visualizer_inventory","recorded",active_count=len(facts))
        selected=[(viz,fact) for viz,fact in zip(visualizers,facts) if fact["visualizer_type"]=="kit"
            and fact["pumps_app_update"] is True]
        if not selected or any(fact["initialized"] is not True or fact["is_running"] is not True for _,fact in selected):
            raise RuntimeError("Native capture requires an initialized running Kit visualizer; see visualizer_inventory.json")
        return [viz for viz,_ in selected]

    def _initialize(self,width,height):
        kits=self._operation("inspect_visualizers",self._validated_kit_visualizers)
        import omni.timeline
        self.timeline=omni.timeline.get_timeline_interface()
        self._initial_timeline_time=float(self.timeline.get_current_time())
        import omni.replicator.core as rep
        self.rep=rep
        from isaacsim.core.rendering_manager import ViewportManager
        self.viewport_manager=ViewportManager
        self.receipt["kit_visualizers"]=[{"type":type(viz).__name__,"initialized":bool(viz._is_initialized),
            "source_sha256":hashlib.sha256(Path(inspect.getfile(type(viz))).read_bytes()).hexdigest()} for viz in kits]
        self.receipt["simulation_context_sha256"]=hashlib.sha256(Path(inspect.getfile(type(self.env.sim))).read_bytes()).hexdigest()
        self.receipt["physics_manager_sha256"]=hashlib.sha256(Path(inspect.getfile(self.env.sim.physics_manager.step)).read_bytes()).hexdigest()
        self._operation("enable_native_capture",lambda:self.rep.orchestrator.set_capture_on_play(True))
        self.product=self._operation("create_perspective_render_product",lambda:rep.create.render_product(CAMERA_PRIM_PATH,(width,height)))
        self.rgb=self._operation("create_cpu_rgb_annotator",lambda:rep.AnnotatorRegistry.get_annotator("rgb",device="cpu"))
        self._operation("attach_cpu_rgb_annotator",lambda:self.rgb.attach([self.product]))
        self.resolution=(width,height)

    def __call__(self,index=0,width=1280,height=720):
        if type(index) is not int or not 0<=index<self.env.num_envs:
            raise ValueError("Camera robot index is outside the native environment")
        if type(width) is not int or type(height) is not int or width<=0 or height<=0:
            raise ValueError("Native RGB dimensions must be positive integers")
        if self.resolution is not None and self.resolution!=(width,height):
            raise ValueError("A native render product cannot silently change resolution")
        with self._deadline():
            return self._capture(index,width,height)

    def _capture(self,index,width,height):
        before=None;started=time.monotonic()
        entry={"call":self.calls,"env_index":index,"attempts":[]}
        timeline_before=None
        try:
            before=self._operation("native_snapshot",self._snapshot)
            entry["physics_counter"]=before[0]
            entry["physics_dt_s"]=float(self.env.cfg.physics_dt)
            entry["physics_time_s"]=before[0]*entry["physics_dt_s"]
            if self.rgb is None:
                self._operation("initialize_camera",lambda:self._initialize(width,height))
                self._assert_unchanged(before,"camera graph construction")
                timeline_before=self._initial_timeline_time
            else:
                timeline_before=float(self.timeline.get_current_time())
            entry["visual_timeline"]={"before_s":timeline_before,"diagnostic_only":True}
            position=before[1]["get_root_transforms"][index,:3]
            self._operation("set_perspective_camera_view",lambda:self.viewport_manager.set_camera_view(CAMERA_PRIM_PATH,
                eye=[float(position[0])+.8,float(position[1])+1.,.65],
                target=[float(position[0]),float(position[1])-.1,.1]))
            self._assert_unchanged(before,"camera pose graph authoring")
            for attempt in range(self.max_capture_attempts):
                # Installed KitVisualizer.step disables playSimulations around
                # its app.update. Never replace this with a naked app.update.
                # Two pumps let the perspective product publish after pose sync.
                for pump in range(2):
                    self._operation(f"render_{attempt+1}_{pump+1}",self.env.sim.render)
                    self._operation(f"audit_render_{attempt+1}_{pump+1}",lambda:self._assert_unchanged(before,"Kit perspective render"))
                # Kit's visual clock advances during guarded app pumping. Native
                # integration uses sim.cfg.dt and is audited above; never use
                # this renderer clock for a physical sample timestamp.
                visual_time=float(self.timeline.get_current_time())
                entry["visual_timeline"].update(after_s=visual_time,delta_s=visual_time-timeline_before)
                data=self._operation(f"read_cpu_rgb_{attempt+1}",self.rgb.get_data)
                if isinstance(data,dict):data=data.get("data",np.empty(0,dtype=np.uint8))
                pixels=_array(data)
                if pixels.ndim==1 and pixels.size in (height*width*3,height*width*4):
                    pixels=pixels.reshape(height,width,-1)
                valid_shape=pixels.ndim==3 and pixels.shape[:2]==(height,width) and pixels.shape[2] in (3,4)
                rgb=pixels[...,:3] if valid_shape else np.empty(0,dtype=np.uint8)
                numeric=valid_shape and np.issubdtype(rgb.dtype,np.number) and np.isfinite(rgb).all()
                stats={"attempt":attempt+1,"shape":list(pixels.shape),"dtype":str(pixels.dtype),
                    "visual_timeline_after_s":visual_time,
                    "rgb_min":float(rgb.min()) if numeric else None,
                    "rgb_max":float(rgb.max()) if numeric else None,
                    "rgb_std":float(rgb.std()) if numeric else None}
                entry["attempts"].append(stats)
                self._progress("rgb_readback","observed",**stats)
                if numeric and rgb.dtype==np.uint8 and float(rgb.std())>=1.:
                    self._assert_unchanged(before,"completed RGB readback")
                    entry.update(status="captured",wall_seconds=time.monotonic()-started)
                    self.receipt["successful_frames"]+=1
                    self.receipt["last_frame"]=entry
                    return rgb.copy()
            raise RuntimeError("Missing or blank native Isaac RGB after bounded Kit perspective captures")
        except BaseException as error:
            entry.update(status="failed",failure=repr(error),wall_seconds=time.monotonic()-started)
            try:
                if before is not None:self._assert_unchanged(before,"failed capture final audit")
            except BaseException as audit_error:
                entry["final_native_audit_failure"]=repr(audit_error)
            self.receipt["failure"]=entry
            _save(self.output/f"failure_{self.calls:05d}.json",entry)
            raise
        finally:
            with (self.output/"frames.jsonl").open("a") as stream:
                stream.write(json.dumps(entry,allow_nan=False)+"\n")
                stream.flush();os.fsync(stream.fileno())
            _save(self.output/"receipt.json",self.receipt)
            self._progress("capture",entry.get("status","failed"))
            self.calls+=1


def prepare_policy_scene(env):
    """Lighting and visual-only meter grid for an actual native policy recording."""
    from pxr import UsdGeom, UsdLux, Gf
    stage = env.sim.stage
    light = UsdLux.DomeLight.Define(stage, "/PolicyVideoLight")
    light.CreateIntensityAttr(1800.)
    light.CreateColorAttr(Gf.Vec3f(.90, .94, 1.))
    ground = UsdGeom.Mesh(stage.GetPrimAtPath("/Ground"))
    ground.CreateDisplayColorAttr([Gf.Vec3f(.18, .20, .23)])
    # These strips have no CollisionAPI or rigid body. They provide a visible
    # distance reference; all robot-ground contact remains the original mesh.
    for axis in range(2):
        for i in range(-10, 11):
            line = UsdGeom.Cube.Define(stage, f"/PolicyVideoGrid/axis{axis}_{i+10:02d}")
            line.CreateSizeAttr(1.)
            transform = UsdGeom.Xformable(line.GetPrim())
            transform.AddTranslateOp().Set(Gf.Vec3d(i*.5 if axis == 0 else 0.,
                                                  i*.5 if axis == 1 else 0., .00015))
            transform.AddScaleOp().Set(Gf.Vec3f(.002 if axis == 0 else 10.,
                                              .002 if axis == 1 else 10., .0001))
            line.CreateDisplayColorAttr([Gf.Vec3f(.30, .33, .36)])
    _save(env.output / "video_scene.json", {"lighting": "dome1800", "grid_spacing_m": .5,
        "grid_has_collision": False, "robot_physics_changed": False})
