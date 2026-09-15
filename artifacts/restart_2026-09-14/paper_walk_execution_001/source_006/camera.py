"""Native RGB capture with explicit Replicator rendering and zero physics time."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np


STEP_ARGUMENTS={"rt_subframes":4,"delta_time":0.,"pause_timeline":False,"wait_for_render":True}
NATIVE_GETTERS=("get_dof_positions","get_dof_velocities","get_root_transforms",
                "get_root_velocities","get_link_transforms","get_dof_actuation_forces")


def _array(value):
    if hasattr(value,"detach"):
        value=value.detach().cpu().numpy()
    return np.array(value,copy=True)


def _save(path,value):
    temporary=path.with_suffix(path.suffix+".part")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


class NativePolicyCamera:
    """Callable replacement for env.render; never steps or writes robot state."""
    def __init__(self,env,output=None,*,max_capture_attempts=16):
        if not env.cfg.render:
            raise ValueError("Native cameras must be enabled before app startup")
        if type(max_capture_attempts) is not int or not 1<=max_capture_attempts<=32:
            raise ValueError("Camera warmup attempts must be bounded1..32")
        self.env=env
        self.output=Path(output) if output is not None else Path(env.output)/"camera"
        self.output.mkdir(parents=True,exist_ok=False)
        self.max_capture_attempts=max_capture_attempts
        self.rep=self.timeline=self.camera=self.product=self.rgb=None
        self.resolution=None;self.calls=0
        self.receipt={"schema":"canonical_native_policy_camera_v1",
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "capture_arguments":STEP_ARGUMENTS,"max_capture_attempts":max_capture_attempts,
            "native_getters_checked":list(NATIVE_GETTERS),"physics_state_comparison":"exact array equality",
            "robot_pose_writes":False,"simulation_steps":False,"successful_frames":0,
            "synchronization":"SimulationContext.render(skip_app_pumping=True), then Replicator zero-time capture",
            "api_reference":"https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_getting_started.html"}
        _save(self.output/"receipt.json",self.receipt)

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

    def _initialize(self,width,height):
        import omni.timeline
        self.timeline=omni.timeline.get_timeline_interface()
        self._initial_timeline_time=float(self.timeline.get_current_time())
        import omni.replicator.core as rep
        self.rep=rep
        self.rep.orchestrator.set_capture_on_play(False)
        self.camera=rep.create.camera(position=(1.,1.,.6),look_at=(0.,0.,.1),focal_length=24.)
        self.product=rep.create.render_product(self.camera,(width,height))
        self.rgb=rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb.attach([self.product])
        self.resolution=(width,height)

    def __call__(self,index=0,width=1280,height=720):
        if type(index) is not int or not 0<=index<self.env.num_envs:
            raise ValueError("Camera robot index is outside the native environment")
        if type(width) is not int or type(height) is not int or width<=0 or height<=0:
            raise ValueError("Native RGB dimensions must be positive integers")
        if self.resolution is not None and self.resolution!=(width,height):
            raise ValueError("A native render product cannot silently change resolution")
        before=self._snapshot();started=time.monotonic()
        entry={"call":self.calls,"env_index":index,"physics_counter":before[0],"attempts":[]}
        timeline_before=None
        try:
            if self.rgb is None:
                self._initialize(width,height)
                self._assert_unchanged(before,"camera graph construction")
                timeline_before=self._initial_timeline_time
            else:
                timeline_before=float(self.timeline.get_current_time())
            if float(self.timeline.get_current_time())!=timeline_before:
                raise RuntimeError("Camera graph construction changed timeline time")
            position=before[1]["get_root_transforms"][index,:3]
            with self.camera:
                self.rep.modify.pose(position=(float(position[0])+.8,float(position[1])+1.,.65),
                    look_at=(float(position[0]),float(position[1])-.1,.1))
            self._assert_unchanged(before,"camera pose graph authoring")
            for attempt in range(self.max_capture_attempts):
                # Flush the actual native transforms to the renderer without a
                # free-running Kit app pump or another native integration step.
                self.env.sim.render(skip_app_pumping=True)
                self._assert_unchanged(before,"visual state synchronization")
                self.rep.orchestrator.step(**STEP_ARGUMENTS)
                self._assert_unchanged(before,"zero-time Replicator capture")
                if float(self.timeline.get_current_time())!=timeline_before:
                    raise RuntimeError("Replicator RGB capture changed timeline time")
                pixels=_array(self.rgb.get_data())
                valid_shape=pixels.ndim==3 and pixels.shape[:2]==(height,width) and pixels.shape[2] in (3,4)
                rgb=pixels[...,:3] if valid_shape else np.empty(0,dtype=np.uint8)
                numeric=valid_shape and np.issubdtype(rgb.dtype,np.number) and np.isfinite(rgb).all()
                stats={"attempt":attempt+1,"shape":list(pixels.shape),"dtype":str(pixels.dtype),
                    "rgb_min":float(rgb.min()) if numeric else None,
                    "rgb_max":float(rgb.max()) if numeric else None,
                    "rgb_std":float(rgb.std()) if numeric else None}
                entry["attempts"].append(stats)
                if numeric and rgb.dtype==np.uint8 and float(rgb.std())>=1.:
                    self._assert_unchanged(before,"completed RGB readback")
                    entry.update(status="captured",timeline_time=timeline_before,wall_seconds=time.monotonic()-started)
                    self.receipt["successful_frames"]+=1
                    self.receipt["last_frame"]=entry
                    return rgb.copy()
            raise RuntimeError("Missing or blank native Isaac RGB after bounded zero-time Replicator captures")
        except BaseException as error:
            entry.update(status="failed",failure=repr(error),wall_seconds=time.monotonic()-started)
            try:
                self._assert_unchanged(before,"failed capture final audit")
            except BaseException as audit_error:
                entry["final_native_audit_failure"]=repr(audit_error)
            self.receipt["failure"]=entry
            _save(self.output/f"failure_{self.calls:05d}.json",entry)
            raise
        finally:
            with (self.output/"frames.jsonl").open("a") as stream:
                stream.write(json.dumps(entry,allow_nan=False)+"\n")
            self.calls+=1
            _save(self.output/"receipt.json",self.receipt)
