"""CPU-only lineage checks shared by the external policy recorder and supervisor."""
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

TASK_ID = "Isaac-Velocity-Flat-Hexapod-MKII-Fourbar-V1-Direct-v0"
MODEL_ID = "mkii_fourbar_v5"
SCHEMA = "hexapod.admitted_policy_capture.v1"
TOOL_FILES = ("capture_common.py", "record_admitted_policy.py", "capture_policy.py")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def reject(value):
        raise ValueError(f"Nonfinite JSON: {value}")
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=reject)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_module(path, name):
    path = Path(path).resolve()
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path:
            raise ValueError(f"Module {name} was loaded from a different source")
        return module
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def source_api(source):
    source = Path(source).resolve(strict=True)
    sys.path[:0] = [str(source / p) for p in
                   ("tools", "isaaclab", "packages/hexapod_core", "packages/hexapod_env")]
    return load_module(source / "tools/mkii_training_contract.py", "mkii_training_contract")


def tool_identity(directory):
    directory = Path(directory).resolve()
    return {name: digest(directory / name) for name in TOOL_FILES}


def environment_layout(source, runtime):
    """Retain the admitted placement recipe even for a one-environment recording."""
    descriptor = runtime.get("resolved_environment_layout")
    selected = descriptor.get("layout_id") if type(descriptor) is dict else "grid_2m_v1"
    helper = Path(source) / "packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1/environment_layout.py"
    if helper.is_file():
        module = load_module(helper, "capture_source_environment_layout")
        module.validate_layout_runtime(runtime, selected)
    elif "resolved_environment_layout" in runtime or selected != "grid_2m_v1":
        raise ValueError("Frozen source cannot validate the admitted environment layout")
    return selected


def validate_training(training, admission, sidecar, contract):
    """A paused/partial run or merely existing checkpoint does not admit capture."""
    if (training.get("task_id") != TASK_ID or training.get("contract") != contract
            or training.get("mode") != "provisional_physical_fourbar_ppo"
            or training.get("pass") is not True or training.get("errors") != []
            or training.get("paused") is not False or training.get("hardware_admission") is not False
            or any(training.get(key) is not True for key in
                   ("checkpoint_verified", "checkpoint_roundtrip_pass", "policy_changed"))):
        raise ValueError("Capture requires completed successful physical PPO, not a paused run")
    requested, completed, start, nxt = [training.get(key) for key in
        ("iterations_requested", "iterations_completed", "start_iteration", "next_iteration")]
    if (any(type(v) is not int for v in (requested, completed, start, nxt))
            or not 1 <= requested <= 10000 or completed != requested or start < 0
            or nxt != start + completed or nxt != sidecar["next_iteration"]):
        raise ValueError("Training iteration completion/checkpoint mismatch")
    if training.get("checkpoint_sha256") != sidecar["checkpoint_sha256"]:
        raise ValueError("Training report names different checkpoint bytes")
    for key in ("policy_after_sha256", "algorithm_after_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(training.get(key, ""))):
            raise ValueError(f"Missing trained-state digest: {key}")
    probe = training.get("inference_probe", {})
    if probe.get("finite") is not True or type(probe.get("steps")) is not int or probe["steps"] != 100:
        raise ValueError("Training did not finish its checkpoint inference probe")
    runtime = training.get("runtime_manifest")
    if (type(runtime) is not dict or runtime.get("schema") != "hexapod.physical_fourbar_runtime.v1"
            or runtime.get("task_id") != TASK_ID or runtime.get("model_id") != MODEL_ID
            or runtime.get("observation_dim") != 84
            or len(runtime.get("active_motor_names", [])) != 18
            or len(set(runtime.get("active_motor_names", []))) != 18
            or len(runtime.get("tree_joint_names", [])) != 30
            or len(set(runtime.get("tree_joint_names", []))) != 30
            or canonical(runtime) != canonical(admission.get("runtime_manifest"))):
        raise ValueError("Training/admission physical v5 runtime mismatch")
    return runtime


def verify_inputs(source, checkpoint, admission_path, training_path):
    api = source_api(source)
    contract = api.identity(Path(source))
    # Read strict JSON first, then retain the frozen source's authoritative gates.
    admission, training = read_json(admission_path), read_json(training_path)
    sidecar_path = Path(str(checkpoint) + ".json")
    sidecar = read_json(sidecar_path)
    if api.require_admission(admission_path, contract) != admission:
        raise ValueError("Admission changed while reading")
    if api.require_checkpoint(checkpoint, contract) != sidecar:
        raise ValueError("Checkpoint metadata changed while reading")
    runtime = validate_training(training, admission, sidecar, contract)
    inputs = {"checkpoint": digest(checkpoint), "checkpoint_sidecar": digest(sidecar_path),
              "admission": digest(admission_path), "training_report": digest(training_path)}
    return {"contract": contract, "runtime_manifest": runtime, "training": training,
            "admission": admission, "sidecar": sidecar, "input_sha256": inputs}


def require_same_inputs(before, after):
    if canonical(before) != canonical(after):
        raise ValueError("Source/checkpoint/admission/training evidence changed during capture")


def validate_dimensions(seconds, width, height, dt=.02):
    if (not math.isfinite(seconds) or not 10 <= seconds <= 20
            or any(type(v) is not int or v % 2 or not 256 <= v <= 1920 for v in (width, height))
            or not math.isfinite(dt) or dt <= 0):
        raise ValueError("Capture requires 10–20 seconds and even 256–1920 pixel dimensions")
    count = round(seconds / dt)
    if not math.isclose(count * dt, seconds, abs_tol=1e-9):
        raise ValueError("Duration must be an integer number of policy steps")
    return count


def check_frame(frame, width, height):
    import numpy as np
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.shape != (height, width, 3):
        raise ValueError("Renderer did not return the expected uint8 RGB frame")
    if int(frame.max()) <= 2 or int(frame.max()) - int(frame.min()) < 3:
        raise ValueError("Renderer returned a blank/unpopulated frame")
    return frame


def validate_states(states, count, dt):
    """Check transition alignment independently of the policy/renderer objects."""
    import numpy as np
    widths = {"joint_pos_rad": 30, "joint_vel_rad_s": 30, "root_pos_w_m": 3,
              "root_quat_w_xyzw": 4, "command_navigation": 3}
    shapes = {prefix+name: (count,width) for prefix in ("pre_","post_") for name,width in widths.items()}
    shapes.update({"observation_policy": (count,84), "policy_action": (count,18),
        "processed_target_endpoint_rad": (count,18), "post_applied_motor_torque_nm": (count,18),
        "camera_eye_target_w_m": (count,6), "reward": (count,), "done": (count,),
        "frame_index": (count,), "simulation_time_s": (count,)})
    if set(states) != set(shapes):
        raise ValueError("State archive fields do not match the reviewed transition schema")
    for key, shape in shapes.items():
        array = states[key]
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid state array: {key}")
    if not np.array_equal(states["frame_index"], np.arange(count)):
        raise ValueError("State/video frame indices are not continuous")
    if not np.allclose(states["simulation_time_s"], (np.arange(count)+1)*dt, rtol=0, atol=1e-12):
        raise ValueError("State/video simulation timestamps do not match policy time")
    if not np.array_equal(states["pre_command_navigation"], states["observation_policy"][:,9:12]):
        raise ValueError("Recorded command differs from the command actually supplied to the policy")
    if not np.isin(states["done"], [0,1]).all():
        raise ValueError("Episode-reset flags must be boolean")
    for prefix in ("pre_","post_"):
        if not np.allclose(np.linalg.norm(states[prefix+"root_quat_w_xyzw"],axis=-1),1.,rtol=0,atol=1e-5):
            raise ValueError("Recorded root quaternion is not normalized")
    return {"samples":count,"command_observation_alignment":True,"frame_time_alignment":True}


def validate_capture_report(path, verified, expected_frames):
    result = read_json(path)
    if (result.get("schema") != SCHEMA or result.get("pass") is not True or result.get("errors") != []
            or result.get("input_sha256") != verified["input_sha256"]
            or result.get("contract") != verified["contract"]
            or canonical(result.get("runtime_manifest")) != canonical(verified["runtime_manifest"])
            or result.get("frames_written") != expected_frames or result.get("controls_completed") != expected_frames
            or result.get("encoder_finalized") is not True or result.get("checkpoint_unchanged") is not True
            or result.get("policy_state_unchanged") is not True
            or result.get("physics_coverage_pass") is not True):
        raise ValueError("Capture report is missing, failed, incomplete, or has different lineage")
    for name in ("policy.mp4", "states.npz", "metadata.json"):
        record = result.get("artifacts", {}).get(name, {})
        artifact = Path(path).parent / name
        if not artifact.is_file() or record.get("sha256") != digest(artifact) or record.get("bytes") != artifact.stat().st_size:
            raise ValueError(f"Capture artifact is missing or changed: {name}")
    return result
