"""Read-only CPU replay of the recorded evaluate_007 terminal predicate.

No simulator import, physics step, raw-file mutation or qualification override.
The only output is the explicitly requested, previously nonexistent JSON file.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import math
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
A = ROOT / "artifacts/restart_2026-09-14/paper_walk_execution_001"
RUN = A / "results_evaluate_007"
INPUTS = {}


def read(path):
    path = Path(path)
    value = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(value).hexdigest()
    return value


def document(path):
    return json.loads(read(path))


def arrays(path):
    import io
    with np.load(io.BytesIO(read(path)), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def audit():
    read(Path(__file__))
    report = document(RUN / "standing/evaluation/report.json")
    capture = document(RUN / "standing/evaluation/native400hz/capture.json")
    identity = document(RUN / "standing/identity.json")
    readback = document(RUN / "standing/native/native_readback.json")
    errors = document(RUN / "standing/native/native_errors.json")
    model = document(ROOT / "robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json")
    native = RUN / "standing/evaluation/native400hz"
    controls = arrays(RUN / "standing/evaluation/control_trace.npz")
    chunks = [arrays(native / name) for name in capture["substep_files"]]
    raw = {k: np.concatenate([x[k] for x in chunks]) for k in chunks[0]}
    count = len(raw["sequence"])
    joints = capture["joint_names"]
    named = {j["name"]: j for j in model["joints"]}
    limits = np.asarray([[named[n]["lower"], named[n]["upper"]] for n in joints], np.float32)
    q = raw["joint_position_rad"][:, 0]
    joint_bad = (q < limits[:, 0] - 2e-6) | (q > limits[:, 1] + 2e-6)
    root = raw["root_pose_xyzw"][:, 0]
    upright = 1 - 2 * (root[:, 3] ** 2 + root[:, 4] ** 2)
    tilt = np.arccos(np.clip(upright, -1, 1))
    height_bad = root[7::8, 2] < .045
    tilt_bad = -upright[7::8] > -math.cos(.85)
    terminal_joint_bad = joint_bad[7::8].any(-1)

    source = read(A / "source_013/env.py")
    config = read(A / "source_013/env_config.py")
    assert hashlib.sha256(source).hexdigest() == identity["physics_source_files"]["env.py"]
    assert hashlib.sha256(config).hexdigest() == identity["physics_source_files"]["env_config.py"]
    # Execute only the hash-bound, pure NumPy servo equation for this CPU audit;
    # the archived native module is never imported or initialized.
    tree = ast.parse(source)
    servo = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_diagnostic_servo")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[servo], type_ignores=[]), "bound_servo_equation", "exec"), namespace)
    kd = next(ast.literal_eval(n.value) for n in ast.parse(config).body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "KD" for t in n.targets))
    wanted = namespace["_diagnostic_servo"](
        *(raw[k].reshape(-1, 18) for k in ("pre_joint_position_rad", "pre_joint_velocity_rad_s", "joint_target_rad")),
        np.full(18, 12., np.float32), np.asarray(kd, np.float32))
    checks = {
        "finite_arrays": all(np.isfinite(v).all() for v in raw.values()) and all(np.isfinite(v).all() for v in controls.values()),
        "sequence": np.array_equal(raw["sequence"], np.arange(count)),
        "counter": np.array_equal(raw["explicit_counter"], np.arange(capture["initial_counter"] + 1, capture["final_counter"] + 1)),
        "control_index": np.array_equal(raw["control_index"], np.arange(count) // 8),
        "substep_index": np.array_equal(raw["substep_index"], np.arange(count) % 8),
        "pre_position_continuity": np.array_equal(raw["pre_joint_position_rad"][1:], raw["joint_position_rad"][:-1]),
        "pre_velocity_continuity": np.array_equal(raw["pre_joint_velocity_rad_s"][1:], raw["joint_velocity_rad_s"][:-1]),
        "actual_native_motor_input": np.array_equal(raw["native_input_pre_nm"], raw["applied_torque_nm"]),
        "terminal_predicate_matches": np.array_equal(controls["terminated"][:, 0], terminal_joint_bad | height_bad | tilt_bad),
        "native_errors_empty": errors == [],
        "capture_failure_absent": capture["failure"] is None,
    }
    for key in ("joint_position_rad", "joint_velocity_rad_s", "joint_target_rad", "root_pose_xyzw", "computed_torque_nm", "applied_torque_nm"):
        checks["control_endpoint_" + key] = np.array_equal(controls[key], raw[key][7::8])
    for key, expected in zip(("computed_torque_nm", "applied_torque_nm", "effort_ceiling_nm"), wanted):
        checks["exact_servo_" + key] = np.array_equal(raw[key].reshape(-1, 18), expected)
    patches = [json.loads(line) for line in read(native / "contacts.jsonl").splitlines()]
    checks["contact_sequence"] = np.array_equal([p["sequence"] for p in patches], raw["sequence"])
    checks["contact_counter"] = np.array_equal([p["explicit_counter"] for p in patches], raw["explicit_counter"])
    verified = 0
    for parent, receipt in ((RUN / "standing/evaluation", report), (native, capture)):
        for name, sha in receipt["files"].items():
            assert hashlib.sha256(read(parent / name)).hexdigest() == sha, name
            verified += 1
    camera = document(RUN / "standing/native/camera/receipt.json")
    events = []
    for step, joint in np.argwhere(joint_bad):
        event = {"sequence": int(step), "control_index": int(raw["control_index"][step]),
                 "substep_index": int(raw["substep_index"][step]), "time_s": float(raw["time_s"][step]),
                 "joint": joints[joint], "lower_rad": float(limits[joint, 0]), "upper_rad": float(limits[joint, 1]),
                 "below_lower_rad": float(limits[joint, 0] - q[step, joint]), "tolerance_rad": 2e-6}
        for key in ("pre_joint_position_rad", "pre_joint_velocity_rad_s", "joint_position_rad", "joint_velocity_rad_s", "joint_target_rad", "computed_torque_nm", "applied_torque_nm", "effort_ceiling_nm"):
            event[key] = float(raw[key][step, 0, joint])
        events.append(event)
    checks = {k: bool(v) for k, v in checks.items()}
    assert all(checks.values()), checks
    return {
        "schema": "canonical_evaluate007_terminal_cpu_audit_v1", "stage2_complete": False,
        "scope": "Read-only CPU terminal and recorder audit of the preserved failed native trial; no new physics or acceptance.",
        "conclusion": "The only terminal predicate is a left-middle tibia lower-limit crossing. Neither root-height nor tilt fall predicate triggered.",
        "inputs_sha256": INPUTS, "recorder_checks": checks, "verified_receipt_files": verified,
        "recorded_controls": len(controls["time_s"]), "recorded_native_steps": count,
        "joint_limit_events": events, "recorded_terminal_control_indices": np.flatnonzero(controls["terminated"][:, 0]).tolist(),
        "fall_height_control_indices": np.flatnonzero(height_bad).tolist(), "fall_tilt_control_indices": np.flatnonzero(tilt_bad).tolist(),
        "truncations": int(controls["truncated"].sum()), "resets_during_trial": int(controls["reset"].sum()),
        "terminal_time_s": float(controls["time_s"][-1]), "terminal_root_height_m": float(root[-1, 2]),
        "minimum_root_height_m": float(root[:, 2].min()), "fall_height_threshold_m": .045,
        "terminal_tilt_degrees": float(np.degrees(tilt[-1])), "maximum_tilt_degrees": float(np.degrees(tilt).max()),
        "fall_tilt_threshold_rad": .85,
        "physical_bounds": {
            "joint_violation_steps": int(joint_bad.any(-1).sum()),
            "speed_violation_steps": int((abs(raw["joint_velocity_rad_s"]) > np.asarray(readback["native_max_velocity"]) + 2e-6).any(-1).sum()),
            "recorded_nonfoot_steps": int(raw["nonfoot_contact"].sum()),
            "minimum_non_toe_floor_m": float(raw["minimum_non_toe_floor_m"].min()),
            "peak_requested_nm": float(abs(raw["computed_torque_nm"]).max()),
            "peak_applied_nm": float(abs(raw["applied_torque_nm"]).max()),
            "applied_cap_gate_nm": 1.60001,
            "requested_above_1_6_fraction_all_steps_and_joints": float((abs(raw["computed_torque_nm"]) > 1.6).mean()),
            "maximum_held_target_step_rad": float(abs(np.diff(controls["joint_target_rad"], axis=0)).max()),
            "original_native_motor_and_joint_pass": report["results"][0]["native_motor_and_joint_checks_pass"],
        },
        "camera": {"successful_captures_including_initialization": camera["successful_frames"],
                   "recorded_video_frames": report["video_frames"], "last_frame_physics_time_s": camera["last_frame"]["physics_time_s"],
                   "state_invariance_scope": camera["physics_state_comparison"], "robot_pose_writes": camera["robot_pose_writes"]},
        "limitations": ["This verifies saved native contact classifications and packet continuity; it does not rerun geometric patch classification.",
                        "It identifies the termination condition, not the causal origin of the oscillation or a physical joint-limit fix.",
                        "The requested 20-second trial stopped at13.38 seconds; original incomplete-window and behavior failures remain."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit()
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
