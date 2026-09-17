"""CPU evaluation of the new robot against explicitly named, unchanged screens.

The old C-study direction/transition screen, Stage2C forward screen, and the
canonical native standing screen are different contracts. This module reports
each separately. A video or successful PPO allocation never sets Stage2 complete.
Arrays are post-physics, before any reset, shaped [time, env, ...] (one-env
arrays [time, ...] are also accepted). Native quaternions are explicitly XYZW.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

DT = 0.02
PHYSICS_DT = 0.0025
SUBSTEPS = 8
URDF_SHA256 = "9492fde54c50170e940b49ccb3037d7b413743a5b77bc1d8707a539d44349e78"
QUIET_GATES = {
    "max_planar_excursion_m": .01,
    "max_heading_excursion_deg": 2.,
    "max_joint_velocity_rms_rad_s": .03,
    "max_joint_position_range_rad": .02,
    "max_target_step_abs_p95_rad_per_20ms": .002,
    "max_requested_torque_saturation_fraction": .005,
    "max_applied_torque_nm": 1.60001,
}
MOVING_RMS = {"base_height_std_m": .0035, "vertical_velocity_rms_mps": .070,
              "roll_pitch_angular_velocity_rms_radps": .28, "tilt_rms_degrees": .90}
STAND_RMS = {"base_height_std_m": .001, "vertical_velocity_rms_mps": .020,
             "roll_pitch_angular_velocity_rms_radps": .050, "tilt_rms_degrees": .50}
TAIL = {"base_height_peak_to_peak_m": .0105, "vertical_velocity_abs_p95_mps": .11,
        "roll_pitch_angular_velocity_p95_radps": .45, "tilt_p95_degrees": 1.40,
        "maximum_tilt_degrees": 1.70, "yaw_rate_rmse_radps": .080}
STAGE2C_TORQUE = {"peak_abs_computed_nm": 4.40, "computed_demand_over_rating_fraction": .15,
                 "maximum_computed_over_rating_burst_s": .14,
                 "max_per_joint_rms_applied_nm": 1.40,
                 "maximum_per_joint_computed_demand_over_rating_fraction": .25}

ACCEPTANCE_MATRIX = {
    "omni_static": {"source": "experiments/c_length_study/tools/omni_flat_math.py:scenario_gate",
                    "cases": 77, "duration_s": 20., "settle_s": 2.,
                    "scope": "Historical full signed direction, yaw and combined-command numeric screen."},
    "transition": {"source": "experiments/c_length_study/tools/omni_flat_evaluation.py:evaluate_omni",
                   "scope": "All 14 segments in one reset-free episode; last-second errors <= .03m/s and .08rad/s."},
    "quiet_stand": {"source": "canonical standing source005 standing_score.py + inherited quiet_metrics.py",
                    "duration_s": 20., "settle_s": 4., "quiet_window_s": 16.,
                    "scope": "Every original 400Hz distal contact, nonfoot, clearance, motor and native admission check remains required."},
    "stage2_long_quiet": {"source": "experiments/c_length_study/tools/omni_quiet_review.py:evaluate_quiet_review",
                          "duration_s": 32., "settle_s": 2., "minimum_quiet_window_s": 29.9,
                          "scope": "Historical Stage2 long quiet screen; separate from the20-second canonical standing admission. Preserve every sample in the30-second quiet window and every early failure."},
    "stop_to_stand": {"source": "experiments/c_length_study/tools/omni_quiet_review.py",
                      "duration_s": 21., "moving_until_s": 8., "settle_after_zero_s": 2.,
                      "minimum_quiet_window_s": 10.,
                      "scope": "No cropping to a convenient quiet suffix; score all samples after the fixed settling allowance."},
    "stage2c_formal": {"source": "isaaclab/grade_stage2c_stable_forward.py",
                       "seed": 60, "duration_s": 10., "warmup_controls": 25,
                       "measured_controls": 475, "target_slew_rad_per_20ms": .040,
                       "scope": "Historical forward absolute comparison only; old model/checkpoint-relative admission cannot transfer."},
    "visual": {"source": "ARCHITECTURE.md sections 2 and 6",
               "scope": "Compare every direction, start/reversal/stop with accepted forward gait; human acceptance is separate."},
}


def evaluation_scenarios():
    """The unchanged 77-case historical full omni command matrix."""
    rows = [{"name": "stand", "command": [0., 0., 0.]}]
    for speed in (.10, .20):
        for i in range(16):
            a = 2 * math.pi * i / 16
            rows.append({"name": f"translate_{speed:.2f}_{i*22.5:g}deg",
                         "command": [speed * math.cos(a), speed * math.sin(a), 0.]})
    for yaw in (-.4, -.2, .2, .4):
        rows.append({"name": f"turn_{yaw:+.2f}", "command": [0., 0., yaw]})
    for i in range(4):
        a = math.pi * i / 2
        for yaw in (-.2, .2):
            rows.append({"name": f"combined_{i*90}deg_{yaw:+.2f}",
                         "command": [.10 * math.cos(a), .10 * math.sin(a), yaw]})
    for speed, yaw_magnitude, label in ((.15, .05, 'gentle_arc'), (.20, .40, 'tight_arc')):
        for i in range(8):
            a = math.pi * i / 4
            for yaw in (-yaw_magnitude, yaw_magnitude):
                rows.append({'name': f'{label}_{i*45}deg_{yaw:+.2f}',
                             'command': [speed*math.cos(a), speed*math.sin(a), yaw]})
    return rows


def transition_sequence():
    return [
        ("stand", 3., [0., 0., 0.]), ("forward", 4., [.10, 0., 0.]),
        ("left", 4., [0., .10, 0.]), ("reverse", 4., [-.10, 0., 0.]),
        ("right", 4., [0., -.10, 0.]), ("turn_left", 4., [0., 0., .25]),
        ("turn_right", 4., [0., 0., -.25]),
        ("diagonal_turn_left", 4., [.071, .071, .20]),
        ("reverse_diagonal_turn_right", 4., [-.071, -.071, -.20]),
        ("gentle_arc", 6., [.15, 0., .05]), ("strafe_arc", 6., [0., .10, .15]),
        ("s_curve", 10., [.12, 0., 0.]), ("fixed_heading_bend", 8., [.10, 0., 0.]),
        ("stop", 5., [0., 0., 0.]),
    ]


def trajectory_command(name, elapsed, duration, command):
    if name == 's_curve':
        return [.12, 0., .15 * math.sin(2 * math.pi * elapsed / duration)]
    if name == 'fixed_heading_bend':
        angle = .5 * math.pi * elapsed / duration
        return [.10 * math.cos(angle), .10 * math.sin(angle), 0.]
    return list(command)


def _array(data, name, width, env_index):
    if name not in data:
        return None
    a = np.asarray(data[name])
    rank = 1 if width is None else 2
    if a.ndim == rank + 1:
        a = a[:, env_index]
    if a.ndim != rank or (width is not None and a.shape[-1] != width):
        raise ValueError(f"Malformed {name}: expected [T,env,{width}] or one-env equivalent")
    if not np.isfinite(a).all():
        raise ValueError(f"Nonfinite {name}")
    return a


def _check(report, name, value, bound, *, minimum=False):
    if value is None:
        report["checks"][name] = {"status": "missing", "bound": bound}
    else:
        passed = value >= bound if minimum else value <= bound
        report["checks"][name] = {"status": "pass" if passed else "fail", "value": value,
                                  "bound": bound, "operator": ">=" if minimum else "<="}


def _finish(report):
    report["failed_bounds"] = [k for k, v in report["checks"].items() if v["status"] == "fail"]
    report["missing_evidence"] = [k for k, v in report["checks"].items() if v["status"] == "missing"]
    report["pass"] = not report["failed_bounds"] and not report["missing_evidence"]
    report["stage2_complete"] = False
    return report


def _longest(mask):
    best = np.zeros(mask.shape[1], dtype=int)
    current = np.zeros_like(best)
    for row in mask:
        current = np.where(row, current + 1, 0)
        best = np.maximum(best, current)
    return int(best.max())


def adapt_recording_fields(data: Mapping):
    """Adapt the frozen native recorder's names; never edit or discard raw fields."""
    out = dict(data)
    aliases = {"linear_velocity_nav": "velocity_navigation_mps",
               "angular_velocity_body": "gyro_body_rad_s",
               "torque_square_sum_400hz": "applied_torque_squared_sum_400hz",
               "requested_torque_abs_max_400hz": "computed_torque_abs_max_400hz"}
    for old, new in aliases.items():
        if old in data:
            if new in data and not np.array_equal(data[old], data[new]):
                raise ValueError(f"Conflicting native telemetry alias: {old}/{new}")
            out[new] = data[old]
    # Current native compact recording stores per-joint maxima over each hold.
    # Reduce joints only for the all-joint cap check; preserve every original array.
    one_env = np.asarray(data["root_pose_xyzw"]).ndim == 2
    for key in ("computed_torque_abs_max_400hz", "applied_torque_abs_max_400hz"):
        if key in out:
            value = np.asarray(out[key])
            if value.ndim == 3 or (one_env and value.ndim == 2 and value.shape[-1] == 18):
                if value.shape[-1] != 18:
                    raise ValueError("Native torque maxima have wrong joint count")
                out[key] = value.max(axis=-1)
    return out


def score_recording(data: Mapping, metadata: Mapping):
    """Score one replica. Required metadata: profile, env_index (default 0).

    Optional ``command`` is the held target (not a post-reset resample).
    ``contact_classification`` must be ``exact_distal_points`` to use supplied
    nonfoot/distal classifications. Tibia force is not a distal-foot contact.
    This function never resets, discards terminals, or silently extends a window.
    """
    data = adapt_recording_fields(data)
    profile = metadata["profile"]
    if profile not in {"omni_static", "quiet_stand", "stage2_long_quiet", "stop_to_stand", "stage2c_formal"}:
        raise ValueError("Unknown profile; continuous sequences use score_transitions")
    env = int(metadata.get("env_index", 0))
    root = _array(data, "root_pose_xyzw", 7, env)
    if root is None or len(root) < 2:
        raise ValueError("At least two recorded root poses are required")
    n = len(root)
    expected = {"omni_static": 1000, "quiet_stand": 1000, "stage2_long_quiet": 1600, "stop_to_stand": 1050,
                "stage2c_formal": 500}[profile]
    start = {"omni_static": 100, "quiet_stand": 200, "stage2_long_quiet": 100, "stop_to_stand": 500,
             "stage2c_formal": 25}[profile]
    r = {"schema": "canonical_stage2_cpu_evaluation_v1", "profile": profile,
         "scope": ACCEPTANCE_MATRIX[profile], "environment_index": env,
         "recorded_controls": n, "checks": {}, "metrics": {},
         "lineage": {k: metadata.get(k) for k in ("urdf_sha256", "checkpoint_sha256", "source_sha256")}}
    _check(r, "complete_requested_window", abs(n - expected), 0)
    _check(r, "control_dt_s", abs(float(metadata.get("control_dt_s", DT)) - DT), 1e-12)
    qnorm = np.linalg.norm(root[:, 3:], axis=-1)
    _check(r, "quaternion_unit_norm", float(abs(qnorm - 1).max()), 1e-5)
    for flag in ("terminated", "truncated", "reset"):
        a = _array(data, flag, None, env)
        if a is not None and (len(a) != n or not np.isin(a, (0, 1)).all()):
            raise ValueError(f"Invalid {flag} flags")
        _check(r, flag, None if a is None else int(a.sum()), 0)
    times = _array(data, "time_s", None, env)
    if times is not None and len(times) != n:
        raise ValueError("Timestamp length differs from trace")
    _check(r, "contiguous_time", None if times is None else float(abs(np.diff(times) - DT).max()), 1e-9)
    command = _array(data, "command", 3, env)
    if command is None:
        _check(r, "recorded_command", None, 0)
    elif len(command) != n:
        raise ValueError("Command length differs from trace")
    target = np.asarray(metadata.get("command", command[0] if command is not None else [0., 0., 0.]), float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("Invalid command")
    if profile == "stop_to_stand" and command is not None:
        # Preserve the preceding commanded direction in this stop-case identity.
        # Whether that direction was actually tracked is its separate motion gate.
        _check(r, "preceding_command", None if n < 400 else float(abs(command[100:400] - target).max()), 1e-6)
        r["preceding_command"] = target.tolist()
        r["preceding_motion_scope"] = "Quiet recovery only; the matching signed motion case must independently pass."
        eligible = np.flatnonzero((np.arange(n) >= 400) & (np.max(abs(command), axis=-1) < 1e-6))
        start = int(eligible[0]) + 100 if len(eligible) else n
        _check(r, "zero_command_continuous", None if start >= n else float(abs(command[start:]).max()), 1e-6)
    elif profile in {"quiet_stand", "stage2_long_quiet"} and command is not None:
        _check(r, "zero_command_continuous", float(abs(command).max()), 1e-6)
    if profile in {"omni_static", "stage2c_formal"} and command is not None:
        # The initial command ramp may settle; every measured command remains the declared target.
        _check(r, "held_command", float(abs(command[min(start, n-1):] - target).max()), 1e-6)
    if profile in {"quiet_stand", "stage2_long_quiet", "stop_to_stand"}:
        required_quiet = {"quiet_stand": 800, "stage2_long_quiet": 1495, "stop_to_stand": 500}[profile]
        _check(r, "quiet_window_controls", max(n - start, 0), required_quiet, minimum=True)
    r["window_start_control"] = start
    r["window_controls"] = max(n - start, 0)
    if n <= start:
        return _finish(r)
    sel = slice(start, n)
    fields = {name: _array(data, name, width, env) for name, width in (
        ("velocity_navigation_mps", 3), ("velocity_world_mps", 3), ("gyro_body_rad_s", 3),
        ("joint_position_rad", 18), ("joint_velocity_rad_s", 18), ("joint_target_rad", 18),
        ("computed_torque_nm", 18), ("applied_torque_nm", 18),
        ("saturation_count_400hz", 18), ("applied_torque_squared_sum_400hz", 18),
        ("computed_torque_abs_max_400hz", None), ("applied_torque_abs_max_400hz", None),
        ("nonfoot_contact_count_400hz", None), ("missing_six_toe_count_400hz", None),
        ("minimum_non_toe_floor_m", None), ("nonfoot_contact", None))}
    if any(v is not None and len(v) != n for v in fields.values()):
        raise ValueError("Telemetry length differs from root trace")
    if fields["velocity_world_mps"] is None and "linear_velocity_body" in data:
        body = _array(data, "linear_velocity_body", 3, env)
        if len(body) != n:
            raise ValueError("Body velocity length differs from root trace")
        vector, scalar = root[:, 3:6], root[:, 6:7]
        twice_cross = 2*np.cross(vector, body)
        fields["velocity_world_mps"] = body + scalar*twice_cross + np.cross(vector, twice_cross)
        r["derived_telemetry"] = {"velocity_world_mps": "Rigid rotation of recorded native body-root-origin velocity using recorded XYZW; no differentiation or simulation."}
        r["velocity_measurement_point"] = "body_root_origin; historical COM-velocity comparisons require a separately verified measurement-point binding"
    # Statistics always retain the full declared window, including terminal samples.
    pose = root[sel]
    x, y, z, w = pose[:, 3:].T
    tilt = np.arccos(np.clip(1 - 2*(x*x + y*y), -1., 1.))
    heading = np.unwrap(np.arctan2(-1 + 2*(x*x + z*z), 2*(w*z - x*y)))
    m = r["metrics"]
    m.update(base_height_std_m=float(np.std(pose[:, 2])),
             base_height_peak_to_peak_m=float(np.ptp(pose[:, 2])),
             tilt_rms_degrees=float(np.degrees(np.sqrt(np.mean(tilt**2)))),
             tilt_p95_degrees=float(np.degrees(np.quantile(tilt, .95))),
             maximum_tilt_degrees=float(np.degrees(tilt.max())))
    v, vw, gyro = (fields[k] for k in ("velocity_navigation_mps", "velocity_world_mps", "gyro_body_rad_s"))
    if vw is not None:
        m["vertical_velocity_rms_mps"] = float(np.sqrt(np.mean(vw[sel, 2]**2)))
        m["vertical_velocity_abs_p95_mps"] = float(np.quantile(abs(vw[sel, 2]), .95))
    if v is not None:
        error = np.linalg.norm(v[sel, :2] - target[:2], axis=-1)
        m.update(mean_velocity_mps=v[sel].mean(0).tolist(), planar_error_mps=float(error.mean()),
                 planar_velocity_rmse_mps=float(np.sqrt(np.mean(error**2))))
    if gyro is not None:
        error = gyro[sel, 2] - target[2]
        rp = np.linalg.norm(gyro[sel, :2], axis=-1)
        m.update(yaw_error_rad_s=float(abs(error).mean()), yaw_rate_rmse_radps=float(np.sqrt(np.mean(error**2))),
                 mean_yaw_rate_rad_s=float(gyro[sel, 2].mean()),
                 roll_pitch_angular_velocity_rms_radps=float(np.sqrt(np.mean(rp**2))),
                 roll_pitch_angular_velocity_p95_radps=float(np.quantile(rp, .95)))
    requested, applied = fields["computed_torque_nm"], fields["applied_torque_nm"]
    sat = fields["saturation_count_400hz"]
    if sat is not None:
        if np.any((sat < 0) | (sat > SUBSTEPS) | (sat != np.floor(sat))):
            raise ValueError("Invalid native substep saturation counts")
        m["requested_saturation_fraction_400hz"] = float(sat[sel].mean()/SUBSTEPS)
        m["maximum_requested_saturation_fraction_400hz"] = float((sat[sel].mean(0)/SUBSTEPS).max())
    if requested is not None:
        over = abs(requested[sel]) > 1.6
        m.update(computed_demand_over_rating_fraction=float(over.mean()),
                 maximum_per_joint_computed_demand_over_rating_fraction=float(over.mean(0).max()),
                 maximum_computed_over_rating_burst_s=_longest(over)*DT,
                 peak_abs_computed_nm=float(abs(requested[sel]).max()))
    if applied is not None:
        m["max_per_joint_rms_applied_nm"] = float(np.sqrt(np.mean(applied[sel]**2, axis=0)).max())
    exact = metadata.get("contact_classification") == "exact_distal_points"
    nonfoot = fields["nonfoot_contact"] if exact else None
    m["nonfoot_fraction"] = None if nonfoot is None else float(nonfoot[sel].mean())
    if profile == "omni_static":
        bounds = {"planar_error_mps": max(.025, .25*math.hypot(*target[:2])),
                  "yaw_error_rad_s": max(.06, .25*abs(target[2])), "tilt_rms_degrees": 5.,
                  "vertical_velocity_rms_mps": .04, "nonfoot_fraction": .001,
                  "computed_demand_over_rating_fraction": .005}
        for key, bound in bounds.items():
            _check(r, key, m.get(key), bound)
    elif profile == "stage2c_formal":
        _check(r, "seed_60", None if "seed" not in metadata else abs(metadata["seed"] - 60), 0)
        _check(r, "formal_limiter", None if "target_slew_rad" not in metadata else abs(metadata["target_slew_rad"] - .040), 1e-12)
        if not any(np.allclose(target, c, rtol=0, atol=1e-12) for c in ((0,0,0),(.16,0,0),(.20,0,0),(.30,0,0))):
            raise ValueError("Command is not a historical Stage2C formal command")
        for key, bound in {**(STAND_RMS if not target.any() else MOVING_RMS), **TAIL,
                           **STAGE2C_TORQUE, "planar_velocity_rmse_mps": .10}.items():
            _check(r, key, m.get(key), bound)
        if v is not None:
            speed = float(v[sel, 0].mean())
            if target.any():
                _check(r, "forward_fraction_minimum", speed, .8*target[0], minimum=True)
                _check(r, "forward_fraction_maximum", speed, 1.3*target[0])
            else:
                _check(r, "stand_planar_speed", float(np.linalg.norm(v[sel, :2].mean(0))), .03)
                yaw_mean = m.get("mean_yaw_rate_rad_s")
                _check(r, "stand_abs_yaw_rate", None if yaw_mean is None else abs(yaw_mean), .08)
        else:
            _check(r, "command_achievement", None, 0)
        r["historical_relative_comparison"] = "Not transferable: old seed/model cannot qualify this model; no synthetic baseline used."
    else:
        q, dq, targets = (fields[k] for k in ("joint_position_rad", "joint_velocity_rad_s", "joint_target_rad"))
        m.update(max_planar_excursion_m=float(np.linalg.norm(pose[:, :2] - pose[0, :2], axis=-1).max()),
                 max_heading_excursion_deg=float(np.degrees(abs(heading - heading[0]).max())))
        if q is not None:
            m["max_joint_position_range_rad"] = float(np.ptp(q[sel], axis=0).max())
        if dq is not None:
            m["max_joint_velocity_rms_rad_s"] = float(np.sqrt(np.mean(dq[sel]**2, axis=0)).max())
        if targets is not None and len(targets[sel]) > 1:
            m["max_target_step_abs_p95_rad_per_20ms"] = float(np.quantile(abs(np.diff(targets[sel], axis=0)), .95, axis=0).max())
        if requested is not None:
            m["max_requested_torque_saturation_fraction"] = float((abs(requested[sel]) > 1.6).mean(0).max())
        if applied is not None:
            m["max_applied_torque_nm"] = float(abs(applied[sel]).max())
        for key, bound in QUIET_GATES.items():
            _check(r, key, m.get(key), bound)
        _check(r, "requested_saturation_400hz", m.get("maximum_requested_saturation_fraction_400hz"), .005)
        _check(r, "requested_saturation_mean_400hz", m.get("requested_saturation_fraction_400hz"), .005)
        cap = fields["applied_torque_abs_max_400hz"]
        _check(r, "applied_cap_all_substeps", None if cap is None else float(cap.max()), 1.60001)
        for key, window in (("missing_six_toe_count_400hz", sel), ("nonfoot_contact_count_400hz", slice(None))):
            a = fields[key] if exact else None
            if a is not None and np.any((a < 0) | (a > SUBSTEPS) | (a != np.floor(a))):
                raise ValueError("Invalid native contact substep counts")
            _check(r, key, None if a is None else int(a[window].sum()), 0)
        clearance = fields["minimum_non_toe_floor_m"] if exact else None
        _check(r, "minimum_non_toe_floor_m", None if clearance is None else float(clearance.min()), -.001, minimum=True)
        _check(r, "minimum_plate_height_m", float(root[:, 2].min()), .055, minimum=True)
    r["native_admission"] = "Separate exact-model native 1/32 standing, solver, geometry, full400Hz and asset receipts are required."
    return _finish(r)


def score_transitions(data: Mapping, metadata: Mapping):
    """Score the complete original uninterrupted 14-segment command program."""
    data = adapt_recording_fields(data)
    env = int(metadata.get("env_index", 0))
    v = _array(data, "velocity_navigation_mps", 3, env)
    gyro = _array(data, "gyro_body_rad_s", 3, env)
    command = _array(data, "command", 3, env)
    if v is None or gyro is None or command is None or len(v) != len(gyro) or len(v) != len(command):
        raise ValueError("Complete velocity, gyro and command traces required")
    n = len(v)
    r = {"schema": "canonical_stage2_cpu_evaluation_v1", "profile": "transition",
         "scope": ACCEPTANCE_MATRIX["transition"], "checks": {}, "segments": []}
    _check(r, "complete_program", abs(n - round(sum(s[1] for s in transition_sequence())/DT)), 0)
    for name in ("terminated", "truncated", "reset"):
        flags = _array(data, name, None, env)
        if flags is not None and (len(flags) != n or not np.isin(flags, (0, 1)).all()):
            raise ValueError(f"Invalid {name} flags")
        _check(r, name, None if flags is None else int(flags.sum()), 0)
    times = _array(data, "time_s", None, env)
    if times is not None and len(times) != n:
        raise ValueError("Timestamp length differs from trace")
    _check(r, "contiguous_time", None if times is None else float(abs(np.diff(times)-DT).max()), 1e-9)
    offset = 0
    for name, duration, target in transition_sequence():
        count = round(duration/DT)
        if offset + count > n:
            _check(r, name + ".complete_segment", 1, 0)
            break
        expected = np.array([trajectory_command(name, i*DT, duration, target) for i in range(count)])
        # The original command smoother may lag the requested trajectory. Validate
        # requested_command separately and compare late motion with its live target.
        requested = _array(data, "requested_command", 3, env)
        if requested is not None and len(requested) != n:
            raise ValueError("Requested command length differs from trace")
        _check(r, name + ".requested_command", None if requested is None else float(abs(requested[offset:offset+count]-expected).max()), 1e-6)
        late = slice(offset+count-50, offset+count)
        lp = float(np.linalg.norm(v[late, :2]-expected[-50:, :2], axis=-1).mean())
        ly = float(abs(gyro[late, 2]-expected[-50:, 2]).mean())
        _check(r, name + ".last_second_planar_error_mps", lp, .03)
        _check(r, name + ".last_second_yaw_error_rad_s", ly, .08)
        r["segments"].append({"name": name, "duration_s": duration,
                              "last_second_planar_error_mps": lp, "last_second_yaw_error_rad_s": ly})
        offset += count
    return _finish(r)


def summarize_suite(results, *, required_cases, evidence=None):
    """Fail closed on missing/duplicate cases and external qualification evidence.

    ``required_cases`` is frozen before dispatch, never inferred from successes.
    Evidence values are independently verified receipts supplied by the owner.
    """
    evidence = dict(evidence or {})
    names = [x["case_id"] for x in results]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate evaluation case identity")
    if len(required_cases) != len(set(required_cases)):
        raise ValueError("Duplicate required case identity")
    missing = sorted(set(required_cases) - set(names))
    extra = sorted(set(names) - set(required_cases))
    prerequisites = ("exact_model_bound", "matching_single_standing", "matching_batch_standing",
                     "full_direction_matrix", "full_transition_matrix", "all_stop_cases",
                     "exact_contact_and_motor_evidence", "formal_forward_comparison", "visual_comparison_accepted")
    absent = [k for k in prerequisites if evidence.get(k) is not True]
    failed = [x["case_id"] for x in results if x.get("pass") is not True]
    passed = bool(results) and not (missing or extra or failed or absent)
    return {"schema": "canonical_stage2_suite_v1", "stage2_complete": passed,
            "required_cases": list(required_cases), "missing_cases": missing,
            "unexpected_cases": extra, "failed_cases": failed, "missing_qualification_evidence": absent,
            "scope": "Simulation Stage2 only; no hardware or terrain admission."}
