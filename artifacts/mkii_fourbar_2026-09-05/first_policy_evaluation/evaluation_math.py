"""Fixed scenarios and descriptive scoring; stdlib only, no simulator imports."""
from __future__ import annotations

import math
import statistics
import json

SEEDS = (101, 202, 303)
POLICY_DT_S = .02
STEPS = 500
FALL_REASONS = frozenset(("base_contact", "low_height", "upside_down"))
MODEL_INTEGRITY_REASONS = frozenset(("nonfinite_state", "invalid_contact", "invalid_motor_input",
                                   "invalid_motor_budget", "closure_coordinate_error"))


def scenarios():
    result = [{"name": "stand", "command": [0., 0., 0.], "reversal": False}]
    for axis, label, speed in ((0, "forward", .1), (1, "left", .1), (2, "yaw", .2)):
        for sign in (1, -1):
            command = [0., 0., 0.]
            command[axis] = sign * speed
            result.append({"name": label + ("_positive" if sign > 0 else "_negative"),
                           "command": command, "reversal": False})
    for x, y in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        result.append({"name": f"diagonal_{x:+d}_{y:+d}",
                       "command": [x*.1/math.sqrt(2), y*.1/math.sqrt(2), 0.], "reversal": False})
    for axis, label, speed in ((0, "forward", .1), (1, "left", .1), (2, "yaw", .2)):
        command = [0., 0., 0.]
        command[axis] = speed
        result.append({"name": label + "_reversal", "command": command, "reversal": True})
    return result


def scheduled_command(scenario, step):
    if type(step) is not int or not 0 <= step < STEPS:
        raise ValueError("Scenario step must be an integer in [0,500)")
    if scenario["name"] == "stand":
        return [0., 0., 0.], "stand"
    if step < 100:
        return [0., 0., 0.], "startup"
    if scenario["reversal"]:
        if step < 250:
            return list(scenario["command"]), "positive"
        if step < 400:
            return [-v for v in scenario["command"]], "negative"
    elif step < 300:
        return list(scenario["command"]), "motion"
    return [0., 0., 0.], "stop"


def quantiles(values):
    if not values:
        return None
    ordered = sorted(values)
    return {"median": statistics.median(ordered), "p95": ordered[math.ceil(.95*len(ordered))-1],
            "max": ordered[-1]}


def quaternion_attitude(q):
    """Internal WXYZ quaternion: body roll/pitch and anatomical forward heading."""
    w, x, y, z = q
    norm = math.sqrt(sum(v*v for v in q))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid root quaternion")
    w, x, y, z = (v/norm for v in (w, x, y, z))
    roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(max(-1., min(1., 2*(w*y-z*x))))
    # Anatomical forward is local -Y, not body +X.
    heading = math.atan2(-(1-2*(x*x+z*z)), -2*(x*y-w*z))
    return roll, pitch, heading


def native_xyzw_to_wxyz(q):
    """Isaac Lab 3 native root_quat_w is XYZW; conversion is always explicit."""
    if len(q) != 4 or not all(math.isfinite(value) for value in q):
        raise ValueError("Native quaternion must have four finite XYZW components")
    x, y, z, w = q
    return [w, x, y, z]


def verify_reset_identity(native_quaternions):
    """Check the actual reset buffer against this task's known identity rotation."""
    if not native_quaternions:
        raise ValueError("Missing native reset quaternions")
    converted = [native_xyzw_to_wxyz(q) for q in native_quaternions]
    error = max(max(abs(abs(q[0])-1), *(abs(value) for value in q[1:])) for q in converted)
    if error > 1e-6:
        raise ValueError("Native XYZW reset quaternion is not the frozen task's identity orientation")
    return {"native_format": "XYZW", "internal_format": "WXYZ", "checked_instances": len(converted),
            "max_identity_component_error": error, "pass": True}


def reference_step(pose, command, dt=POLICY_DT_S):
    """Exact planar body-twist integration: forward/left/yaw, metres/radians."""
    x, y, heading = pose
    forward, left, yaw = command
    angle = yaw*dt
    if abs(yaw) < 1e-10:
        dx, dy = forward*dt, left*dt
    else:
        dx = (forward*math.sin(angle) + left*(math.cos(angle)-1))/yaw
        dy = (forward*(1-math.cos(angle)) + left*math.sin(angle))/yaw
    return [x+math.cos(heading)*dx-math.sin(heading)*dy,
            y+math.sin(heading)*dx+math.cos(heading)*dy, heading+angle]


def _rms(values):
    return math.sqrt(sum(v*v for v in values)/len(values)) if values else None


def score_trial(scenario, seed, initial, rows):
    """Score until first terminal transition, including that pre-reset state."""
    json.dumps([initial, rows], allow_nan=False)
    accepted = []
    for index, row in enumerate(rows):
        if row["step"] != index:
            raise ValueError("Trial trace is not a contiguous sequence from step zero")
        expected, window = scheduled_command(scenario, index)
        if any(abs(a-b) > 1e-6 for a, b in zip(row["command"], expected)) or len(row["command"]) != 3:
            raise ValueError("Executed command differs from fixed scenario")
        if row["window"] != window:
            raise ValueError("Trial window differs from fixed scenario")
        accepted.append(row)
        if row["terminated"] or row["truncated"]:
            break
    if not accepted:
        raise ValueError("Cannot score an empty trial")
    ref = [*initial["position_world_m"][:2], quaternion_attitude(initial["quaternion_wxyz"])[2]]
    errors, cross_track, path_errors, windows = [[], [], []], [], [], {}
    prior_pos = initial["position_world_m"]
    stop_distance = 0.
    for row in accepted:
        ref = reference_step(ref, row["command"])
        measured = [*row["velocity_navigation_m_s"][:2], row["yaw_rate_rad_s"]]
        bucket = windows.setdefault(row["window"], {"count": 0, "measured_sum": [0., 0., 0.],
                                                     "squared_error_sum": [0., 0., 0.]})
        bucket["count"] += 1
        for axis in range(3):
            difference = measured[axis]-row["command"][axis]
            bucket["measured_sum"][axis] += measured[axis]
            bucket["squared_error_sum"][axis] += difference*difference
            if row["window"] in ("motion", "positive", "negative"):
                errors[axis].append(abs(difference))
        delta = [row["position_world_m"][i]-ref[i] for i in (0, 1)]
        path_errors.append(math.hypot(*delta))
        # Cross-track is only defined for translational reference segments.
        if math.hypot(*row["command"][:2]) > 0:
            track_heading = ref[2]+math.atan2(row["command"][1], row["command"][0])
            cross_track.append(abs(-math.sin(track_heading)*delta[0]+math.cos(track_heading)*delta[1]))
        if row["window"] == "stop":
            stop_distance += math.dist(prior_pos[:2], row["position_world_m"][:2])
        prior_pos = row["position_world_m"]
    for bucket in windows.values():
        bucket["duration_s"] = bucket["count"]*POLICY_DT_S
        bucket["mean_measured_forward_left_yaw"] = [v/bucket["count"] for v in bucket.pop("measured_sum")]
        bucket["rmse_forward_left_yaw"] = [math.sqrt(v/bucket["count"]) for v in bucket.pop("squared_error_sum")]
    reversal_delay = None
    if scenario["reversal"]:
        axis = next(i for i, v in enumerate(scenario["command"]) if v)
        for row in accepted:
            measured = [*row["velocity_navigation_m_s"][:2], row["yaw_rate_rad_s"]]
            if row["window"] == "negative" and measured[axis] < 0:
                reversal_delay = (row["step"]+1-250)*POLICY_DT_S
                break
    terminal = accepted[-1]
    attitudes = [quaternion_attitude(row["quaternion_wxyz"]) for row in accepted]
    accelerations = [math.dist(a["velocity_world_m_s"], b["velocity_world_m_s"])/POLICY_DT_S
                     for a, b in zip(accepted, accepted[1:])]
    return {"seed": seed, "scenario": scenario["name"], "steps_scored": len(accepted),
        "completed": len(accepted) == STEPS and not (terminal["terminated"] or terminal["truncated"]),
        "terminated": terminal["terminated"], "truncated": terminal["truncated"],
        "fall": bool(FALL_REASONS.intersection(terminal["termination_reasons"])),
        "model_integrity_termination_reasons": sorted(MODEL_INTEGRITY_REASONS.intersection(terminal["termination_reasons"])),
        "other_termination_reasons": sorted(set(terminal["termination_reasons"])-FALL_REASONS-MODEL_INTEGRITY_REASONS),
        "termination_reasons": terminal["termination_reasons"], "windows": windows,
        "absolute_motion_tracking_errors": {axis: quantiles(values) for axis, values in zip(("forward_m_s", "left_m_s", "yaw_rad_s"), errors)},
        "path_position_error_m": quantiles(path_errors), "cross_track_error_m": quantiles(cross_track),
        "stop_travel_m": stop_distance, "stop_final_speed_m_s": math.hypot(*terminal["velocity_navigation_m_s"][:2]) if terminal["window"] == "stop" else None,
        "reversal_delay_s": reversal_delay, "reversal_reached": reversal_delay is not None if scenario["reversal"] else None,
        "roll_rms_rad": _rms([a[0] for a in attitudes]), "pitch_rms_rad": _rms([a[1] for a in attitudes]),
        "base_acceleration_rms_m_s2_50hz": _rms(accelerations),
        "height_range_m": [min(row["height_m"] for row in accepted), max(row["height_m"] for row in accepted)],
        "support_count_50hz": quantiles([row["support_count"] for row in accepted]),
        "nonfoot_contact_control_samples": sum(row["nonfoot_contacts"] > 0 for row in accepted),
        "loaded_foot_slip_rms_m_s_50hz": _rms([row["loaded_foot_slip_rms_m_s"] for row in accepted]),
        "max_raw_demand_nm": max(row["max_raw_demand_nm"] for row in accepted),
        "max_applied_nm": max(row["max_applied_nm"] for row in accepted),
        "max_clipping_nm": max(row["max_clipping_nm"] for row in accepted),
        "max_continuous_overload_nm": max(row["max_continuous_overload_nm"] for row in accepted),
        "max_estimated_phase_current_arms_proxy": max(row["max_estimated_phase_current_arms"] for row in accepted),
        "min_headroom": min(row["min_headroom"] for row in accepted),
        "max_motor_speed_rad_s": max(row["max_motor_speed_rad_s"] for row in accepted),
        "max_motor_overload_exposure_s": terminal["max_motor_overload_exposure_s"],
        "max_motor_peak_exposure_s": terminal["max_motor_peak_exposure_s"],
        "reward_total": sum(row["reward"] for row in accepted),
        "reward_component_totals": {key: sum(row["reward_components"][key]*POLICY_DT_S for row in accepted)
                                    for key in terminal["reward_components"]}}
