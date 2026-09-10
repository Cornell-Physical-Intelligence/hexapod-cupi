"""Exact inherited quiet scorer AST; SDK rates remain raw. No old environment import."""
import numpy as np

QUIET_GATES = {
    "max_planar_excursion_m": .01,
    "max_heading_excursion_deg": 2.,
    "max_joint_velocity_rms_rad_s": .03,
    "max_joint_position_range_rad": .02,
    "max_target_step_abs_p95_rad_per_20ms": .002,
    "max_requested_torque_saturation_fraction": .005,
    "max_applied_torque_nm": 1.60001,
}

def quiet_metrics(data, env_index, start_step, joint_names, dt):
    """Score one contiguous window without deleting failures or restarting time."""
    take = lambda key: data[key][start_step:, env_index]
    q = take("joint_position_rad")
    target = take("joint_target_rad")
    velocity = take("joint_velocity_rad_s")
    position = take("position_world_m")
    quat = take("quaternion_world_wxyz")
    if len(q) < 2:
        raise ValueError("Quiet window needs at least two samples")
    w, x, y, z = quat.T
    heading = np.unwrap(np.arctan2(-1 + 2*(x*x+z*z), 2*(w*z-x*y)))
    joint_rms = np.sqrt(np.mean(velocity**2, axis=0))
    qrange = np.ptp(q, axis=0)
    target_p95 = np.quantile(np.abs(np.diff(target, axis=0)) * .02 / dt, .95, axis=0)
    requested = take("computed_torque_nm")
    if not all(np.isfinite(value).all() for value in (q, target, velocity, position, quat, requested, take("applied_torque_nm"))):
        raise ValueError("Nonfinite quiet-review state")
    row = {
        "window_samples": len(q), "window_duration_s": len(q) * dt,
        "max_planar_excursion_m": float(np.linalg.norm(position[:, :2] - position[0, :2], axis=1).max()),
        "max_heading_excursion_deg": float(np.degrees(np.abs(heading - heading[0]).max())),
        "max_joint_velocity_rms_rad_s": float(joint_rms.max()),
        "max_joint_position_range_rad": float(qrange.max()),
        "max_target_step_abs_p95_rad_per_20ms": float(target_p95.max()),
        "max_requested_torque_saturation_fraction": float((np.abs(requested) > 1.6).mean(0).max()),
        "mean_requested_torque_saturation_fraction": float((np.abs(requested) > 1.6).mean()),
        "max_applied_torque_nm": float(np.abs(take("applied_torque_nm")).max()),
        "requested_torque_abs_max_nm": float(np.abs(requested).max()),
        # Any failure anywhere in the trial invalidates the result, even if the
        # robot is quiet after an automatic reset inside/before the scored window.
        "terminations": int(data["terminated"][:, env_index].sum()),
        "truncations": int(data["truncated"][:, env_index].sum()),
        "joints": {name: {"velocity_rms_rad_s": float(joint_rms[j]),
                          "position_range_rad": float(qrange[j]),
                          "target_step_abs_p95_rad_per_20ms": float(target_p95[j]),
                          "saturation_fraction": float((np.abs(requested[:, j]) > 1.6).mean())}
                   for j, name in enumerate(joint_names)},
    }
    row["failed_bounds"] = [key for key, bound in QUIET_GATES.items() if row[key] > bound]
    row["pass"] = not row["failed_bounds"] and row["terminations"] == 0 and row["truncations"] == 0
    return row
