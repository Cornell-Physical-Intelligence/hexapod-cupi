"""Undisturbed quiet-stand and stop-to-stand review, isolated from training.

These are explicit simulation engineering bounds. Disturbance recovery requires
separate tests; this module never freezes joints or overwrites physical state.
"""

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from pathlib import Path
import numpy as np
import torch

from experiments.c_length_study.tools.omni_diagnostics import diagnostic_scenarios


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


@torch.inference_mode()
def evaluate_quiet_review(env, runner, plan, output, checkpoint_sha):
    from tensordict import TensorDict
    from experiments.c_length_study.tools.omni_flat_evaluation import save
    scenarios = diagnostic_scenarios()
    if env.num_envs % len(scenarios):
        raise ValueError("Quiet review requires a multiple of 12 environments")
    replicas = env.num_envs // len(scenarios)
    options = plan["omni"].get("quiet_stand_review", {})
    if set(options) - {"seed"}:
        raise ValueError("Quiet review accepts only seed; acceptance bounds are fixed in this version")
    seed = options.get("seed", 27057)
    if type(seed) is not int:
        raise ValueError("Quiet review seed must be integer")
    policy = runner.get_inference_policy(device=env.device)
    dt = env.step_dt
    output = Path(output)
    report = {"complete": False, "kind": "quiet_stand_and_stop_review_v1",
              "checkpoint_sha256": checkpoint_sha, "gates": QUIET_GATES,
              "seed": seed, "replicas_per_stop_case": replicas,
              "gate_status": "engineering simulation acceptance bounds; not measured hardware limits",
              "static": [], "transitions": [], "trace_files": [],
              "notes": ["Direct policy actions; no pose follower, rigid freezing, or body-position prescription.",
                        "Every termination/truncation invalidates its replica even if a later reset appears quiet.",
                        "Undisturbed quiet standing and recovery from disturbances are separate requirements."]}
    env.omni_diagnostic_enabled = True
    try:
        for trial, duration, moving_until in (("quiet_stand", 32., 0.), ("stop_to_stand", 21., 8.)):
            env.reset(seed=seed + (trial == "stop_to_stand"))
            env.episode_length_buf.zero_()
            targets = torch.tensor([row["command"] for row in scenarios], device=env.device).repeat_interleave(replicas, 0)
            snapshots = []
            first_zero = np.full(env.num_envs, -1, dtype=np.int64)
            for step in range(round(duration / dt)):
                requested = targets if step * dt < moving_until else torch.zeros_like(targets)
                env.set_evaluation_targets(requested)
                observations = TensorDict(env._get_observations(), batch_size=[env.num_envs])
                env.step(policy(observations))
                env.set_evaluation_targets(requested)
                snapshot = env.omni_diagnostic_sample
                if not all(np.isfinite(value).all() for value in snapshot.values()):
                    raise RuntimeError("Nonfinite quiet-review snapshot")
                snapshots.append(snapshot)
                if step * dt >= moving_until:
                    zero = np.max(np.abs(snapshot["command"]), axis=-1) < 1e-6
                    first_zero[(first_zero < 0) & zero] = step
            data = {key: np.stack([snapshot[key] for snapshot in snapshots]) for key in snapshots[0]}
            for env_index in range(env.num_envs):
                start = round(2. / dt) if trial == "quiet_stand" else int(first_zero[env_index]) + round(2. / dt)
                if first_zero[env_index] < 0:
                    raise RuntimeError("Stop command did not reach zero within the bounded trial")
                metrics = quiet_metrics(data, env_index, start, list(env._robot.joint_names), dt)
                if metrics["window_duration_s"] < (29.9 if trial == "quiet_stand" else 10.):
                    metrics["pass"] = False
                    metrics["failed_bounds"].append("quiet_window_too_short")
                report["static"].append({"name": trial,
                                         "preceding_command": ([0., 0., 0.] if trial == "quiet_stand" else scenarios[env_index // replicas]["command"]),
                                         "preceding_case": ("stand" if trial == "quiet_stand" else scenarios[env_index // replicas]["name"]),
                                         "replica": env_index % replicas,
                                         "environment_index": env_index,
                                         "zero_command_time_s": float(first_zero[env_index] * dt), **metrics})
            ids = [0] if trial == "quiet_stand" else list(range(0, env.num_envs, replicas))
            trace_path = output / f"{trial}_trace.npz"
            np.savez_compressed(trace_path, **{key: value[:, ids] for key, value in data.items()},
                                joint_names=np.array(env._robot.joint_names), trace_env_ids=np.array(ids),
                                time_s=(np.arange(len(snapshots)) + 1) * dt)
            report["trace_files"].append(trace_path.name)
            save(output / "quiet_stand.json", report)
    finally:
        env.omni_diagnostic_enabled = False
    report["complete"] = True
    report["all_scenarios_pass"] = all(row["pass"] for row in report["static"])
    report["stage2_complete"] = False
    save(output / "quiet_stand.json", report)
    # The guarded runner expects evaluation.json for a non-diagnostic evaluate
    # job. Its explicit kind distinguishes this review from the 154-row suite.
    save(output / "evaluation.json", report)
    return report
