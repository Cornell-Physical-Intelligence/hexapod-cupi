"""Short, pre-reset motor/observation probe; does not replace qualification.

Only NumPy/Torch are imported until the Isaac rollout is requested, so the
configuration and reset-exclusion rules remain testable on a CPU host.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch


def diagnostic_scenarios():
    return [
        {"name": name, "command": command}
        for name, command in (
            ("stand", [0., 0., 0.]), ("forward", [.10, 0., 0.]),
            ("reverse", [-.10, 0., 0.]), ("left", [0., .10, 0.]),
            ("right", [0., -.10, 0.]), ("forward_fast", [.20, 0., 0.]),
            ("turn_left", [0., 0., .20]), ("turn_right", [0., 0., -.20]),
            ("arc_left", [.10, 0., .20]), ("arc_right", [.10, 0., -.20]),
            ("strafe_arc", [0., .10, .20]), ("diagonal", [.071, .071, 0.]),
        )
    ]


def diagnostic_options(value):
    if not isinstance(value, dict):
        raise ValueError("omni.diagnostics must be an object")
    allowed = {"duration_s", "settle_s", "seed", "trace_envs_per_scenario", "controller"}
    if set(value) - allowed:
        raise ValueError(f"Unknown diagnostic options: {sorted(set(value) - allowed)}")
    result = {"duration_s": 12., "settle_s": 2., "seed": 7057,
              "trace_envs_per_scenario": 1, "controller": "policy", **value}
    if not (4 <= result["duration_s"] <= 30 and .5 <= result["settle_s"] < result["duration_s"] - 1):
        raise ValueError("Diagnostic duration must be 4–30 s; settle must be >=0.5 s and leave >1 s")
    if type(result["seed"]) is not int or type(result["trace_envs_per_scenario"]) is not int:
        raise ValueError("Diagnostic seed and trace_envs_per_scenario must be integers")
    if not 1 <= result["trace_envs_per_scenario"] <= 8:
        raise ValueError("Trace replicas must be 1–8")
    if result["controller"] not in ("policy", "zero"):
        raise ValueError("Diagnostic controller must be policy or zero")
    return result


def apply_omni_overrides(cfg, overrides):
    """Explicit, bounded experiments. Actuator caps and failure gates are immutable."""
    if not isinstance(overrides, dict):
        raise ValueError("omni.overrides must be an object")
    allowed = {"target_slew_rad_per_20ms", "reward_weights", "external_forces_every_iteration",
               "observation_noise_scale", "target_filter_time_constant_s"}
    if set(overrides) - allowed:
        raise ValueError(f"Unknown omni overrides: {sorted(set(overrides) - allowed)}")
    if "target_slew_rad_per_20ms" in overrides:
        value = overrides["target_slew_rad_per_20ms"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not .01 <= value <= .06:
            raise ValueError("Target slew must remain within 0.01–0.06 rad per 20 ms")
        cfg.processed_joint_target_slew_limit_rad_per_20ms = float(value)
    for key, field, lower, upper in (
        ("observation_noise_scale", "omni_observation_noise_scale", 0., 1.),
        ("target_filter_time_constant_s", "omni_target_filter_time_constant_s", 0., .15),
    ):
        if key in overrides:
            value = overrides[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{key} must be finite and within {lower}–{upper}")
            setattr(cfg, field, float(value))
    weights = overrides.get("reward_weights", {})
    if not isinstance(weights, dict):
        raise ValueError("reward_weights must be an object")
    if set(weights) - set(cfg.omni_reward_weights):
        raise ValueError("Reward overrides may only change existing omni terms")
    for key, value in weights.items():
        original = cfg.omni_reward_weights[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid reward weight {key}")
        # Keep cost signs, and bound strength to avoid accidental unit errors.
        # Progress bonuses may be removed to test whether they encourage speed bias.
        if original < 0 and not 10 * original <= value <= original / 10:
            raise ValueError(f"Cost {key} must keep its sign and stay within 0.1–10x default")
        if original > 0 and not 0 <= value <= 4 * original:
            raise ValueError(f"Bonus {key} must be nonnegative and at most 4x default")
        if original == 0:
            bounds = {"stand_joint_velocity": (-2., 0.), "stand_target_velocity": (-1., 0.), "stand_raw_action": (-4., 0.)}
            if key not in bounds or not bounds[key][0] <= value <= bounds[key][1]:
                raise ValueError(f"Opt-in reward {key} exceeds its explicit nonpositive bounds")
        cfg.omni_reward_weights[key] = float(value)
    if "external_forces_every_iteration" in overrides:
        value = overrides["external_forces_every_iteration"]
        if type(value) is not bool:
            raise ValueError("external_forces_every_iteration must be boolean")
        # Filled against the installed Isaac API; never silently invent a field.
        physics = cfg.sim.physics
        field = "enable_external_forces_every_iteration"
        if not hasattr(physics, field):
            raise ValueError(f"Installed PhysxCfg does not expose {field}; numerical probe unavailable")
        setattr(physics, field, value)


def sample_masks(age_s, terminated, settle_s):
    """Every reset has its own settling interval; terminal samples remain auditable."""
    age = np.asarray(age_s)
    terminal = np.asarray(terminated, dtype=bool)
    return {"all": np.ones(age.shape, dtype=bool), "post_settle": age >= settle_s,
            "post_settle_nonterminal": (age >= settle_s) & ~terminal,
            "reset_transient": age < settle_s}


def capture_step(env, terms):
    """Called from rewards before command resampling and automatic reset."""
    d = env._robot.data
    from isaaclab.utils.math import quat_apply_inverse, quat_apply
    fd_world = (d.root_pos_w.torch - env.omni_diagnostic_start_position) / env.step_dt
    fd_body = quat_apply_inverse(d.root_quat_w.torch, fd_world)
    forward_axis = torch.tensor([0., -1., 0.], device=env.device).expand(env.num_envs, 3)
    forward_before = quat_apply(env.omni_diagnostic_start_quaternion, forward_axis)
    forward_after = quat_apply(d.root_quat_w.torch, forward_axis)
    heading_delta = torch.atan2(forward_after[:, 1], forward_after[:, 0]) - torch.atan2(forward_before[:, 1], forward_before[:, 0])
    fd_heading_rate = torch.atan2(heading_delta.sin(), heading_delta.cos()) / env.step_dt
    forces = env._base_contact_sensor.data.net_forces_w_history.torch
    base_contact = forces.norm(dim=-1).amax(1)[:, 0] > 5.
    limit = float(env.cfg.terminate_on_computed_torque_demand_duration_s)
    sample = {
        "age_s": env._episode_elapsed_s + env.step_dt,
        "command": env._commands,
        "target_command": env.omni_targets,
        "joint_position_rad": d.joint_pos.torch,
        "joint_velocity_rad_s": d.joint_vel.torch,
        "joint_target_rad": env._processed_actions,
        "action": env._actions,
        "computed_torque_nm": d.computed_torque.torch,
        "applied_torque_nm": d.applied_torque.torch,
        "velocity_navigation_mps": env._vector_in_command_frame(d.root_lin_vel_b.torch),
        "finite_difference_velocity_navigation_mps": env._vector_in_command_frame(fd_body),
        "gyro_navigation_rad_s": env._vector_in_command_frame(d.root_ang_vel_b.torch),
        "finite_difference_heading_rate_rad_s": fd_heading_rate,
        "position_world_m": d.root_pos_w.torch,
        "quaternion_world_wxyz": d.root_quat_w.torch,
        "projected_gravity": d.projected_gravity_b.torch,
        "terminated": env.reset_terminated,
        "truncated": env.reset_time_outs,
        "reason_base_contact": base_contact,
        "reason_too_low": d.root_pos_w.torch[:, 2] < .055,
        "reason_upside_down": d.projected_gravity_b.torch[:, 2] > -.45,
        "reason_torque_duration": env._torque_demand_excess_duration_s >= limit,
        "target_slew_limited_fraction": env._joint_target_slew_limited_fraction,
    }
    sample.update({"reward_term_" + key: value for key, value in terms.items()})
    if hasattr(env, "omni_raw_policy_action"):
        sample["raw_policy_action"] = env.omni_raw_policy_action
    if hasattr(env, "omni_filtered_target"):
        limits = d.soft_joint_pos_limits.torch
        requested = (env.cfg.action_scale * env._actions + d.default_joint_pos.torch).clamp(limits[:, :, 0], limits[:, :, 1])
        sample["filtered_target_rad"] = (env.omni_filtered_target if env.cfg.omni_target_filter_time_constant_s > 0 else requested)
        sample["unfiltered_target_rad"] = requested
    # A CPU copy here fixes the pre-reset state rather than retaining live views.
    env.omni_diagnostic_sample = {key: value.detach().cpu().numpy().copy() for key, value in sample.items()}


def _summary(data, mask, joint_names):
    count = int(mask.sum())
    if not count:
        return {"samples": 0}
    take = lambda name: data[name][mask]
    velocity = take("velocity_navigation_mps")
    fd_velocity = take("finite_difference_velocity_navigation_mps")
    gyro = take("gyro_navigation_rad_s")
    command = take("command")
    computed = take("computed_torque_nm")
    applied = take("applied_torque_nm")
    qerror = take("joint_target_rad") - take("joint_position_rad")
    dq = take("joint_velocity_rad_s")
    saturation = np.abs(computed) > 1.6
    gravity = take("projected_gravity")
    result = {
        "samples": count,
        "mean_velocity_mps": velocity.mean(0).tolist(),
        "velocity_std_mps": velocity.std(0).tolist(),
        "mean_gyro_rad_s": gyro.mean(0).tolist(),
        "gyro_std_rad_s": gyro.std(0).tolist(),
        "planar_error_mps": float(np.linalg.norm(velocity[:, :2] - command[:, :2], axis=-1).mean()),
        "yaw_error_rad_s": float(np.abs(gyro[:, 2] - command[:, 2]).mean()),
        "finite_difference_planar_error_mps": float(np.linalg.norm(fd_velocity[:, :2] - command[:, :2], axis=-1).mean()),
        "reported_minus_finite_difference_velocity_rms_mps": np.sqrt(np.mean((velocity - fd_velocity)**2, axis=0)).tolist(),
        "torque_saturation_fraction": float(saturation.mean()),
        "computed_torque_abs_max_nm": float(np.abs(computed).max()),
        "applied_torque_abs_max_nm": float(np.abs(applied).max()),
        "positive_mechanical_power_w": float(np.maximum(applied * dq, 0).sum(-1).mean()),
        "tilt_rms_deg": float(np.degrees(np.sqrt(np.mean(np.arccos(np.clip(-gravity[:, 2], -1, 1))**2)))),
        "joints": {
            name: {"saturation_fraction": float(saturation[:, i].mean()),
                   "computed_torque_abs_p95_nm": float(np.quantile(np.abs(computed[:, i]), .95)),
                   "computed_torque_abs_max_nm": float(np.abs(computed[:, i]).max()),
                   "applied_torque_abs_max_nm": float(np.abs(applied[:, i]).max()),
                   "target_error_rms_rad": float(np.sqrt(np.mean(qerror[:, i]**2))),
                   "velocity_rms_rad_s": float(np.sqrt(np.mean(dq[:, i]**2)))}
            for i, name in enumerate(joint_names)
        },
    }
    result["mean_raw_reward_terms"] = {key.removeprefix("reward_term_"): float(data[key][mask].mean())
                                       for key in data if key.startswith("reward_term_")}
    if "finite_difference_heading_rate_rad_s" in data:
        fd_heading = take("finite_difference_heading_rate_rad_s")
        result["finite_difference_heading_rate_std_rad_s"] = float(fd_heading.std())
        result["reported_gyro_z_minus_fd_heading_rate_rms_rad_s"] = float(np.sqrt(np.mean((gyro[:, 2] - fd_heading)**2)))
    return result


@torch.inference_mode()
def evaluate_diagnostics(env, runner, plan, output, checkpoint_sha):
    from tensordict import TensorDict
    from omni_flat_evaluation import save
    options = diagnostic_options(plan["omni"]["diagnostics"])
    scenarios = diagnostic_scenarios()
    if env.num_envs % len(scenarios):
        raise ValueError("Diagnostic environments must contain equal replicas of 12 scenarios")
    replicas = env.num_envs // len(scenarios)
    if options["trace_envs_per_scenario"] > replicas:
        raise ValueError("trace_envs_per_scenario exceeds the available replicas")
    targets = torch.tensor([row["command"] for row in scenarios], device=env.device).repeat_interleave(replicas, 0)
    policy = runner.get_inference_policy(device=env.device)
    env.omni_diagnostic_enabled = True
    env.reset(seed=options["seed"])
    env.episode_length_buf.zero_()
    env.set_evaluation_targets(targets)
    samples = []
    observation_audit = {"actor_width": 315, "critic_width": 318,
                         "max_same_step_repeat_difference": 0., "max_command_slice_difference": 0.,
                         "max_history_shift_difference": 0.}
    previous_actor = previous_done = None
    try:
        for step in range(round(options["duration_s"] / env.step_dt)):
            obs = env._get_observations()
            repeat = env._get_observations()
            actor = obs["policy"]
            observation_audit["max_same_step_repeat_difference"] = max(
                observation_audit["max_same_step_repeat_difference"], float((actor - repeat["policy"]).abs().max()))
            scaled_command = env._commands * torch.tensor([5., 5., 2.5], device=env.device)
            observation_audit["max_command_slice_difference"] = max(
                observation_audit["max_command_slice_difference"], float((actor[:, -63+6:-63+9] - scaled_command).abs().max()))
            if previous_actor is not None and (~previous_done).any():
                valid = ~previous_done
                observation_audit["max_history_shift_difference"] = max(
                    observation_audit["max_history_shift_difference"],
                    float((actor[valid, :-63] - previous_actor[valid, 63:]).abs().max()))
            previous_actor = actor.clone()
            action = policy(TensorDict(obs, batch_size=[env.num_envs]))
            if options["controller"] == "zero":
                action = torch.zeros_like(action)
            _, _, terminated, truncated, _ = env.step(action)
            previous_done = (terminated | truncated).clone()
            env.set_evaluation_targets(targets)
            sample = env.omni_diagnostic_sample
            if not all(np.isfinite(value).all() for value in sample.values()):
                raise RuntimeError("Nonfinite diagnostic sample")
            # Verify the preserved event sample matches the returned reset flags.
            if not np.array_equal(sample["terminated"], terminated.cpu().numpy()):
                raise RuntimeError("Diagnostic snapshot does not match pre-reset termination events")
            samples.append(sample)
    finally:
        env.omni_diagnostic_enabled = False
    data = {key: np.stack([sample[key] for sample in samples]) for key in samples[0]}
    joint_names = list(env._robot.joint_names)
    masks = sample_masks(data["age_s"], data["terminated"], options["settle_s"])
    rows = []
    for index, scenario in enumerate(scenarios):
        selected = slice(index * replicas, (index + 1) * replicas)
        subset = {key: value[:, selected] for key, value in data.items()}
        row = {**scenario, "replicas": replicas,
               "terminations": int(subset["terminated"].sum()), "truncations": int(subset["truncated"].sum()),
               "termination_reasons": {key.removeprefix("reason_"): int((value & subset["terminated"]).sum())
                                       for key, value in subset.items() if key.startswith("reason_")},
               "windows": {name: _summary(subset, mask[:, selected], joint_names) for name, mask in masks.items()}}
        rows.append(row)
    trace_ids = [index * replicas + replica for index in range(len(scenarios))
                 for replica in range(options["trace_envs_per_scenario"])]
    trace_path = Path(output) / "diagnostic_trace.npz"
    np.savez_compressed(trace_path, **{key: value[:, trace_ids] for key, value in data.items()},
                        trace_env_ids=np.array(trace_ids), joint_names=np.array(joint_names),
                        time_s=(np.arange(len(samples)) + 1) * env.step_dt)
    report = {"complete": True, "kind": "diagnostic_not_qualification", "checkpoint_sha256": checkpoint_sha,
              "options": options, "overrides": plan["omni"].get("overrides", {}),
              "reward_weights": env.cfg.omni_reward_weights,
              "joint_names": joint_names, "control_sample_dt_s": env.step_dt,
              "physics_dt_s": env.cfg.sim.dt, "trace_file": trace_path.name,
              "observation_audit": observation_audit, "scenarios": rows,
              "notes": ["State captured in reward calculation before command advance and automatic reset.",
                        "All failure counts retained; post-settle windows exclude each episode's reset transient.",
                        "Computed torque is requested; applied torque is the motor-clipped simulator command.",
                        "Finite-difference velocity spans one control step; reported velocity is its endpoint.",
                        "Control-rate trace cannot identify oscillation above its Nyquist frequency.",
                        "Termination reason counts overlap when multiple causes coincide."]}
    save(Path(output) / "diagnostics.json", report)
    print("OMNI_DIAGNOSTICS_DONE " + json.dumps({"checkpoint": checkpoint_sha,
          "terminations": sum(row["terminations"] for row in rows), "trace": str(trace_path)}), flush=True)
    return report
