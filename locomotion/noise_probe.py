"""Hold fixed action means under graded exploration noise and record the reward inputs.

No optimizer runs. Each replica holds one declared pose and draws independent Gaussian action
noise at one declared deviation, so pose and noise vary separately. The recorded telemetry lets
any reward version be scored on a workstation: training runs change pose and noise together and
cannot separate them (docs/REWARD_V2_TABLE1_AUDIT.md section 15).
"""
import json
from pathlib import Path
import time

import numpy as np
import torch

SCHEMA = "hexapod_noise_probe_v1"
CONTROLS = 600
SETTLE_CONTROLS = 100
REPLICAS_PER_CELL = 8
STANDARD_DEVIATIONS = (0., .05, .1, .15)
# Action units: one unit is 0.35 rad from neutral. Per leg: coxa yaw, femur pitch, tibia pitch.
POSES = {
    "neutral": (0., 0., 0.),
    "femur_raised": (0., -1., 0.),
    "crouch": (0., 1., -1.),
    "beyond_bound": (0., -1.5, 0.),
}
POSE_SCOPE = {
    "neutral": "The reset stance.",
    "femur_raised": "Six femur means at the lower action bound, the direction retained runs B and C reached.",
    "crouch": "Femur means at the upper bound and tibia means at the lower bound, as in retained run D.",
    "beyond_bound": "Six femur means 0.5 units past the bound, as an unbounded mean allows (retained run A).",
}
FIELDS = ("linear_velocity_nav", "angular_velocity_body", "root_pose_xyzw", "joint_position_rad",
          "joint_velocity_rad_s", "joint_target_rad", "torque_square_sum_400hz",
          "requested_torque_abs_max_400hz", "saturation_count_400hz", "other_body_force_max_400hz",
          "tibia_floor_force_world_n", "tibia_floor_force_min_norm_400hz", "action")


def plan(num_envs=128):
    """One row per replica: pose name, action mean and noise deviation."""
    cells = [(pose, std) for pose in POSES for std in STANDARD_DEVIATIONS]
    if num_envs != len(cells) * REPLICAS_PER_CELL:
        raise ValueError("The probe plan needs 128 replicas")
    rows = []
    for index in range(num_envs):
        pose, std = cells[index // REPLICAS_PER_CELL]
        rows.append({"replica": index, "pose": pose, "standard_deviation": std, "mean": list(POSES[pose]) * 6})
    return rows


def declaration(seed):
    return {"schema": SCHEMA, "seed": seed, "controls": CONTROLS, "settle_controls": SETTLE_CONTROLS,
        "replicas_per_cell": REPLICAS_PER_CELL, "standard_deviations": list(STANDARD_DEVIATIONS),
        "poses": {name: {"leg_action_mean": list(value), "scope": POSE_SCOPE[name]} for name, value in POSES.items()},
        "noise": "Independent Gaussian action noise per replica, joint and control; the environment applies "
                 "its unchanged clamp, joint limits and 0.040 rad limiter.",
        "commands": "The held command is zero. The open-loop means ignore it, so offline scoring may assume any command.",
        "fields": list(FIELDS), "optimizer": None, "policy": None, "stage2_complete": False,
        "scope": "Reward-input measurement at fixed means. No policy is trained, loaded or qualified."}


def run(env, output, *, seed, guard=None, max_wall_seconds=None, progress=None):
    """Step the declared plan and save one array per telemetry field, shaped controls x replicas x width."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = plan(env.num_envs)
    record = declaration(seed)
    (output/"plan.json").write_text(json.dumps({**record, "replicas": rows}, indent=2, allow_nan=False)+"\n")
    mean = torch.tensor([row["mean"] for row in rows], dtype=torch.float32, device=env.device)
    std = torch.tensor([row["standard_deviation"] for row in rows], dtype=torch.float32, device=env.device)[:, None]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    env.commands.zero_()
    env.reset()
    started = time.monotonic()
    saved = {name: [] for name in FIELDS}
    flags = {"terminated": [], "truncated": []}
    for control in range(CONTROLS):
        roots = env.current["root"][:, :3].clone()
        if guard is not None:
            guard.check(roots, "before_control", speeds=torch.linalg.vector_norm(env.current["linear"], dim=-1))
        noise = torch.randn((env.num_envs, 18), generator=generator).to(env.device)
        result = env.step(mean + std * noise)
        if guard is not None:
            guard.check(env.current["root"][:, :3], "after_control", previous=roots)
        for name in FIELDS:
            saved[name].append(env.telemetry[name].detach().to("cpu", torch.float32).numpy())
        for name in flags:
            flags[name].append(result[name].detach().cpu().numpy())
        # A fallen replica restarts at neutral; its rows stay in the record with the flag set.
        fallen = result["terminated"].nonzero(as_tuple=False).flatten()
        if len(fallen):
            env.reset(fallen)
        if progress is not None and (control + 1) % 100 == 0:
            progress({"probe_controls": control + 1, "replicas": env.num_envs})
        if max_wall_seconds is not None and time.monotonic() - started >= max_wall_seconds:
            raise TimeoutError("Noise probe exceeded its deadline")
    arrays = {name: np.stack(values) for name, values in saved.items()}
    arrays.update({name: np.stack(values) for name, values in flags.items()})
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("Nonfinite noise-probe telemetry")
    np.savez_compressed(output/"telemetry.npz", **arrays)
    summary = {**record, "controls_completed": CONTROLS, "terminations": int(arrays["terminated"].sum()),
               "cells": summarize(arrays, rows)}
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


def summarize(arrays, rows):
    """Per-cell motion statistics after the settle window; rewards are scored offline."""
    cells = {}
    for row in rows:
        cells.setdefault((row["pose"], row["standard_deviation"]), []).append(row["replica"])
    result = []
    window = slice(SETTLE_CONTROLS, None)
    for (pose, std), replicas in cells.items():
        take = lambda name: arrays[name][window][:, replicas].astype(np.float64)
        velocity, gyro, target = take("linear_velocity_nav"), take("angular_velocity_body"), take("joint_target_rad")
        result.append({"pose": pose, "standard_deviation": std, "replicas": replicas,
            "planar_speed_rms_mps": float(np.sqrt((velocity[..., :2]**2).sum(-1).mean())),
            "planar_velocity_mean_mps": velocity[..., :2].mean((0, 1)).tolist(),
            "vertical_velocity_rms_mps": float(np.sqrt((velocity[..., 2]**2).mean())),
            "roll_pitch_rate_rms_rad_s": float(np.sqrt((gyro[..., :2]**2).sum(-1).mean())),
            "yaw_rate_rms_rad_s": float(np.sqrt((gyro[..., 2]**2).mean())),
            "joint_velocity_rms_rad_s": float(np.sqrt((take("joint_velocity_rad_s")**2).mean())),
            "executed_target_step_rms_rad": float(np.sqrt((np.diff(target, axis=0)**2).mean())),
            "root_height_mean_m": float(take("root_pose_xyzw")[..., 2].mean()),
            "applied_torque_rms_nm": float(np.sqrt((take("torque_square_sum_400hz") / 8).mean())),
            "terminations": int(arrays["terminated"][:, replicas].sum())})
    return result
