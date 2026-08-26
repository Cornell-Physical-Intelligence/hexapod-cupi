#!/usr/bin/env python3
"""Run a deterministic standing/contact sanity check before RL training."""

from __future__ import annotations

import argparse
import math
import sys

import gymnasium as gym
import torch

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

from hexapod_rl.asset_cfg import FEMUR_JOINTS, TIBIA_JOINTS
from hexapod_rl.register import TASK_ID, register_envs


parser = argparse.ArgumentParser(description="Validate the RobStride hexapod articulation.")
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument(
    "--root-height-m",
    type=float,
    help="Override reset root height for a controlled stance validation.",
)
parser.add_argument(
    "--femur-angle-rad",
    type=float,
    help="Override all six reset/default femur angles.",
)
parser.add_argument(
    "--tibia-angle-rad",
    type=float,
    help="Override all six reset/default tibia angles.",
)
add_launcher_args(parser)
parser.set_defaults(visualizer=[])
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args
register_envs()


def main() -> None:
    if args_cli.steps < 1:
        raise ValueError("--steps must be at least 1")
    if args_cli.num_envs < 1:
        raise ValueError("--num_envs must be at least 1")
    if args_cli.root_height_m is not None and not (
        math.isfinite(args_cli.root_height_m) and args_cli.root_height_m > 0.0
    ):
        raise ValueError("--root-height-m must be finite and positive")
    if args_cli.femur_angle_rad is not None and not (
        math.isfinite(args_cli.femur_angle_rad)
        and 0.0 <= args_cli.femur_angle_rad <= 1.74533
    ):
        raise ValueError("--femur-angle-rad must lie within [0, 1.74533]")
    if args_cli.tibia_angle_rad is not None and not (
        math.isfinite(args_cli.tibia_angle_rad)
        and 0.0 <= args_cli.tibia_angle_rad <= 2.53073
    ):
        raise ValueError("--tibia-angle-rad must lie within [0, 2.53073]")
    env_cfg, _ = resolve_task_config(TASK_ID, "")
    env_cfg.seed = 0
    env_cfg.scene.num_envs = args_cli.num_envs
    root_x, root_y, default_root_height = env_cfg.robot.init_state.pos
    root_height = float(
        args_cli.root_height_m
        if args_cli.root_height_m is not None
        else default_root_height
    )
    femur_angle = float(
        args_cli.femur_angle_rad
        if args_cli.femur_angle_rad is not None
        else env_cfg.robot.init_state.joint_pos[FEMUR_JOINTS[0]]
    )
    tibia_angle = float(
        args_cli.tibia_angle_rad
        if args_cli.tibia_angle_rad is not None
        else env_cfg.robot.init_state.joint_pos[TIBIA_JOINTS[0]]
    )
    env_cfg.robot.init_state.pos = (root_x, root_y, root_height)
    env_cfg.robot.init_state.joint_pos.update(
        {name: femur_angle for name in FEMUR_JOINTS}
    )
    env_cfg.robot.init_state.joint_pos.update(
        {name: tibia_angle for name in TIBIA_JOINTS}
    )
    env_cfg.nominal_height_m = root_height
    env_cfg.episode_length_s = max(
        env_cfg.episode_length_s,
        args_cli.steps * env_cfg.decimation * env_cfg.sim.dt + 1.0,
    )
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(TASK_ID, cfg=env_cfg)
        env.reset(seed=0)
        unwrapped = env.unwrapped
        # DirectRLEnv staggers full-reset episode counters for training.  The
        # validation run must begin at step zero so truncations are meaningful.
        unwrapped.episode_length_buf.zero_()
        if unwrapped._robot.num_joints != 18:
            raise RuntimeError(f"Expected 18 joints, found {unwrapped._robot.num_joints}")
        foot_count = sum(len(sensor.body_names) for sensor in unwrapped._feet_contact_sensors)
        if foot_count != 6:
            raise RuntimeError(
                f"Expected 6 feet, found {foot_count}"
            )

        actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
        min_height = float("inf")
        max_abs_torque = 0.0
        max_abs_computed_torque = 0.0
        post_settle_abs_computed_sum = 0.0
        post_settle_sample_count = 0
        post_settle_saturated_count = 0
        terminated_count = 0
        truncated_count = 0
        post_settle_samples = 0
        post_settle_height_sum = torch.zeros(
            unwrapped.num_envs, device=unwrapped.device
        )
        post_settle_height_squared_sum = torch.zeros_like(post_settle_height_sum)
        post_settle_vertical_velocity_squared_sum = torch.zeros_like(
            post_settle_height_sum
        )
        post_settle_roll_pitch_rate_squared_sum = torch.zeros_like(
            post_settle_height_sum
        )
        post_settle_tilt_squared_sum = torch.zeros_like(post_settle_height_sum)
        post_settle_non_foot_contact_env_steps = 0
        # Always leave at least one post-settling sample, including short smoke tests.
        settling_steps = min(max(1, args_cli.steps // 5), args_cli.steps - 1)
        for step in range(args_cli.steps):
            with torch.inference_mode():
                observations, _, terminated, truncated, _ = env.step(actions)
            policy_obs = observations["policy"]
            if not torch.isfinite(policy_obs).all():
                raise RuntimeError("Non-finite observation detected")
            min_height = min(min_height, unwrapped._robot.data.root_pos_w.torch[:, 2].min().item())
            max_abs_torque = max(
                max_abs_torque,
                unwrapped._robot.data.applied_torque.torch.abs().max().item(),
            )
            abs_computed_torque = unwrapped._robot.data.computed_torque.torch.abs()
            max_abs_computed_torque = max(
                max_abs_computed_torque, abs_computed_torque.max().item()
            )
            if step >= settling_steps:
                post_settle_abs_computed_sum += abs_computed_torque.sum().item()
                post_settle_sample_count += abs_computed_torque.numel()
                post_settle_saturated_count += int(
                    torch.count_nonzero(abs_computed_torque > 1.6).item()
                )
                root_height_sample = unwrapped._robot.data.root_pos_w.torch[:, 2]
                vertical_velocity = unwrapped._robot.data.root_lin_vel_w.torch[:, 2]
                roll_pitch_rate = torch.linalg.norm(
                    unwrapped._robot.data.root_ang_vel_b.torch[:, :2], dim=1
                )
                upright_cosine = torch.clamp(
                    -unwrapped._robot.data.projected_gravity_b.torch[:, 2],
                    -1.0,
                    1.0,
                )
                tilt = torch.acos(upright_cosine)
                post_settle_samples += 1
                post_settle_height_sum += root_height_sample
                post_settle_height_squared_sum += torch.square(root_height_sample)
                post_settle_vertical_velocity_squared_sum += torch.square(
                    vertical_velocity
                )
                post_settle_roll_pitch_rate_squared_sum += torch.square(
                    roll_pitch_rate
                )
                post_settle_tilt_squared_sum += torch.square(tilt)

                _, shaft_contact, _ = unwrapped._get_foot_contact_state()
                coxa_forces = unwrapped._coxa_contact_sensor.data.net_forces_w_history.torch
                femur_forces = torch.cat(
                    [
                        sensor.data.net_forces_w_history.torch
                        for sensor in unwrapped._femur_contact_sensors
                    ],
                    dim=2,
                )
                non_foot_contact = torch.cat(
                    (
                        torch.linalg.norm(coxa_forces, dim=-1).max(dim=1)[0] > 1.0,
                        torch.linalg.norm(femur_forces, dim=-1).max(dim=1)[0] > 1.0,
                        shaft_contact,
                    ),
                    dim=1,
                ).any(dim=1)
                post_settle_non_foot_contact_env_steps += int(
                    torch.count_nonzero(non_foot_contact).item()
                )
            terminated_count += int(torch.count_nonzero(terminated).item())
            truncated_count += int(torch.count_nonzero(truncated).item())

        mean_height = unwrapped._robot.data.root_pos_w.torch[:, 2].mean().item()
        per_env_mean_height = post_settle_height_sum / post_settle_samples
        per_env_height_std = torch.sqrt(
            torch.clamp(
                post_settle_height_squared_sum / post_settle_samples
                - torch.square(per_env_mean_height),
                min=0.0,
            )
        )
        vertical_velocity_rms = torch.sqrt(
            post_settle_vertical_velocity_squared_sum / post_settle_samples
        )
        roll_pitch_rate_rms = torch.sqrt(
            post_settle_roll_pitch_rate_squared_sum / post_settle_samples
        )
        tilt_rms = torch.sqrt(post_settle_tilt_squared_sum / post_settle_samples)
        non_foot_contact_fraction = (
            post_settle_non_foot_contact_env_steps
            / (post_settle_samples * unwrapped.num_envs)
        )
        print(f"joint_count={unwrapped._robot.num_joints}")
        print(f"body_count={unwrapped._robot.num_bodies}")
        print(f"foot_count={foot_count}")
        print(f"min_base_height_m={min_height:.6f}")
        print(f"mean_base_height_m={mean_height:.6f}")
        print(f"stance_root_height_m={root_height:.6f}")
        print(f"stance_femur_angle_rad={femur_angle:.6f}")
        print(f"stance_tibia_angle_rad={tibia_angle:.6f}")
        print(f"post_settle_mean_base_height_m={per_env_mean_height.mean().item():.6f}")
        print(f"post_settle_mean_height_std_m={per_env_height_std.mean().item():.6f}")
        print(f"post_settle_max_height_std_m={per_env_height_std.max().item():.6f}")
        print(
            "post_settle_mean_vertical_velocity_rms_mps="
            f"{vertical_velocity_rms.mean().item():.6f}"
        )
        print(
            "post_settle_mean_roll_pitch_rate_rms_radps="
            f"{roll_pitch_rate_rms.mean().item():.6f}"
        )
        print(f"post_settle_mean_tilt_rms_deg={torch.rad2deg(tilt_rms).mean().item():.6f}")
        print(f"post_settle_non_foot_contact_fraction={non_foot_contact_fraction:.8f}")
        print(f"max_abs_torque_nm={max_abs_torque:.6f}")
        print(f"max_abs_computed_torque_nm={max_abs_computed_torque:.6f}")
        print(
            "post_settle_mean_abs_computed_torque_nm="
            f"{post_settle_abs_computed_sum / post_settle_sample_count:.6f}"
        )
        saturation_fraction = post_settle_saturated_count / post_settle_sample_count
        print(f"post_settle_torque_saturation_fraction={saturation_fraction:.8f}")
        print(f"unexpected_terminations={terminated_count}")
        print(f"unexpected_truncations={truncated_count}")
        if max_abs_torque > 1.61:
            raise RuntimeError("Actuator exceeded the RS05 1.6 N*m continuous limit")
        if saturation_fraction > 0.005:
            raise RuntimeError(
                "Standing validation failed: sustained torque demand exceeds the RS05 rating"
            )
        if post_settle_non_foot_contact_env_steps:
            raise RuntimeError(
                "Standing validation failed: coxa, femur, or tibia shaft contacted the ground"
            )
        if min_height < 0.055 or terminated_count or truncated_count:
            raise RuntimeError("Standing validation failed: robot fell or contacted its base")
        print("VALIDATION_PASS")
        env.close()


if __name__ == "__main__":
    main()
