#!/usr/bin/env python3
"""Measure Mid-360 self-occlusion in Isaac Sim at one mount height.

`tools/lidar_placement_study.py` answers the mount question offline, from the
URDF and its meshes. This script answers the same question inside the simulator
that will actually carry the sensor, so the two can be compared. It holds the
robot at its default stance on flat ground, casts one Mid-360 surrogate scan,
and classifies every ray by where it landed.

The classification is unambiguous because the scene is flat: the ground is the
plane z = 0, the calibration wall is parked far outside lidar range, so any
finite hit above `--robot-hit-z` came off the robot itself.

Reported metrics line up one-for-one with the offline study:

* `blocked_fraction_below_horizon` - share of below-horizon rays that struck the
  robot instead of reaching the ground. Compare with the study's `blk<0`.
* `nearest_ground_m` - closest ground return, horizontally, from the optical
  centre. Compare with the study's `ground fwd` / `ground med`.
* `ground_visible_azimuth_fraction` - share of 1-degree azimuth sectors with at
  least one ground return.

The Mid-360 surrogate pattern samples uniformly in solid angle, so a plain ray
count is already a solid-angle weighting and the two tools are comparable
without reweighting.

Run one process per height, for example::

    ./isaaclab.sh -p /workspace/hexapod/isaaclab/phase3_lidar_placement_probe.py \
        --kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true" \
        --headless --enable_cameras --mount-height 0.101 \
        --out /workspace/hexapod/artifacts/sensor_placement/livox_mid360/sim_probe_0101.json

`--mount-height` is the optical centre above the deck top, which is what
`Phase3MountAssumptions.lidar_pos_b_m` stores. The offline study is
parameterized by plate height instead; the two differ by the 25.92 mm
optical-centre offset recorded in
`robot/sensors/livox_mid360/config/mid360_sensor.json`.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Measure Mid-360 self-occlusion at one mount height.")
parser.add_argument(
    "--mount-height",
    type=float,
    default=None,
    help="Optical centre height above the deck top, in metres. Defaults to the configured mount.",
)
parser.add_argument("--steps", type=int, default=40, help="Physics steps to settle before sampling.")
parser.add_argument(
    "--robot-hit-z",
    type=float,
    default=0.02,
    help="Hits above this world height are attributed to the robot rather than the ground.",
)
parser.add_argument("--wall-distance", type=float, default=-500.0, help="Where to park the calibration wall.")
parser.add_argument("--out", type=str, default=None, help="Write the result as JSON here.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, enable_cameras=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Isaac Lab imports must follow application launch."""

import json
import math

import torch

from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from hexapod_phase3.sensor_cfg import (
    PHASE3_MOUNTS,
    PHYSICS_DT_S,
    HexapodPhase3SensorSceneCfg,
    make_phase3_sim_cfg,
)


def _reset_robot(scene: InteractiveScene) -> None:
    robot = scene["robot"]
    root_pose = robot.data.default_root_pose.torch.clone()
    root_pose[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim_index(root_pose=root_pose)
    robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())
    robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
    robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
    scene.reset()


def run() -> dict[str, object]:
    mount_height = (
        PHASE3_MOUNTS.lidar_pos_b_m[2] if args_cli.mount_height is None else args_cli.mount_height
    )

    scene_cfg = HexapodPhase3SensorSceneCfg(
        num_envs=1, env_spacing=3.0, lazy_sensor_update=False, replicate_physics=True
    )
    scene_cfg.lidar_mount.init_state.pos = (
        PHASE3_MOUNTS.lidar_pos_b_m[0],
        PHASE3_MOUNTS.lidar_pos_b_m[1],
        mount_height,
    )
    # The wall exists to give the smoke test a finite target. It would sit
    # inside the field of view here, so move it beyond lidar range.
    scene_cfg.calibration_wall.init_state.pos = (0.0, args_cli.wall_distance, 0.25)

    sim = SimulationContext(make_phase3_sim_cfg(args_cli.device))
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    _reset_robot(scene)

    robot = scene["robot"]
    for _ in range(args_cli.steps):
        robot.set_joint_position_target_index(target=robot.data.default_joint_pos.torch)
        scene.write_data_to_sim()
        sim.step()
        scene.update(PHYSICS_DT_S)

    lidar = scene["lidar"]
    origin = lidar.data.pos_w.torch[0]
    hits = lidar.data.ray_hits_w.torch[0]

    # Ray directions come from the pattern itself, so misses are still counted
    # in the denominator. The mount is upright and the robot is level, so the
    # sensor-frame elevation is the world elevation.
    pattern_cfg = scene_cfg.lidar.pattern_cfg
    _, directions = pattern_cfg.func(pattern_cfg, str(origin.device))
    elevation = torch.asin(directions[:, 2].clamp(-1.0, 1.0))

    finite = torch.isfinite(hits).all(dim=-1)
    from_robot = finite & (hits[:, 2] > args_cli.robot_hit_z)
    from_ground = finite & ~from_robot
    below = elevation < 0.0

    below_count = int(below.sum().item())
    blocked_below = int((below & from_robot).sum().item())

    ground_offsets = hits[from_ground, :2] - origin[:2]
    ground_radius = torch.linalg.norm(ground_offsets, dim=-1)
    # Navigation frame: forward is body -Y, lateral is body +X.
    ground_azimuth = torch.rad2deg(torch.atan2(ground_offsets[:, 0], -ground_offsets[:, 1]))
    sectors = torch.floor((ground_azimuth + 180.0)).clamp(0, 359).long()
    covered = torch.zeros(360, dtype=torch.bool, device=hits.device)
    covered[sectors] = True

    forward = ground_radius[(ground_azimuth.abs() <= 15.0)]

    result = {
        "mount_height_above_deck_m": round(float(mount_height), 4),
        "optical_centre_height_w_m": round(float(origin[2].item()), 4),
        "rays_total": int(directions.shape[0]),
        "rays_below_horizon": below_count,
        "rays_hitting_robot": int(from_robot.sum().item()),
        "blocked_fraction_below_horizon": round(blocked_below / max(below_count, 1), 4),
        "blocked_fraction_all_rays": round(int(from_robot.sum().item()) / directions.shape[0], 4),
        "nearest_ground_m": (
            round(float(ground_radius.min().item()), 3) if ground_radius.numel() else None
        ),
        "nearest_ground_forward_m": (
            round(float(forward.min().item()), 3) if forward.numel() else None
        ),
        "ground_visible_azimuth_fraction": round(float(covered.float().mean().item()), 4),
        "ground_returns": int(from_ground.sum().item()),
        "unobstructed_blind_radius_m": round(
            float(origin[2].item()) / math.tan(math.radians(7.0)), 3
        ),
    }
    return result


def main() -> None:
    result = run()
    print(json.dumps(result, indent=2))
    if args_cli.out:
        with open(args_cli.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        print(f"wrote {args_cli.out}")


main()
simulation_app.close()
