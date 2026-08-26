#!/usr/bin/env python3
"""Short, non-training smoke test for the isolated Phase 3 sensor stack.

Run through Isaac Lab with cameras enabled, for example::

    ./isaaclab.sh -p /workspace/hexapod/isaaclab/phase3_sensor_smoke.py \
        --kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true --/app/extensions/excluded/4=omni.kit.telemetry" \
        --headless --enable_cameras --steps 300

The test holds the robot at its default stance, verifies ideal sensor shapes,
passes copies through the latency/noise/dropout model, and checks delivery age,
finite values, and the explicit absence of contact/foot sensors.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke-test Phase 3 camera, LiDAR, and IMU.")
parser.add_argument("--steps", type=int, default=300, help="Number of 200 Hz physics steps.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of sensor scenes.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True, enable_cameras=True)
args_cli = parser.parse_args()
if args_cli.steps < 100:
    parser.error("--steps must be at least 100 to cover camera and LiDAR latency warm-up.")
if args_cli.num_envs < 1:
    parser.error("--num_envs must be positive.")
if args_cli.num_envs > 4:
    parser.error(
        "--num_envs must be at most 4: the D455 path creates two "
        "independent RTX render products per environment."
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Isaac Lab imports must follow application launch."""

import json
import omni.kit.app
import torch
from pxr import PhysxSchema, Usd

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
import isaaclab.utils.math as math_utils

# ``--headless --enable_cameras`` selects Isaac Lab's minimal rendering
# experience, which does not depend on the experimental RTX sensor extension.
# Enable it before importing the D455 wrapper so its Python module is registered.
RTX_SENSOR_EXTENSION = "isaacsim.sensors.experimental.rtx"
extension_manager = omni.kit.app.get_app().get_extension_manager()
if not extension_manager.set_extension_enabled_immediate(
    RTX_SENSOR_EXTENSION, True
):
    raise RuntimeError(f"Failed to enable required extension: {RTX_SENSOR_EXTENSION}")

from hexapod_phase3.d455 import D455_RESOLUTION_HW, D455RigBatch
from hexapod_phase3.sensor_cfg import (
    PHASE3_HARDWARE_ACCOUNTING,
    PHASE3_MOUNTS,
    PHYSICS_DT_S,
    HexapodPhase3SensorSceneCfg,
    make_phase3_sim_cfg,
)
from hexapod_phase3.sensor_model import (
    DEFAULT_PHASE3_SENSOR_MODEL_CFG,
    Phase3SensorModel,
)


def _reset_robot(scene: InteractiveScene) -> None:
    robot = scene["robot"]
    root_pose = robot.data.default_root_pose.torch.clone()
    root_pose[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim_index(root_pose=root_pose)
    robot.write_root_velocity_to_sim_index(
        root_velocity=robot.data.default_root_vel.torch.clone()
    )
    robot.write_joint_position_to_sim_index(
        position=robot.data.default_joint_pos.torch.clone()
    )
    robot.write_joint_velocity_to_sim_index(
        velocity=robot.data.default_joint_vel.torch.clone()
    )
    scene.reset()


def _check_latency(name: str, ages: list[float]) -> None:
    transport = getattr(DEFAULT_PHASE3_SENSOR_MODEL_CFG, name).transport
    if not ages:
        raise RuntimeError(f"No delayed {name} frames were delivered.")
    lower = transport.latency_s - transport.latency_jitter_s - 1.0e-6
    upper = transport.latency_s + transport.latency_jitter_s + PHYSICS_DT_S + 1.0e-6
    if min(ages) < lower or max(ages) > upper:
        raise RuntimeError(
            f"{name} delivery age [{min(ages):.4f}, {max(ages):.4f}] "
            f"fell outside [{lower:.4f}, {upper:.4f}]."
        )


def _robot_contact_report_paths(num_envs: int) -> list[str]:
    """Return any contact-report APIs remaining under cloned robot prims."""
    stage = sim_utils.get_current_stage()
    contact_paths: list[str] = []
    for env_index in range(num_envs):
        root_path = f"/World/envs/env_{env_index}/Robot"
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            raise RuntimeError(f"Missing cloned robot prim: {root_path}")
        contact_paths.extend(
            str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
        )
    return contact_paths


def run() -> dict[str, object]:
    sim = SimulationContext(make_phase3_sim_cfg(args_cli.device))
    scene = InteractiveScene(
        HexapodPhase3SensorSceneCfg(
            num_envs=args_cli.num_envs,
            env_spacing=3.0,
            lazy_sensor_update=True,
            replicate_physics=True,
        )
    )
    # The D455 API is separate from InteractiveScene because only Isaac Sim's
    # SingleViewDepthCameraSensor executes the authored stereo disparity model.
    d455 = D455RigBatch.create_for_hexapod_scene(
        num_envs=args_cli.num_envs,
        translation_b_m=PHASE3_MOUNTS.camera_pos_b_m,
        orientation_b_wxyz=PHASE3_MOUNTS.camera_quat_b_wxyz,
    )
    sim.reset()
    _reset_robot(scene)

    sensor_names = set(scene.sensors)
    expected_sensor_names = {"lidar", "imu"}
    if sensor_names != expected_sensor_names:
        raise RuntimeError(
            f"Expected only {sorted(expected_sensor_names)}, got {sorted(sensor_names)}."
        )
    if scene.cfg.robot.spawn.activate_contact_sensors:
        raise RuntimeError(
            "The Phase 3 sensor scene unexpectedly enabled PhysX contact reporting."
        )
    contact_report_paths = _robot_contact_report_paths(args_cli.num_envs)
    if contact_report_paths:
        raise RuntimeError(
            "The Phase 3 robot still contains PhysX contact-report APIs: "
            f"{contact_report_paths}"
        )

    model = Phase3SensorModel(seed=20260824)
    periods_s = {
        name: getattr(DEFAULT_PHASE3_SENSOR_MODEL_CFG, name).transport.sample_period_s
        for name in ("camera", "lidar", "imu")
    }
    expected_lidar_rays = int(
        round(
            scene.cfg.lidar.pattern_cfg.points_per_second
            / scene.cfg.lidar.pattern_cfg.scan_rate_hz
        )
    )
    # Begin on the first real sample boundary.  Capturing every stream at t=0
    # manufactured a pre-period frame, and for the 10 Hz lazy ray caster it
    # could label a previously computed buffer with a new capture timestamp.
    next_capture_s = dict(periods_s)
    delivered = {name: 0 for name in periods_s}
    last_capture: dict[str, torch.Tensor | None] = {
        name: None for name in periods_s
    }
    delivery_ages: dict[str, list[float]] = {name: [] for name in periods_s}
    observed_shapes: dict[str, list[int]] = {}
    max_valid_samples = {"camera_depth": 0, "lidar": 0}
    max_lidar_origin_error_m = 0.0
    delivered_imu_accel_norms: list[float] = []
    delivered_imu_gyro_norms: list[float] = []
    sim_time_s = 0.0

    for step in range(args_cli.steps):
        robot = scene["robot"]
        robot.set_joint_position_target_index(
            target=robot.data.default_joint_pos.torch
        )
        scene.write_data_to_sim()
        sim.step()
        scene.update(PHYSICS_DT_S)
        sim_time_s += PHYSICS_DT_S

        if sim_time_s + 1.0e-9 >= next_capture_s["camera"]:
            camera_frame = d455.get_torch_frame()
            if camera_frame is not None:
                model.capture_camera(
                    camera_frame.rgb,
                    camera_frame.depth_m,
                    sim_time_s,
                )
                while next_capture_s["camera"] <= sim_time_s + 1.0e-9:
                    next_capture_s["camera"] += periods_s["camera"]
        if sim_time_s + 1.0e-9 >= next_capture_s["lidar"]:
            lidar_data = scene["lidar"].data
            mount_offset_b = torch.tensor(
                PHASE3_MOUNTS.lidar_pos_b_m,
                dtype=robot.data.root_pos_w.torch.dtype,
                device=robot.data.root_pos_w.torch.device,
            ).expand(args_cli.num_envs, -1)
            expected_lidar_pos_w = robot.data.root_pos_w.torch + math_utils.quat_apply(
                robot.data.root_quat_w.torch, mount_offset_b
            )
            lidar_origin_error_m = torch.linalg.norm(
                lidar_data.pos_w.torch - expected_lidar_pos_w, dim=-1
            )
            max_lidar_origin_error_m = max(
                max_lidar_origin_error_m,
                float(torch.max(lidar_origin_error_m).item()),
            )
            if max_lidar_origin_error_m > 1.0e-4:
                raise RuntimeError(
                    "LiDAR rays and scalar ranges do not share the configured "
                    f"optical origin (max error {max_lidar_origin_error_m:.6f} m)."
                )
            model.capture_lidar(
                lidar_data.ray_hits_w.torch,
                lidar_data.pos_w.torch,
                sim_time_s,
            )
            while next_capture_s["lidar"] <= sim_time_s + 1.0e-9:
                next_capture_s["lidar"] += periods_s["lidar"]
        if sim_time_s + 1.0e-9 >= next_capture_s["imu"]:
            imu_data = scene["imu"].data
            model.capture_imu(
                imu_data.ang_vel_b.torch,
                imu_data.lin_acc_b.torch,
                sim_time_s,
                PHYSICS_DT_S,
            )
            while next_capture_s["imu"] <= sim_time_s + 1.0e-9:
                next_capture_s["imu"] += periods_s["imu"]

        for name, frame in model.read(sim_time_s).items():
            if last_capture[name] is None:
                last_capture[name] = torch.full_like(
                    frame.capture_time_s, -torch.inf
                )
            newly_delivered = frame.valid & (
                frame.capture_time_s > last_capture[name] + 1.0e-7
            )
            if torch.any(newly_delivered):
                delivered[name] += int(torch.count_nonzero(newly_delivered).item())
                delivery_ages[name].extend(
                    frame.age_s[newly_delivered].detach().cpu().tolist()
                )
                last_capture[name] = torch.where(
                    newly_delivered, frame.capture_time_s, last_capture[name]
                )
            for field_name, value in frame.values.items():
                observed_shapes[f"{name}.{field_name}"] = list(value.shape)
                if value.dtype != torch.bool and not torch.isfinite(value).all():
                    raise RuntimeError(f"Non-finite value in {name}.{field_name}.")

            if name == "camera":
                rgb = frame.values["rgb"]
                depth = frame.values["depth_m"]
                expected_rgb_shape = (
                    args_cli.num_envs,
                    D455_RESOLUTION_HW[0],
                    D455_RESOLUTION_HW[1],
                    3,
                )
                if rgb.shape != expected_rgb_shape:
                    raise RuntimeError(f"Unexpected RGB shape: {tuple(rgb.shape)}")
                expected_depth_shape = expected_rgb_shape[:-1] + (1,)
                if depth.shape != expected_depth_shape:
                    raise RuntimeError(f"Unexpected depth shape: {tuple(depth.shape)}")
                if rgb.min() < 0.0 or rgb.max() > 1.0:
                    raise RuntimeError("Corrupted RGB left the normalized [0, 1] range.")
                valid_depth = frame.values["depth_valid"] & frame.valid.reshape(
                    (-1, 1, 1, 1)
                )
                max_valid_samples["camera_depth"] = max(
                    max_valid_samples["camera_depth"],
                    int(torch.count_nonzero(valid_depth).item()),
                )
            elif name == "lidar":
                ranges = frame.values["ranges_m"]
                if ranges.shape != (args_cli.num_envs, expected_lidar_rays):
                    raise RuntimeError(f"Unexpected LiDAR shape: {tuple(ranges.shape)}")
                if ranges.min() < DEFAULT_PHASE3_SENSOR_MODEL_CFG.lidar.min_range_m:
                    raise RuntimeError("LiDAR range below configured minimum.")
                if ranges.max() > DEFAULT_PHASE3_SENSOR_MODEL_CFG.lidar.max_range_m:
                    raise RuntimeError("LiDAR range above configured maximum.")
                valid_hits = frame.values["hit_valid"] & frame.valid.unsqueeze(-1)
                max_valid_samples["lidar"] = max(
                    max_valid_samples["lidar"],
                    int(torch.count_nonzero(valid_hits).item()),
                )
            elif name == "imu":
                expected_imu_shape = (args_cli.num_envs, 3)
                angular_velocity = frame.values["angular_velocity_rad_s"]
                linear_acceleration = frame.values["linear_acceleration_m_s2"]
                if angular_velocity.shape != expected_imu_shape:
                    raise RuntimeError("Unexpected IMU angular velocity shape.")
                if linear_acceleration.shape != expected_imu_shape:
                    raise RuntimeError("Unexpected IMU linear acceleration shape.")
                if torch.any(newly_delivered):
                    delivered_imu_gyro_norms.extend(
                        torch.linalg.norm(
                            angular_velocity[newly_delivered], dim=-1
                        ).detach().cpu().tolist()
                    )
                    delivered_imu_accel_norms.extend(
                        torch.linalg.norm(
                            linear_acceleration[newly_delivered], dim=-1
                        ).detach().cpu().tolist()
                    )

    for name in periods_s:
        _check_latency(name, delivery_ages[name])
    if min(delivered.values()) < args_cli.num_envs:
        raise RuntimeError(f"Too few delivered frames: {delivered}")
    if min(max_valid_samples.values()) < 1:
        raise RuntimeError(
            "The calibration wall/ground produced no valid camera-depth or LiDAR "
            f"returns: {max_valid_samples}. Check the unmeasured mount rotations."
        )
    if not delivered_imu_accel_norms:
        raise RuntimeError("No valid IMU acceleration samples were delivered.")
    median_imu_accel_norm = float(torch.tensor(delivered_imu_accel_norms).median().item())
    median_imu_gyro_norm = float(torch.tensor(delivered_imu_gyro_norms).median().item())
    if not 2.0 <= median_imu_accel_norm <= 20.0:
        raise RuntimeError(
            "Stationary IMU proper-acceleration magnitude is implausible: "
            f"median {median_imu_accel_norm:.3f} m/s^2."
        )

    report = {
        "status": "pass",
        "physics_steps": args_cli.steps,
        "simulated_seconds": round(sim_time_s, 6),
        "num_envs": args_cli.num_envs,
        "scene_sensors": sorted(sensor_names),
        "external_camera_sensor": "NVIDIA RealSense D455 USD stereo depth",
        "d455_depth_model_parameters": asdict(d455.depth_model_parameters),
        "lidar_model": "Livox Mid-360 near-hemispherical coverage surrogate",
        "mount_assumptions": asdict(PHASE3_MOUNTS),
        "hardware_accounting": PHASE3_HARDWARE_ACCOUNTING.as_report(),
        "foot_contact_sensors_present": False,
        "physx_contact_reporting_enabled": False,
        "maximum_lidar_origin_error_m": round(max_lidar_origin_error_m, 9),
        "median_delivered_imu_acceleration_norm_m_s2": round(
            median_imu_accel_norm, 6
        ),
        "median_delivered_imu_angular_velocity_norm_rad_s": round(
            median_imu_gyro_norm, 6
        ),
        "delivered_environment_frames": delivered,
        "maximum_valid_samples_in_a_delivered_frame": max_valid_samples,
        "delivery_age_s": {
            name: {
                "min": round(min(ages), 6),
                "max": round(max(ages), 6),
            }
            for name, ages in delivery_ages.items()
        },
        "observed_shapes": observed_shapes,
    }
    return report


if __name__ == "__main__":
    try:
        print(json.dumps(run(), indent=2, sort_keys=True))
    finally:
        simulation_app.close()
