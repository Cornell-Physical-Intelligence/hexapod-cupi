#!/usr/bin/env python3
"""Run all morphology candidates together and capture an actual Isaac frame.

This is a finite standing comparison, NOT PPO training or a dynamics admission
gate. Failed postures remain visible and are recorded; no policy is loaded.
Launch using launch_length_study_spark.py, which checks shared GPU occupancy.
"""
from __future__ import annotations

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--package", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--capture-step", type=int, default=500)
parser.add_argument("--limit", type=int, default=49, help="Use 1 for the first smoke test")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 0 < args.capture_step <= args.steps or not 0 < args.limit <= 49:
    parser.error("Require 0 < capture-step <= steps and 1 <= limit <= 49")
args.enable_cameras = True
app = AppLauncher(args).app

import imageio.v3 as iio
import numpy as np
import torch
import omni.replicator.core as rep
import isaaclab.sim as sim_utils
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import DCMotorCfg
from experiments.c_length_study.tools.repair_length_study_inertias import repair_and_verify


def array(value):
    return value.torch if hasattr(value, "torch") else value


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.package / "manifest.json").read_text())
    records = manifest["variants"][:args.limit]
    if args.limit == 1:
        records = [next(v for v in manifest["variants"] if v["is_unchanged_length_baseline"])]
    report = {"status": "importing", "kind": "standing_comparison_no_policy", "training": False,
              "robot_count": len(records), "physics_steps_completed": 0, "variants": {}}
    save(args.output / "state.json", report)
    usd_paths = []
    for record in records:
        name = record["variant"]
        urdf = args.package / record["urdf"]
        if hashlib.sha256(urdf.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"URDF hash mismatch: {name}")
        config = UrdfConverterCfg(
            asset_path=str(urdf), usd_dir=str(args.package / "usd"),
            force_usd_conversion=True, fix_base=False, merge_fixed_joints=True,
            link_density=0.0, self_collision=False, collision_from_visuals=False,
            collision_type="Convex Hull", run_multi_physics_conversion=False,
            ros_package_paths=[{"name": "hexapod_mkii_length_study", "path": str(args.package)}],
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force", target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)))
        converter = UrdfConverter(config)
        usd = Path(converter.usd_path)
        if not usd.is_file():
            raise RuntimeError(f"URDF import produced no USD: {name}")
        tensor_audit = repair_and_verify(urdf, usd)
        report["variants"][name] = {"inertia_roundtrip": tensor_audit, "urdf_sha256": record["sha256"]}
        usd_paths.append(usd)
        save(args.output / "state.json", report)
        print(f"IMPORTED {name} tensor_roundtrip=PASS", flush=True)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.0025, render_interval=8, device=args.device))
    ground = sim_utils.GroundPlaneCfg(color=(0.16, 0.19, 0.22))
    ground.func("/World/Ground", ground)
    light = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.9, 0.93, 1.0))
    light.func("/World/Light", light)
    entities = []
    side = math.ceil(math.sqrt(len(records)))
    spacing = 1.05
    for index, (record, usd) in enumerate(zip(records, usd_paths)):
        column, row = index % side, index // side
        position = ((column - (side - 1) / 2) * spacing, (row - (side - 1) / 2) * spacing, record["reset_root_height_m"])
        motor = DCMotorCfg(**manifest["actuator_config_snapshot"])
        cfg = ArticulationCfg(
            prim_path=f"/World/Variant_{record['variant']}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd), activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=4)),
            init_state=ArticulationCfg.InitialStateCfg(pos=position, joint_pos=record["screening"]["joint_positions_rad"], joint_vel={".*": 0.0}),
            soft_joint_pos_limit_factor=0.95,
            actuators={"legs": motor})
        sim_utils.create_prim(f"/World/Variant_{record['variant']}", "Xform")
        entities.append(Articulation(cfg))
    view_scale = max(1.0, side / 7.0)
    eye = (7.0 * view_scale, -8.0 * view_scale, 10.0 * view_scale)
    sim.set_camera_view(eye=eye, target=(0, 0, 0))
    camera = rep.create.camera(position=eye, look_at=(0, 0, 0), focal_length=26.0)
    product = rep.create.render_product(camera, (2200, 1800))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([product])
    sim.reset()
    for record, robot in zip(records, entities):
        if robot.num_joints != 18 or robot.num_bodies != 19:
            raise RuntimeError("Imported articulation layout changed")
        if set(robot.joint_names) != set(record["screening"]["joint_positions_rad"]):
            raise RuntimeError("Imported joint-name set mismatch")
        positions = array(robot.data.default_joint_pos).clone()
        robot.write_joint_position_to_sim_index(position=positions)
        robot.write_joint_velocity_to_sim_index(velocity=array(robot.data.default_joint_vel).clone())
        robot.reset()
        report["variants"][record["variant"]].update({"observed_joint_order": robot.joint_names, "body_names": robot.body_names,
                                                        "max_abs_applied_torque_nm": 0.0, "max_abs_computed_torque_nm": 0.0})
    report["status"] = "simulating"
    report["started_utc_unix"] = time.time()
    save(args.output / "state.json", report)
    for step in range(1, args.steps + 1):
        for robot in entities:
            robot.set_joint_position_target_index(target=array(robot.data.default_joint_pos))
            robot.write_data_to_sim()
        sim.step()
        for record, robot in zip(records, entities):
            robot.update(sim.get_physics_dt())
            entry = report["variants"][record["variant"]]
            position = array(robot.data.joint_pos)
            if not bool(torch.isfinite(position).all()):
                raise RuntimeError(f"Nonfinite joint state: {record['variant']}")
            for data_name, metric in (("applied_torque", "max_abs_applied_torque_nm"), ("computed_torque", "max_abs_computed_torque_nm")):
                value = float(array(getattr(robot.data, data_name)).abs().max().item())
                if not math.isfinite(value):
                    raise RuntimeError(f"Nonfinite torque: {record['variant']}")
                entry[metric] = max(entry[metric], value)
        if step % 100 == 0:
            report["physics_steps_completed"] = step
            save(args.output / "state.json", report)
            print(f"SIMULATING robots={len(entities)} steps={step}", flush=True)
        if step == args.capture_step:
            for _ in range(12):
                sim.render()
            pixels = rgb.get_data()
            if pixels is None or np.asarray(pixels).size == 0:
                raise RuntimeError("Isaac render returned no image")
            screenshot = args.output / "isaac_length_study.png"
            iio.imwrite(screenshot, np.asarray(pixels)[..., :3])
            report["screenshot"] = str(screenshot)
            report["screenshot_physics_step"] = step
            report["screenshot_utc_unix"] = time.time()
            save(args.output / "state.json", report)
            print(f"SCREENSHOT {screenshot}", flush=True)
    for record, robot in zip(records, entities):
        entry = report["variants"][record["variant"]]
        entry["final_root_position_m"] = array(robot.data.root_pos_w).cpu().tolist()
        entry["final_joint_positions_rad"] = array(robot.data.joint_pos).cpu().tolist()
    report.update(status="completed", physics_steps_completed=args.steps, completed_utc_unix=time.time(), simulation_acceptance_passed=False)
    save(args.output / "state.json", report)
    print("COMPLETED standing comparison; training acceptance has not been evaluated.", flush=True)


try:
    main()
finally:
    app.close()
