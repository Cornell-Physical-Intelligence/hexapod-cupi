#!/usr/bin/env python3
"""Hold the exact admitted full C robot on an imported fixture's flat start.

No policy optimizer, traversal or asset mutation. Requires both the full flat
standing admission and a matching successful fixture geometry/contact smoke.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
from pathlib import Path
import time
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--package", type=Path, required=True)
parser.add_argument("--admission", type=Path, required=True)
parser.add_argument("--fixture-admission", type=Path, required=True)
parser.add_argument("--catalog", type=Path, required=True)
parser.add_argument("--fixture-id", default="train_ramp_1103")
parser.add_argument("--variant", default="f050_t060")
parser.add_argument("--stance-index", type=int, default=0)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1000:
    parser.error("The terrain standing smoke requires at least 1000 control steps")
if args.output.exists():
    parser.error("Use a fresh output directory")
args.output.mkdir(parents=True)
faulthandler.enable()
faulthandler.dump_traceback_later(90, repeat=True)
app = AppLauncher(args).app
faulthandler.cancel_dump_traceback_later()

import numpy as np
import torch
from isaaclab.actuators import DCMotorCfg
from isaaclab.utils.io import dump_yaml
from hexapod_rl import env as env_module
from hexapod_rl.env_cfg import _contact_sensor
from hexapod_rl.phase1_v5_cfg import HexapodPhase1V5EnvCfg
from hexapod_terrain.fixture_adapter import MildTerrainSpec, adapt_flat_cfg_for_fixture_smoke
from hexapod_terrain.robot_smoke import LEGS, audit_robot_usd, check_start_footprint, load_admitted_study
from omni_flat_env import OmniFlatEnv, configure_omni
from terrain_fixture_checks import collision_meshes, digest, load_catalog


def save(state):
    temporary = args.output / "state.tmp"
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output / "state.json")


def main():
    state = dict(status="initializing", started_unix=time.time(), mode="terrain_standing_smoke",
                 ready_for_terrain_training=False, policy_training_started=False)
    env = None
    try:
        manifest, plan, record, urdf, xml, stance, identity, admission = load_admitted_study(
            args.package, args.variant, args.stance_index, args.admission)
        state.update(identity)
        fixture_gate = json.loads(args.fixture_admission.read_text())
        if (fixture_gate.get("status") != "completed" or not fixture_gate.get("all_fixture_smokes_passed")
                or fixture_gate.get("catalog_sha256") != digest(args.catalog)
                or not any(r["id"] == args.fixture_id and r["passed"] for r in fixture_gate.get("runtime_rows", []))):
            raise ValueError("The selected fixture needs a matching passed Isaac geometry/contact smoke")
        spec = MildTerrainSpec(args.catalog, args.fixture_id)
        terrain = load_catalog(args.catalog, [args.fixture_id])[0]
        state["start_footprint"] = check_start_footprint(args.package, xml, manifest, record, stance, spec, terrain)
        usd = args.package / "training_usd" / args.variant / (args.variant + ".usda")
        state["inertia_audit"] = audit_robot_usd(urdf, usd)
        state["fixture_id"] = args.fixture_id
        state["fixture_sha256"] = terrain[0]["sha256"]
        state["fixture_admission_sha256"] = digest(args.fixture_admission)
        # Match the existing study launcher's physical setup, preserving the
        # current plan's bounded omni overrides and the original actuator caps.
        cfg = HexapodPhase1V5EnvCfg()
        cfg.seed = plan["training_seed"]
        cfg.events = None
        cfg.sim.dt = plan["physics_dt_s"]
        cfg.decimation = plan["decimation"]
        cfg.sim.render_interval = cfg.decimation
        cfg.sim.device = args.device
        cfg.episode_length_s = max(25., args.steps * cfg.sim.dt * cfg.decimation + 2.)
        cfg.robot.spawn.usd_path = str(usd)
        cfg.robot.spawn.articulation_props.solver_position_iteration_count = 16
        cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = 4
        cfg.robot.init_state.pos = (0., 0., stance["suggested_reset_root_height_m"])
        cfg.robot.init_state.joint_pos = stance["joint_positions_rad"]
        cfg.robot.actuators = {"legs": DCMotorCfg(**manifest["actuator_config_snapshot"])}
        cfg.nominal_height_m = stance["root_height_at_contact_m"]
        cfg.distal_foot_min_y_m = record["tibia_length_m"] * manifest["study"]["distal_foot_fraction"]
        cfg.swing_clearance_pad_offset_y_m = record["tibia_length_m"]
        names = tuple(tuple(manifest["link_joint_mapping"][leg]["links"][kind]
                            for kind in ("coxa", "femur", "tibia")) for leg in LEGS)
        env_module.LEG_LINK_NAMES = names
        root = "/World/envs/env_.*/Robot/Geometry/body_mock"
        cfg.base_contact_sensor = _contact_sensor(root)
        cfg.coxa_contact_sensor = _contact_sensor(root + "/coxa.*")
        cfg.feet_contact_sensors = tuple(_contact_sensor(f"{root}/{coxa}/{femur}/{tibia}", track_air_time=True,
            track_contact_points=True, track_friction_forces=True) for coxa, femur, tibia in names)
        cfg.femur_contact_sensors = tuple(_contact_sensor(f"{root}/{coxa}/{femur}") for coxa, femur, _ in names)
        for sensor in (cfg.base_contact_sensor, cfg.coxa_contact_sensor, *cfg.feet_contact_sensors, *cfg.femur_contact_sensors):
            sensor.update_period = cfg.sim.dt
        configure_omni(cfg, plan["omni"].get("overrides"))
        cfg = adapt_flat_cfg_for_fixture_smoke(cfg, spec, admission=admission, asset_identity=identity)
        dump_yaml(str(args.output / "environment.yaml"), cfg)
        env = OmniFlatEnv(cfg=cfg, evaluation=True)
        if (env._robot.num_joints != 18 or env._robot.num_bodies != 19
                or set(env._robot.joint_names) != set(stance["joint_positions_rad"])
                or sum(len(s.body_names) for s in env._feet_contact_sensors) != 6):
            raise ValueError("Runtime robot joint/body/foot names or counts differ")
        collision_meshes(env.sim.stage, "/World/ground")
        state.update(status="running", observed_joint_order=list(env._robot.joint_names), body_count=19, foot_count=6)
        save(state)
        obs, _ = env.reset(seed=0)
        env.episode_length_buf.zero_()
        zero = torch.zeros((1, 18), device=env.device)
        env.set_evaluation_targets(torch.zeros((1, 3), device=env.device))
        settle = args.steps // 5
        saturated = samples = bad_contacts = base_contacts = terms = truncs = 0
        applied_max = computed_max = 0.
        root_min, root_sum, root_samples = float("inf"), 0., 0
        support_min = 6
        for step in range(args.steps):
            with torch.inference_mode():
                obs, _, term, trunc, _ = env.step(zero)
            if not all(torch.isfinite(v).all() for v in obs.values()):
                raise RuntimeError("Nonfinite observations")
            if torch.count_nonzero(env._commands) or torch.count_nonzero(env.omni_targets):
                raise RuntimeError("Standing smoke received a nonzero command")
            data = env._robot.data
            terms += int(term.sum())
            truncs += int(trunc.sum())
            applied_max = max(applied_max, float(data.applied_torque.torch.abs().max()))
            root_min = min(root_min, float(data.root_pos_w.torch[:, 2].min()))
            if step >= settle:
                torque = data.computed_torque.torch.abs()
                computed_max = max(computed_max, float(torque.max()))
                saturated += int((torque > 1.6).sum())
                samples += torque.numel()
                distal, shaft, _ = env._get_foot_contact_state()
                coxa = env._coxa_contact_sensor.data.net_forces_w_history.torch.norm(dim=-1).amax(1) > 1.
                femur = torch.cat([s.data.net_forces_w_history.torch.norm(dim=-1).amax(1) > 1.
                                    for s in env._femur_contact_sensors], dim=1)
                bad_contacts += int(torch.cat((coxa, femur, shaft), dim=1).any(1).sum())
                base_contacts += int((env._base_contact_sensor.data.net_forces_w_history.torch.norm(dim=-1).amax(1) > 1.).sum())
                support_min = min(support_min, int(distal.sum()))
                root_sum += float(data.root_pos_w.torch[:, 2].sum())
                root_samples += 1
            if (step + 1) % 100 == 0:
                state["control_steps"] = step + 1
                save(state)
                print(f"TERRAIN_STAND {args.fixture_id} {step + 1}/{args.steps}", flush=True)
        gate = dict(joint_count=18, body_count=19, foot_count=6, num_envs=1, control_steps=args.steps,
            max_abs_applied_torque_nm=applied_max, post_settle_max_abs_computed_torque_nm=computed_max,
            post_settle_torque_saturation_fraction=saturated / samples, post_settle_nonfoot_contact_env_steps=bad_contacts,
            post_settle_base_contact_env_steps=base_contacts, post_settle_min_distal_support_contacts=support_min,
            terminations=terms, truncations=truncs, min_root_height_m=root_min,
            post_settle_mean_root_height_m=root_sum / root_samples)
        gate["passed"] = (applied_max <= 1.61 and saturated / samples <= .005 and bad_contacts == 0
            and base_contacts == 0 and support_min == 6 and terms == 0 and truncs == 0 and root_min >= .055)
        state.update(status="completed" if gate["passed"] else "rejected", gate=gate, finished_unix=time.time(),
                     scope="Full C robot holds its admitted stance on fixture flat start only; no traversal qualification")
        save(state)
        print(json.dumps({"status": state["status"], "gate": gate}), flush=True)
        return 0 if gate["passed"] else 1
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        save(state)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        result = main()
    finally:
        app.close()
    raise SystemExit(result)
