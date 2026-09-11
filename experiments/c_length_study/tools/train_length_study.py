#!/usr/bin/env python3
"""One isolated morphology: full standing gate, scratch PPO, or matched evaluation.

Invoked by the guarded Spark campaign. No default CAD asset or task is edited.
The tested existing HexapodEnv supplies observations, actions and contact rewards.
"""
from __future__ import annotations

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import faulthandler
import hashlib
from importlib import metadata
import json
from pathlib import Path
import time

from tools.c_study_runtime import bootstrap_c_study_runtime
C_STUDY_RUNTIME = bootstrap_c_study_runtime()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--package", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--variant", required=True)
parser.add_argument("--mode", choices=("validate", "probe", "train", "evaluate", "video"), required=True)
parser.add_argument("--stance-index", type=int, default=0)
parser.add_argument("--admission", type=Path)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--iterations", type=int)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.mode == "video":
    args.enable_cameras = True
faulthandler.enable()
faulthandler.dump_traceback_later(90, repeat=True)
app = AppLauncher(args).app
faulthandler.cancel_dump_traceback_later()

import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg
from isaaclab.utils.io import dump_yaml
from hexapod_rl import env as env_module
from hexapod_rl.env_cfg import _contact_sensor
from hexapod_rl.phase1_v5_cfg import HexapodPhase1V5EnvCfg, HexapodPhase1V5PPORunnerCfg
from isaaclab.actuators import DCMotorCfg
from experiments.c_length_study.tools.repair_length_study_inertias import repair_and_verify


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contacts(env):
    distal, shaft, slip = env._get_foot_contact_state()
    coxa = env._coxa_contact_sensor.data.net_forces_w_history.torch
    femur = torch.cat([s.data.net_forces_w_history.torch for s in env._femur_contact_sensors], dim=2)
    nonfoot = torch.cat((torch.linalg.norm(coxa, dim=-1).amax(dim=1)>1,
                        torch.linalg.norm(femur, dim=-1).amax(dim=1)>1, shaft),dim=1).any(dim=1)
    return distal, nonfoot, slip


def validate(env, plan, state):
    obs, _ = env.reset(seed=0)
    env.episode_length_buf.zero_()
    actions = torch.zeros((env.num_envs, 18), device=env.device)
    steps = plan["validation_control_steps"]
    settle = steps // 5
    saturation = samples = nonfoot_count = terminated = truncated = 0
    maximum = 0.0
    max_computed = 0.0
    min_height = float("inf")
    heights = []
    for step in range(steps):
        with torch.inference_mode():
            obs, _, term, trunc, _ = env.step(actions)
        if not bool(torch.isfinite(obs["policy"]).all()):
            raise RuntimeError("Nonfinite standing observation")
        data = env._robot.data
        terminated += int(term.sum())
        truncated += int(trunc.sum())
        maximum = max(maximum, float(data.applied_torque.torch.abs().max()))
        min_height = min(min_height, float(data.root_pos_w.torch[:,2].min()))
        if step >= settle:
            torque = data.computed_torque.torch.abs()
            saturation += int((torque>1.6).sum())
            samples += torque.numel()
            max_computed = max(max_computed, float(torque.max()))
            _, bad, _ = contacts(env)
            nonfoot_count += int(bad.sum())
            heights.append(float(data.root_pos_w.torch[:,2].mean()))
        if (step+1)%100 == 0:
            state.update(control_steps=step+1)
            save(args.output/"state.json",state)
            print(f"VALIDATE {args.variant} {step+1}/{steps}",flush=True)
    gate = {"joint_count":env._robot.num_joints,"body_count":env._robot.num_bodies,
            "foot_count":sum(len(s.body_names) for s in env._feet_contact_sensors),
            "num_envs":env.num_envs,"control_steps":steps,
            "max_abs_applied_torque_nm":maximum,"post_settle_max_abs_computed_torque_nm":max_computed,
            "post_settle_torque_saturation_fraction":saturation/samples,
            "post_settle_nonfoot_contact_env_steps":nonfoot_count,
            "terminations":terminated,"truncations":truncated,"min_root_height_m":min_height,
            "post_settle_mean_root_height_m":float(np.mean(heights))}
    gate["passed"] = (gate["joint_count"]==18 and gate["body_count"]==19 and gate["foot_count"]==6
                      and maximum<=1.61 and saturation/samples<=0.005
                      and nonfoot_count==0 and terminated==0 and truncated==0 and min_height>=0.055)
    save(args.output/"admission.json",{**state,"gate":gate})
    return gate


@torch.inference_mode()
def evaluate(env, runner, plan):
    policy = runner.get_inference_policy(device=env.device)
    reports = []
    for speed in plan["evaluation_forward_speeds_mps"]:
        obs, _ = env.reset(seed=plan["evaluation_seed"])
        env.episode_length_buf.zero_()
        total = {k:0.0 for k in ("abs_forward_error_mps","abs_lateral_velocity_mps","abs_yaw_rate_rad_s",
                                "tilt_squared_rad2","vertical_velocity_squared_m2_s2","saturation_fraction",
                                "positive_mechanical_power_w","slip_speed_sum_mps","distal_contact_count",
                                "nonfoot_contact_fraction","achieved_forward_velocity_mps")}
        fell = torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
        terminations = count = 0
        for step in range(plan["evaluation_control_steps_per_speed"]):
            # Reset may sample a training command; every policy observation and
            # transition here instead receives the same exact test command.
            env._commands[:,0] = speed
            env._commands[:,1:] = 0
            obs = TensorDict(env._get_observations(),batch_size=[env.num_envs])
            with torch.inference_mode():
                action = policy(obs)
                _, _, term, _, _ = env.step(action)
            fell |= term
            terminations += int(term.sum())
            d = env._robot.data
            if not bool(torch.isfinite(d.joint_pos.torch).all()):
                raise RuntimeError("Nonfinite evaluation state")
            if step < 100:
                continue
            v = env._vector_in_command_frame(d.root_lin_vel_b.torch)
            torque = d.applied_torque.torch
            distal, bad, slip = contacts(env)
            tilt = torch.acos(torch.clamp(-d.projected_gravity_b.torch[:,2],-1,1))
            values = {
                "abs_forward_error_mps":(v[:,0]-speed).abs().mean(),
                "achieved_forward_velocity_mps":v[:,0].mean(),
                "abs_lateral_velocity_mps":v[:,1].abs().mean(),
                "abs_yaw_rate_rad_s":d.root_ang_vel_b.torch[:,2].abs().mean(),
                "tilt_squared_rad2":tilt.square().mean(),
                "vertical_velocity_squared_m2_s2":d.root_lin_vel_w.torch[:,2].square().mean(),
                "saturation_fraction":(d.computed_torque.torch.abs()>1.6).float().mean(),
                "positive_mechanical_power_w":(torque*d.joint_vel.torch).clamp_min(0).sum(dim=1).mean(),
                "slip_speed_sum_mps":(slip*distal).sum(),"distal_contact_count":distal.sum(),
                "nonfoot_contact_fraction":bad.float().mean(),
            }
            for key,value in values.items():
                total[key] += float(value)
            count += 1
        summary = {key:value/count for key,value in total.items() if key not in ("slip_speed_sum_mps","distal_contact_count")}
        summary["distal_slip_speed_mps"] = total["slip_speed_sum_mps"]/max(total["distal_contact_count"],1)
        summary["tilt_rms_deg"] = float(np.degrees(np.sqrt(summary.pop("tilt_squared_rad2"))))
        summary["vertical_velocity_rms_mps"] = float(np.sqrt(summary.pop("vertical_velocity_squared_m2_s2")))
        summary.update(command_mps=speed,fall_fraction=float(fell.float().mean()),terminations=terminations,
                       num_envs=env.num_envs,control_steps=plan["evaluation_control_steps_per_speed"])
        reports.append(summary)
        save(args.output/"evaluation.json",{"variant":args.variant,"seed":plan["evaluation_seed"],"speeds":reports,"complete":False})
    save(args.output/"evaluation.json",{"variant":args.variant,"seed":plan["evaluation_seed"],"speeds":reports,"complete":True})
    return reports


@torch.inference_mode()
def record_video(env, runner, plan):
    """Actual deterministic checkpoint rollout at real-time playback speed."""
    import imageio.v2 as imageio
    from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view
    policy = runner.get_inference_policy(device=env.device)
    env.reset(seed=plan["evaluation_seed"])
    env.episode_length_buf.zero_()
    if env.render() is None:
        raise RuntimeError("Isaac RGB backend returned no frame")
    trajectory = []
    fps = 25
    steps = 600
    with imageio.get_writer(str(args.output/"rollout.mp4"), fps=fps, codec="libx264", quality=8) as writer:
        for step in range(steps):
            env._commands[:,0] = 0.2
            env._commands[:,1:] = 0
            obs = TensorDict(env._get_observations(), batch_size=[env.num_envs])
            with torch.inference_mode():
                _, _, term, trunc, _ = env.step(policy(obs))
            d = env._robot.data
            pos = d.root_pos_w.torch[0].cpu().tolist()
            velocity = float(env._vector_in_command_frame(d.root_lin_vel_b.torch)[0,0])
            trajectory.append({"time_s":(step+1)*env.step_dt,"position_m":pos,
                               "forward_velocity_mps":velocity,"terminated":bool(term[0]),"truncated":bool(trunc[0])})
            if step % 2 == 1:
                set_kit_renderer_camera_view(eye=(pos[0]+0.9,pos[1]+1.0,0.72),target=(pos[0],pos[1]-0.10,0.15))
                frame = env.render()
                if frame is None or frame.ndim != 3 or float(np.std(frame)) < 1:
                    raise RuntimeError("Missing or blank Isaac recording frame")
                writer.append_data(frame[:,:,:3])
                if step == 1:
                    imageio.imwrite(str(args.output/"first_frame.png"), frame[:,:,:3])
            if (step+1)%100 == 0:
                print(f"VIDEO {step+1}/{steps}",flush=True)
    report = {"complete":True,"variant":args.variant,"checkpoint_sha256":digest(args.checkpoint),
              "command_mps":0.2,"fps":fps,"frames":steps//2,"duration_s":steps*env.step_dt,
              "trajectory":trajectory}
    save(args.output/"video.json",report)


def main():
    args.output.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((args.package/"manifest.json").read_text())
    plan = json.loads((args.package/"training_plan.json").read_text())
    record = next(r for r in manifest["variants"] if r["variant"]==args.variant)
    urdf = args.package/record["urdf"]
    if digest(urdf)!=record["sha256"]:
        raise RuntimeError("URDF hash mismatch")
    stance = plan["variants"][args.variant]["stances"][args.stance_index]
    # The campaign copies the verified second batch USDs into one fixed path.
    usd = args.package/"training_usd"/args.variant/(args.variant+".usda")
    tensor_audit = repair_and_verify(urdf,usd)
    state = {"variant":args.variant,"mode":args.mode,"status":"initializing","started_unix":time.time(),
             "runtime_binding":C_STUDY_RUNTIME,
             "urdf_sha256":digest(urdf),"plan_sha256":digest(args.package/"training_plan.json"),
             "stance_index":args.stance_index,"inertia_audit":tensor_audit}
    if args.mode != "validate":
        admission = json.loads(args.admission.read_text()) if args.admission else {}
        if not (admission.get("gate",{}).get("passed") and all(admission.get(k)==state[k] for k in ("variant","urdf_sha256","plan_sha256","stance_index"))):
            raise RuntimeError("Training/evaluation requires matching passed full standing admission")
    save(args.output/"state.json",state)
    cfg = HexapodPhase1V5EnvCfg()
    cfg.seed = plan["training_seed"]
    cfg.events = None  # Exact CAD mass budget and matched material in this first screen.
    cfg.sim.dt = plan["physics_dt_s"]
    cfg.decimation = plan["decimation"]
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.num_envs = 1 if args.mode=="video" else plan[{"validate":"validation_num_envs","probe":"validation_num_envs","train":"training_num_envs","evaluate":"evaluation_num_envs"}[args.mode]]
    if args.mode=="video":
        cfg.video_recorder.window_width = 1280
        cfg.video_recorder.window_height = 720
    cfg.sim.device = args.device
    omni = plan.get("omni")
    cfg.episode_length_s = 20.0 if args.mode=="train" else (90.0 if omni else 25.0)
    cfg.robot.spawn.usd_path = str(usd)
    cfg.robot.spawn.articulation_props.solver_position_iteration_count = 16
    cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = 4
    cfg.robot.init_state.pos = (0,0,stance["suggested_reset_root_height_m"])
    cfg.robot.init_state.joint_pos = stance["joint_positions_rad"]
    cfg.robot.actuators = {"legs":DCMotorCfg(**manifest["actuator_config_snapshot"])}
    cfg.nominal_height_m = stance["root_height_at_contact_m"]
    cfg.distal_foot_min_y_m = record["tibia_length_m"]*manifest["study"]["distal_foot_fraction"]
    cfg.swing_clearance_pad_offset_y_m = record["tibia_length_m"]
    cfg.command_lin_vel_x_range_mps = (0.1,0.3)
    cfg.command_frame = "navigation"
    cfg.action_scale = 0.20
    cfg.rated_torque_excess_reward_scale = -0.12
    cfg.torque_saturation_reward_scale = -0.5
    cfg.max_joint_rated_torque_excess_reward_scale = -0.05
    cfg.fall_penalty = -2.0
    names = tuple(tuple(manifest["link_joint_mapping"][leg]["links"][k] for k in ("coxa","femur","tibia")) for leg in ("lf","lm","lr","rf","rm","rr"))
    # Process-local adapter; no changes to the reusable CAD task's source.
    env_module.LEG_LINK_NAMES = names
    root = "/World/envs/env_.*/Robot/Geometry/body_mock"
    cfg.base_contact_sensor = _contact_sensor(root)
    cfg.coxa_contact_sensor = _contact_sensor(root+"/coxa.*")
    cfg.feet_contact_sensors = tuple(_contact_sensor(f"{root}/{c}/{f}/{t}",track_air_time=True,track_contact_points=True,track_friction_forces=True) for c,f,t in names)
    cfg.femur_contact_sensors = tuple(_contact_sensor(f"{root}/{c}/{f}") for c,f,_ in names)
    for sensor in (cfg.base_contact_sensor,cfg.coxa_contact_sensor,*cfg.feet_contact_sensors,*cfg.femur_contact_sensors):
        sensor.update_period = cfg.sim.dt
    reference=plan.get("reference_controller")
    if reference:
        reference_path=args.package/reference["file"]
        if digest(reference_path)!=reference["sha256"]:raise RuntimeError("Reference hash mismatch")
        cfg.command_lin_vel_x_range_mps=(.10,.20)
        cfg.action_scale=reference["residual_scale_rad"]
        cfg.include_gait_phase_observation=True
        cfg.observation_space=68
        cfg.gait_phase_contact_reward_scale=.5
        cfg.swing_clearance_reward_scale=.1
        cfg.gait_duty_factor=.65
        cfg.gait_cycles_per_meter=6.5
        cfg.gait_min_frequency_hz=.65
        cfg.gait_max_frequency_hz=1.3
        cfg.swing_clearance_target_m=.02
        cfg.lin_vel_tracking_std_mps=.10
        cfg.lin_vel_reward_scale=6.
        cfg.deck_stability_reward_scale=.5
        cfg.base_height_reward_scale=-30.
    if omni:
        if reference:
            raise RuntimeError("Omni task must not inherit the forward-only reference")
        from experiments.c_length_study.tools.omni_flat_env import configure_omni
        configure_omni(cfg, omni.get("overrides"))
        if "diagnostics" in omni:
            from experiments.c_length_study.tools.omni_diagnostics import diagnostic_options, diagnostic_scenarios
            diagnostic_options(omni["diagnostics"])
            if args.mode == "evaluate" and cfg.scene.num_envs % len(diagnostic_scenarios()):
                raise ValueError("Diagnostic evaluation_num_envs must be a multiple of 12")
    dump_yaml(str(args.output/"environment.yaml"),cfg)
    torch.manual_seed(cfg.seed)
    if omni:
        from experiments.c_length_study.tools.omni_flat_env import OmniFlatEnv
        env = OmniFlatEnv(cfg=cfg, render_mode="rgb_array" if args.mode=="video" else None,
                          evaluation=args.mode!="train")
    elif reference:
        from experiments.c_length_study.tools.length_reference_env import ReferenceGaitEnv
        env=ReferenceGaitEnv(cfg=cfg,render_mode="rgb_array" if args.mode=="video" else None,
                             reference_path=reference_path,reference_enabled=args.mode!="validate",warmup_s=reference["warmup_s"])
    else:
        env = env_module.HexapodEnv(cfg=cfg, render_mode="rgb_array" if args.mode=="video" else None)
    if env._robot.num_joints!=18 or set(env._robot.joint_names)!=set(stance["joint_positions_rad"]):
        raise RuntimeError("Joint layout mismatch")
    state.update(status="running",observed_joint_order=list(env._robot.joint_names))
    if omni:
        state.update(architecture=omni["architecture"], observation_actor=315, observation_critic=318)
    save(args.output/"state.json",state)
    if args.mode=="validate":
        gate = validate(env,plan,state)
        state.update(status="completed" if gate["passed"] else "rejected",gate=gate)
    elif args.mode=="probe":
        class ZeroRunner:
            def get_inference_policy(self,device):
                return lambda obs: torch.zeros((env.num_envs,18),device=device)
        evaluate(env,ZeroRunner(),plan)
        state.update(status="completed",controller="reference_only_untrained")
    else:
        runner_cfg = HexapodPhase1V5PPORunnerCfg()
        runner_cfg.seed = plan["training_seed"]
        runner_cfg.device = env.device
        runner_cfg.max_iterations = args.iterations or plan["training_iterations"]
        runner_cfg.save_interval = 25
        runner_cfg.logger = "tensorboard"
        runner_cfg.obs_groups = {"actor": ["policy"], "critic": ["critic"] if omni else ["policy"]}
        if omni:
            runner_cfg.experiment_name = "hexapod_c_omni_flat_history_v1"
            runner_cfg.actor.distribution_cfg.init_std = .20
            runner_cfg.algorithm.learning_rate = 3.e-4
            runner_cfg.algorithm.entropy_coef = .005
        repair = omni.get("repair_training") if omni and args.mode == "train" else None
        if repair is not None:
            from experiments.c_length_study.tools.omni_repair_training import verified_checkpoint
            repair = verified_checkpoint(args.checkpoint, repair)
            runner_cfg.actor.distribution_cfg.init_std = repair["exploration_std"]
            runner_cfg.algorithm.entropy_coef = repair["entropy_coef"]
            runner_cfg.algorithm.learning_rate = repair["learning_rate"]
            if runner_cfg.clip_actions is not None:
                raise RuntimeError("Raw-action experiment requires no wrapper clipping")
        # Match Isaac Lab's own trainer: discard deprecated pre-v5 stochastic
        # fields while preserving the explicit Gaussian distribution config.
        runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, metadata.version("rsl-rl-lib"))
        dump_yaml(str(args.output/"agent.yaml"),runner_cfg)
        wrapped = RslRlVecEnvWrapper(env,clip_actions=runner_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped,runner_cfg.to_dict(),log_dir=str(args.output/"policy"),device=env.device)
        if args.mode=="train":
            if repair is not None:
                from experiments.c_length_study.tools.omni_repair_training import load_repair_checkpoint
                state["repair_initialization"] = load_repair_checkpoint(runner, args.checkpoint, repair)
                save(args.output / "repair_initialization.json", state["repair_initialization"])
                save(args.output / "state.json", state)
            elif args.checkpoint:
                runner.load(str(args.checkpoint))
            runner.learn(num_learning_iterations=runner_cfg.max_iterations,init_at_random_ep_len=True)
            checkpoint = args.output/"policy"/"final.pt"
            runner.save(str(checkpoint))
            state.update(status="completed",iterations=runner_cfg.max_iterations,checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint))
        else:
            runner.load(str(args.checkpoint))
            if omni:
                from experiments.c_length_study.tools.omni_flat_evaluation import evaluate_omni, record_omni
                function = record_omni if args.mode=="video" else evaluate_omni
                function(env, runner, plan, args.output, digest(args.checkpoint))
            elif args.mode=="video":
                record_video(env,runner,plan)
            else:
                evaluate(env,runner,plan)
            state.update(status="completed",checkpoint_sha256=digest(args.checkpoint))
    state["finished_unix"] = time.time()
    save(args.output/"state.json",state)
    env.close()
    print(f"RESULT {args.variant} {args.mode} {state['status']}",flush=True)


try:
    main()
except Exception as exc:
    args.output.mkdir(parents=True,exist_ok=True)
    save(args.output/"failure.json",{"variant":args.variant,"mode":args.mode,"error":repr(exc),"time":time.time()})
    raise
finally:
    app.close()
