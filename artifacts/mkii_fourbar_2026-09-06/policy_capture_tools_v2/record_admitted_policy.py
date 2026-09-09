#!/usr/bin/env python3
"""External real-policy capture v2 using the frozen training runner compatibility adapter."""
from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
from pathlib import Path
import sys

from capture_common import (SCHEMA, TASK_ID, check_frame, digest, environment_layout, load_module, read_json,
    require_same_inputs, tool_identity, validate_dimensions, validate_states, verify_inputs, write_json)


def make_runner_config(training_api, agent_cfg):
    helper = getattr(training_api, "runner_config_dict", None)
    if not callable(helper):
        raise ValueError("Capture v2 requires a frozen training source with runner_config_dict")
    result = helper(agent_cfg, "5.0.1")
    obsolete = {"stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"}
    for name in ("actor", "critic"):
        model = result.get(name)
        if not isinstance(model, dict) or obsolete.intersection(model):
            raise ValueError(f"Training compatibility adapter left unsupported {name} constructor fields")
    if result.get("check_for_nan") is not True:
        raise ValueError("Training runner configuration omitted finite-value checking")
    return result


def parser(add_launcher_args=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source-dir", "checkpoint", "admission", "training-report", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--seconds", type=float, default=15.)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    if add_launcher_args:
        add_launcher_args(p)
        p.set_defaults(visualizer=[], enable_cameras=True)
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    early, _ = parser().parse_known_args(argv)
    out = early.output_dir.resolve(strict=True)
    report_path = out / "report.json"
    if any((out / name).exists() for name in ("report.json", "policy.mp4", "states.npz", "metadata.json")):
        raise ValueError("Refusing to overwrite capture evidence")
    report = {"schema": SCHEMA, "task_id": TASK_ID, "pass": False, "errors": [],
              "hardware_admission": False, "navigation_or_terrain_qualification": False,
              "frames_written": 0, "controls_completed": 0, "encoder_finalized": False}
    persisted = False

    def finish(error=None):
        nonlocal persisted
        if persisted:
            return
        if error is not None:
            report["errors"].append(f"{type(error).__name__}: {error}")
        report["pass"] = (not report["errors"] and report.get("encoder_finalized") is True
            and report.get("checkpoint_unchanged") is True and report.get("policy_state_unchanged") is True
            and report.get("physics_coverage_pass") is True
            and report["frames_written"] == report.get("expected_frames"))
        write_json(report_path, report)
        persisted = True
        print("POLICY_CAPTURE_RESULT", report["pass"], report["frames_written"], flush=True)

    try:
        verified = verify_inputs(early.source_dir, early.checkpoint, early.admission, early.training_report)
        layout = environment_layout(early.source_dir, verified["runtime_manifest"])
        if os.environ.get("HEXAPOD_MKII_ENVIRONMENT_LAYOUT", "grid_2m_v1") != layout:
            raise ValueError("Container environment layout differs from admitted layout")
        tools_before = tool_identity(Path(__file__).parent)
        request = read_json(out / "request.json")
        if (not (out / "admitted").is_file() or request.get("input_sha256") != verified["input_sha256"]
                or request.get("capture_tools_sha256") != tools_before
                or request.get("contract") != verified["contract"]
                or request.get("seconds") != early.seconds or request.get("width") != early.width
                or request.get("height") != early.height):
            raise ValueError("Missing or mismatched guarded capture request")
        report.update(contract=verified["contract"], input_sha256=verified["input_sha256"],
                      capture_tools_sha256=tools_before, deterministic_policy=True, environment_layout=layout)
        source = early.source_dir.resolve()
        training_api = load_module(source / "isaaclab/train_mkii_fourbar.py", "train_mkii_fourbar")
        if not callable(getattr(training_api, "runner_config_dict", None)):
            raise ValueError("Capture v2 requires the corrected frozen training runner configuration helper")
        from hexapod_env.tasks.mkii_fourbar_v1.register import register_mkii_fourbar_v1
        register_mkii_fourbar_v1()
        from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli
        saved = sys.argv
        try:
            sys.argv = [__file__, *argv]
            args, overrides = setup_preset_cli(parser(add_launcher_args))
            if overrides:
                raise ValueError("Capture does not allow task or physics configuration overrides")
            if args.device != "cuda:0" or args.visualizer not in (None, []):
                raise ValueError("Capture requires the reviewed headless cuda:0 configuration")
            args.enable_cameras = True
            sys.argv = [__file__]
            cfg, agent_cfg = resolve_task_config(TASK_ID, "rsl_rl_cfg_entry_point")
        finally:
            sys.argv = saved
        cfg.scene.num_envs = 1
        cfg.seed = agent_cfg.seed = verified["training"]["seed"]
        cfg.viewer.eye, cfg.viewer.lookat = (.90, 1.05, .58), (0., 0., .15)
        cfg.viewer.resolution = (args.width, args.height)
        if cfg.video_recorder is None:
            raise ValueError("Installed task has no RGB recorder")
        cfg.video_recorder.window_width, cfg.video_recorder.window_height = args.width, args.height
        cfg.video_recorder.backend_source = "renderer"
        count = validate_dimensions(args.seconds, args.width, args.height, cfg.sim.dt * cfg.decimation)
        report.update(expected_frames=count, seconds=args.seconds, width=args.width, height=args.height,
                      num_envs=1, policy_dt_s=cfg.sim.dt * cfg.decimation)
        if importlib.metadata.version("rsl-rl-lib") != "5.0.1":
            raise ValueError("Only the reviewed RSL-RL 5.0.1 inference API is supported")
        if "pxr" in sys.modules:
            raise ValueError("Standalone USD imported before Kit")
        with launch_simulation(cfg, args):
            env = writer = guard = None
            rows = []
            try:
                import numpy as np
                import torch
                import gymnasium as gym
                from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
                from rsl_rl.runners import OnPolicyRunner
                from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
                from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view
                from validate_mkii_fourbar import PhysicalMetrics

                env = gym.make(TASK_ID, cfg=cfg, render_mode="rgb_array")
                raw = env.unwrapped
                report["runtime_manifest"] = raw.runtime_manifest
                comparison = training_api.compare_admitted_runtime(verified["runtime_manifest"], raw.runtime_manifest)
                report["admitted_runtime_comparison"] = comparison
                if not comparison["pass"]:
                    raise ValueError("Recording runtime differs from the admitted physical runtime")
                if raw._robot.num_bodies != 31 or raw.num_envs != 1:
                    raise ValueError("Capture requires one physical 31-body robot")
                wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
                agent_cfg.device = str(raw.device)
                agent_dict = make_runner_config(training_api, agent_cfg)
                runner = OnPolicyRunner(wrapped, agent_dict, log_dir=None, device=agent_cfg.device)
                infos = runner.load(str(args.checkpoint), strict=True, map_location=agent_cfg.device)
                if infos != {"contract": verified["contract"], "next_iteration": verified["sidecar"]["next_iteration"]}:
                    raise ValueError("Embedded checkpoint identity differs from verified sidecar")
                training_api.restore_adaptive_learning_rate(runner.alg)
                policy_hash = training_api.parameters_digest(runner)
                algorithm_hash = training_api.algorithm_state_digest(runner)
                if (policy_hash != verified["training"]["policy_after_sha256"]
                        or algorithm_hash != verified["training"]["algorithm_after_sha256"]):
                    raise ValueError("Loaded policy, normalization, critic or optimizer differs from completed training")
                policy = runner.get_inference_policy(device=agent_cfg.device)
                obs = wrapped.get_observations()

                def tensor(value):
                    return value.torch if hasattr(value, "torch") else value

                def array(value):
                    return tensor(value).detach().cpu().numpy().copy()

                def state():
                    data = raw._robot.data
                    return {"joint_pos_rad": array(data.joint_pos)[0],
                            "joint_vel_rad_s": array(data.joint_vel)[0],
                            "root_pos_w_m": array(data.root_pos_w)[0],
                            "root_quat_w_xyzw": array(data.root_quat_w)[0],
                            "command_navigation": array(raw._commands)[0]}

                anchor = array(raw._robot.data.root_pos_w)[0, :2]
                alpha = -math.expm1(-raw.step_dt / .20)

                def camera():
                    nonlocal anchor
                    anchor = anchor + alpha * (array(raw._robot.data.root_pos_w)[0, :2] - anchor)
                    eye = (float(anchor[0])+.90, float(anchor[1])+1.05, .58)
                    target = (float(anchor[0]), float(anchor[1]), .15)
                    set_kit_renderer_camera_view(eye=eye, target=target)
                    return np.asarray([*eye, *target], dtype=np.float64)

                # Warmup creates the render product. No physics/control advances are allowed.
                before = state()
                counters = (raw.common_step_counter, raw._sim_step_counter)
                warmup_valid = False
                for attempt in range(120):
                    camera()
                    frame = env.render()
                    try:
                        check_frame(frame, args.width, args.height)
                        warmup_valid = True
                        break
                    except ValueError:
                        pass
                if not warmup_valid:
                    raise ValueError("RGB renderer never produced a populated frame during bounded warmup")
                if ((raw.common_step_counter, raw._sim_step_counter) != counters
                        or any(not np.array_equal(before[key], state()[key]) for key in before)):
                    raise ValueError("Render warmup changed robot state or simulation counters")
                report["warmup"] = {"render_calls": attempt + 1, "physics_steps": 0, "state_unchanged": True}
                metrics = PhysicalMetrics(raw, raw.kinematics)
                metrics.window = "policy_capture"
                guard = training_api.PhysicalTrainingGuard(raw, metrics)
                writer = FFMPEG_VideoWriter(str(out / "policy.mp4"), (args.width, args.height),
                    fps=1. / raw.step_dt, codec="libx264", preset="fast", threads=2,
                    ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
                guard.__enter__()
                with torch.inference_mode():
                    for step in range(count):
                        if (out / "stop_requested").exists():
                            raise RuntimeError("Guarded supervisor requested capture stop")
                        row = {"pre_" + key: value for key, value in state().items()}
                        # RSL-RL 5 returns TensorDict, which is not a Python dict.
                        row["observation_policy"] = array(obs["policy"])[0]
                        actions = policy(obs, stochastic_output=False)
                        if not bool(torch.isfinite(actions).all()):
                            raise ValueError("Nonfinite policy action")
                        row["policy_action"] = array(actions)[0]
                        obs, rewards, dones, _ = wrapped.step(actions)
                        if not bool(torch.isfinite(rewards).all()):
                            raise ValueError("Nonfinite policy reward")
                        row.update({"post_" + key: value for key, value in state().items()})
                        row["processed_target_endpoint_rad"] = array(raw._processed_actions)[0]
                        row["post_applied_motor_torque_nm"] = array(raw.motor_state("applied_torque"))[0]
                        row["reward"] = array(rewards)[0]
                        row["done"] = array(dones)[0]
                        row["camera_eye_target_w_m"] = camera()
                        row["frame_index"] = step
                        row["simulation_time_s"] = (step + 1) * raw.step_dt
                        frame = check_frame(env.render(), args.width, args.height)
                        writer.write_frame(frame)
                        rows.append(row)
                        report["frames_written"] += 1
                        report["controls_completed"] += 1
                        policy.reset(dones)
                        if (step + 1) % 50 == 0:
                            write_json(out / "progress.json", {"controls_completed": step + 1,
                                "seconds": (step + 1) * raw.step_dt, "physical_metrics": metrics.windows})
                guard.require_coverage(count)
                report["physics_coverage_pass"] = True
                report["physics_substeps"] = guard.total
                report["physical_metrics"] = metrics.windows
                report["policy_state_unchanged"] = (training_api.parameters_digest(runner) == policy_hash
                    and training_api.algorithm_state_digest(runner) == algorithm_hash)
                if not report["policy_state_unchanged"]:
                    raise ValueError("Inference changed checkpoint policy/normalization/algorithm state")
                process = writer.proc
                writer.close()
                writer = None
                if process.returncode != 0:
                    raise ValueError("MP4 encoder exited unsuccessfully")
                report["encoder_finalized"] = True
                states = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
                report["state_archive_validation"] = validate_states(states, count, raw.step_dt)
                np.savez_compressed(out / "states.npz", **states)
                metadata = {"schema": SCHEMA, "task_id": TASK_ID, "input_sha256": verified["input_sha256"],
                    "source_sha256": verified["contract"]["sha256"], "capture_tools_sha256": tools_before,
                    "tree_joint_names": list(raw._robot.joint_names), "active_motor_names": list(raw.active_joint_names),
                    "body_names": list(raw._robot.body_names), "root_quaternion_order": "XYZW",
                    "commands": "original task sampler; navigation vx forward=-bodyY, vy left=+bodyX, yaw about +Z",
                    "command_units": ["m/s", "m/s", "rad/s"], "command_override": False,
                    "state_alignment": "pre fields precede action; post fields/frame follow the transition and any recorded auto-reset",
                    "processed_target_semantics": "post-step endpoint; a done transition may contain reset defaults",
                    "post_torque_semantics": "final cached actuator value; use physics_metrics for every-substep coverage",
                    "frame_timestamp_semantics": "frame k represents post-transition time (k+1)*policy_dt",
                    "fps": 1./raw.step_dt, "frames": count, "seconds": args.seconds,
                    "resolution": [args.width, args.height], "codec": "H.264/libx264", "pixel_format": "yuv420p",
                    "camera": {"eye_offset": [.90, 1.05, .58], "target_offset": [0., 0., .15],
                               "xy_follow_time_constant_s": .20, "z_fixed_world": True},
                    "environment_origin_w_m": array(raw.scene.env_origins)[0].tolist(),
                    "done_transitions": [i for i, row in enumerate(rows) if bool(row["done"])],
                    "episode_length_s": cfg.episode_length_s,
                    "versions": {"torch": torch.__version__, "rsl_rl": "5.0.1",
                                 "moviepy": importlib.metadata.version("moviepy")},
                    "interpretation": "actual deterministic policy rollout; video appearance does not qualify terrain/navigation/hardware"}
                write_json(out / "metadata.json", metadata)
                report["artifacts"] = {name: {"sha256": digest(out/name), "bytes": (out/name).stat().st_size}
                    for name in ("policy.mp4", "states.npz", "metadata.json")}
                after = verify_inputs(args.source_dir, args.checkpoint, args.admission, args.training_report)
                require_same_inputs(verified, after)
                if tools_before != tool_identity(Path(__file__).parent):
                    raise ValueError("Capture tools changed while running")
                report["checkpoint_unchanged"] = True
                finish()
            except BaseException as error:
                if guard is not None:
                    report["physical_metrics"] = guard.metrics.windows
                    report["physics_substeps"] = guard.total
                finish(error)
                raise
            finally:
                if guard is not None:
                    guard.__exit__(None, None, None)
                if writer is not None:
                    writer.close()
                # All report/MP4 writes above happen before native Kit shutdown.
                if env is not None:
                    env.close()
    except BaseException as error:
        finish(error)
        raise
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
