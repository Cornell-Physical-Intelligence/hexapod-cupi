#!/usr/bin/env python3
"""Bounded native canonical standing, AMP training and actual policy recording."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = Path(__file__).resolve().parent.name

from .env_config import EnvConfig, MODEL_SHA256, USD_SHA256, sha, verify_assets


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_identity():
    root = Path(__file__).resolve().parent
    return {p.name: sha(p) for p in sorted(root.glob("*.py"))}


def require_admission(path, identity, cfg):
    """New training can consume only authentic matching one/batch standing passes."""
    if path is None:
        raise ValueError("Training/recording requires fresh model/source-bound standing admission")
    value = json.loads(Path(path).read_text())
    if value.get("schema") != "canonical_paper_walk_standing_admission_v1":
        raise ValueError("Wrong standing admission schema")
    if value.get("model_sha256") != MODEL_SHA256 or value.get("usd_sha256") != USD_SHA256:
        raise ValueError("Standing admission model differs")
    if value.get("physics_source_files") != identity["physics_source_files"] or value.get("prior_metadata_sha256") != identity["prior_metadata_sha256"]:
        raise ValueError("Standing source or neutral stance differs")
    if value.get("physics_config") != identity["physics_config"]:
        raise ValueError("Standing physics configuration differs")
    for key in ("geometry_sha256", "geometry_extrema_sha256"):
        if value.get(key) != identity[key]:
            raise ValueError("Standing geometry identity differs")
    if type(value.get("num_envs")) is not int or value["num_envs"] < 32:
        raise ValueError("At least 32 same-source standing replicas are required")
    for name, count in (("one", 1), ("batch", value.get("num_envs"))):
        row = value.get(name, {})
        report_path = Path(row.get("report_path", ""))
        if not report_path.is_file() or sha(report_path) != row.get("report_sha256"):
            raise ValueError("Missing/changed actual standing report: " + name)
        report = json.loads(report_path.read_text())
        if report.get("all_pass") is not True or report.get("num_envs") != count:
            raise ValueError("Standing quality rejected: " + name)
    if cfg.num_envs != 1 and value.get("num_envs") != cfg.num_envs:
        raise ValueError("Training replica layout lacks matching standing admission")
    return value


def record_policy(env, learner, output, seconds):
    """A reset-free deterministic rollout; capture is actual Isaac RGB, 25fps."""
    import numpy as np
    import torch
    import imageio.v2 as imageio
    observation = env.reset()
    frames, rows = 0, []
    env.render()
    started_counter = env.sim.get_physics_step_count()
    with imageio.get_writer(str(output / "rollout.mp4"), fps=25, codec="libx264", quality=8) as writer:
        for step in range(round(seconds / env.cfg.control_dt)):
            with torch.inference_mode():
                action = learner.act(observation["obs"], deterministic=True)
                result = env.step(action)
            row = {k: v.detach().cpu().numpy().copy() for k, v in env.telemetry.items()}
            row["time_s"] = np.array((step + 1) * env.cfg.control_dt)
            row["reset"] = np.zeros(env.num_envs, bool)
            rows.append(row)
            if step % 2 == 1:
                frame = env.render()
                writer.append_data(frame)
                frames += 1
                if frames == 1:
                    imageio.imwrite(str(output / "first_frame.png"), frame)
            observation = result
            if bool(result["terminated"].any() or result["truncated"].any()):
                break
    np.savez_compressed(output / "rollout_trace.npz", **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
    record = {"schema": "canonical_paper_walk_native_video_v1", "frames": frames, "fps": 25,
        "duration_s": frames / 25, "physics_steps": env.sim.get_physics_step_count() - started_counter,
        "controls": len(rows), "deterministic_policy": True, "pose_forcing": False,
        "resets_during_rollout": 0, "terminated": bool(rows[-1]["terminated"].any()),
        "truncated": bool(rows[-1]["truncated"].any()), "contact_evidence_complete": False,
        "classification": "tibia_force_only", "stage2_complete": False,
        "model_sha256": MODEL_SHA256, "usd_sha256": USD_SHA256,
        "video_sha256": sha(output / "rollout.mp4"), "trace_sha256": sha(output / "rollout_trace.npz")}
    save(output / "video.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--mode", choices=("diagnostic", "train", "video"), required=True)
    for name in ("asset", "model", "geometry", "geometry-extrema", "prior", "prior-metadata", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--standing-admission", type=Path)
    parser.add_argument("--source-freeze-sha256")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--seconds", type=float, default=24.)
    parser.add_argument("--command", nargs=3, type=float, default=[.04, 0., 0.])
    parser.add_argument("--max-wall-seconds", type=float, default=3600.)
    parser.add_argument("--preflight-only", action="store_true")
    if "--preflight-only" in (argv if argv is not None else sys.argv[1:]):
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", default="cuda:0")
    else:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    cfg = EnvConfig(num_envs=args.num_envs, render=args.mode == "video", episode_seconds=20. if args.mode == "train" else 60.,
                    device=args.device, command_forward_mps=args.command[0],
                    command_left_mps=args.command[1], command_yaw_rad_s=args.command[2])
    verify_assets(args.asset, args.model)
    metadata = json.loads(args.prior_metadata.read_text())
    identity = {"source_files": source_identity(), "model_sha256": MODEL_SHA256, "usd_sha256": USD_SHA256,
        "prior_sha256": sha(args.prior), "prior_metadata_sha256": sha(args.prior_metadata),
        "geometry_sha256": sha(args.geometry), "geometry_extrema_sha256": sha(args.geometry_extrema), "config": cfg.declaration()}
    identity["physics_source_files"] = {key: identity["source_files"][key] for key in ("env.py", "env_config.py")}
    identity["physics_config"] = {"physics_dt": cfg.physics_dt, "decimation": cfg.decimation,
        "spacing_m": cfg.spacing_m, "target_slew_rad": cfg.target_slew_rad, "action_scale_rad": cfg.action_scale_rad,
        "solver_position_iterations": 32, "solver_velocity_iterations": 0,
        "floor": "80m_two_triangle_mesh_y_equals_x_seam", "material_friction": [1., 1.],
        "restitution": 0., "external_forces_every_iteration": True,
        "neutral_joint_position_rad": metadata["nominal_joint_position_rad"],
        "reset_root_height_m": metadata["reset_root_height_m"]}
    if args.source_freeze_sha256 is not None:
        freeze = Path(__file__).resolve().parent / "FREEZE_SHA256.json"
        if not freeze.is_file() or sha(freeze) != args.source_freeze_sha256:
            raise ValueError("Actual source-freeze manifest differs")
    if args.mode != "diagnostic":
        identity["standing_admission"] = require_admission(args.standing_admission, identity, cfg)
    if args.mode == "video" and (args.checkpoint is None or cfg.num_envs != 1):
        raise ValueError("Video requires one robot and an actual learned checkpoint")
    if args.preflight_only:
        print(json.dumps(identity, indent=2)); return 0
    if not args.headless:
        raise ValueError("Use the bounded headless native allocation")
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"schema": "canonical_paper_walk_run_v1", "status": "initializing", "identity": identity,
             "mode": args.mode, "stage2_complete": False, "errors": [],
             "runtime_binding": {"runtime_tree_sha256": args.source_freeze_sha256}}
    save(args.output / "state.json", state)
    app, capture = None, None
    started = time.monotonic()
    try:
        args.enable_cameras = cfg.render
        app = AppLauncher(args).app
        print("REFERENCE_SCREEN_APP_READY", flush=True)
        import torch
        from .env import PaperWalkEnv, DiagnosticCapture, score_diagnostic
        torch.manual_seed(cfg.seed)
        env = PaperWalkEnv(cfg, args.asset, args.model, args.geometry, args.output / "native", reference_metadata=metadata)
        save(args.output / "identity.json", identity)
        state["status"] = "running"
        save(args.output / "state.json", state)
        if args.mode == "diagnostic":
            env.commands.zero_()
            capture = DiagnosticCapture(env, args.output / "standing", args.geometry_extrema)
            for control in range(1000):
                env.step(torch.zeros((env.num_envs, 18), device=env.device))
                if (control + 1) % 100 == 0:
                    print(f"PAPER_STANDING controls={control+1} replicas={env.num_envs}", flush=True)
            capture.close(); capture = None
            report = score_diagnostic(args.output / "standing")
            save(args.output / "standing" / "standing_report.json", report)
            state["standing_gate_pass"] = report["all_pass"]
        else:
            from .learner import Config, PPOLearner
            learner_cfg = Config(num_envs=env.num_envs)
            if args.mode == "video":
                checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
                learner_cfg = Config(**checkpoint["config"])
            learner = PPOLearner(env, args.prior, args.output / "learner", learner_cfg, device=env.device)
            if args.checkpoint is not None:
                learner.load(args.checkpoint)
                state["input_checkpoint_sha256"] = sha(args.checkpoint)
            if args.mode == "train":
                def progress(metrics):
                    print("PAPER_PPO " + json.dumps(metrics, allow_nan=False), flush=True)
                    return time.monotonic() - started < args.max_wall_seconds
                state["checkpoint"] = learner.train(args.updates, checkpoint_interval=args.checkpoint_interval, callback=progress)
            else:
                state["video"] = record_policy(env, learner, args.output, args.seconds)
                state["video"]["checkpoint_sha256"] = sha(args.checkpoint)
                save(args.output / "video.json", state["video"])
        if source_identity() != identity["source_files"]:
            raise RuntimeError("Runtime source changed during allocation")
        env.verify_native_recipe("after_controlled_steps")
        state["status"] = "completed"
    except BaseException as error:
        state["status"] = "failed"
        state["errors"].append(repr(error))
        (args.output / "traceback.txt").write_text(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
    finally:
        if capture is not None:
            try:
                capture.close()
            except BaseException as error:
                state["errors"].append("capture close: " + repr(error))
        state["wall_seconds"] = time.monotonic() - started
        save(args.output / "state.json", state)
        if app is not None:
            app.close()
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
