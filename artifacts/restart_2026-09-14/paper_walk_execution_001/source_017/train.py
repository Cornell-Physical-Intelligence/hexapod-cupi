#!/usr/bin/env python3
"""Bounded native canonical standing, AMP training and actual policy recording."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = Path(__file__).resolve().parent.name

from .env_config import EnvConfig, MODEL_SHA256, USD_SHA256, sha, verify_assets

PHYSICS_IDENTITY_KEYS = ("physics_source_files", "physics_config", "prior_metadata_sha256",
                         "model_sha256", "usd_sha256", "geometry_sha256", "geometry_extrema_sha256")
# CLI names only: keep validation available in the stdlib-only preflight. Tests
# check these against evaluate.learning_probe_cases(), which owns trial recipes.
LEARNING_PROBE_CASE_IDS = tuple(f"learning:translate_0.05_{bearing}deg" for bearing in range(0,360,45)) + (
    "learning:yaw_-1", "learning:yaw_+1", "learning:quiet_20s", "learning:quiet_32s",
    "learning:forward_0.05_to_stop")


def physics_identity(identity):
    return {key: identity[key] for key in PHYSICS_IDENTITY_KEYS}


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_identity():
    root = Path(__file__).resolve().parent
    return {p.name: sha(p) for p in sorted(root.glob("*.py"))}


def initial_std_value(value):
    """Validate the existing learner parameter without importing the runtime."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("initial std must be a positive finite number") from error
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("initial std must be a positive finite number")
    return number


def positive_training_value(value):
    """Validate explicit optimizer/KL settings in the stdlib-only preflight."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("training value must be a positive finite number") from error
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("training value must be a positive finite number")
    return number


def nonnegative_training_value(value):
    """Zero explicitly disables an optional fresh BC loss coefficient."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("training value must be a nonnegative finite number") from error
    if isinstance(value, bool) or not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("training value must be a nonnegative finite number")
    return number


def fresh_learner_overrides(mode, checkpoint=None, *, initial_std=None, learning_rate=None,
                            target_kl=None, bc_optimizer=None, bc_learning_rate=None,
                            bc_velocity_coefficient=None, bc_detach_velocity=None):
    """Explicit fresh settings only; never reinterpret a saved configuration."""
    values = {"initial_std":initial_std, "learning_rate":learning_rate, "target_kl":target_kl,
              "bc_optimizer":bc_optimizer, "bc_learning_rate":bc_learning_rate,
              "bc_velocity_coefficient":bc_velocity_coefficient,"bc_detach_velocity":bc_detach_velocity}
    provided = {key:value for key,value in values.items() if value is not None}
    if provided and (mode != "train" or checkpoint is not None):
        flags = ", ".join("--"+key.replace("_","-") for key in provided)
        raise ValueError(flags+" is only for fresh training without --checkpoint")
    for key in ("initial_std", "learning_rate", "target_kl", "bc_learning_rate"):
        if key in provided:
            provided[key] = initial_std_value(provided[key]) if key == "initial_std" else positive_training_value(provided[key])
    if bc_velocity_coefficient is not None:
        provided["bc_velocity_coefficient"] = nonnegative_training_value(bc_velocity_coefficient)
    if bc_detach_velocity is not None and type(bc_detach_velocity) is not bool:
        raise ValueError("BC detach velocity must be a boolean")
    if bc_optimizer is not None and bc_optimizer not in ("shared", "separate"):
        raise ValueError("BC optimizer must be shared or separate")
    if bc_optimizer == "separate" and bc_learning_rate is None:
        raise ValueError("Separate BC optimizer requires explicit --bc-learning-rate")
    if bc_optimizer != "separate" and bc_learning_rate is not None:
        if provided["bc_learning_rate"] != provided.get("learning_rate", 3e-4):
            raise ValueError("Shared BC optimizer requires the PPO learning rate")
    return provided


def learner_configuration(mode, num_envs, initial_std=None, checkpoint=None, *,
                          learning_rate=None, target_kl=None, bc_optimizer=None, bc_learning_rate=None,
                          bc_velocity_coefficient=None, bc_detach_velocity=None):
    """Fresh learning is configurable; checkpoint configuration stays exact."""
    overrides = fresh_learner_overrides(mode, checkpoint, initial_std=initial_std, learning_rate=learning_rate,
        target_kl=target_kl, bc_optimizer=bc_optimizer, bc_learning_rate=bc_learning_rate,
        bc_velocity_coefficient=bc_velocity_coefficient,bc_detach_velocity=bc_detach_velocity)
    from .learner import Config
    if checkpoint is not None:
        from dataclasses import asdict
        cfg = Config(**checkpoint["config"])
        cfg.validate()
        if asdict(cfg) != checkpoint["config"]:
            raise ValueError("Checkpoint configuration must be complete; implicit migration is unsupported")
        if mode == "train" and cfg.num_envs != num_envs:
            raise ValueError("Training checkpoint replica count differs from the native environment")
        return cfg
    if mode != "train":
        raise ValueError("Policy recording/evaluation requires an actual learned checkpoint")
    cfg = Config(num_envs=num_envs, **overrides)
    cfg.validate()
    return cfg


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
        if (report.get("all_pass") is not True or report.get("num_envs") != count
                or report.get("controls") != 1000 or report.get("substeps") != 8000
                or len(report.get("replicas", [])) != count
                or [r.get("env") for r in report["replicas"]] != list(range(count))
                or not all(type(r.get("env")) is int and r.get("pass") is True
                    and r.get("failed_physical_bounds") == []
                    and r.get("quiet", {}).get("pass") is True
                    and r.get("quiet", {}).get("failed_bounds") == []
                    for r in report["replicas"])):
            raise ValueError("Standing quality rejected: " + name)
        state_path=Path(row.get("state_path", ""))
        if not state_path.is_file() or sha(state_path)!=row.get("state_sha256"):
            raise ValueError("Missing/changed native standing acquisition: "+name)
        state=json.loads(state_path.read_text())
        if state.get("status")!='completed' or state.get("standing_gate_pass") is not True or state.get("errors"):
            raise ValueError("Native standing acquisition failed: "+name)
        observed=state.get("identity",{})
        for key in ('physics_source_files','physics_config','prior_metadata_sha256','model_sha256',
                    'usd_sha256','geometry_sha256','geometry_extrema_sha256'):
            if observed.get(key)!=identity[key]:
                raise ValueError("Native standing identity differs: "+name+':'+key)
        if observed.get('config',{}).get('num_envs')!=count:
            raise ValueError("Native standing replica count differs: "+name)
    if cfg.num_envs != 1 and value.get("num_envs") != cfg.num_envs:
        raise ValueError("Training replica layout lacks matching standing admission")
    return value


def require_realized_prior(prior_path, report_path, identity):
    """Bind learning data to a completed, accepted native replay acquisition."""
    if prior_path is None or report_path is None:
        raise ValueError("Training/video requires --realized-prior and --replay-report")
    prior_path, report_path = Path(prior_path), Path(report_path)
    if not prior_path.is_file() or not report_path.is_file():
        raise ValueError("Missing native replay dataset or report")
    report = json.loads(report_path.read_text())
    if (report.get("schema") != "canonical_paper_native_prior_replay_v1"
            or report.get("complete") is not True or report.get("failure") is not None
            or report.get("all_commands_accepted") is not True
            or report.get("pose_forcing_after_initial_reset") is not False
            or report.get("resets_after_initial") != 0):
        raise ValueError("Native replay was incomplete or rejected")
    if (report.get("physics_identity") != physics_identity(identity)
            or report.get("analytic_prior_sha256") != identity["prior_sha256"]
            or report.get("analytic_metadata_sha256") != identity["prior_metadata_sha256"]
            or report.get("source_sha256") != identity["source_files"]["replay.py"]):
        raise ValueError("Native replay physics/source/reference identity differs")
    row = report.get("datasets", {}).get("realized_prior.npz", {})
    if (prior_path.name != "realized_prior.npz" or row.get("sha256") != sha(prior_path)
            or type(row.get("transitions")) is not int or row["transitions"] < 2):
        raise ValueError("Native replay training dataset bytes differ")
    commands = report.get("command_order", [])
    replicas = report.get("replicas", [])
    count = report.get("num_envs")
    if (type(count) is not int or count < len(commands) or not commands
            or len(replicas) != count or [r.get("env") for r in replicas] != list(range(count))
            or report.get("accepted_command_indices") != list(range(len(commands)))):
        raise ValueError("Native replay command/replica coverage differs")
    accepted_ids = set()
    command_indices = report.get("command_index_by_env", [])
    if len(command_indices) != count:
        raise ValueError("Native replay command mapping is incomplete")
    for e, replica in enumerate(replicas):
        ci = command_indices[e]
        if type(ci) is not int or not 0 <= ci < len(commands) or replica.get("command") != commands[ci]:
            raise ValueError("Native replay command mapping differs")
        if replica.get("accepted") is True:
            if (replica.get("rejected_reasons") != [] or replica.get("terminated") is not False
                    or replica.get("truncated") is not False or replica.get("nonfoot_contact") is not False):
                raise ValueError("Native replay has contradictory accepted row")
            accepted_ids.add(ci)
    if accepted_ids != set(range(len(commands))):
        raise ValueError("Native replay lacks accepted commands")
    state_path = report_path.parent.parent / "state.json"
    if not state_path.is_file():
        raise ValueError("Missing completed native replay acquisition state")
    state = json.loads(state_path.read_text())
    if (state.get("status") != "completed" or state.get("mode") != "replay"
            or state.get("errors") or state.get("replay") != report):
        raise ValueError("Native replay acquisition failed or report differs")
    observed = state.get("identity", {})
    if any(observed.get(key) != identity[key] for key in PHYSICS_IDENTITY_KEYS):
        raise ValueError("Native replay acquisition physics identity differs")
    return {"report_path": str(report_path), "report_sha256": sha(report_path),
            "state_path": str(state_path), "state_sha256": sha(state_path),
            "dataset_path": str(prior_path), "dataset_sha256": sha(prior_path),
            "transitions": row["transitions"], "independent_heldout_variant": report.get("independent_heldout_variant")}



def prepare_policy_scene(env):
    """Lighting and visual-only meter grid for an actual native policy recording."""
    from pxr import UsdGeom, UsdLux, Gf
    stage = env.sim.stage
    light = UsdLux.DomeLight.Define(stage, "/PolicyVideoLight")
    light.CreateIntensityAttr(1800.)
    light.CreateColorAttr(Gf.Vec3f(.90, .94, 1.))
    ground = UsdGeom.Mesh(stage.GetPrimAtPath("/Ground"))
    ground.CreateDisplayColorAttr([Gf.Vec3f(.18, .20, .23)])
    # These strips have no CollisionAPI or rigid body. They provide a visible
    # distance reference; all robot-ground contact remains the original mesh.
    for axis in range(2):
        for i in range(-10, 11):
            line = UsdGeom.Cube.Define(stage, f"/PolicyVideoGrid/axis{axis}_{i+10:02d}")
            line.CreateSizeAttr(1.)
            transform = UsdGeom.Xformable(line.GetPrim())
            transform.AddTranslateOp().Set(Gf.Vec3d(i*.5 if axis == 0 else 0.,
                                                  i*.5 if axis == 1 else 0., .00015))
            transform.AddScaleOp().Set(Gf.Vec3f(.002 if axis == 0 else 10.,
                                              .002 if axis == 1 else 10., .0001))
            line.CreateDisplayColorAttr([Gf.Vec3f(.30, .33, .36)])
    save(env.output / "video_scene.json", {"lighting": "dome1800", "grid_spacing_m": .5,
        "grid_has_collision": False, "robot_physics_changed": False})


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
    parser.add_argument("--mode", choices=("diagnostic", "replay", "train", "video", "evaluate"), required=True)
    for name in ("asset", "model", "geometry", "geometry-extrema", "prior", "prior-metadata", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--standing-admission", type=Path)
    parser.add_argument("--source-freeze-sha256")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--eval-scope", choices=("diagnostic", "learning", "full"), default="diagnostic")
    parser.add_argument("--probe-video-case", help="Record this one additional learning-probe case")
    parser.add_argument("--probe-cases", nargs="+", choices=LEARNING_PROBE_CASE_IDS,
        help="Ordered learning-probe subset; omitted probes remain missing (default: all 13)")
    parser.add_argument("--realized-prior", type=Path)
    parser.add_argument("--replay-report", type=Path)
    parser.add_argument("--bc-steps", type=int, default=0)
    parser.add_argument("--initial-std", type=initial_std_value, default=None,
        help="Initial action standard deviation for fresh training (default: 0.4); unavailable with --checkpoint")
    parser.add_argument("--learning-rate", type=positive_training_value, default=None,
        help="PPO learning rate for fresh training (default: 0.0003)")
    parser.add_argument("--target-kl", type=positive_training_value, default=None,
        help="Positive KL target for stopping further PPO steps after a crossing step (default: disabled)")
    parser.add_argument("--bc-optimizer", choices=("shared", "separate"), default=None,
        help="BC optimizer for fresh training (default: shared with PPO)")
    parser.add_argument("--bc-learning-rate", type=positive_training_value, default=None,
        help="BC learning rate for fresh training; required for a separate optimizer")
    parser.add_argument("--bc-velocity-coefficient", type=nonnegative_training_value, default=None,
        help="Native velocity supervision coefficient for fresh BC (default: 0)")
    parser.add_argument("--bc-detach-velocity", action=argparse.BooleanOptionalAction, default=None,
        help="Detach estimated velocity from the action loss during fresh BC (default: false)")
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
    learner_overrides = fresh_learner_overrides(args.mode, args.checkpoint,
        initial_std=args.initial_std, learning_rate=args.learning_rate, target_kl=args.target_kl,
        bc_optimizer=args.bc_optimizer, bc_learning_rate=args.bc_learning_rate,
        bc_velocity_coefficient=args.bc_velocity_coefficient,bc_detach_velocity=args.bc_detach_velocity)
    if args.probe_cases is not None:
        if args.mode != "evaluate" or args.eval_scope != "learning":
            raise ValueError("--probe-cases requires --mode evaluate --eval-scope learning")
        if len(set(args.probe_cases)) != len(args.probe_cases):
            raise ValueError("Duplicate selected learning-probe case")
        if args.probe_video_case is not None and args.probe_video_case not in args.probe_cases:
            raise ValueError("Video case is outside the selected learning probes")
    config_type = EnvConfig
    if args.mode == "evaluate":
        from .evaluation_config import EvaluationEnvConfig
        config_type = EvaluationEnvConfig
    cfg = config_type(num_envs=args.num_envs, render=args.mode in ("video", "evaluate"),
                    episode_seconds=90. if args.mode == "evaluate" else 20. if args.mode == "train" else 60.,
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
    identity['realized_prior_sha256']=sha(args.realized_prior) if args.realized_prior else None
    if args.mode in ("train", "video", "evaluate"):
        identity["learner_configuration_source"] = "checkpoint" if args.checkpoint is not None else "fresh"
        identity["fresh_initial_std"] = (.4 if args.initial_std is None else args.initial_std) if args.checkpoint is None else None
        identity["fresh_learner_overrides"] = learner_overrides if args.checkpoint is None else None
    if args.bc_steps < 0 or args.bc_steps > 5000 or (args.bc_steps and (args.mode != "train" or not args.realized_prior)):
        raise ValueError('BC requires an explicit accepted native replay dataset and bounded steps')
    if args.source_freeze_sha256 is not None:
        freeze = Path(__file__).resolve().parent / "FREEZE_SHA256.json"
        if not freeze.is_file() or sha(freeze) != args.source_freeze_sha256:
            raise ValueError("Actual source-freeze manifest differs")
    if args.mode != "diagnostic":
        identity["standing_admission"] = require_admission(args.standing_admission, identity, cfg)
    if args.mode in ("train", "video", "evaluate"):
        identity["accepted_native_replay"] = require_realized_prior(args.realized_prior, args.replay_report, identity)
    if args.mode in ("video", "evaluate") and (args.checkpoint is None or cfg.num_envs != 1):
        raise ValueError("Policy recording/evaluation requires one robot and an actual learned checkpoint")
    if args.probe_video_case and (args.mode != "evaluate" or args.eval_scope != "learning"):
        raise ValueError("A probe video case requires evaluation scope learning")
    if args.mode == "evaluate":
        identity["evaluation_scope"] = args.eval_scope
        identity["evaluation_episode_timeout_seconds"] = 90.
        if args.eval_scope == "learning":
            identity["learning_probe_case_ids"] = list(LEARNING_PROBE_CASE_IDS) if args.probe_cases is None else args.probe_cases
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
        if cfg.render:
            # Isaac Lab3 maps the deprecated explicit CLI flag to "viz none".
            # Keep the application headless while allowing configured Kit RGB.
            args.headless_explicit = False
            state["camera_launcher"] = {"headless": True, "enable_cameras": True,
                "legacy_disable_visualizers": False}
        # Keep startup stacks on a dedicated descriptor: AppLauncher temporarily
        # redirects stdout, and a stalled native import otherwise leaves no trace.
        import faulthandler
        with (args.output / "startup_tracebacks.log").open("w") as startup_trace:
            faulthandler.enable(file=startup_trace)
            faulthandler.dump_traceback_later(45., repeat=True, file=startup_trace)
            try:
                app = AppLauncher(args).app
            finally:
                faulthandler.cancel_dump_traceback_later()
                faulthandler.disable()
        print("REFERENCE_SCREEN_APP_READY", flush=True)
        import torch
        from .env import PaperWalkEnv, DiagnosticCapture, score_diagnostic
        torch.manual_seed(cfg.seed)
        env = PaperWalkEnv(cfg, args.asset, args.model, args.geometry, args.output / "native", reference_metadata=metadata)
        if cfg.render:
            prepare_policy_scene(env)
            from .camera import NativePolicyCamera
            env.render = NativePolicyCamera(env)
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
        elif args.mode == "replay":
            from .replay import realize_prior
            def replay_progress(metrics):
                print('PAPER_REPLAY '+json.dumps(metrics,allow_nan=False),flush=True)
            state['replay']=realize_prior(env,args.prior,args.prior_metadata,args.output/'native_replay',
                progress=replay_progress, physics_identity=physics_identity(identity))
        else:
            from .learner import PPOLearner
            checkpoint = None if args.checkpoint is None else torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            learner_cfg = learner_configuration(args.mode, env.num_envs, args.initial_std, checkpoint,
                learning_rate=args.learning_rate, target_kl=args.target_kl,
                bc_optimizer=args.bc_optimizer, bc_learning_rate=args.bc_learning_rate,
                bc_velocity_coefficient=args.bc_velocity_coefficient,bc_detach_velocity=args.bc_detach_velocity)
            training_env=env
            if args.mode=='train':
                from .task import TrainingTask
                training_env=TrainingTask(env,output_dir=args.output/'task')
            learning_prior=args.realized_prior
            learner = PPOLearner(training_env, learning_prior, args.output / "learner", learner_cfg, device=env.device)
            if args.checkpoint is not None:
                learner.load(args.checkpoint)
                state["input_checkpoint_sha256"] = sha(args.checkpoint)
            if args.mode == "train":
                if args.bc_steps:
                    learner.pretrain_bc(learning_prior,args.bc_steps)
                def progress(metrics):
                    task_status = training_env.status(reset_interval=True)
                    save(args.output / "task" / "latest_status.json", task_status)
                    print("PAPER_PPO " + json.dumps(metrics, allow_nan=False), flush=True)
                    print("PAPER_TASK " + json.dumps(task_status, allow_nan=False), flush=True)
                    return time.monotonic() - started < args.max_wall_seconds
                state["checkpoint"] = learner.train(args.updates, checkpoint_interval=args.checkpoint_interval, callback=progress)
            elif args.mode == "evaluate":
                from .evaluate import evaluate_suite, run_diagnostic_trial, run_learning_probe_suite
                def evaluation_progress(metrics):
                    print("PAPER_EVALUATION " + json.dumps(metrics, allow_nan=False), flush=True)
                if args.eval_scope == "full":
                    state["evaluation"] = evaluate_suite(env, learner, args.output / "evaluation",
                        args.geometry_extrema, args.checkpoint, progress=evaluation_progress,
                        max_wall_seconds=args.max_wall_seconds)
                elif args.eval_scope == "learning":
                    state["evaluation"] = run_learning_probe_suite(env, learner, args.output / "evaluation",
                        args.geometry_extrema, args.checkpoint, progress=evaluation_progress,
                        record_video=bool(args.probe_video_case), video_case_id=args.probe_video_case,
                        max_wall_seconds=args.max_wall_seconds, selected_case_ids=args.probe_cases)
                else:
                    state["evaluation"] = run_diagnostic_trial(env, learner, args.output / "evaluation",
                        args.geometry_extrema, args.checkpoint, command=args.command,
                        record_video=True, progress=evaluation_progress,
                        max_wall_seconds=args.max_wall_seconds)
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
