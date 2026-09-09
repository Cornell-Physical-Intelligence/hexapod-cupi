#!/usr/bin/env python3
"""External, bounded evaluation driver. Default is CPU preflight; --execute starts Kit.

Run execution only inside an independently supervised, exclusively owned Isaac
container. This file does not create containers, acquire GPU ownership, or admit a
policy. The frozen training snapshot must be a different, read-only directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
from evaluation_math import (SEEDS, STEPS, POLICY_DT_S, scenarios, scheduled_command, score_trial,
                             native_xyzw_to_wxyz, verify_reset_identity)

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def driver_identity():
    names = ("evaluate.py", "evaluation_math.py", "README.md", "sdk_contract_evidence.json")
    records = {name: sha(HERE/name) for name in names}
    return {"files": records, "sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()}


def parser(add_launcher_args=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--admission", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--controller", choices=("learned", "zero"), default="learned")
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    if add_launcher_args:
        add_launcher_args(p)
        p.set_defaults(visualizer=[])
    return p


def frozen_module_paths(source):
    result = {}
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("hexapod_core", "hexapod_env", "hexapod_rl", "hexapod_runtime")) or name in (
                "mkii_training_contract", "train_mkii_fourbar", "validate_mkii_fourbar"):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if not path.is_relative_to(source):
                    raise ValueError(f"Task module escaped frozen snapshot: {name}: {path}")
                result[name] = str(path)
    return result


def verify_installed_sdk(classes):
    """Bind reviewed native quaternion/reset/RSL APIs to their actual source files."""
    import inspect
    evidence = json.loads((HERE/"sdk_contract_evidence.json").read_text())["files"]
    expected = {value["sha256"] for value in evidence.values()}
    observed = {}
    for cls in classes:
        filename = inspect.getsourcefile(cls)
        if filename is None:
            raise ValueError(f"Cannot identify SDK source for {cls}")
        path = Path(filename).resolve()
        checksum = sha(path)
        if checksum not in expected:
            raise ValueError(f"SDK source differs from reviewed quaternion/reset/inference contract: {path}")
        observed[str(path)] = checksum
    return observed


def reset_without_physics(env, seed, guard):
    before = guard.total, guard.policy_samples
    result = env.reset(seed=seed)
    if (guard.total, guard.policy_samples) != before:
        raise ValueError("Explicit reset unexpectedly generated physics samples")
    return result


def prepare(args):
    source = args.source.resolve()
    if HERE.is_relative_to(source) or args.report.resolve().is_relative_to(source):
        raise ValueError("Driver and output must be outside the frozen training snapshot")
    if not (source/"tools/mkii_training_contract.py").is_file():
        raise ValueError("Source is not a physical MKII training snapshot")
    if (args.controller == "learned") != (args.checkpoint is not None):
        raise ValueError("Learned evaluation requires a checkpoint; zero baseline must omit it")
    frozen_module_paths(source)  # Reject an interpreter already using another checkout.
    sys.path[:0] = [str(source/path) for path in (
        "tools", "isaaclab", "packages/hexapod_core", "packages/hexapod_env", "packages/hexapod_runtime")]
    helpers = importlib.import_module("mkii_training_contract")
    contract = helpers.identity(source)
    admission = helpers.require_admission(args.admission, contract)
    checkpoint = helpers.require_checkpoint(args.checkpoint, contract) if args.checkpoint else None
    runtime = admission.get("runtime_manifest", {})
    from hexapod_core.fourbar_v1 import validate_asset_bundle
    bundle = validate_asset_bundle(runtime.get("asset_bundle"))
    if any(contract["files"].get(path) != checksum for path, checksum in bundle["bundle_files_sha256"].items()):
        raise ValueError("Admitted asset bytes differ from frozen source")
    usd = (source/bundle["usd_path_relative"]).resolve()
    if not usd.is_relative_to(source) or sha(usd) != bundle["usd_root_sha256"]:
        raise ValueError("Admitted USD escaped or differs from frozen source")
    frozen_module_paths(source)
    if "pxr" in sys.modules:
        raise ValueError("Preflight imported standalone USD before Kit")
    return source, helpers, contract, admission, checkpoint, usd


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    early, unknown = parser().parse_known_args(argv)
    early.report = early.report.resolve()
    if unknown and not early.execute:
        parser().error("Launcher arguments require --execute")
    if early.report.exists():
        raise ValueError("Refusing to overwrite evaluation evidence")
    output = early.report.resolve().parent
    if output.is_relative_to(early.source.resolve()):
        raise ValueError("Output must be outside the frozen snapshot")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema": "hexapod.first_policy_evaluation.v1", "pass": False,
        "evaluation_complete": False, "physical_integrity_pass": False, "policy_quality_pass": None,
        "simulation_training_admission": False, "hardware_admission": False,
        "controller": early.controller, "errors": [], "trials": [], "control_steps": 0,
        "protocol": {"seeds": list(SEEDS), "scenarios": scenarios(), "steps_per_seed": STEPS,
                     "policy_dt_s": POLICY_DT_S, "trial_count": 42},
        "runtime_execution_exercised": False}
    persisted = False
    helpers = None
    contract = None

    def save_trials(seed, initial, rows):
        trace = output/f"trials_seed_{seed}.json"
        payload = [{"seed": seed, "scenario": scenario, "initial": init, "rows": trial_rows}
                   for scenario, init, trial_rows in zip(scenarios(), initial, rows)]
        helpers.write_json(trace, payload)
        report.setdefault("traces", []).append({"file": trace.name, "sha256": sha(trace)})
        report["trials"] += [score_trial(scenario, seed, init, trial_rows)
                             for scenario, init, trial_rows in zip(scenarios(), initial, rows) if trial_rows]

    def finish(error=None):
        nonlocal persisted
        if persisted:
            return
        if error is not None:
            report["errors"].append(f"{type(error).__name__}: {error}")
        if helpers is not None and contract is not None:
            try:
                if helpers.identity(source) != contract or driver_identity() != report["evaluation_driver"]:
                    raise ValueError("Training snapshot or evaluator changed during evaluation")
                if sha(early.admission) != report["admission_sha256"]:
                    raise ValueError("Admission bytes changed during evaluation")
                if early.checkpoint and helpers.require_checkpoint(early.checkpoint, contract) != report["checkpoint_sidecar"]:
                    raise ValueError("Checkpoint or sidecar changed during evaluation")
                report["source_unchanged"] = True
                report["loaded_task_modules"] = frozen_module_paths(source)
            except Exception as exc:
                report["errors"].append(str(exc))
        if report["errors"]:
            report["evaluation_complete"] = report["physical_integrity_pass"] = False
        temporary = early.report.with_suffix(".json.partial")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        temporary.replace(early.report)
        persisted = True
        print("FIRST_POLICY_EVALUATION_RESULT "+json.dumps({key: report[key] for key in (
            "evaluation_complete", "physical_integrity_pass", "policy_quality_pass", "errors")}), flush=True)

    try:
        report["evaluation_driver"] = driver_identity()
        source, helpers, contract, admission, checkpoint, usd = prepare(early)
        report.update(contract=contract, source=str(source), admission_sha256=sha(early.admission),
                      checkpoint_sidecar=checkpoint, checkpoint_path=str(early.checkpoint) if checkpoint else None,
                      preflight_pass=True, execution="not_started")
        archive = output/"evaluation_driver"
        archive.mkdir(exist_ok=False)
        for name, expected in report["evaluation_driver"]["files"].items():
            data = (HERE/name).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("Evaluator changed before its source archive was saved")
            (archive/name).write_bytes(data)
        if not early.execute:
            finish()
            return 0
        os.environ["HEXAPOD_USD_PATH"] = os.environ["HEXAPOD_MKII_FOURBAR_USD_PATH"] = str(usd)
        from hexapod_env.tasks.mkii_fourbar_v1.register import register_mkii_fourbar_v1
        register_mkii_fourbar_v1()
        from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli
        original_argv = sys.argv
        try:
            sys.argv = [__file__, *argv]
            args, overrides = setup_preset_cli(parser(add_launcher_args))
            if overrides:
                raise ValueError(f"Unreviewed task overrides: {overrides}")
            sys.argv = [__file__]
            cfg, agent_cfg = resolve_task_config(helpers.TASK_ID, "rsl_rl_cfg_entry_point")
        finally:
            sys.argv = original_argv
        cfg.scene.num_envs = len(scenarios())
        cfg.seed = agent_cfg.seed = SEEDS[0]
        cfg.log_dir = str(output/"runtime")
        if cfg.sim.dt*cfg.decimation != POLICY_DT_S or cfg.episode_length_s <= STEPS*POLICY_DT_S:
            raise ValueError("Frozen timing/episode length cannot implement the fixed protocol")
        if cfg.is_finite_horizon:
            raise ValueError("Evaluation requires explicit timeout reporting from the frozen task")
        report["rsl_rl_version"] = importlib.metadata.version("rsl-rl-lib")
        if report["rsl_rl_version"] != "5.0.1" or "pxr" in sys.modules:
            raise ValueError("Unreviewed RSL release or pre-Kit USD import")
        if helpers.identity(source) != contract:
            raise ValueError("Frozen source changed before Kit launch")
        with launch_simulation(cfg, args):
            report["kit_launched"] = True
            env = guard = None
            guard_update = None
            pending_trial = None
            try:
                import torch
                import gymnasium as gym
                from rsl_rl.runners import OnPolicyRunner
                from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
                from train_mkii_fourbar import (PhysicalTrainingGuard, compare_admitted_runtime,
                    algorithm_state_digest, restore_adaptive_learning_rate)
                from validate_mkii_fourbar import PhysicalMetrics
                from hexapod_env.tasks.mkii_fourbar_v1.env import tensor
                from hexapod_env.tasks.mkii_fourbar_v1.math import reward_terms
                from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData
                from isaaclab.envs import DirectRLEnv
                env = gym.make(helpers.TASK_ID, cfg=cfg)
                raw = env.unwrapped
                sdk_classes = [BaseArticulationData, type(raw._robot.data), DirectRLEnv]
                report["runtime_manifest"] = raw.runtime_manifest
                comparison = compare_admitted_runtime(admission.get("runtime_manifest"), raw.runtime_manifest)
                report["admitted_runtime_comparison"] = comparison
                if not comparison["pass"]:
                    raise ValueError("Evaluation runtime differs from admitted nominal runtime")
                if raw.runtime_manifest["observation_dim"] != 84 or len(raw.active_joint_names) != 18:
                    raise ValueError("Evaluator requires the frozen 84-observation/18-action contract")
                wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
                runner = policy = None
                if checkpoint:
                    if helpers.require_checkpoint(early.checkpoint, contract) != checkpoint:
                        raise ValueError("Checkpoint changed before load")
                    agent_cfg.device, agent_cfg.logger = str(raw.device), "tensorboard"
                    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=str(output/"inference"), device=agent_cfg.device)
                    infos = runner.load(str(early.checkpoint.resolve()), strict=True, map_location=agent_cfg.device)
                    if infos != {"contract": contract, "next_iteration": checkpoint["next_iteration"]}:
                        raise ValueError("Embedded checkpoint lineage differs from verified sidecar")
                    restore_adaptive_learning_rate(runner.alg)
                    report["learner_loaded_sha256"] = algorithm_state_digest(runner)
                    policy = runner.get_inference_policy(device=agent_cfg.device)
                    report["learner_before_sha256"] = algorithm_state_digest(runner)
                    if report["learner_before_sha256"] != report["learner_loaded_sha256"]:
                        raise ValueError("Switching actor evaluation mode changed serialized learner state")
                    if any(module.training for module in policy.modules()):
                        raise ValueError("Actor/normalization did not enter deterministic evaluation mode")
                    report["actor_eval_mode_pass"] = True
                    sdk_classes += [OnPolicyRunner, type(runner.alg)]
                report["installed_sdk_source_sha256"] = verify_installed_sdk(sdk_classes)
                metrics = PhysicalMetrics(raw, raw.kinematics)
                guard = PhysicalTrainingGuard(raw, metrics)
                guard.__enter__()
                guard_update = raw.scene.update
                maxima, minimum, frame = {}, None, None

                def capture_update(*values, **keywords):
                    nonlocal minimum, frame
                    result = guard_update(*values, **keywords)
                    current = {"max_raw_demand_nm": raw.motor_telemetry("raw_demand_nm").abs().amax(-1),
                        "max_applied_nm": raw.motor_state("applied_torque").abs().amax(-1),
                        "max_clipping_nm": raw.motor_telemetry("clipping_nm").amax(-1),
                        "max_continuous_overload_nm": raw.motor_telemetry("continuous_overload_nm").amax(-1),
                        "max_estimated_phase_current_arms": raw.motor_telemetry("estimated_phase_current_arms").amax(-1),
                        "max_motor_speed_rad_s": raw.motor_state("joint_vel").abs().amax(-1)}
                    for name, value in current.items():
                        maxima[name] = value.clone() if name not in maxima else torch.maximum(maxima[name], value)
                    value = raw.motor_telemetry("burst_headroom").amin(-1)
                    minimum = value.clone() if minimum is None else torch.minimum(minimum, value)
                    if guard.policy_samples == raw.cfg.decimation:
                        data = raw._robot.data
                        pads, shafts, speeds, _ = raw._contact_state()
                        nonfoot, _ = raw._nonfoot_contact_counts(shafts)
                        linear = raw._vector_in_command_frame(tensor(data.root_lin_vel_b))
                        angular = raw._vector_in_command_frame(tensor(data.root_ang_vel_b))
                        height = tensor(data.root_pos_w)[:,2]-tensor(raw._terrain.env_origins)[:,2]
                        components = reward_terms(command=raw._commands, linear_navigation=linear,
                            angular_navigation=angular, gravity_body=tensor(data.projected_gravity_b),
                            active_torque=raw.motor_state("applied_torque"), active_velocity=raw.motor_state("joint_vel"),
                            active_acceleration=raw.motor_state("joint_acc"), active_position=raw.motor_state("joint_pos"),
                            soft_limits=raw.coordinates.soft_limits, action=raw._actions, previous_action=raw._previous_actions,
                            height=height, nominal_height=raw.cfg.nominal_height_m, nonfoot_contacts=nonfoot,
                            support_count=pads.sum(-1), foot_slip=(speeds.square()*pads).sum(-1),
                            clipping_nm=raw.motor_telemetry("clipping_nm"), overload_nm=raw.motor_telemetry("continuous_overload_nm"))
                        fields = {**maxima, "min_headroom": minimum, "command": raw._commands,
                            "position_world_m": tensor(data.root_pos_w), "quaternion_xyzw": tensor(data.root_quat_w),
                            "velocity_navigation_m_s": linear, "velocity_world_m_s": tensor(data.root_lin_vel_w),
                            "yaw_rate_rad_s": angular[:,2], "height_m": height, "support_count": pads.sum(-1),
                            "nonfoot_contacts": nonfoot,
                            "loaded_foot_slip_rms_m_s": ((speeds.square()*pads).sum(-1)/pads.sum(-1).clamp_min(1)).sqrt(),
                            "max_motor_overload_exposure_s": raw.motor_telemetry("applied_overload_exposure_s").amax(-1),
                            "max_motor_peak_exposure_s": raw.motor_telemetry("applied_peak_exposure_s").amax(-1)}
                        frame = {key: value.detach().cpu().tolist() for key, value in fields.items()}
                        frame["quaternion_wxyz"] = [native_xyzw_to_wxyz(q) for q in frame["quaternion_xyzw"]]
                        frame["reward_components"] = {key: value.detach().cpu().tolist() for key, value in components.items()}
                    return result

                raw.scene.update = capture_update
                report.update(execution="running", runtime_execution_exercised=True,
                              torch_version=torch.__version__, physics_substeps=0)
                started = time.monotonic()
                for seed in SEEDS:
                    reset_without_physics(env, seed, guard)
                    initial_pos = tensor(raw._robot.data.root_pos_w).detach().cpu().tolist()
                    initial_quat = tensor(raw._robot.data.root_quat_w).detach().cpu().tolist()
                    report.setdefault("reset_quaternion_checks", []).append({"seed": seed, **verify_reset_identity(initial_quat)})
                    initial = [{"position_world_m": pos, "quaternion_xyzw": quat,
                                "quaternion_wxyz": native_xyzw_to_wxyz(quat)} for pos, quat in zip(initial_pos, initial_quat)]
                    rows = [[] for _ in scenarios()]
                    pending_trial = (seed, initial, rows)
                    live = [True]*len(rows)
                    metrics.window = f"seed_{seed}"
                    for step in range(STEPS):
                        commands, windows = zip(*(scheduled_command(scenario, step) for scenario in scenarios()))
                        command = torch.tensor(commands, device=raw.device, dtype=torch.float32)
                        raw._commands.copy_(command)
                        raw._command_time_left_s.fill_(1.)
                        obs = wrapped.get_observations()
                        if obs["policy"].shape != (14,84) or not torch.allclose(obs["policy"][:,9:12], command, atol=1e-6, rtol=0):
                            raise ValueError("Policy observations do not contain the scheduled commands")
                        if not bool(torch.isfinite(obs["policy"]).all()):
                            raise ValueError("Nonfinite evaluation observations")
                        maxima, minimum, frame = {}, None, None
                        with torch.inference_mode():
                            actions = policy(obs) if policy else torch.zeros(14,18,device=raw.device)
                            if actions.shape != (14,18) or not bool(torch.isfinite(actions).all()):
                                raise ValueError("Invalid deterministic actor actions")
                            actions = actions.clone()
                            actions[torch.tensor([not value for value in live], device=raw.device)] = 0.
                            _, reward, done, info = wrapped.step(actions)
                        if frame is None or not bool(torch.isfinite(reward).all()):
                            raise ValueError("Missing pre-reset state or nonfinite reward")
                        timeouts = info.get("time_outs", torch.zeros_like(done)).detach().cpu().tolist()
                        dones, rewards = done.detach().cpu().tolist(), reward.detach().cpu().tolist()
                        reasons = {name: value.detach().cpu().tolist() for name, value in raw.last_termination_reasons.items()}
                        for i in range(14):
                            if not live[i]:
                                continue
                            row = {key: value[i] for key, value in frame.items() if key != "reward_components"}
                            row.update(step=step, time_s=(step+1)*POLICY_DT_S, window=windows[i],
                                reward=rewards[i], terminated=bool(dones[i] and not timeouts[i]), truncated=bool(timeouts[i]),
                                termination_reasons=[name for name, values in reasons.items() if values[i]],
                                reward_components={key: values[i] for key, values in frame["reward_components"].items()})
                            rows[i].append(row)
                            live[i] = not dones[i]
                        report["control_steps"] += 1
                        if (step+1) % 100 == 0:
                            helpers.write_json(output/"progress.json", {"seed": seed, "seed_steps": step+1,
                                "control_steps": report["control_steps"], "physics_substeps": guard.total,
                                "elapsed_seconds": time.monotonic()-started})
                    guard.require_coverage(report["control_steps"])
                    save_trials(seed, initial, rows)
                    pending_trial = None
                    helpers.write_json(output/"progress.json", {"seeds_completed": len(report["traces"]),
                        "control_steps": report["control_steps"], "physics_substeps": guard.total,
                        "elapsed_seconds": time.monotonic()-started})
                if runner:
                    report["learner_after_sha256"] = algorithm_state_digest(runner)
                    if report["learner_after_sha256"] != report["learner_before_sha256"]:
                        raise ValueError("Inference changed policy/normalization/optimizer state")
                if len(report["trials"]) != 42 or report["control_steps"] != 1500:
                    raise ValueError("Incomplete fixed evaluation protocol")
                report.update(evaluation_complete=True, physical_integrity_pass=True, execution="finished",
                    trials_completed=sum(trial["completed"] for trial in report["trials"]),
                    trials_fallen=sum(trial["fall"] for trial in report["trials"]),
                    trials_terminated=sum(trial["terminated"] for trial in report["trials"]),
                    trials_truncated=sum(trial["truncated"] for trial in report["trials"]))
            except BaseException as error:
                if guard:
                    report.update(physical_metrics=guard.metrics.windows, physics_substeps=guard.total)
                if pending_trial and any(pending_trial[2]):
                    try:
                        save_trials(*pending_trial)
                    except Exception as trace_error:
                        report["errors"].append(f"Partial trace persistence: {trace_error}")
                finish(error)
                raise
            finally:
                if guard:
                    report.update(physical_metrics=guard.metrics.windows, physics_substeps=guard.total)
                    if guard_update:
                        raw.scene.update = guard_update
                    guard.__exit__()
                finish()  # Native Kit shutdown may exit Python before the outer finally.
                if env:
                    env.close()
    except Exception as error:
        import traceback
        traceback.print_exc()
        if not persisted:
            finish(error)
        return 1
    finish()
    return 0 if report["evaluation_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
