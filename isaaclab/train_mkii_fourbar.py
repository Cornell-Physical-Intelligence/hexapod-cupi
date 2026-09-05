#!/usr/bin/env python3
"""Bounded, lineage-checked PPO for the physical MKII four-bar task.

Run only through deploy/run-mkii-fourbar. Checkpoints include optimizer state,
observation normalization, source/asset identity and an explicit next iteration.
This is provisional simulation learning, never a hardware admission.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ("tools", "isaaclab", "packages/hexapod_core", "packages/hexapod_env")]
from mkii_training_contract import TASK_ID, digest, identity, require_admission, require_checkpoint, write_json


class PauseRequested(Exception):
    pass


def parser(add_launcher_args=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--admission", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--report", type=Path, required=True)
    if add_launcher_args:
        add_launcher_args(p)
        p.set_defaults(visualizer=[])
    return p


def parameters_digest(runner):
    h = hashlib.sha256()
    for key, tensor in sorted(runner.alg.get_policy().named_parameters()):
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def finite_state_digest(value):
    """Digest/check every tensor and scalar, including critic, moments and buffers."""
    import torch
    h = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            data = item.detach().cpu().contiguous()
            if not bool(torch.isfinite(data).all()):
                raise ValueError("Nonfinite model/optimizer/normalization state")
            h.update(f"tensor:{data.dtype}:{tuple(data.shape)}:".encode())
            h.update(data.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            h.update(b"dict{")
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(item[key])
            h.update(b"}")
        elif isinstance(item, (tuple, list)):
            h.update(type(item).__name__.encode() + b"[")
            for child in item:
                visit(child)
            h.update(b"]")
        elif item is None or isinstance(item, (str, bool, int, float)):
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("Nonfinite model/optimizer scalar state")
            h.update(json.dumps([type(item).__name__, item], allow_nan=False).encode())
        else:
            raise ValueError(f"Unsupported checkpoint state value: {type(item).__name__}")

    visit(value)
    return h.hexdigest()


def algorithm_state_digest(runner):
    return finite_state_digest({"saved_algorithm": runner.alg.save(),
                                "adaptive_learning_rate": runner.alg.learning_rate})


def restore_adaptive_learning_rate(algorithm):
    """RSL5.0.1 loads optimizer LR but omits its adaptive-scheduler scalar."""
    rates = [group["lr"] for group in algorithm.optimizer.param_groups]
    if not rates or any(not math.isfinite(rate) or rate <= 0 or rate != rates[0] for rate in rates):
        raise ValueError("Resume requires consistent finite positive optimizer learning rates")
    algorithm.learning_rate = rates[0]


def check_training_physics(windows):
    """Exploration may fall; it may not exploit broken loops or motor bounds."""
    limits = {"max_closure_point_m": .0001, "max_closure_axis_chord": math.radians(.1),
              "max_envelope_excess_nm": 1e-5, "invalid_samples": 0}
    for window in windows.values():
        for key, maximum in limits.items():
            value = window[key]
            if not math.isfinite(value) or value > maximum:
                raise ValueError(f"Training physical guard: {key}={value} exceeds {maximum}")


class PhysicalTrainingGuard:
    """Capture fresh physics state after every scene update, before any reset."""
    def __init__(self, raw, metrics):
        self.raw, self.metrics, self.total = raw, metrics, 0
        self.policy_counter, self.policy_samples = None, 0
        self.installed = False

    def __enter__(self):
        if (getattr(self.raw, "_physics_handles_decimation", None) is not False
                or self.raw.cfg.sim.dt != .005 or self.raw.cfg.decimation != 4):
            raise ValueError("Training physical guard requires explicit4x5ms updates")
        if any(not 0 <= s.cfg.update_period <= .005 for s in self.raw._body_contact_sensors.values()):
            raise ValueError("Training contact sensors cannot cover every physics substep")
        self.original = self.raw.scene.update
        self.had_override = "update" in vars(self.raw.scene)
        self.previous = vars(self.raw.scene).get("update")

        def update(*args, **kwargs):
            dt = kwargs.get("dt", args[0] if args else None)
            if dt != .005:
                raise ValueError("Unexpected scene update during training")
            counter = self.raw.common_step_counter
            if counter != self.policy_counter:
                if self.policy_counter is not None and self.policy_samples != 4:
                    raise ValueError("Missing training physics substep")
                self.policy_counter, self.policy_samples = counter, 0
            if self.policy_samples >= 4:
                raise ValueError("Extra training physics substep")
            result = self.original(*args, **kwargs)
            self.metrics.capture()
            self.total += 1
            self.policy_samples += 1
            if self.policy_samples == 4:
                self.metrics.drain()
                check_training_physics(self.metrics.windows)
            return result

        self.raw.scene.update = update
        self.installed = True
        return self

    def require_coverage(self, expected_control_steps):
        if (self.total != expected_control_steps * 4 or self.metrics.pending
                or (self.total and self.policy_samples != 4)):
            raise ValueError("Incomplete training/inference physical sample coverage")

    def __exit__(self, *_):
        if self.installed:
            if self.had_override:
                self.raw.scene.update = self.previous
            else:
                del self.raw.scene.update
            self.installed = False


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    early, _ = parser().parse_known_args(argv)
    if early.report.exists():
        raise ValueError("Refusing to overwrite run evidence")
    output = early.report.parent
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "task_id": TASK_ID, "mode": "provisional_physical_fourbar_ppo",
              "pass": False, "errors": [], "hardware_admission": False, "paused": False,
              "iterations_requested": early.iterations, "iterations_completed": 0}
    persisted = False

    def finish(error=None):
        nonlocal persisted
        if persisted:
            return
        if error:
            report["errors"].append(f"{type(error).__name__}: {error}")
        report["pass"] = (not report["errors"] and report.get("checkpoint_verified") is True
                          and report["iterations_completed"] > 0)
        write_json(early.report, report)
        persisted = True
        print(f"FOURBAR_TRAIN_RESULT {json.dumps({k: report.get(k) for k in ('pass', 'paused', 'iterations_completed', 'errors')})}", flush=True)

    try:
        report["contract"] = contract = identity()
        require_admission(early.admission, contract)
        resume = require_checkpoint(early.checkpoint, contract) if early.checkpoint else None
        if not 1 <= early.num_envs <= 4096 or not 1 <= early.iterations <= 10000:
            raise ValueError("Invalid bounded training workload")
        from hexapod_env.tasks.mkii_fourbar_v1.register import register_mkii_fourbar_v1
        register_mkii_fourbar_v1()
        from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli
        saved = sys.argv
        try:
            sys.argv = [__file__, *argv]
            args, overrides = setup_preset_cli(parser(add_launcher_args))
            if overrides:
                raise ValueError(f"Unreviewed config overrides: {overrides}")
            sys.argv = [__file__]
            cfg, agent_cfg = resolve_task_config(TASK_ID, "rsl_rl_cfg_entry_point")
        finally:
            sys.argv = saved
        cfg.scene.num_envs = args.num_envs
        cfg.seed = 42
        agent_cfg.seed = 42
        agent_cfg.max_iterations = args.iterations
        cfg.log_dir = str(output / "ppo")
        report.update(num_envs=args.num_envs, seed=cfg.seed, rsl_rl_version=importlib.metadata.version("rsl-rl-lib"))
        # API/callback semantics below are verified against this installed release.
        if report["rsl_rl_version"] != "5.0.1":
            raise ValueError("Review runner callback/checkpoint semantics before changing RSL-RL release")
        if "pxr" in sys.modules:
            raise ValueError("Configuration imported standalone USD before Kit; refusing native ABI collision")
        with launch_simulation(cfg, args):
            env = None
            physical_guard = None
            try:
                import gymnasium as gym
                import torch
                from rsl_rl.runners import OnPolicyRunner
                from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
                from isaaclab.utils.io import dump_yaml

                class GuardedRunner(OnPolicyRunner):
                    def save(self, path, infos=None):
                        super().save(path, infos={"contract": contract, "next_iteration": self.current_learning_iteration + 1})
                        write_json(Path(path).with_suffix(".pt.json"), {"contract": contract,
                            "checkpoint_sha256": digest(path), "next_iteration": self.current_learning_iteration + 1})

                    def load(self, *args, **kwargs):
                        infos = super().load(*args, **kwargs)
                        restore_adaptive_learning_rate(self.alg)
                        algorithm_state_digest(self)
                        return infos

                env = gym.make(TASK_ID, cfg=cfg)
                report["runtime_manifest"] = env.unwrapped.runtime_manifest
                report["torch_version"] = torch.__version__
                wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
                agent_cfg.device = str(env.unwrapped.device)
                agent_cfg.save_interval = 10
                agent_cfg.logger = "tensorboard"
                agent_dict = agent_cfg.to_dict()
                agent_dict["check_for_nan"] = True
                runner = GuardedRunner(wrapped, agent_dict, log_dir=str(output / "ppo"), device=agent_cfg.device)
                from validate_mkii_fourbar import PhysicalMetrics
                kinematics = json.loads((ROOT / "configs/mkii_fourbar_v3_kinematics.json").read_text())
                physical_metrics = PhysicalMetrics(env.unwrapped, kinematics)
                physical_metrics.window = "learning"
                physical_guard = PhysicalTrainingGuard(env.unwrapped, physical_metrics)
                physical_guard.__enter__()
                start_it = 0
                if resume:
                    infos = runner.load(str(args.checkpoint), strict=True, map_location=agent_cfg.device)
                    if infos != {"contract": contract, "next_iteration": resume["next_iteration"]}:
                        raise ValueError("Embedded checkpoint lineage differs from verified sidecar")
                    runner.current_learning_iteration = start_it = resume["next_iteration"]
                report["resumed_from_sha256"] = resume["checkpoint_sha256"] if resume else None
                report["start_iteration"] = start_it
                report["policy_before_sha256"] = parameters_digest(runner)
                report["algorithm_before_sha256"] = algorithm_state_digest(runner)
                dump_yaml(str(output / "env.yaml"), cfg)
                dump_yaml(str(output / "agent.yaml"), agent_cfg)
                original_log = runner.logger.log
                started = time.monotonic()
                last_loss = {}

                def checked_log(**values):
                    nonlocal last_loss
                    last_loss = {k: float(v) for k, v in values["loss_dict"].items()}
                    if not {"value", "surrogate", "entropy"}.issubset(last_loss):
                        raise ValueError("PPO logger omitted required loss metrics")
                    if not all(torch.isfinite(torch.tensor(v)) for v in last_loss.values()):
                        raise ValueError("Nonfinite PPO loss")
                    algorithm_state_digest(runner)
                    original_log(**values)
                    report["iterations_completed"] = values["it"] - start_it + 1
                    physical_guard.require_coverage(report["iterations_completed"] * agent_cfg.num_steps_per_env)
                    report["physical_metrics"] = physical_metrics.windows
                    report["physics_substeps"] = physical_guard.total
                    write_json(output / "progress.json", {"iterations_completed": report["iterations_completed"],
                        "last_iteration": values["it"], "elapsed_seconds": time.monotonic() - started,
                        "losses": last_loss, "collect_seconds": values["collect_time"], "learn_seconds": values["learn_time"],
                        "physical_metrics": physical_metrics.windows, "physics_substeps": physical_guard.total})
                    if (output / "stop_requested").exists():
                        raise PauseRequested("Shared compute coordination requested a pause")

                runner.logger.log = checked_log
                try:
                    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
                except PauseRequested:
                    report["paused"] = True
                checkpoint = output / "checkpoint.pt"
                runner.save(str(checkpoint))
                require_checkpoint(checkpoint, contract)
                report.update(checkpoint_verified=True, checkpoint_sha256=digest(checkpoint),
                    next_iteration=runner.current_learning_iteration + 1,
                    policy_after_sha256=parameters_digest(runner), losses=last_loss,
                    training_wall_seconds=time.monotonic() - started)
                report["algorithm_after_sha256"] = algorithm_state_digest(runner)
                report["policy_changed"] = report["policy_before_sha256"] != report["policy_after_sha256"]
                if not report["policy_changed"]:
                    raise ValueError("PPO completed without changing policy state")
                # Reload immediately: verifies model, optimizer and normalization
                # compatibility before the separate resumed process is attempted.
                runner.load(str(checkpoint), strict=True, map_location=agent_cfg.device)
                if parameters_digest(runner) != report["policy_after_sha256"]:
                    raise ValueError("Checkpoint roundtrip changed policy state")
                if algorithm_state_digest(runner) != report["algorithm_after_sha256"]:
                    raise ValueError("Checkpoint roundtrip changed critic/optimizer/normalization/scheduler state")
                report["checkpoint_roundtrip_pass"] = True
                if not report["paused"]:
                    physical_metrics.window = "checkpoint_inference"
                    policy = runner.get_inference_policy(device=agent_cfg.device)
                    obs = wrapped.get_observations()
                    total_reward = done_count = 0.0
                    with torch.inference_mode():
                        for _ in range(100):
                            actions = policy(obs)
                            obs, rewards, dones, _ = wrapped.step(actions)
                            if not bool(torch.isfinite(actions).all() and torch.isfinite(rewards).all()):
                                raise ValueError("Nonfinite checkpoint inference")
                            total_reward += float(rewards.sum())
                            done_count += int(dones.sum())
                    report["inference_probe"] = {"steps": 100, "finite": True, "terminations_or_timeouts": done_count,
                        "mean_reward": total_reward / (100 * args.num_envs), "navigation_or_terrain_qualification": False}
                physical_guard.require_coverage(report["iterations_completed"] * agent_cfg.num_steps_per_env
                                                + (0 if report["paused"] else 100))
            except BaseException as error:
                if physical_guard is not None:
                    report["physical_metrics"] = physical_guard.metrics.windows
                    report["physics_substeps"] = physical_guard.total
                finish(error)
                raise
            finally:
                if physical_guard is not None:
                    report["physical_metrics"] = physical_guard.metrics.windows
                    report["physics_substeps"] = physical_guard.total
                    physical_guard.__exit__()
                finish()
                if env is not None:
                    env.close()
    except Exception as error:
        import traceback
        traceback.print_exc()
        if not persisted:
            finish(error)
        return 1
    finish()
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
