"""Replay actions open loop on the surrogate and score stored telemetry under any reward.

Three subcommands (run from the repository root with
``uv run python -m locomotion.surrogate.replay ...``):

  trace     Replay a recorded action or target trace (npz). A native ``control_trace.npz``
            needs no conversion (field ``policy_action``, or ``joint_target_rad`` with
            ``--field target``).
  scripted  Constant mean plus iid Gaussian noise: ``--mean 0 --std 0.15``.
  score     Score a saved ``telemetry.npz`` (surrogate or native probe) under v1, v2, v3, v4,
            ``v4:key=value,...``, ``module:Class``, ``module:factory(text)`` or ``module:function``.

``trace`` and ``scripted`` write ``telemetry.npz`` (every ``env.telemetry`` key as
[control, replica, ...]) and ``summary.json``. The replay resets a replica that terminates or
times out before the next control, as ``ppo.VanillaVecEnv`` does.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from locomotion import task as task_v1
from locomotion import task_v4
from locomotion.env import executed_action_feature, rotate
from locomotion.surrogate.env import Renderer, SurrogateConfig, SurrogateEnv, contact_overrides, make_env
from locomotion.surrogate.train import resolve, thread_count


REWARD_TASKS = {"v1": "locomotion.task:TrainingTask", "v2": "locomotion.task_v2:TrainingTaskV2",
                "v3": "locomotion.task_v3:TrainingTaskV3", "v4": "locomotion.task_v4:TrainingTaskV4"}
# Fields that are flags or bookkeeping, not env.telemetry entries.
NOT_TELEMETRY = ("terminated", "truncated", "reset_before", "time_s", "command")


def reward_task(name):
    """The task class of a reward name, or a plain function for the older ``module:function`` form.

    ``v1`` to ``v4`` name the repository tasks, ``v4:key=value,...`` is ``task_v4.variant`` of that
    text, and ``module:Class`` or ``module:factory(text)`` resolves as the trainer's ``--task``
    does. Scoring runs the class itself, so its memory rules follow the checked-out revision.
    """
    version, _, options = name.partition(":")
    version = "v" + version if version in ("1", "2", "3", "4") else version
    if version in REWARD_TASKS and not options:
        return resolve(REWARD_TASKS[version])
    if version == "v4":
        return task_v4.variant(options)
    module, _, attribute = name.partition(":")
    if "(" not in attribute:
        value = importlib.import_module(module)
        for part in attribute.split("."):
            value = getattr(value, part)
        if not isinstance(value, type):
            return value
    return resolve(name)


class TelemetryPlayer(SurrogateEnv):
    """A SurrogateEnv whose ``step`` returns stored telemetry in place of physics.

    The repository task classes run on it unchanged and score a native or surrogate record with
    their own memory rules. A record without toe positions (the native noise probe) receives them
    from forward kinematics of the stored root pose and joint positions; link poses from this
    kinematics equal native link poses to 3e-7 m.
    """

    def __init__(self, arrays):
        controls, replicas = arrays["action"].shape[:2]
        self._reset_current = None
        super().__init__(SurrogateConfig(num_envs=replicas, episode_seconds=120.), threads=0, substep_state=False)
        self._reset_current = {key: value.clone() for key, value in self.current.items()}
        self.arrays = {key: value for key, value in arrays.items()
                       if value.ndim >= 2 and value.shape[:2] == (controls, replicas)}
        reset_before = self.arrays.get("reset_before")
        if reset_before is None:
            done = self.arrays["terminated"] | self.arrays.get("truncated", np.zeros_like(self.arrays["terminated"]))
            reset_before = np.concatenate((np.ones((1, replicas), dtype=bool), done[:-1]))
        self.reset_before = reset_before.astype(bool)
        steps, count = np.zeros((controls, replicas), dtype=np.int64), np.zeros(replicas, dtype=np.int64)
        for control in range(controls):
            count = np.where(self.reset_before[control], 0, count) + 1
            steps[control] = count
        self._episode_steps_at, self.cursor = steps, 0

    def reset(self, indices=None):
        if self._reset_current is None:
            return super().reset(indices)
        indices = torch.arange(self.num_envs) if indices is None else torch.as_tensor(indices, dtype=torch.long)
        for key, value in self.current.items():
            value[indices] = self._reset_current[key][indices]
        self.held[indices] = self.neutral
        self.previous_action[indices] = 0
        self.episode_steps[indices] = 0
        self.history[indices] = self._proprio(self.current)[indices, None].expand(-1, 5, -1)
        return self._observations(self.current)

    def step(self, action):
        row = self.cursor
        take = lambda key: torch.from_numpy(np.ascontiguousarray(self.arrays[key][row]))
        pose, nav = take("root_pose_xyzw").to(torch.float64), take("linear_velocity_nav").to(torch.float64)
        state = self._state
        state[:] = 0.
        state[:, 1:4] = (pose[:, :3] - self._origins64).numpy()
        state[:, 4], state[:, 5:8] = pose[:, 6].numpy(), pose[:, 3:6].numpy()
        state[:, 1 + self._qpos_columns] = self.arrays["joint_position_rad"][row]
        # Navigation axes: forward = -body Y, left = body X.
        body_velocity = torch.stack((nav[:, 1], -nav[:, 0], nav[:, 2]), dim=-1)
        state[:, 26:29] = rotate(pose[:, 3:], body_velocity).numpy()
        state[:, 29:32] = self.arrays["angular_velocity_body"][row]
        state[:, 26 + self._qvel_columns] = self.arrays["joint_velocity_rad_s"][row]
        current = self._read(full=True)
        telemetry = {key: take(key) for key in self.arrays if key not in NOT_TELEMETRY}
        telemetry.setdefault("toe_xyz_world", current["toe_world"].clone())
        telemetry.setdefault("toe_xyz_body", current["toe_body"].clone())
        telemetry.setdefault("linear_velocity_body", current["linear"].clone())
        terminated = take("terminated").to(torch.bool)
        truncated = take("truncated").to(torch.bool) if "truncated" in self.arrays else torch.zeros_like(terminated)
        telemetry.update(command=self.commands.clone(), terminated=terminated, truncated=truncated)
        self.telemetry, self.current = telemetry, current
        self.held = telemetry["joint_target_rad"].to(torch.float32).clone()
        self.previous_action = executed_action_feature(self.held, self.neutral, self.cfg.action_scale_rad).clone()
        self.history = torch.cat((self.history[:, 1:], self._proprio(current)[:, None]), dim=1)
        self.episode_steps = torch.from_numpy(self._episode_steps_at[row].copy())
        result = self._observations(current)
        result.update(terminated=terminated, truncated=truncated)
        return result


class FunctionScorer:
    """A plain ``function(telemetry, command, terminated) -> (reward, components)`` on stored telemetry.

    Its telemetry also holds ``previous_action``, ``previous_joint_velocity_rad_s`` and
    ``previous_joint_target_rad`` (zero action, zero rate and the neutral target after a restart).
    A reward that needs more memory belongs in a task class.
    """

    def __init__(self, function, replicas, neutral):
        self.function, self.neutral = function, neutral
        self.previous_action = torch.zeros(replicas, 18)
        self.previous_rate = torch.zeros(replicas, 18)
        self.previous_target = neutral.expand(replicas, -1).clone()

    def restart(self, indices):
        self.previous_action[indices] = 0
        self.previous_rate[indices] = 0
        self.previous_target[indices] = self.neutral

    def __call__(self, telemetry, command, terminated):
        result = self.function({**telemetry, "previous_action": self.previous_action,
                                "previous_joint_velocity_rad_s": self.previous_rate,
                                "previous_joint_target_rad": self.previous_target}, command, terminated)
        self.previous_action = telemetry["action"].clone()
        self.previous_rate = telemetry["joint_velocity_rad_s"].clone()
        self.previous_target = telemetry["joint_target_rad"].clone()
        return result


def score_arrays(data, names, nominal_height=None, neutral=None):
    """Reward components for stored telemetry arrays [control, replica, ...].

    ``data`` needs the fields of ``noise_probe.FIELDS``, ``command`` and ``terminated``;
    ``reset_before`` marks the replicas that restarted before a control (the default restarts
    after each stored termination or truncation). Returns {name: {"reward" and components}}.
    """
    controls, replicas = data["action"].shape[:2]
    player = TelemetryPlayer(data)
    commands = torch.from_numpy(np.ascontiguousarray(data["command"])).to(torch.float32)
    actions = torch.from_numpy(np.ascontiguousarray(data["action"])).to(torch.float32)
    scorers = {}
    for name in names:
        target = reward_task(name)
        if isinstance(target, type):
            player.commands[:] = commands[0]
            scorers[name] = target(player, task_v1.TaskConfig(), None)
            # A record can stack separate one-robot trials as replicas; their spacing means nothing here.
            scorers[name].proximity.check = lambda *arguments, **keywords: None
        else:
            scorers[name] = FunctionScorer(target, replicas, player.neutral)
    rows = {name: [] for name in names}
    for control in range(controls):
        restarted = torch.from_numpy(player.reset_before[control]).nonzero().flatten()
        player.cursor = control
        for name, scorer in scorers.items():
            if isinstance(scorer, FunctionScorer):
                if len(restarted):
                    scorer.restart(restarted)
                    player.reset(restarted)
                player.commands[:] = commands[control]
                output = player.step(actions[control])
                reward, components = scorer(player.telemetry, commands[control], output["terminated"])
            else:
                if len(restarted):
                    scorer.reset(restarted)
                # The task draws its own commands; scoring holds the stored one.
                player.commands[:] = commands[control]
                reward = scorer.step(actions[control])["reward"]
                components = scorer.last_components
            rows[name].append({"reward": reward.detach().clone(), **{key: value.detach().clone() for key, value in components.items()}})
    player.close()
    return {name: {key: torch.stack([row[key] for row in values]).numpy() for key in values[0]}
            for name, values in rows.items()}


def reward_summary(scores):
    return {name: {key: float(value.mean()) for key, value in components.items()} for name, components in scores.items()}


def run(env, action_at, controls, output, *, rewards=(), video=False, view="three_quarter"):
    """Step ``controls`` controls with ``action_at(control) -> [replicas, 18]`` and record telemetry."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    replicas = env.num_envs
    records = []
    reset_before = torch.ones(replicas, dtype=torch.bool)
    writer = renderer = None
    if video:
        import cv2
        renderer = Renderer(env, view=view)
    started = time.perf_counter()
    for control in range(controls):
        action = action_at(control)
        result = env.step(action)
        telemetry = env.telemetry
        records.append({**{key: value.numpy().copy() for key, value in telemetry.items()},
                        "reset_before": reset_before.numpy().copy()})
        if renderer is not None and control % 2 == 1:
            frame = renderer.frame(0)
            if writer is None:
                writer = cv2.VideoWriter(str(output / "replay.mp4"), cv2.VideoWriter_fourcc(*"avc1"), 25, frame.shape[1::-1])
            writer.write(frame[:, :, ::-1])
        done = result["terminated"] | result["truncated"]
        reset_before = done.clone()
        if bool(done.any()):
            env.reset(done.nonzero().flatten())
    if writer is not None:
        writer.release()
    data = {key: np.stack([row[key] for row in records]) for key in records[0]}
    data["time_s"] = (np.arange(controls) + 1) * env.cfg.control_dt
    np.savez_compressed(output / "telemetry.npz", **data)
    # The repository task classes score the stored record with their own memory rules.
    scores = score_arrays(data, list(rewards)) if rewards else {}
    if scores:
        np.savez_compressed(output / "rewards.npz", **{f"{name}/{key}": value for name, values in scores.items() for key, value in values.items()})
    settle = min(100, controls // 2)
    nav, gyro = data["linear_velocity_nav"][settle:], data["angular_velocity_body"][settle:]
    steps = np.abs(np.diff(data["joint_target_rad"], axis=0))[~data["reset_before"][1:]]
    rms = np.sqrt(data["torque_square_sum_400hz"][settle:] / 8)
    summary = {"schema": "hexapod_surrogate_replay_v1", "surrogate": True, "controls": controls, "replicas": replicas,
        "window_start_control": settle, "command_first_control": data["command"][0].tolist() if replicas <= 4 else data["command"][0, :4].tolist(),
        "mean_velocity_nav_mps": nav.mean((0, 1)).tolist(), "mean_yaw_rate_rad_s": float(gyro[..., 2].mean()),
        "achieved_planar_speed_mps": float(np.linalg.norm(nav[..., :2], axis=-1).mean()),
        "planar_velocity_std_xy_mps": [float(nav[..., 0].std()), float(nav[..., 1].std())],
        "vertical_velocity_rms_mps": float(np.sqrt((nav[..., 2] ** 2).mean())),
        "yaw_rate_std_rad_s": float(gyro[..., 2].std()),
        "roll_pitch_rate_mean_norm_rad_s": float(np.linalg.norm(gyro[..., :2], axis=-1).mean()),
        "root_height_mean_m": float(data["root_pose_xyzw"][settle:, :, 2].mean()),
        "final_root_height_m": data["root_pose_xyzw"][-1, :4, 2].tolist(),
        "final_displacement_forward_left_m": [(-(data["root_pose_xyzw"][-1, :4, 1] - env.origins[:4, 1].numpy())).tolist(),
                                              (data["root_pose_xyzw"][-1, :4, 0] - env.origins[:4, 0].numpy()).tolist()],
        "limiter_fraction_coxa_femur_tibia": [float((np.abs(steps[:, k::3] - env.cfg.target_slew_rad) <= 2e-6).mean()) for k in range(3)],
        "requested_saturation_fraction_400hz": float(data["saturation_count_400hz"][settle:].mean() / 8),
        "applied_torque_rms_coxa_femur_tibia_nm": [float(np.sqrt((rms[..., k::3] ** 2).mean())) for k in range(3)],
        "torque_rms_norm_mean_nm": float(np.linalg.norm(rms, axis=-1).mean()),
        "toe_loaded_all_substeps_fraction": (data["surrogate_toe_cap_force_min_400hz"][settle:] > 1.).mean((0, 1)).tolist(),
        "nonfoot_control_fraction": float((data["other_body_force_max_400hz"] > 1.).mean()),
        "terminations": int(data["terminated"].sum()), "truncations": int(data["truncated"].sum()),
        "speed_clamp_events": env.speed_clamp_events, "peak_joint_speed_rad_s": env.peak_joint_speed,
        "touchdown_events": env.touchdown_events, "touchdown_overshoot_events": env.overshoot_events,
        "reward_component_means": reward_summary(scores),
        "controls_per_second": controls * replicas / (time.perf_counter() - started),
        "scope": "CPU MuJoCo surrogate; design evidence."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def load_actions(path, field, env):
    """Actions [control, replica, 18] and commands [control, replica, 3] or None from an npz trace."""
    with np.load(path, allow_pickle=False) as data:
        names = data.files
        if field == "target":
            target = data["joint_target_rad"].astype(np.float32)
            actions = (target - env.neutral.numpy()) / env.cfg.action_scale_rad
        else:
            key = field if field in names else next(k for k in ("policy_action", "action") if k in names)
            actions = data[key].astype(np.float32)
        commands = data["command"].astype(np.float32) if "command" in names else None
    if actions.ndim == 2:
        actions = actions[:, None]
    if commands is not None and commands.ndim == 2:
        commands = commands[:, None]
    return actions, commands


def parse_mean(text):
    path = Path(text)
    if path.suffix == ".json" and path.exists():
        value = np.asarray(json.loads(path.read_text()), dtype=np.float32)
    elif path.suffix == ".npy" and path.exists():
        value = np.load(path).astype(np.float32)
    else:
        value = np.asarray([float(x) for x in text.split(",")], dtype=np.float32)
    if value.shape not in ((1,), (18,)):
        raise ValueError("The mean is one value or 18 values in JOINT_NAMES order")
    return torch.from_numpy(np.broadcast_to(value, (18,)).copy())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    sub = parser.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output", type=Path, required=True)
    common.add_argument("--controls", type=int)
    common.add_argument("--command", type=float, nargs=3, metavar=("FORWARD", "LEFT", "YAW"))
    common.add_argument("--rewards", nargs="*", default=["v2", "v3", "v4"], help="v1, v2, v3, v4, v4:key=value,... or module:function.")
    common.add_argument("--episode-seconds", type=float, default=60.)
    common.add_argument("--threads", type=int, default=2, help="Physics and torch threads for this process (1 to 6).")
    common.add_argument("--video", action="store_true", help="Render replica 0 to replay.mp4.")
    common.add_argument("--view", choices=["three_quarter", "side"], default="three_quarter")
    common.add_argument("--contact", action="append", default=[], metavar="FIELD=VALUE")
    trace = sub.add_parser("trace", parents=[common])
    trace.add_argument("--trace", type=Path, required=True)
    trace.add_argument("--field", default="policy_action", help="policy_action, action, or target (joint_target_rad).")
    scripted = sub.add_parser("scripted", parents=[common])
    scripted.add_argument("--mean", default="0", help="One value, 18 comma-separated values, or a .json/.npy file.")
    scripted.add_argument("--std", type=float, default=.15)
    scripted.add_argument("--replicas", type=int, default=128)
    scripted.add_argument("--seed", type=int, default=20260917)
    score = sub.add_parser("score")
    score.add_argument("--telemetry", type=Path, required=True)
    score.add_argument("--rewards", nargs="+", default=["v2", "v3", "v4"])
    score.add_argument("--output", type=Path, help="Optional new npz file for per-control components.")
    score.add_argument("--command", type=float, nargs=3, metavar=("FORWARD", "LEFT", "YAW"),
                       help="Score under this held command in place of the stored one (required meaning for a native probe record).")
    args = parser.parse_args(argv)
    if args.mode == "score":
        if args.output is not None:
            # numpy appends the suffix to a name without it; the result keeps a fresh name either way.
            args.output = args.output if args.output.suffix == ".npz" else args.output.with_name(args.output.name + ".npz")
            if args.output.exists():
                raise FileExistsError(f"{args.output} exists; choose a new output file")
        with np.load(args.telemetry, allow_pickle=False) as file:
            data = {key: file[key] for key in file.files}
        if "command" not in data:
            # A native probe record holds a zero command; --command names the one to score under.
            controls, replicas = data["action"].shape[:2]
            data["command"] = np.broadcast_to(np.asarray(args.command or (0., 0., 0.), dtype=np.float32), (controls, replicas, 3)).copy()
        elif args.command is not None:
            data["command"] = np.broadcast_to(np.asarray(args.command, dtype=np.float32), data["command"].shape).copy()
        scores = score_arrays(data, args.rewards)
        if args.output:
            np.savez_compressed(args.output, **{f"{name}/{key}": value for name, values in scores.items() for key, value in values.items()})
        print(json.dumps(reward_summary(scores), indent=2))
        return 0
    # The same count reaches torch and the MuJoCo thread pool.
    torch.set_num_threads(thread_count(args.threads))
    contact = contact_overrides(args.contact)
    if args.mode == "trace":
        probe = make_env(1, threads=0)
        actions, commands = load_actions(args.trace, args.field, probe)
        replicas = actions.shape[1]
        controls = min(args.controls or len(actions), len(actions))
        env = make_env(replicas, threads=min(args.threads, replicas) if replicas > 1 else 0, contact=contact,
                       episode_seconds=args.episode_seconds, substep_state=False)
        actions = torch.from_numpy(actions)

        def action_at(control):
            if args.command is not None:
                env.commands[:] = torch.tensor(args.command)
            elif commands is not None:
                env.commands[:] = torch.from_numpy(commands[control])
            else:
                env.commands.zero_()
            return actions[control]
    else:
        controls = args.controls or 500
        env = make_env(args.replicas, threads=args.threads, contact=contact, episode_seconds=args.episode_seconds, substep_state=False)
        env.commands[:] = torch.tensor(args.command if args.command is not None else [0., 0., 0.])
        mean = parse_mean(args.mean)
        generator = torch.Generator().manual_seed(args.seed)

        def action_at(control):
            return mean + args.std * torch.randn(args.replicas, 18, generator=generator)
    summary = run(env, action_at, controls, args.output, rewards=args.rewards, video=args.video, view=args.view)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
