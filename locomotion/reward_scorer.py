"""Score recorded native rollouts offline under the current and candidate rewards.

Reads evaluation ``control_trace.npz`` files, rebuilds the per-control telemetry
that ``task.measured_reward`` consumes, and scores each reward on the recorded
behaviour and on a motionless counterfactual under the same commands. No
simulator is needed. Restore archived traces with ``tools/archive.py restore``.

Candidate rewards share ``measured_reward``'s signature:
``fn(telemetry, commands, previous_target, terminated, config, nominal_height)``
returning ``(reward, components)``. Coefficient edits exported by
``locomotion.reward_workbench`` run through the unchanged ``measured_reward``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from .env import LocomotionEnv, inverse_rotate, navigation, rotate
from .task import REWARD_VERSION, TaskConfig, measured_reward

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "offline_reward_score_v1"
REQUIRED_FIELDS = ("root_pose_xyzw", "velocity_world_mps", "gyro_body_rad_s", "joint_velocity_rad_s",
                   "joint_target_rad", "applied_torque_squared_sum_400hz", "nonfoot_contact_count_400hz",
                   "command", "terminated")
RECONSTRUCTION_TOLERANCE_MPS = 1e-5
# env.LocomotionEnv's AMP state: q, then joint, body linear and body angular velocity.
AMP_VELOCITY_SLICES = (slice(18, 36), slice(36, 39), slice(39, 42))
DECLARED_GAPS = (
    "Non-foot contact: training penalizes >1 N floor force on non-tibia bodies; traces keep only "
    "evaluation patch-classified contact counts, so the scorer passes zero force and reports "
    "contact controls separately.",
    "Control 0 is excluded because its previous joint target is not recorded.",
    "The motionless counterfactual zeroes body and joint velocities, target changes and action "
    "changes but keeps recorded pose, torque and contact; compare its tracking terms, not its penalty terms.",
    "Requested-torque peaks: training reads each joint's largest requested torque over the eight "
    "substeps; traces keep only the last substep's, so the scorer uses that.",
)
REWARD_CONFIG_SCHEMA = "reward_config_v1"
REWARD_FIELDS = ("linear_tracking_weight", "linear_error_variance", "yaw_tracking_weight", "yaw_error_variance",
                 "quiet_joint_rate_weight", "quiet_joint_rate_scale_rad_s", "quiet_target_step_weight",
                 "quiet_target_step_scale_rad", "tilt_weight", "angular_xy_weight", "vertical_velocity_weight",
                 "height_weight", "normalized_effort_weight", "target_movement_weight", "nonfoot_event_weight",
                 "nonfoot_force_weight", "terminal_penalty")


@dataclass
class Trace:
    path: Path
    sha256: str
    data: dict
    case_ids: list


def root_com_local(root=ROOT):
    """Root-link centre of mass in the link frame, from the approved model."""
    selector = json.loads((root / "robot/active_model.json").read_text())
    model = json.loads((root / selector["model"]["path"]).read_text())
    return next(link["com"] for link in model["links"] if link["parent"] is None)


def load_trace(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as file:
        data = {key: file[key] for key in file.files}
    missing = [key for key in REQUIRED_FIELDS if key not in data]
    if missing:
        raise ValueError(f"{path} lacks trace fields: {missing}")
    if data["command"].shape[0] < 2:
        raise ValueError(f"{path} needs at least two controls")
    report = path.with_name("report.json")
    case_ids = json.loads(report.read_text()).get("assigned_case_ids", []) if report.exists() else []
    return Trace(path, hashlib.sha256(path.read_bytes()).hexdigest(), data, case_ids)


def origin_velocity_nav(trace, com_local):
    """Training measures the root-link origin; evaluation records the root centre of mass."""
    pose = torch.as_tensor(trace.data["root_pose_xyzw"], dtype=torch.float64)
    quaternion = pose[..., 3:]
    world_angular = rotate(quaternion, torch.as_tensor(trace.data["gyro_body_rad_s"], dtype=torch.float64))
    com = torch.as_tensor(com_local, dtype=torch.float64).expand_as(world_angular)
    velocity = torch.as_tensor(trace.data["velocity_world_mps"], dtype=torch.float64)
    origin = velocity - torch.cross(world_angular, rotate(quaternion, com), dim=-1)
    return navigation(inverse_rotate(quaternion, origin))


def reconstruction_error(trace, com_local):
    """Largest difference from the critic's recorded velocity, which row t+1 observes."""
    critic = trace.data.get("critic_observation")
    if critic is None:
        return None
    recorded = critic[1:, ..., LocomotionEnv.observation_width:LocomotionEnv.critic_width]
    return float(np.abs(origin_velocity_nav(trace, com_local)[:-1].numpy() - recorded).max())


def reward_inputs(trace, com_local, *, motionless=False):
    """Controls 1..T-1 flattened across replicas into measured_reward's batch layout."""
    def flat(value):
        value = torch.as_tensor(value)
        return value.reshape(-1, *value.shape[2:])

    data = trace.data
    target = torch.as_tensor(data["joint_target_rad"], dtype=torch.float32)
    joint_velocity = torch.as_tensor(data["joint_velocity_rad_s"], dtype=torch.float32)
    telemetry = {
        "linear_velocity_nav": flat(origin_velocity_nav(trace, com_local)[1:].to(torch.float32)),
        "angular_velocity_body": flat(torch.as_tensor(data["gyro_body_rad_s"][1:], dtype=torch.float32)),
        "root_pose_xyzw": flat(torch.as_tensor(data["root_pose_xyzw"][1:], dtype=torch.float32)),
        "joint_velocity_rad_s": flat(joint_velocity[1:]),
        "previous_joint_velocity_rad_s": flat(joint_velocity[:-1]),
        "joint_target_rad": flat(target[1:]),
        "torque_square_sum_400hz": flat(torch.as_tensor(data["applied_torque_squared_sum_400hz"][1:], dtype=torch.float32)),
        "requested_torque_abs_max_400hz": flat(torch.as_tensor(data["computed_torque_nm"][1:], dtype=torch.float32).abs()),
        "command": flat(torch.as_tensor(data["command"][1:], dtype=torch.float32)),
    }
    if "policy_action" in data:
        action = torch.as_tensor(data["policy_action"], dtype=torch.float32)
        telemetry["action"], telemetry["previous_action"] = flat(action[1:]), flat(action[:-1])
    telemetry["replicas"] = data["command"].shape[1]
    if "amp_state_before" in data:
        telemetry["amp_state"] = flat(torch.as_tensor(data["amp_state_before"][1:], dtype=torch.float32))
        telemetry["next_amp_state"] = flat(torch.as_tensor(data["amp_state_after"][1:], dtype=torch.float32))
    telemetry["other_body_force_max_400hz"] = torch.zeros(len(telemetry["command"]))
    previous_target = flat(target[:-1])
    if motionless:
        for key in ("linear_velocity_nav", "angular_velocity_body", "joint_velocity_rad_s", "previous_joint_velocity_rad_s"):
            telemetry[key] = torch.zeros_like(telemetry[key])
        previous_target = telemetry["joint_target_rad"].clone()
        if "action" in telemetry:
            telemetry["previous_action"] = telemetry["action"].clone()
        if "amp_state" in telemetry:
            still = telemetry["amp_state"].clone()
            for columns in AMP_VELOCITY_SLICES:
                still[:, columns] = 0
            telemetry["amp_state"], telemetry["next_amp_state"] = still, still.clone()
    terminated = flat(torch.as_tensor(data["terminated"][1:], dtype=torch.bool))
    return telemetry, previous_target, terminated


def evaluate(trace, reward, config, nominal_height, com_local, *, motionless=False):
    telemetry, previous_target, terminated = reward_inputs(trace, com_local, motionless=motionless)
    return reward(telemetry, telemetry["command"], previous_target, terminated, config, nominal_height)


def load_reward(spec):
    """MODULE:FUNCTION, or MODULE:FACTORY:ARGUMENT for rewards built from a file such as a checkpoint."""
    module, _, rest = spec.partition(":")
    attribute, _, argument = rest.partition(":")
    if not module or not attribute:
        raise ValueError(f"Reward must be MODULE:FUNCTION, got {spec!r}")
    target = getattr(importlib.import_module(module), attribute)
    return target(argument) if argument else target


def config_reward(overrides):
    """The current measured_reward under edited coefficients that training would accept."""
    unknown = sorted(set(overrides) - set(REWARD_FIELDS))
    if unknown:
        raise ValueError(f"Not reward coefficients: {unknown}")
    config = replace(TaskConfig(), **{key: float(value) for key, value in overrides.items()})
    config.validate()

    def reward(telemetry, commands, previous_target, terminated, _config, nominal_height):
        return measured_reward(telemetry, commands, previous_target, terminated, config, nominal_height)
    return reward


def load_reward_config(path):
    data = json.loads(Path(path).read_text())
    if data.get("schema") != REWARD_CONFIG_SCHEMA:
        raise ValueError(f"{path} is not a {REWARD_CONFIG_SCHEMA} file")
    if data.get("base_reward_version") != REWARD_VERSION:
        raise ValueError(f"{path} edits {data.get('base_reward_version')!r}, not the current {REWARD_VERSION!r}")
    return config_reward(data["overrides"])


def add_reward_arguments(parser):
    parser.add_argument("--reward", action="append", default=[], metavar="NAME=MODULE:FUNCTION",
                        help="Candidate reward function beside the current one; repeatable.")
    parser.add_argument("--reward-config", action="append", default=[], metavar="NAME=PATH",
                        help="Coefficient edits exported by locomotion.reward_workbench; repeatable.")


def selected_rewards(parser, args):
    rewards = {f"current ({REWARD_VERSION})": measured_reward}
    for option, specs, load in (("--reward", args.reward, load_reward),
                                ("--reward-config", args.reward_config, load_reward_config)):
        for spec in specs:
            name, separator, target = spec.partition("=")
            if not separator or not name or not target:
                parser.error(f"{option} must be NAME=..., got {spec!r}")
            if name in rewards:
                parser.error(f"Reward name {name!r} is used twice")
            rewards[name] = load(target)
    return rewards


def _mean(values, mask):
    return float(values[mask].to(torch.float64).mean())


def score_trace(trace, rewards, config, nominal_height, com_local):
    error = reconstruction_error(trace, com_local)
    if error is not None and error > RECONSTRUCTION_TOLERANCE_MPS:
        raise ValueError(f"{trace.path}: reconstructed velocity differs from the recorded critic "
                         f"observation by {error:.3g} m/s; the trace may belong to another model")
    telemetry, _, _ = reward_inputs(trace, com_local)
    commands, velocity = telemetry["command"], telemetry["linear_velocity_nav"]
    scored = {name: {phase: evaluate(trace, fn, config, nominal_height, com_local, motionless=phase == "motionless")
                     for phase in ("recorded", "motionless")} for name, fn in rewards.items()}
    keys = [tuple(np.round(row, 6).tolist()) for row in commands.numpy()]
    groups = []
    for command in sorted(set(keys)):
        mask = torch.tensor([key == command for key in keys])
        speed = float(np.hypot(*command[:2]))
        group = {"command": list(command), "rows": int(mask.sum()),
                 "planar_speed_mps": _mean(torch.linalg.vector_norm(velocity[:, :2], dim=-1), mask),
                 "yaw_rate_rad_s": _mean(telemetry["angular_velocity_body"][:, 2], mask),
                 "along_command_mps": (_mean(velocity[:, :2] @ torch.tensor(command[:2]) / speed, mask)
                                       if speed > 0 else None),
                 "rewards": {}}
        for name, phases in scored.items():
            group["rewards"][name] = {}
            for phase, (reward, components) in phases.items():
                group["rewards"][name][phase] = {"mean": _mean(reward, mask),
                    "components": {key: _mean(value, mask) for key, value in components.items()}}
        groups.append(group)
    return {"path": str(trace.path), "sha256": trace.sha256, "case_ids": trace.case_ids,
            "controls_scored": len(commands), "velocity_reconstruction_max_error_mps": error,
            "nonfoot_contact_controls": int((trace.data["nonfoot_contact_count_400hz"][1:] > 0).sum()),
            "terminated_controls": int(trace.data["terminated"][1:].sum()),
            "totals": {name: {phase: float(reward.to(torch.float64).mean()) for phase, (reward, _) in phases.items()}
                       for name, phases in scored.items()},
            "groups": groups}


def score(paths, rewards, nominal_height, *, config=None, com_local=None):
    config = TaskConfig() if config is None else config
    com_local = root_com_local() if com_local is None else com_local
    traces = [score_trace(load_trace(path), rewards, config, nominal_height, com_local) for path in paths]
    ranking = {name: [t["path"] for t in sorted(traces, key=lambda t: t["totals"][name]["recorded"], reverse=True)]
               for name in rewards}
    return {"schema": SCHEMA, "nominal_height_m": nominal_height, "root_com_local_m": list(com_local),
            "declared_gaps": list(DECLARED_GAPS), "traces": traces, "ranking": ranking}


def _format(result):
    lines = []
    for trace in result["traces"]:
        lines.append(f"{trace['path']}")
        lines.append(f"  cases {trace['case_ids'] or '-'}; controls {trace['controls_scored']}; "
                     f"velocity check {trace['velocity_reconstruction_max_error_mps']}; "
                     f"non-foot contact controls {trace['nonfoot_contact_controls']}; "
                     f"terminated {trace['terminated_controls']}")
        for group in trace["groups"]:
            along = group["along_command_mps"]
            lines.append(f"  command {group['command']} ({group['rows']} rows): planar speed "
                         f"{group['planar_speed_mps']:.4f} m/s" + (f", along command {along:.4f} m/s" if along is not None else ""))
            for name, phases in group["rewards"].items():
                recorded, motionless = phases["recorded"]["mean"], phases["motionless"]["mean"]
                lines.append(f"    {name:<24} recorded {recorded:+.4f}  motionless {motionless:+.4f}  "
                             f"advantage {recorded - motionless:+.4f}")
    lines.append("ranking by mean recorded reward:")
    for name, paths in result["ranking"].items():
        names = [Path(path).name for path in paths]
        lines.append(f"  {name}: " + " > ".join(names if len(set(names)) == len(names) else paths))
    lines.append("declared gaps:")
    lines.extend(f"  - {gap}" for gap in result["declared_gaps"])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--nominal-height", type=float, required=True,
                        help="Nominal root height (m) the reward was trained with; the run's "
                             "task_definition.json records it as nominal_plate_height_m.")
    add_reward_arguments(parser)
    parser.add_argument("--json", type=Path, help="Write the full score report here.")
    args = parser.parse_args(argv)
    result = score(args.traces, selected_rewards(parser, args), args.nominal_height)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(_format(result))


if __name__ == "__main__":
    main()
