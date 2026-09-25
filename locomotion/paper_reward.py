"""Liu et al. Table I task and penalty reward (arXiv:2511.03167, p. 3) on this robot's telemetry.

A candidate beside ``task.measured_reward``; training still uses ``task.py``.
It shares ``measured_reward``'s signature, so ``locomotion.reward_scorer`` and
``locomotion.reward_viewer`` score it with ``--reward
paper=locomotion.paper_reward:paper_reward``. ``ADAPTATIONS`` records every
departure from the printed table and ``OMITTED`` every term left out.

``paper_reward_calibrated`` keeps the paper's terms but rescales the continuous
penalties by the rule in ``CALIBRATION``; reproduce its weights with
``python -m locomotion.paper_reward <traces...>``.

``variant`` builds any configuration from a spec, including the audit's
command-scaled tracking kernels, for example
``--reward c=locomotion.paper_reward:variant:kernel=stride,penalties=calibrated``.
The stride kernel averages each replica's velocity over a contiguous trace;
training would keep that window per replica in ``task.py``.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace
import json

import torch

REWARD_VERSION = "paper_table1_task_penalty_v1"
ADAPTATIONS = (
    "Tracking kernels use exp(-||e||/0.15): the printed exponent has no minus sign, which would "
    "grow with tracking error.",
    "Planar and vertical velocity are measured at the root-link origin, as the training reward does.",
    "||tau|| uses each joint's RMS applied torque over the control's eight physics substeps.",
    "||q_ddot|| is the joint-velocity change between consecutive 50 Hz controls divided by 0.02 s.",
    "The action rate uses raw policy actions, before clipping and the 0.040 rad target limiter.",
    "n_collision counts one event when any non-tibia body presses the ground with more than 1 N "
    "during the control; the simulation does not report self-collisions.",
    "The torque limit applies to requested torque before the motor cap, against the 1.6 N·m "
    "software cap; applied torque never exceeds that cap.",
    "The joint-velocity limit is the URDF limit, 50.27 rad/s, checked at each control's end.",
)
OMITTED = (
    "Style r^s needs the AMP discriminator (docs/TRAINING.md Step 5).",
    "The foot contact-force limit: control traces record no foot force and the paper states no limit.",
)
REQUIRED = ("linear_velocity_nav", "angular_velocity_body", "torque_square_sum_400hz", "joint_velocity_rad_s",
            "previous_joint_velocity_rad_s", "action", "previous_action", "other_body_force_max_400hz",
            "requested_torque_abs_max_400hz", "command")


@dataclass(frozen=True)
class PaperRewardConfig:
    linear_tracking_weight: float = 1.
    yaw_tracking_weight: float = .5
    tracking_scale: float = .15
    vertical_velocity_weight: float = 1.
    roll_pitch_weight: float = .08
    torque_weight: float = 2e-6
    acceleration_weight: float = 1.5e-7
    action_rate_weight: float = .01
    collision_weight: float = .05
    torque_limit_weight: float = .05
    velocity_limit_weight: float = .5
    torque_limit_nm: float = 1.6
    velocity_limit_rad_s: float = 50.26548245743669
    collision_force_n: float = 1.
    control_dt_s: float = .02
    substeps: int = 8
    # docs/REWARD_V2_TABLE1_AUDIT.md section 5: "paper" is variant A, "scaled" B, "stride" C.
    tracking_kernel: str = "paper"
    scale_k: float = .4
    minimum_speed_mps: float = .025
    minimum_yaw_rate_rad_s: float = .15
    stride_controls: int = 60


PAPER_CONFIG = PaperRewardConfig()
KERNELS = ("paper", "scaled", "stride")


def stride_average(values, commands, replicas, window):
    """Causal mean over each replica's last ``window`` controls, restarted when its command changes.

    Rows arrive control-major, as the scorer flattens them: row = control * replicas + replica.
    """
    controls = len(values) // replicas
    values = values.reshape(controls, replicas, -1)
    commands = commands.reshape(controls, replicas, -1)
    result = torch.empty_like(values)
    start = torch.zeros(replicas, dtype=torch.long)
    for control in range(controls):
        if control:
            changed = (commands[control] != commands[control - 1]).any(-1)
            start[changed] = control
        for replica in range(replicas):
            first = max(int(start[replica]), control - window + 1)
            result[control, replica] = values[first:control + 1, replica].mean(0)
    return result.reshape(controls * replicas, -1)


def _tracking(velocity_xy, yaw_rate, commands, telemetry, c):
    if c.tracking_kernel not in KERNELS:
        raise ValueError(f"Tracking kernel must be one of {KERNELS}, got {c.tracking_kernel!r}")
    if c.tracking_kernel == "paper":
        return c.tracking_scale, c.tracking_scale, velocity_xy, yaw_rate
    if c.tracking_kernel == "stride":
        if "replicas" not in telemetry:
            raise ValueError("Stride-averaged tracking needs the telemetry replica count")
        replicas = int(telemetry["replicas"])
        velocity_xy = stride_average(velocity_xy, commands, replicas, c.stride_controls)
        yaw_rate = stride_average(yaw_rate[:, None], commands, replicas, c.stride_controls)[:, 0]
    linear_scale = c.scale_k * torch.linalg.vector_norm(commands[:, :2], dim=-1).clamp_min(c.minimum_speed_mps)
    yaw_scale = c.scale_k * commands[:, 2].abs().clamp_min(c.minimum_yaw_rate_rad_s)
    return linear_scale, yaw_scale, velocity_xy, yaw_rate


def paper_reward(telemetry, commands, previous_target, terminated, config=None, nominal_height=None,
                 *, paper_config=PAPER_CONFIG):
    """Table I task plus penalty terms; the shared TaskConfig argument is unused."""
    n = commands.shape[0]
    missing = [key for key in REQUIRED if key not in telemetry]
    if missing:
        raise ValueError(f"Paper reward telemetry lacks {missing}")
    t = {key: torch.as_tensor(telemetry[key]).to(commands.device, torch.float32) for key in REQUIRED}
    if not torch.equal(t["command"], commands):
        raise ValueError("Reward command differs from the native completed hold")
    c = paper_config
    velocity, gyro = t["linear_velocity_nav"], t["angular_velocity_body"]
    rms_torque = (t["torque_square_sum_400hz"] / c.substeps).sqrt()
    acceleration = (t["joint_velocity_rad_s"] - t["previous_joint_velocity_rad_s"]) / c.control_dt_s
    torque_excess = (t["requested_torque_abs_max_400hz"].abs() - c.torque_limit_nm).clamp_min(0)
    velocity_excess = (t["joint_velocity_rad_s"].abs() - c.velocity_limit_rad_s).clamp_min(0)
    norm = torch.linalg.vector_norm
    linear_scale, yaw_scale, tracked_xy, tracked_yaw = _tracking(velocity[:, :2], gyro[:, 2], commands, telemetry, c)
    components = {
        "linear_tracking": c.linear_tracking_weight * torch.exp(-norm(tracked_xy - commands[:, :2], dim=-1) / linear_scale),
        "yaw_tracking": c.yaw_tracking_weight * torch.exp(-(tracked_yaw - commands[:, 2]).abs() / yaw_scale),
        "vertical_velocity": -c.vertical_velocity_weight * velocity[:, 2].square(),
        "roll_pitch_rate": -c.roll_pitch_weight * norm(gyro[:, :2], dim=-1),
        "joint_torque": -c.torque_weight * norm(rms_torque, dim=-1),
        "joint_acceleration": -c.acceleration_weight * norm(acceleration, dim=-1),
        "action_rate": -c.action_rate_weight * norm(t["action"] - t["previous_action"], dim=-1),
        "collisions": -c.collision_weight * (t["other_body_force_max_400hz"] > c.collision_force_n).to(torch.float32),
        "torque_limit": -c.torque_limit_weight * norm(torque_excess, dim=-1),
        "velocity_limit": -c.velocity_limit_weight * norm(velocity_excess, dim=-1),
    }
    if any(value.shape != (n,) for value in components.values()):
        raise ValueError("Paper reward rows differ from the command rows")
    reward = sum(components.values())
    if not bool(torch.isfinite(reward).all()):
        raise FloatingPointError("Nonfinite paper reward")
    return reward, components


# Continuous penalties and the weight each one scales; limit and event terms are
# sparse, so an average-magnitude rule would make a single violation enormous.
CALIBRATED_TERMS = {"vertical_velocity": "vertical_velocity_weight", "roll_pitch_rate": "roll_pitch_weight",
                    "joint_torque": "torque_weight", "joint_acceleration": "acceleration_weight",
                    "action_rate": "action_rate_weight"}
TRACKING_TERMS = ("linear_tracking", "yaw_tracking")


def calibrate(traces, *, ratio=.8, base=PAPER_CONFIG, com_local=None):
    """Weights giving the continuous penalties, in equal shares, ``ratio`` of the mean tracking reward."""
    from . import reward_scorer
    from .task import TaskConfig
    com_local = reward_scorer.root_com_local() if com_local is None else com_local
    sums = {name: 0. for name in (*TRACKING_TERMS, *CALIBRATED_TERMS)}
    rows = 0
    for trace in traces:
        def reward(*args, **kwargs):
            return paper_reward(*args, paper_config=base)
        _, components = reward_scorer.evaluate(trace, reward, TaskConfig(), None, com_local)
        rows += len(components["linear_tracking"])
        for name in sums:
            sums[name] += float(components[name].double().abs().sum())
    tracking = sum(sums[name] for name in TRACKING_TERMS) / rows
    share = ratio * tracking / len(CALIBRATED_TERMS)
    weights = {field: float(f"{getattr(base, field) * share / (sums[term] / rows):.3g}")
               for term, field in CALIBRATED_TERMS.items()}
    return {"rule": f"Continuous penalties together average {ratio:g} of the mean tracking reward, in equal shares",
            "ratio": ratio, "controls": rows, "mean_tracking_reward": tracking,
            "traces": [{"path": str(trace.path), "sha256": trace.sha256} for trace in traces], "weights": weights}


# Frozen output of calibrate() on the seven recorded focus-case rollouts: the
# trajectory-optimizer forward replay and both 2026-09-17 PPO arms at update 1200.
CALIBRATION = {
    "rule": "Continuous penalties together average 0.8 of the mean tracking reward, in equal shares",
    "ratio": .8,
    "controls": 6583,
    "mean_tracking_reward": 1.1411,
    "trace_sha256": {
        "trajectory_optimizer_20260917/replay_001": "dc737dbfabe20bea65684319b2e6a3a80ea9b7f159dcb2cf321cf98975316dac",
        "forward_example_ppo_20260917/example_evaluate_update001200_001/evaluation_00": "643fcc9ed1cfcde8e7fed64bdec30e2430c9eee95684a931515389e02347d627",
        "forward_example_ppo_20260917/example_evaluate_update001200_001/evaluation_01": "5c98ad2fd2cad543b1488285dee0f350ba67674d602fc49eecf9377a6f2e970c",
        "forward_example_ppo_20260917/example_evaluate_update001200_001/evaluation_02": "8655623ae7cb0be831d30ccf4fa2333391b563d32e4adbd4f33c9e0b2868ce12",
        "forward_example_ppo_20260917/scratch_evaluate_update001200_001/evaluation_00": "b869357f63f53939b6ec93ac6f883a67ac39a04504d6ffd310dbcca3524f12e1",
        "forward_example_ppo_20260917/scratch_evaluate_update001200_001/evaluation_01": "60c5b699af37d760065baeaa8b99cc094e74d5af9fe27efbd13f9dcbb7b90098",
        "forward_example_ppo_20260917/scratch_evaluate_update001200_001/evaluation_02": "71ce79033a4e327051c728d6526be3cea9210ef5f1521f7e772f846ac154f8af",
    },
    "weights": {"vertical_velocity_weight": 137., "roll_pitch_weight": 1.18, "torque_weight": .0691,
                "acceleration_weight": .000741, "action_rate_weight": .215},
}
CALIBRATED_CONFIG = replace(PAPER_CONFIG, **CALIBRATION["weights"])


def paper_reward_calibrated(telemetry, commands, previous_target, terminated, config=None, nominal_height=None):
    """The paper reward with continuous penalties rescaled by ``CALIBRATION``."""
    return paper_reward(telemetry, commands, previous_target, terminated, config, nominal_height,
                        paper_config=CALIBRATED_CONFIG)


def variant_config(spec):
    """Parse ``kernel=...,penalties=paper|calibrated,tracking=<multiplier>,<field>=<value>,...``."""
    options = {}
    for item in filter(None, spec.split(",")):
        key, separator, value = item.partition("=")
        if not separator or not key or key in options:
            raise ValueError(f"Variant options are unique key=value pairs, got {item!r}")
        options[key] = value
    penalties = options.pop("penalties", "paper")
    if penalties not in ("paper", "calibrated"):
        raise ValueError(f"penalties must be paper or calibrated, got {penalties!r}")
    base = CALIBRATED_CONFIG if penalties == "calibrated" else PAPER_CONFIG
    multiplier = float(options.pop("tracking", 1))
    if "kernel" in options:
        options["tracking_kernel"] = options.pop("kernel")
    types = {field.name: {"float": float, "int": int, "str": str}[field.type] for field in fields(PaperRewardConfig)}
    unknown = sorted(set(options) - set(types))
    if unknown:
        raise ValueError(f"Unknown variant options: {unknown}")
    config = replace(base, **{key: types[key](value) for key, value in options.items()})
    config = replace(config, linear_tracking_weight=config.linear_tracking_weight * multiplier,
                     yaw_tracking_weight=config.yaw_tracking_weight * multiplier)
    if config.tracking_kernel not in KERNELS:
        raise ValueError(f"Tracking kernel must be one of {KERNELS}, got {config.tracking_kernel!r}")
    return config


def variant(spec):
    """Reward factory for ``--reward NAME=locomotion.paper_reward:variant:<spec>``."""
    config = variant_config(spec)

    def reward(telemetry, commands, previous_target, terminated, _config=None, nominal_height=None):
        return paper_reward(telemetry, commands, previous_target, terminated, paper_config=config)
    reward.config = config
    return reward


def main(argv=None):
    from . import reward_scorer
    parser = argparse.ArgumentParser(description="Recompute the calibrated penalty weights from recorded traces.")
    parser.add_argument("traces", nargs="+", help="Evaluation control_trace.npz files.")
    parser.add_argument("--ratio", type=float, default=.8, help="Penalty-to-tracking magnitude ratio.")
    args = parser.parse_args(argv)
    result = calibrate([reward_scorer.load_trace(path) for path in args.traces], ratio=args.ratio)
    result["paper_weights"] = {field: asdict(PAPER_CONFIG)[field] for field in CALIBRATED_TERMS.values()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
