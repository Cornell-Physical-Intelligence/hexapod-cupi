"""Versioned command/reward wrapper; native physics and Stage 2 gates are unchanged.

Commands are held through complete controls. Rewards use the held command c[t];
only the next actor/critic command fields change when a timer expires. No gait
phase, reference target, reset or simulator step is inserted by the wrapper.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import torch

SCHEMA = "canonical_paper_walking_task_v1"


@dataclass(frozen=True)
class TaskConfig:
    seed: int = 20260914
    hold_min_seconds: float = 2.
    hold_max_seconds: float = 5.
    zero_command_fraction: float = .15
    speed_levels_mps: tuple = (.025, .05)
    maximum_speed_mps: float = .05
    yaw_rate_rad_s: float = .2
    arc_speed_mps: float = .04
    arc_yaw_rate_rad_s: float = .15
    linear_error_variance: float = .0009
    yaw_error_variance: float = .04
    linear_tracking_weight: float = 1.
    yaw_tracking_weight: float = .3
    quiet_joint_rate_scale_rad_s: float = .03
    quiet_joint_rate_weight: float = .05
    quiet_target_step_scale_rad: float = .002
    quiet_target_step_weight: float = .02
    tilt_weight: float = .5
    angular_xy_weight: float = .01
    vertical_velocity_weight: float = .05
    height_weight: float = 5.
    normalized_effort_weight: float = .01
    target_movement_weight: float = .005
    nonfoot_event_weight: float = 1.
    nonfoot_force_weight: float = .02
    terminal_penalty: float = 2.

    def validate(self):
        if self.hold_min_seconds != 2. or self.hold_max_seconds != 5.:
            raise ValueError("Version1 command holds are fixed at 2 to 5 seconds")
        if self.zero_command_fraction != .15:
            raise ValueError("Version1 reserves 15 percent of command draws for zero")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Nonnegative integer seed required")
        if tuple(self.speed_levels_mps) != (.025, .05):
            raise ValueError("Version1 declares .025/.05 m/s curriculum levels")
        if self.maximum_speed_mps not in self.speed_levels_mps:
            raise ValueError("Choose one declared curriculum maximum explicitly")
        for key, value in asdict(self).items():
            if isinstance(value, float) and (not math.isfinite(value) or value <= 0):
                raise ValueError("Positive finite task coefficient required: " + key)


def command_bank(config):
    """Nonzero navigation-frame commands; zero has a separate reserved quota."""
    config.validate()
    result = []
    for speed in config.speed_levels_mps:
        if speed <= config.maximum_speed_mps:
            for bearing in range(8):
                angle = bearing * math.pi / 4
                result.append([speed * math.cos(angle), speed * math.sin(angle), 0.])
    result.extend([[0., 0., -config.yaw_rate_rad_s], [0., 0., config.yaw_rate_rad_s]])
    if config.arc_speed_mps <= config.maximum_speed_mps:
        result.extend([[config.arc_speed_mps, 0., -config.arc_yaw_rate_rad_s],
                       [config.arc_speed_mps, 0., config.arc_yaw_rate_rad_s]])
    return result


def measured_reward(telemetry, commands, previous_target, terminated, config, nominal_height):
    """Actual measured endpoint states and all-eight-substep motor summaries only."""
    n, device = commands.shape[0], commands.device
    def field(name, shape):
        value = telemetry[name]
        if not torch.is_tensor(value) or tuple(value.shape) != (n, *shape) or not bool(torch.isfinite(value).all()):
            raise ValueError("Missing/nonfinite task telemetry: " + name)
        return value.to(device)
    velocity = field("linear_velocity_nav", (3,))
    gyro = field("angular_velocity_body", (3,))
    pose = field("root_pose_xyzw", (7,))
    rate = field("joint_velocity_rad_s", (18,))
    target = field("joint_target_rad", (18,))
    torque_square = field("torque_square_sum_400hz", (18,))
    other_force = field("other_body_force_max_400hz", ())
    held = field("command", (3,))
    if not torch.equal(held, commands):
        raise ValueError("Reward command differs from the native completed hold")
    quiet = (commands == 0).all(-1)
    target_delta = target - previous_target
    quaternion = pose[:, 3:] / torch.linalg.vector_norm(pose[:, 3:], dim=-1, keepdim=True)
    x, y, z, w = quaternion.unbind(-1)
    gravity_xy_square = 4 * ((x*z-w*y).square() + (y*z+w*x).square())
    components = {
        "linear_tracking": config.linear_tracking_weight * torch.exp(-(velocity[:, :2]-commands[:, :2]).square().sum(-1)/config.linear_error_variance),
        "yaw_tracking": config.yaw_tracking_weight * torch.exp(-(gyro[:, 2]-commands[:, 2]).square()/config.yaw_error_variance),
        "quiet_joint_rate": -config.quiet_joint_rate_weight * quiet * (rate/config.quiet_joint_rate_scale_rad_s).square().mean(-1),
        "quiet_target_motion": -config.quiet_target_step_weight * quiet * (target_delta/config.quiet_target_step_scale_rad).square().mean(-1),
        "tilt": -config.tilt_weight * gravity_xy_square,
        "roll_pitch_rate": -config.angular_xy_weight * gyro[:, :2].square().mean(-1),
        "vertical_velocity": -config.vertical_velocity_weight * velocity[:, 2].square(),
        "height": -config.height_weight * (pose[:, 2]-nominal_height).square(),
        "effort": -config.normalized_effort_weight * torque_square.mean(-1)/(8*1.6**2),
        "target_motion": -config.target_movement_weight * (target_delta/.04).square().mean(-1),
        "nonfoot_contact": -config.nonfoot_event_weight * (other_force > 1.) - config.nonfoot_force_weight * other_force.clamp_min(0),
        "termination": -config.terminal_penalty * terminated.to(torch.float32),
    }
    reward = sum(components.values())
    if not bool(torch.isfinite(reward).all()):
        raise FloatingPointError("Nonfinite measured task reward")
    return reward, components


class TrainingTask:
    def __init__(self, env, config=None, output_dir=None):
        self.env = env
        self.config = TaskConfig() if config is None else config
        self.config.validate()
        if env.cfg.control_dt != .02 or env.cfg.decimation != 8:
            raise ValueError("Task requires the unchanged 50/400Hz environment")
        self.bank = torch.tensor(command_bank(self.config), dtype=torch.float32, device=env.device)
        self.rng = torch.Generator(device="cpu").manual_seed(self.config.seed)
        self.remaining_controls = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.previous_target = env.held.detach().clone()
        self.command_draws = 0
        self.zero_command_draws = 0
        self.controls_completed = 0
        self.last_components = {}
        self.last_held_command = None
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.nominal_height = float(env.reference_metadata.get("root_height_m", env.reset_height))
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / "task_definition.json"
            content = json.dumps(self.declaration(), indent=2, allow_nan=False) + "\n"
            if path.exists() and path.read_text() != content:
                raise ValueError("Existing task definition differs")
            path.write_text(content)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def declaration(self):
        return {"schema": SCHEMA, "config": asdict(self.config), "nonzero_command_bank": self.bank.cpu().tolist(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "command_frame": "navigation forward=-nativeY,left=nativeX,yaw=+Z",
            "zero_sampling": "Balanced cumulative quota ceil(0.15*draws), shuffled within each draw batch; at least15% of draws, not a guarantee of time occupancy.",
            "hold_controls_inclusive": [100, 250], "command_columns": [210, 213],
            "reward_attribution": "Reward uses held c[t]; post-control command fields expose c[t+1].",
            "nominal_plate_height_m": self.nominal_height,
            "maximum_speed_curriculum": "Explicit configuration; no automatic advance or qualification claim.",
            "nonfoot_scope": "Native body/coxa/femur floor forces; tibia shaft/toe separation is unavailable in compact training telemetry.",
            "physics_changed": False, "stage2_gates_changed": False, "stage2_complete": False}

    def _indices(self, indices):
        result = torch.arange(self.env.num_envs, device=self.env.device) if indices is None else torch.as_tensor(indices, dtype=torch.long, device=self.env.device)
        if result.ndim != 1 or bool((result < 0).any()) or bool((result >= self.env.num_envs).any()) or len(result.unique()) != len(result):
            raise ValueError("Invalid command/reset row indices")
        return result

    def _resample(self, indices):
        if not len(indices):
            return
        size = len(indices)
        bank_indices = torch.randint(len(self.bank), (size,), generator=self.rng)
        values = self.bank[bank_indices.to(self.env.device)].clone()
        # An integer quota avoids a finite pilot accidentally seeing no stops.
        before = math.ceil(self.config.zero_command_fraction*self.command_draws)
        after = math.ceil(self.config.zero_command_fraction*(self.command_draws+size))
        zero_count = after-before
        selected_zero = torch.randperm(size, generator=self.rng)[:zero_count].to(self.env.device)
        values[selected_zero] = 0
        durations = torch.randint(100, 251, (size,), generator=self.rng, dtype=torch.long).to(self.env.device)
        self.env.commands[indices] = values
        self.remaining_controls[indices] = durations
        self.command_draws += size
        self.zero_command_draws += zero_count

    def _next_command_observation(self, output):
        obs, critic = output["obs"].clone(), output["critic"].clone()
        if obs.shape != (self.env.num_envs, 231) or critic.shape != (self.env.num_envs, 234):
            raise ValueError("Task actor/critic ABI differs")
        if not torch.equal(critic[:, :231], obs):
            raise ValueError("Task received inconsistent actor/critic prefix")
        obs[:, 210:213] = self.env.commands
        critic[:, :231] = obs
        return {**output, "obs": obs, "critic": critic}

    def reset(self, indices=None):
        selected = self._indices(indices)
        output = self.env.reset(selected)
        self._resample(selected)
        self.previous_target[selected] = self.env.held[selected]
        return self._next_command_observation(output)

    def step(self, action):
        if bool((self.remaining_controls <= 0).any()):
            raise RuntimeError("Call task.reset before collecting a rollout")
        command = self.env.commands.detach().clone()
        output = self.env.step(action)
        reward, components = measured_reward(self.env.telemetry, command, self.previous_target,
                                             output["terminated"], self.config, self.nominal_height)
        self.previous_target = self.env.telemetry["joint_target_rad"].detach().clone()
        self.last_held_command = command
        self.last_components = {key: value.detach().clone() for key, value in components.items()}
        self.remaining_controls -= 1
        done = output["terminated"] | output["truncated"]
        # Terminal observations remain attributed to their final command. The
        # learner supplies an explicit selected reset before collecting again.
        expired = ((self.remaining_controls == 0) & ~done).nonzero(as_tuple=False).squeeze(-1)
        self._resample(expired)
        self.controls_completed += 1
        result = self._next_command_observation(output)
        result["reward"] = reward
        return result

    def status(self):
        return {"schema": SCHEMA, "controls_completed": self.controls_completed,
                "command_draws": self.command_draws, "zero_command_draws": self.zero_command_draws,
                "zero_draw_fraction": self.zero_command_draws/max(1, self.command_draws),
                "remaining_controls": self.remaining_controls.detach().cpu().tolist(),
                "commands": self.env.commands.detach().cpu().tolist(), "stage2_complete": False}
