"""Reward version 4: tracking and penalties sized for exploration noise, plus a stepping term.

Versions 2 and 3 calibrated their penalty weights on deterministic rollouts and paid them on
rollouts that carry exploration noise. PPO then earned reward by quieting its own noise and never
moved (docs/REWARD_V2_TABLE1_AUDIT.md section 15). Version 4 keeps Table I's term list and changes
how each term is measured and weighted; ``DEPARTURES`` lists every change with its reason.

``measured_reward_v4`` is pure: the task passes its per-replica memory in the telemetry.
"""
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import math
from pathlib import Path

import torch

from .task import TaskConfig, TrainingTask
from .task_v2 import ACTUATOR, CONTROL_DT_S, SUBSTEPS, StrideWindow

REWARD_VERSION = "noise_calibrated_stepping_v4"
KERNELS = ("quadratic", "absolute", "exponential")
SCALES = ("command", "fixed")
RATED_TORQUE_NM = 1.6
DEPARTURES = (
    "Tracking error uses the mean displacement velocity over the last tracking_window_controls "
    "controls since the latest command change or reset, in the body frame. Table I tracks the "
    "instantaneous velocity. At this robot's 0.025 to 0.05 m/s commands, exploration shakes the "
    "body by about 0.035 m/s per control, which exceeds the command; a position difference removes "
    "that jitter and a short window keeps the reward close to the action that earned it.",
    "The quadratic kernel 1 - e^2 replaces exp(-|e| / 0.15). Its expectation under zero-mean "
    "velocity noise equals the noise-free value minus a constant, so the gain from walking does "
    "not shrink with noise. The printed exponent has no minus sign and its 0.15 m/s scale pays a "
    "motionless robot 72 to 85 percent at these commands (audit sections 2 and 4).",
    "The error scale equals the commanded speed or yaw rate, so a motionless robot earns zero "
    "tracking reward on every moving command. A zero component uses a fixed scale (0.05 m/s, "
    "0.2 rad/s): version 2's floors (0.025 m/s, 0.15 rad/s with k 0.4) made stillness under a "
    "zero command the term that exploration noise taxed most.",
    "A stepping term pays each touchdown its preceding air time minus a threshold, on moving "
    "commands. Table I has no such term: the paper's gait comes from its adversarial style reward, "
    "which vanilla PPO omits. The term reads measured foot forces and no reference motion.",
    "Each continuous penalty is a mean square against a declared scale. Table I prints unsquared "
    "norms, which charge exploration noise in first order. Weights are set on rollouts that carry "
    "the training noise, so the noise-driven total stays below a declared share of the tracking "
    "reward; version 2's weights were 15 to 34,550 times the printed values.",
    "The target-rate penalty reads the executed joint target after the action clamp and the "
    "0.040 rad limiter. Version 2 read the raw sample, which charged the policy for noise that "
    "never reached the joints.",
    "A height term keeps the plate near its nominal height. Table I has none; the 400 Hz gate "
    "counts tibia-shaft contact as non-foot contact and compact training telemetry cannot.",
    "Quiet terms penalize joint rate and target change under a zero command, as version 1 does, "
    "because the stop gates bound both.",
    "Any non-tibia floor contact above 1 N costs collision_weight per control; the simulation "
    "reports no self-collision.",
    "Table I has no termination term. PPO bootstraps zero after a termination, so termination_weight "
    "exceeds the discounted loss of the largest per-control reward.",
)
OMITTED = (
    "Style r^s: no discriminator and no demonstration enter vanilla PPO.",
    "The foot contact-force limit: the paper states no limit.",
    "Joint velocity limit: the 50.27 rad/s URDF limit is 8 times the fastest recorded joint speed.",
)


@dataclass(frozen=True)
class RewardV4Config:
    tracking_kernel: str = "quadratic"
    tracking_scale: str = "command"
    tracking_window_controls: int = 10
    scale_k: float = 1.
    fixed_speed_scale_mps: float = .05
    fixed_yaw_scale_rad_s: float = .2
    tracking_floor: float = 1.
    linear_tracking_weight: float = 1.
    yaw_tracking_weight: float = .5
    air_time_weight: float = 0.
    air_time_threshold_s: float = .1
    air_time_cap_s: float = .5
    swing_weight: float = 0.
    swing_feet: int = 3
    stale_swing_weight: float = 0.
    quiet_contact_weight: float = 1.
    swing_travel_weight: float = 1.
    swing_travel_clip: float = 6.
    stance_slip_weight: float = 0.
    stance_slip_scale_mps: float = .02
    schedule_weight: float = 2.
    schedule_period_controls: int = 60
    schedule_swing_fraction: float = .4
    schedule_load_n: float = 12.2
    contact_force_n: float = 1.
    vertical_velocity_weight: float = .1
    vertical_velocity_scale_mps: float = .04
    roll_pitch_weight: float = .05
    roll_pitch_scale_rad_s: float = .3
    torque_weight: float = .05
    acceleration_weight: float = .05
    acceleration_scale_rad_s2: float = 100.
    target_rate_weight: float = .05
    target_rate_scale_rad: float = .04
    height_weight: float = .2
    height_scale_m: float = .02
    collision_weight: float = 1.
    torque_limit_weight: float = .5
    quiet_joint_rate_weight: float = .2
    quiet_joint_rate_scale_rad_s: float = 1.
    quiet_target_weight: float = .2
    quiet_target_scale_rad: float = .04
    termination_weight: float = 20.

    def validate(self):
        if self.tracking_kernel not in KERNELS or self.tracking_scale not in SCALES:
            raise ValueError("Unknown reward v4 tracking kernel or scale")
        if type(self.tracking_window_controls) is not int or not 1 <= self.tracking_window_controls <= 250:
            raise ValueError("The tracking window needs 1 to 250 controls")
        if type(self.swing_feet) is not int or not 1 <= self.swing_feet <= 3:
            raise ValueError("The swing term counts one to three feet")
        if type(self.schedule_period_controls) is not int or not 10 <= self.schedule_period_controls <= 250:
            raise ValueError("The contact schedule period needs 10 to 250 controls")
        if not 0 < self.schedule_swing_fraction <= .5:
            raise ValueError("Each tripod swings for at most half the schedule period")
        for key, value in asdict(self).items():
            if isinstance(value, float) and (not math.isfinite(value) or value < 0):
                raise ValueError("Nonnegative finite reward v4 coefficient required: " + key)
        scales = [key for key in asdict(self) if "_scale_" in key] + ["schedule_load_n"]
        if any(getattr(self, key) <= 0 for key in scales + ["scale_k", "air_time_cap_s", "contact_force_n"]):
            raise ValueError("Positive reward v4 scales required")
        if self.air_time_cap_s <= self.air_time_threshold_s:
            raise ValueError("The air-time cap must exceed its threshold")
        # A fall must cost more than the discounted loss of the largest per-control reward.
        ceiling = self.linear_tracking_weight + self.yaw_tracking_weight
        if self.termination_weight and self.termination_weight < 10 * ceiling:
            raise ValueError("The termination weight must reach ten times the tracking ceiling")


REWARD_V4_CONFIG = RewardV4Config()


def reward_declaration(config):
    return {"version": REWARD_VERSION, "config": asdict(config),
        "formulas": {
            "tracked_velocity": "mean over the last tracking_window_controls controls, since the latest command "
                "change or reset, of the body-frame root displacement and heading change per control divided by 0.02 s",
            "error": {"command": "e_lin = ||v - c_xy|| / (k ||c_xy||), or / s_lin when c_xy is zero; "
                                 "e_yaw = |w - c_yaw| / (k |c_yaw|), or / s_yaw when c_yaw is zero",
                      "fixed": "e_lin = ||v - c_xy|| / s_lin; e_yaw = |w - c_yaw| / s_yaw"}[config.tracking_scale],
            "kernel": {"quadratic": "max(1 - e^2, -floor)", "absolute": "max(1 - e, -floor)",
                       "exponential": "exp(-e^2)"}[config.tracking_kernel],
            "linear_tracking": "w_lin kernel(e_lin)", "yaw_tracking": "w_yaw kernel(e_yaw)",
            "air_time": "w sum_feet [touchdown this control] (min(t_air, cap) - threshold), moving commands only",
            "swing": "w min(feet off the floor for at most cap, swing_feet) / swing_feet, moving commands only",
            "stale_swing": "-w (feet off the floor for longer than cap), moving commands only",
            "quiet_contact": "-w [zero command] (feet off the floor) / 6",
            "swing_travel": "w min(sum_feet [off the floor] clip(u_foot . d_foot / |d_foot|^2, -clip, clip) / 6, 1), "
                "with u_foot the toe velocity over the floor, in body axes, and d_foot = c_xy + c_yaw z x r_foot the "
                "commanded body velocity at that toe; a gait that walks at the command scores 1; moving commands only",
            "gait_schedule": "w (sum over scheduled-swing feet of u - sum over scheduled-stance feet of u) / 3, with "
                "u = clip(1 - toe force / load, 0, 1) the unloaded share of a foot; tripod lf, lr, rm swings for the "
                "first swing fraction of the period and tripod lm, rf, rr half a period later; moving commands only",
            "stance_slip": "-w mean_feet([on the floor at this and the previous control] (planar toe speed / s)^2)",
            "vertical_velocity": "-w (v_z / s)^2", "roll_pitch_rate": "-w mean((w_xy / s)^2)",
            "joint_torque": "-w mean(sum_substeps tau^2 / 8) / 1.6^2",
            "joint_acceleration": "-w mean(((dq - dq_prev) / 0.02 / s)^2)",
            "target_rate": "-w mean(((q_target - q_target_prev) / s)^2), executed targets",
            "height": "-w ((z - z_nominal) / s)^2",
            "collisions": "-w [max non-tibia floor force > 1 N]",
            "torque_limit": "-w mean(max(|tau_requested| - 1.6, 0) / 1.6)",
            "quiet_joint_rate": "-w [zero command] mean((dq / s)^2)",
            "quiet_target_motion": "-w [zero command] mean(((q_target - q_target_prev) / s)^2)",
            "termination": "-w [height, tilt or joint-limit termination this control]"},
        "departures_from_table_1": list(DEPARTURES), "omitted": list(OMITTED), "actuator": ACTUATOR,
        "review": {"status": "experimental", "basis": "docs/REWARD_V2_TABLE1_AUDIT.md section 15"}}


def heading(quaternion_xyzw):
    """World yaw of the body, from a unit XYZW quaternion."""
    x, y, z, w = quaternion_xyzw.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def displacement_velocity(pose, previous_pose):
    """Navigation-frame planar velocity and yaw rate from two consecutive root poses."""
    from .env import inverse_rotate, navigation
    planar = navigation(inverse_rotate(pose[:, 3:], (pose[:, :3] - previous_pose[:, :3]) / CONTROL_DT_S))[:, :2]
    turn = heading(pose[:, 3:]) - heading(previous_pose[:, 3:])
    turn = torch.atan2(torch.sin(turn), torch.cos(turn))
    return torch.cat((planar, (turn / CONTROL_DT_S)[:, None]), -1)


# Leg order lf, lm, lr, rf, rm, rr: the first tripod is lf, lr, rm, as in tripod.py.
TRIPOD_A = (True, False, True, False, True, False)


def scheduled_swing(episode_steps, config=REWARD_V4_CONFIG):
    """Six booleans per replica: the feet that the contact schedule wants off the floor at this control."""
    fraction = (episode_steps.to(torch.float32) % config.schedule_period_controls) / config.schedule_period_controls
    first = torch.tensor(TRIPOD_A, device=episode_steps.device)
    swing_a = fraction < config.schedule_swing_fraction
    swing_b = (fraction >= .5) & (fraction < .5 + config.schedule_swing_fraction)
    return torch.where(first, swing_a[:, None], swing_b[:, None])


def tracking_errors(tracked, commands, config=REWARD_V4_CONFIG):
    """Dimensionless planar and yaw errors of already selected velocities."""
    c, norm = config, torch.linalg.vector_norm
    linear = torch.full_like(commands[:, 0], c.fixed_speed_scale_mps)
    yaw = torch.full_like(commands[:, 0], c.fixed_yaw_scale_rad_s)
    if c.tracking_scale == "command":
        speed, turn = norm(commands[:, :2], dim=-1), commands[:, 2].abs()
        linear = torch.where(speed > 0, c.scale_k * speed, linear)
        yaw = torch.where(turn > 0, c.scale_k * turn, yaw)
    return norm(tracked[:, :2] - commands[:, :2], dim=-1) / linear, (tracked[:, 2] - commands[:, 2]).abs() / yaw


def kernel(error, config=REWARD_V4_CONFIG):
    if config.tracking_kernel == "quadratic":
        return (1 - error.square()).clamp_min(-config.tracking_floor)
    if config.tracking_kernel == "absolute":
        return (1 - error).clamp_min(-config.tracking_floor)
    return torch.exp(-error.square())


def measured_reward_v4(telemetry, commands, terminated, config=REWARD_V4_CONFIG, nominal_height=None):
    """Version 4 terms from one control's telemetry plus the caller's memory.

    Memory fields: ``tracked_velocity_nav`` (planar velocity and yaw rate, already averaged),
    ``previous_joint_velocity_rad_s``, ``previous_joint_target_rad``, ``touchdown`` (six booleans),
    ``air_time_s`` (the air time each foot had completed before this control's contact) and
    ``swing`` (six booleans: off the floor now, for no longer than the air-time cap),
    ``airborne`` (six booleans: off the floor now), ``toe_velocity_nav`` (planar toe velocity over the floor, in
    body axes) and ``toe_xyz_nav`` (planar toe position in the body frame).
    """
    n, device = commands.shape[0], commands.device
    def field(name, shape, dtype=torch.float32):
        value = telemetry.get(name)
        if not torch.is_tensor(value) or tuple(value.shape) != (n, *shape) or not bool(torch.isfinite(value.to(torch.float32)).all()):
            raise ValueError("Missing/nonfinite reward v4 telemetry: " + name)
        return value.to(device, dtype)
    velocity = field("linear_velocity_nav", (3,))
    gyro = field("angular_velocity_body", (3,))
    pose = field("root_pose_xyzw", (7,))
    rate = field("joint_velocity_rad_s", (18,))
    previous_rate = field("previous_joint_velocity_rad_s", (18,))
    target = field("joint_target_rad", (18,))
    previous_target = field("previous_joint_target_rad", (18,))
    torque_square = field("torque_square_sum_400hz", (18,))
    requested = field("requested_torque_abs_max_400hz", (18,))
    other_force = field("other_body_force_max_400hz", ())
    tracked = field("tracked_velocity_nav", (3,))
    touchdown = field("touchdown", (6,), torch.bool)
    air_time = field("air_time_s", (6,))
    # Records made before the swing term existed carry no swing field; they score with weight zero alone.
    swing = (field("swing", (6,), torch.bool) if "swing" in telemetry or config.swing_weight
             else torch.zeros(n, 6, dtype=torch.bool, device=device))
    lifted = (field("airborne", (6,), torch.bool) if config.stale_swing_weight or config.quiet_contact_weight
              else torch.zeros(n, 6, dtype=torch.bool, device=device))
    if not torch.equal(field("command", (3,)), commands):
        raise ValueError("Reward command differs from the native completed hold")
    terminated = torch.as_tensor(terminated, device=device)
    if tuple(terminated.shape) != (n,) or terminated.dtype != torch.bool:
        raise ValueError("Reward v4 needs one boolean termination flag per row")
    if nominal_height is None:
        raise ValueError("Reward v4 needs the nominal plate height")
    c = config
    if c.swing_travel_weight:
        airborne = field("airborne", (6,), torch.bool)
        toe_velocity = field("toe_velocity_nav", (6, 2))
        toe = field("toe_xyz_nav", (6, 2))
        # Commanded body velocity at each toe: translation plus yaw rate times the lever arm.
        wanted = commands[:, None, :2] + commands[:, None, 2:] * torch.stack((-toe[..., 1], toe[..., 0]), -1)
        ratio = (toe_velocity * wanted).sum(-1) / wanted.square().sum(-1).clamp_min(1e-8)
        travel = (ratio.clamp(-c.swing_travel_clip, c.swing_travel_clip) * airborne).sum(-1) / 6
    else:
        travel = torch.zeros(n, device=device)
    if c.schedule_weight:
        # A foot that carries less than its even share of the weight counts as partly lifted.
        toe_force = torch.linalg.vector_norm(field("tibia_floor_force_world_n", (6, 3)), dim=-1)
        unloaded = (1 - toe_force / c.schedule_load_n).clamp(0, 1)
        wanted_swing = field("scheduled_swing", (6,), torch.bool).to(torch.float32)
        schedule = (unloaded * (2 * wanted_swing - 1)).sum(-1) / 3
    else:
        schedule = torch.zeros(n, device=device)
    if c.stance_slip_weight:
        slip = ((field("toe_slip_mps", (6,)) / c.stance_slip_scale_mps).square() * field("planted", (6,), torch.bool)).mean(-1)
    else:
        slip = torch.zeros(n, device=device)
    quiet = (commands == 0).all(-1).to(torch.float32)
    moving = 1 - quiet
    linear_error, yaw_error = tracking_errors(tracked, commands, c)
    target_step = target - previous_target
    stride = (air_time.clamp_max(c.air_time_cap_s) - c.air_time_threshold_s) * touchdown
    components = {
        "linear_tracking": c.linear_tracking_weight * kernel(linear_error, c),
        "yaw_tracking": c.yaw_tracking_weight * kernel(yaw_error, c),
        "air_time": c.air_time_weight * moving * stride.sum(-1),
        "swing": c.swing_weight * moving * swing.sum(-1).clamp_max(c.swing_feet) / c.swing_feet,
        "stale_swing": -c.stale_swing_weight * moving * (lifted & ~swing).sum(-1),
        "quiet_contact": -c.quiet_contact_weight * quiet * lifted.sum(-1) / 6,
        "swing_travel": c.swing_travel_weight * moving * travel.clamp_max(1.),
        "gait_schedule": c.schedule_weight * moving * schedule,
        "stance_slip": -c.stance_slip_weight * slip,
        "vertical_velocity": -c.vertical_velocity_weight * (velocity[:, 2] / c.vertical_velocity_scale_mps).square(),
        "roll_pitch_rate": -c.roll_pitch_weight * (gyro[:, :2] / c.roll_pitch_scale_rad_s).square().mean(-1),
        "joint_torque": -c.torque_weight * (torque_square / SUBSTEPS).mean(-1) / RATED_TORQUE_NM**2,
        "joint_acceleration": -c.acceleration_weight * ((rate - previous_rate) / CONTROL_DT_S / c.acceleration_scale_rad_s2).square().mean(-1),
        "target_rate": -c.target_rate_weight * (target_step / c.target_rate_scale_rad).square().mean(-1),
        "height": -c.height_weight * ((pose[:, 2] - nominal_height) / c.height_scale_m).square(),
        "collisions": -c.collision_weight * (other_force > c.contact_force_n).to(torch.float32),
        "torque_limit": -c.torque_limit_weight * ((requested - RATED_TORQUE_NM).clamp_min(0) / RATED_TORQUE_NM).mean(-1),
        "quiet_joint_rate": -c.quiet_joint_rate_weight * quiet * (rate / c.quiet_joint_rate_scale_rad_s).square().mean(-1),
        "quiet_target_motion": -c.quiet_target_weight * quiet * (target_step / c.quiet_target_scale_rad).square().mean(-1),
        "termination": -c.termination_weight * terminated.to(torch.float32),
    }
    reward = sum(components.values())
    if not bool(torch.isfinite(reward).all()):
        raise FloatingPointError("Nonfinite reward v4")
    return reward, components


class TrainingTaskV4(TrainingTask):
    """Version 1's command, guard and reporting task with reward version 4."""

    reward_config = REWARD_V4_CONFIG

    def __init__(self, env, config=None, output_dir=None, reward_config=None):
        task_config = TaskConfig() if config is None else config
        self.reward_config = type(self).reward_config if reward_config is None else reward_config
        self.reward_config.validate()
        n, device = env.num_envs, env.device
        self.previous_joint_velocity = env.current["dq"].detach().clone().to(device)
        self.previous_pose = env.current["root"].detach().clone().to(device)
        self.window = StrideWindow(n, self.reward_config.tracking_window_controls, 3, device)
        self.air_time = torch.zeros(n, 6, device=device)
        self.in_contact = torch.ones(n, 6, dtype=torch.bool, device=device)
        self.previous_toe = self._toe_nav(env.current["toe_body"]).to(device)
        self.previous_toe_world = env.current["toe_world"].detach().clone().to(device)
        self.gait_totals = {}
        super().__init__(env, task_config, output_dir)

    def declaration(self):
        result = super().declaration()
        del result["quiet_cost_tail"]
        result.update(reward_version=REWARD_VERSION, reward=reward_declaration(self.reward_config),
            reward_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            unused_version1_reward_fields=[name for name in (f.name for f in fields(TaskConfig))
                                           if name.endswith(("_weight", "_variance", "_scale_rad_s", "_scale_rad", "_penalty"))])
        return result

    @staticmethod
    def _toe_nav(toe_body):
        """Planar toe positions in the navigation axes of the body frame."""
        from .env import navigation
        return navigation(toe_body.detach())[..., :2].clone()

    def reset(self, indices=None):
        output = super().reset(indices)
        selected = self._indices(indices)
        self.previous_toe[selected] = self._toe_nav(self.env.current["toe_body"])[selected].to(self.previous_toe)
        self.previous_toe_world[selected] = self.env.current["toe_world"][selected].detach().to(self.previous_toe_world)
        self.previous_joint_velocity[selected] = self.env.current["dq"][selected].to(self.previous_joint_velocity)
        self.previous_pose[selected] = self.env.current["root"][selected].to(self.previous_pose)
        self.window.restart(selected, self.env.commands[selected])
        self.air_time[selected] = 0
        self.in_contact[selected] = True
        return output

    def reward_telemetry(self, command):
        """Current telemetry plus this task's memory; advances the tracking window and foot timers."""
        t = self.env.telemetry
        pose = t["root_pose_xyzw"].to(self.previous_pose)
        tracked = self.window.update(displacement_velocity(pose, self.previous_pose), command)
        contact = torch.linalg.vector_norm(t["tibia_floor_force_world_n"], dim=-1) > self.reward_config.contact_force_n
        touchdown = contact & ~self.in_contact
        planted = contact & self.in_contact
        world = t["toe_xyz_world"].detach().to(self.previous_toe_world)
        previous_world = self.previous_toe_world
        slip = torch.linalg.vector_norm((world - previous_world)[..., :2], dim=-1) / CONTROL_DT_S
        self.previous_toe_world = world.clone()
        completed = self.air_time.clone()
        self.air_time = torch.where(contact, torch.zeros_like(self.air_time), self.air_time + CONTROL_DT_S)
        self.in_contact = contact
        swing = ~contact & (self.air_time <= self.reward_config.air_time_cap_s + 1e-6)
        # The control just completed started at episode step minus one.
        wanted = (scheduled_swing(self.env.episode_steps - 1, self.reward_config).to(contact.device)
                  if self.reward_config.schedule_weight else torch.zeros_like(contact))
        from .env import inverse_rotate, navigation
        toe = self._toe_nav(t["toe_xyz_body"]).to(self.previous_toe)
        self.previous_toe = toe
        # Toe velocity over the floor, expressed in the body's navigation axes.
        heading = pose[:, None, 3:].expand(-1, 6, -1)
        toe_velocity = navigation(inverse_rotate(heading, (world - previous_world) / CONTROL_DT_S))[..., :2]
        return {**t, "tracked_velocity_nav": tracked, "previous_joint_velocity_rad_s": self.previous_joint_velocity,
                "previous_joint_target_rad": self.previous_target, "touchdown": touchdown, "air_time_s": completed,
                "swing": swing, "airborne": ~contact, "toe_velocity_nav": toe_velocity, "toe_xyz_nav": toe,
                "planted": planted, "toe_slip_mps": slip, "scheduled_swing": wanted}

    def step(self, action):
        # Version 1's step with measured_reward_v4 in place of measured_reward.
        if bool((self.remaining_controls <= 0).any()):
            raise RuntimeError("Call task.reset before collecting a rollout")
        roots_before = self.env.current["root"][:, :3].clone()
        self.proximity.check(roots_before, "before_control",
            speeds=torch.linalg.vector_norm(self.env.current["linear"], dim=-1))
        command = self.env.commands.detach().clone()
        output = self.env.step(action)
        self.proximity.check(self.env.current["root"][:, :3], "after_control", previous=roots_before)
        telemetry = self.reward_telemetry(command)
        reward, components = measured_reward_v4(telemetry, command, output["terminated"], self.reward_config,
                                                self.nominal_height)
        metric_previous_target = self.previous_target
        self.previous_target = self.env.telemetry["joint_target_rad"].detach().clone()
        self.previous_joint_velocity = self.env.telemetry["joint_velocity_rad_s"].detach().clone().to(self.previous_joint_velocity)
        self.previous_pose = self.env.telemetry["root_pose_xyzw"].detach().clone().to(self.previous_pose)
        self.last_held_command = command
        self.last_components = {key: value.detach().clone() for key, value in components.items()}
        self.remaining_controls -= 1
        done = output["terminated"] | output["truncated"]
        expired = ((self.remaining_controls == 0) & ~done).nonzero(as_tuple=False).squeeze(-1)
        self._resample(expired)
        self.controls_completed += 1
        self._metrics(command, output, reward, components, previous_target=metric_previous_target)
        self._gait_metrics(command, telemetry)
        result = self._next_command_observation(output)
        result["reward"] = reward
        return result

    def _gait_metrics(self, command, telemetry):
        """Reporting accumulators for stepping; these values never enter the learning objective."""
        moving = (command != 0).any(-1)
        tracked = telemetry["tracked_velocity_nav"].to(torch.float64)
        speed = torch.linalg.vector_norm(command[:, :2], dim=-1).to(torch.float64)
        along = (tracked[:, :2] * command[:, :2].to(torch.float64)).sum(-1) / speed.clamp_min(1e-8)
        translating = speed > 1e-8
        values = {"moving_rows": moving.sum().to(torch.float64),
            "moving_foot_contact_rows": (self.in_contact & moving[:, None]).sum(0).to(torch.float64),
            "moving_touchdowns": (telemetry["touchdown"] & moving[:, None]).sum(0).to(torch.float64),
            "moving_air_time_s": (telemetry["air_time_s"] * (telemetry["touchdown"] & moving[:, None])).sum(0).to(torch.float64),
            "translating_rows": translating.sum().to(torch.float64),
            "tracked_speed_ratio_sum": (along[translating] / speed[translating]).sum()}
        for key, value in values.items():
            self.gait_totals[key] = self.gait_totals.get(key, torch.zeros_like(value)) + value

    def status(self, *, reset_interval=False):
        result = {**super().status(reset_interval=reset_interval), "reward_version": REWARD_VERSION}
        totals = {key: value.detach().cpu() for key, value in self.gait_totals.items()}
        if totals:
            rows, touchdowns = float(totals["moving_rows"]), totals["moving_touchdowns"]
            translating = float(totals["translating_rows"])
            result["interval_gait"] = {"moving_command_rows": rows,
                "foot_contact_fraction": (totals["moving_foot_contact_rows"] / max(1., rows)).tolist(),
                "touchdowns_per_foot_per_second": (touchdowns / max(1., rows) / CONTROL_DT_S).tolist(),
                "mean_air_time_s": (totals["moving_air_time_s"] / touchdowns.clamp_min(1.)).tolist(),
                "tracked_speed_ratio": float(totals["tracked_speed_ratio_sum"]) / translating if translating else None,
                "scope": "Moving-command controls in this interval; contact is the 50 Hz endpoint toe force above the "
                         "declared threshold. Reporting values, not acceptance."}
        if reset_interval:
            self.gait_totals = {}
        return result


def reward_config(spec=""):
    """``RewardV4Config`` from ``key=value,...`` overrides of the defaults."""
    types = {f.name: f.type for f in fields(RewardV4Config)}
    options = {}
    for item in filter(None, spec.split(",")):
        key, separator, value = item.partition("=")
        if not separator or key not in types or key in options:
            raise ValueError(f"Reward v4 options are unique RewardV4Config key=value pairs, got {item!r}")
        options[key] = types[key](value)
    config = replace(REWARD_V4_CONFIG, **options)
    config.validate()
    return config


def variant(spec):
    """A task class with overridden reward coefficients, for design comparisons."""
    return type("TrainingTaskV4Variant", (TrainingTaskV4,), {"reward_config": reward_config(spec)})
