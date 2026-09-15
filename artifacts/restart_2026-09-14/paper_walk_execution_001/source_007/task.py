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
import numpy as np
import torch

SCHEMA = "canonical_paper_walking_task_v1"
METRIC_COMMAND_CLASSES = ("zero", "linear_0.025", "linear_0.05", "yaw", "arc", "other")


def articulation_reach(model, geometry_path):
    """All-pose spherical bound: path translation lengths plus exact mesh radius.

    Joint rotations preserve each vector length, so the triangle inequality
    encloses every articulated pose; this does not assume a reference gait.
    """
    geometry_path = Path(geometry_path)
    geometry = json.loads(geometry_path.read_text())
    path = geometry_path.parent / "geometry_extrema.npz"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != geometry.get("extrema_sha256"):
        raise ValueError("Proximity geometry extrema identity differs")
    lengths = {"body": 0.}
    pending = list(model["joints"])
    while pending:
        before = len(pending)
        for joint in pending[:]:
            if joint["parent"] in lengths:
                if joint["child"] in lengths:
                    raise ValueError("Proximity model has a repeated child")
                lengths[joint["child"]] = lengths[joint["parent"]] + float(np.linalg.norm(joint["xyz"]))
                pending.remove(joint)
        if len(pending) == before:
            raise ValueError("Proximity model is not a connected tree")
    if set(lengths) != set(geometry["body_names"]):
        raise ValueError("Proximity mesh/body mapping differs")
    bounds = {}
    with np.load(path, allow_pickle=False) as clouds:
        for name, length in lengths.items():
            points = clouds["all__"+name]
            if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
                raise ValueError("Incomplete proximity mesh extrema")
            bounds[name] = length + float(np.linalg.norm(points, axis=1).max())
    skin = max(float(shape["contact_offset_m"]) for shape in geometry["shapes"])
    return {"mesh_radius_m": max(bounds.values()), "contact_skin_m": skin,
            "radius_m": max(bounds.values())+skin, "body_radius_bounds_m": bounds,
            "geometry_extrema_sha256": digest}


class ProximityGuard:
    """Abort on conservative world-root sphere proximity; no simulation writes."""
    buffer_m = .20

    def __init__(self, env, output_dir):
        self.env, self.output_dir = env, output_dir
        self.geometry = articulation_reach(env.model, env.geometry_path)
        self.radius = self.geometry["radius_m"]
        self.minimum_gap = None
        self.checks = 0
        self.failure = None
        self.check(env.current["root"][:, :3], "initial")

    def declaration(self):
        return {**self.geometry, "buffer_m": self.buffer_m, "collision_filter_present": False,
            "scope": "Conservative all-pose spheres before/after each control and prospective reset; measured linear root sweep is also checked. This abort guard is not collision filtering, continuous collision detection, or proof of replica independence.",
            "failure_action": "Save failure and abort entire training attempt; never reset to hide proximity."}

    def check(self, positions, phase, *, previous=None, speeds=None):
        if self.failure is not None:
            raise RuntimeError("Inter-replica proximity attempt already failed")
        if positions.shape != (self.env.num_envs, 3) or not bool(torch.isfinite(positions).all()):
            raise ValueError("Invalid native roots for proximity guard")
        self.checks += 1
        if self.env.num_envs < 2:
            return
        relative = positions[:, None]-positions[None, :]
        if previous is not None:
            start = previous[:, None]-previous[None, :]
            delta = relative-start
            fraction = (-torch.sum(start*delta, -1)/torch.sum(delta*delta, -1).clamp_min(1e-20)).clamp(0, 1)
            relative = start+fraction[..., None]*delta
        gap = torch.linalg.vector_norm(relative, dim=-1)-2*self.radius
        gap.fill_diagonal_(torch.inf)
        threshold = torch.full_like(gap, self.buffer_m)
        if speeds is not None:
            if speeds.shape != (self.env.num_envs,) or not bool(torch.isfinite(speeds).all()):
                raise ValueError("Invalid measured speeds for proximity lookahead")
            threshold += (speeds[:, None]+speeds[None, :])*self.env.cfg.control_dt
        minimum = float(gap.min())
        self.minimum_gap = minimum if self.minimum_gap is None else min(self.minimum_gap, minimum)
        violation = gap <= threshold
        if bool(violation.any()):
            pair = violation.nonzero(as_tuple=False)[0]
            i, j = (int(x) for x in pair)
            self.failure = {"phase": phase, "pair": [i, j], "sphere_gap_m": float(gap[i, j]),
                "required_gap_m": float(threshold[i, j]), "root_positions_world_m": positions.detach().cpu().tolist(),
                "previous_root_positions_world_m": None if previous is None else previous.detach().cpu().tolist(),
                "controls_completed": getattr(self.env, "total_controls", None),
                "physics_steps": self.env.sim.get_physics_step_count() if hasattr(self.env, "sim") else None,
                "declaration": self.declaration(), "contaminated_control_returned_to_learner": False,
                "stage2_complete": False}
            if self.output_dir is not None:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                (self.output_dir/"proximity_failure.json").write_text(json.dumps(self.failure, indent=2, allow_nan=False)+"\n")
            raise RuntimeError("Inter-replica proximity guard rejected pair " + str([i, j]) + " during " + phase)

    def status(self):
        return {"checks": self.checks, "minimum_sphere_gap_m": self.minimum_gap,
                "failure": self.failure, "collision_filter_present": False}


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
        self.proximity = ProximityGuard(env, self.output_dir)
        self.metric_totals, self.metric_interval = {}, {}
        self.metric_rows = self.metric_interval_rows = 0
        self.metric_component_names = None
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
            "proximity_guard": self.proximity.declaration(),
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
        self.proximity.check(self.env.current["root"][:, :3], "before_reset")
        prospective = self.env.current["root"][:, :3].clone()
        prospective[selected] = self.env.origins[selected]
        prospective[selected, 2] = self.env.reset_height
        self.proximity.check(prospective, "prospective_reset")
        output = self.env.reset(selected)
        self.proximity.check(self.env.current["root"][:, :3], "after_reset")
        self._resample(selected)
        self.previous_target[selected] = self.env.held[selected]
        return self._next_command_observation(output)

    def step(self, action):
        if bool((self.remaining_controls <= 0).any()):
            raise RuntimeError("Call task.reset before collecting a rollout")
        roots_before = self.env.current["root"][:, :3].clone()
        self.proximity.check(roots_before, "before_control",
            speeds=torch.linalg.vector_norm(self.env.current["linear"], dim=-1))
        command = self.env.commands.detach().clone()
        output = self.env.step(action)
        self.proximity.check(self.env.current["root"][:, :3], "after_control", previous=roots_before)
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
        self._metrics(command, output, reward, components)
        result = self._next_command_observation(output)
        result["reward"] = reward
        return result

    def _metrics(self, command, output, reward, components):
        t = self.env.telemetry
        actual = t["linear_velocity_nav"][:, :2]
        error = actual-command[:, :2]
        yaw_error = t["angular_velocity_body"][:, 2]-command[:, 2]
        requested_speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
        moving = requested_speed > 1e-8
        projection = (actual*command[:, :2]).sum(-1)/requested_speed.clamp_min(1e-8)
        zero = (command == 0).all(-1)
        turning = command[:, 2] != 0
        masks = [zero,
            moving & ~turning & torch.isclose(requested_speed, requested_speed.new_tensor(.025), rtol=0, atol=1e-6),
            moving & ~turning & torch.isclose(requested_speed, requested_speed.new_tensor(.05), rtol=0, atol=1e-6),
            ~moving & turning, moving & turning]
        masks.append(~torch.stack(masks).any(0))
        classes = torch.stack(masks).to(torch.float64)
        names = tuple(components)
        if self.metric_component_names is not None and names != self.metric_component_names:
            raise ValueError("Reward metric component names changed during an attempt")
        if reward.shape != (len(command),) or any(value.shape != reward.shape for value in components.values()):
            raise ValueError("Reward metric rows differ")
        component_values = torch.stack(tuple(components.values()), dim=-1).detach().to(torch.float64)
        reward_values = reward.detach().to(torch.float64)
        nonfoot = t["other_body_force_max_400hz"] > 1.
        # Reduce in float64 so long cumulative runs do not lose small component
        # contributions. These tensors never enter the learning objective.
        values = {"signed_navigation_velocity_error_mps": error.sum(0),
            "absolute_navigation_velocity_error_mps": error.abs().sum(0),
            "achieved_planar_speed_mps": torch.linalg.vector_norm(actual, dim=-1).sum(),
            "requested_planar_speed_mps": requested_speed.sum(),
            "signed_command_direction_speed_mps": projection.sum(),
            "moving_command_rows": moving.sum(), "signed_yaw_error_rad_s": yaw_error.sum(),
            "absolute_yaw_error_rad_s": yaw_error.abs().sum(),
            "requested_saturation_fraction": t["saturation_count_400hz"].sum()/(8*18),
            "terminations": output["terminated"].sum(), "truncations": output["truncated"].sum(),
            "zero_command_rows": zero.sum(), "nonfoot_event_rows": nonfoot.sum(),
            "_moving_projection_sum": projection[moving].to(torch.float64).sum(),
            "_reward_sum": reward_values.sum(),
            "_component_sums": component_values.sum(0),
            "_component_absolute_sums": component_values.abs().sum(0),
            "_class_rows": classes.sum(-1),
            "_class_reward_sums": classes @ reward_values,
            "_class_component_sums": classes @ component_values,
            "_class_component_absolute_sums": classes @ component_values.abs(),
            "_class_requested_speed_sums": classes @ requested_speed.to(torch.float64),
            "_class_projection_sums": classes @ (projection*moving).to(torch.float64),
            "_class_moving_rows": classes @ moving.to(torch.float64),
            "_class_nonfoot_rows": classes @ nonfoot.to(torch.float64),
            "_zero_joint_rate_square_sums": t["joint_velocity_rad_s"][zero].to(torch.float64).square().sum(0)}
        if not bool(torch.isfinite(torch.cat([value.reshape(-1).to(torch.float64) for value in values.values()])).all()):
            raise FloatingPointError("Nonfinite task reporting metrics")
        self.metric_component_names = names
        for dest in (self.metric_totals, self.metric_interval):
            for key, value in values.items():
                dest[key] = dest.get(key, torch.zeros_like(value)) + value.detach()
        self.metric_rows += len(command)
        self.metric_interval_rows += len(command)

    def status(self, *, reset_interval=False):
        def summary(values, rows):
            counts = {"moving_command_rows", "terminations", "truncations", "zero_command_rows", "nonfoot_event_rows"}
            report = {"environment_controls": rows, **{key: (value if key in counts else value/max(1, rows)).cpu().tolist()
                for key, value in values.items() if not key.startswith("_")}}
            if not rows:
                return report
            raw = {key: value.detach().cpu() for key, value in values.items() if key.startswith("_")}
            moving_rows, zero_rows = report["moving_command_rows"], report["zero_command_rows"]
            names = self.metric_component_names
            report.update(
                moving_signed_command_direction_speed_mps=float(raw["_moving_projection_sum"])/moving_rows if moving_rows else None,
                nonfoot_event_fraction=report["nonfoot_event_rows"]/rows,
                task_reward_sum=float(raw["_reward_sum"]), task_reward_mean=float(raw["_reward_sum"])/rows,
                reward_components={name: {"sum": float(raw["_component_sums"][i]),
                    "absolute_sum": float(raw["_component_absolute_sums"][i]),
                    "mean": float(raw["_component_sums"][i])/rows,
                    "mean_absolute": float(raw["_component_absolute_sums"][i])/rows} for i, name in enumerate(names)},
                command_classes={})
            for i, name in enumerate(METRIC_COMMAND_CLASSES):
                count = int(raw["_class_rows"][i])
                moving_count = int(raw["_class_moving_rows"][i])
                report["command_classes"][name] = {"environment_controls": count, "time_fraction": count/rows,
                    "task_reward_mean": float(raw["_class_reward_sums"][i])/count if count else None,
                    "reward_component_means": {key: float(raw["_class_component_sums"][i, j])/count if count else None
                        for j, key in enumerate(names)},
                    "reward_component_mean_absolute": {key: float(raw["_class_component_absolute_sums"][i, j])/count if count else None
                        for j, key in enumerate(names)},
                    "requested_planar_speed_mps": float(raw["_class_requested_speed_sums"][i])/count if count else None,
                    "moving_command_rows": moving_count,
                    "moving_signed_command_direction_speed_mps": float(raw["_class_projection_sums"][i])/moving_count if moving_count else None,
                    "nonfoot_event_fraction": float(raw["_class_nonfoot_rows"][i])/count if count else None}
            rms = (raw["_zero_joint_rate_square_sums"]/zero_rows).sqrt() if zero_rows else None
            report.update(zero_hold_joint_rate_rms_rad_s=None if rms is None else rms.tolist(),
                zero_hold_worst_joint_rate_rms_rad_s=None if rms is None else float(rms.max()),
                zero_hold_all_joint_rate_rms_rad_s=None if rms is None else float(rms.square().mean().sqrt()),
                diagnostic_scope="Time-weighted 50Hz control endpoints; nonfoot events use each control's maximum 400Hz body/coxa/femur floor force >1N. These are reporting metrics, not Stage2 admission.")
            return report
        result = {"schema": SCHEMA, "controls_completed": self.controls_completed,
                "command_draws": self.command_draws, "zero_command_draws": self.zero_command_draws,
                "zero_draw_fraction": self.zero_command_draws/max(1, self.command_draws),
                "remaining_controls": self.remaining_controls.detach().cpu().tolist(),
                "commands": self.env.commands.detach().cpu().tolist(), "stage2_complete": False,
                "proximity": self.proximity.status(),
                "cumulative_metrics": summary(self.metric_totals, self.metric_rows),
                "interval_metrics": summary(self.metric_interval, self.metric_interval_rows)}
        if reset_interval:
            self.metric_interval, self.metric_interval_rows = {}, 0
        return result
