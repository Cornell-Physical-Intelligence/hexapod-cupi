"""Physical four-bar DirectRLEnv with 18 motors and 30 tree coordinates.

No passive coordinate receives an action, motor reward, or independent reset
noise. Passive motion is simulated by the closed mechanism after reset.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from hexapod_core import fourbar_v1 as contract
from hexapod_core.rs05_v2 import contract_manifest as motor_contract_manifest, verify_runtime_cfg
from ...command_sampling import body_to_navigation_frame
from .math import MotorCoordinates, observations, reward_terms
from .target_schedule import MotorTargetRamp

ROOT = Path(__file__).resolve().parents[5]


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


class HexapodMkiiFourbarEnv(DirectRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        asset_bundle = contract.resolve_asset_bundle(cfg.robot.spawn.usd_path, repo_root=ROOT)
        super().__init__(cfg, render_mode, **kwargs)
        self.kinematics = contract.load_kinematics(cfg.kinematics_path)
        self.coordinates = MotorCoordinates(self._robot.joint_names, self.kinematics, device=self.device)
        self.active_joint_names = list(contract.ACTIVE_JOINT_NAMES)
        # Isaac Lab 3's Warp write kernels require int32 API indices.
        # Keep the coordinate gather indices in their native Torch dtype.
        self.active_joint_ids = self.coordinates.active_indices.to(dtype=torch.int32)
        if self._robot.num_bodies != 31 or set(self._robot.body_names) != set(contract.BODY_NAMES):
            raise ValueError("Physical four-bar requires exactly the 31 named bodies")
        self._motor_model = self._robot.actuators["motors"]
        motor_names = tuple(self._motor_model.joint_names)
        if len(motor_names) != 18 or set(motor_names) != set(self.active_joint_names):
            raise ValueError("Actuator group must contain exactly the 18 named physical motors")
        self._motor_order = torch.tensor([motor_names.index(name) for name in self.active_joint_names],
                                         dtype=torch.long, device=self.device)
        if set(self._robot.actuators) != {"motors"}:
            raise ValueError("Unexpected actuator group; passive joints must not be driven")
        motor_cfg = verify_runtime_cfg(self._motor_model.cfg, self.active_joint_names)
        if (cfg.sim.dt != contract.PHYSICS_DT_S or cfg.sim.dt != motor_cfg["physics_dt_s"]
                or cfg.decimation != contract.DECIMATION or self.step_dt != contract.POLICY_DT_S):
            raise ValueError("Physics/control timing differs from the motor/runtime contract")
        for name, sensor in self._body_contact_sensors.items():
            if list(sensor.body_names) != [name]:
                raise ValueError(f"Incorrect contact sensor binding for {name}: {sensor.body_names}")
        if any(sensor.contact_view.filter_count != 1 for sensor in self._feet_contact_sensors):
            raise ValueError("Each foot must have exactly one ground filter")
        self._feet_body_ids, names = self._robot.find_bodies([f"{leg}_tibia" for leg in contract.LEGS], preserve_order=True)
        if names != [f"{leg}_tibia" for leg in contract.LEGS]:
            raise ValueError("Foot articulation indices do not match named sensors")
        self._actions = torch.zeros(self.num_envs, 18, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._processed_actions = self.coordinates.default.repeat(self.num_envs, 1)
        self._target_schedule = MotorTargetRamp(self._processed_actions, substeps=contract.DECIMATION)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._command_time_left_s = torch.zeros(self.num_envs, device=self.device)
        self._joint_target_slew_limited_fraction = torch.zeros(self.num_envs, device=self.device)
        self._reward_sums = {}
        self.last_termination_reasons = {}
        self._standing_only = bool(cfg.standing_only)
        self._stand_action_scale = 1.0
        if cfg.command_frame != contract.COMMAND_FRAME:
            raise ValueError("Physical policy requires anatomical -Y-forward navigation commands")
        if (cfg.action_scale != contract.ACTION_SCALE_RAD or cfg.stand_action_scale != 1.
                or cfg.processed_joint_target_slew_limit_rad_per_20ms != contract.SLEW_RAD_PER_20MS):
            raise ValueError("Resolved action parameters differ from the physical runtime contract")
        if not math.isfinite(cfg.reset_joint_jitter_rad) or not 0 <= cfg.reset_joint_jitter_rad <= .03:
            raise ValueError("Reset motor jitter must be finite and within [0,.03] radians")
        kin_hash = hashlib.sha256(Path(cfg.kinematics_path).read_bytes()).hexdigest()
        usd_hash = hashlib.sha256(Path(cfg.robot.spawn.usd_path).read_bytes()).hexdigest()
        if contract.resolve_asset_bundle(cfg.robot.spawn.usd_path, repo_root=ROOT) != asset_bundle:
            raise ValueError("Physical bundle changed during environment construction")
        self.runtime_manifest = contract.runtime_manifest(self.kinematics,
            motor_contract_manifest(physics_dt_s=motor_cfg["physics_dt_s"],
                                    assumed_bus_voltage_v=motor_cfg["assumed_bus_voltage_v"]),
            kinematics_sha256=kin_hash, usd_sha256=usd_hash, asset_bundle=asset_bundle)
        self.runtime_manifest["resolved_motor_configuration"] = motor_cfg
        self.runtime_manifest["observed_tree_joint_names"] = list(self._robot.joint_names)
        self.runtime_manifest["observed_motor_model_joint_names"] = list(motor_names)
        self.runtime_manifest["resolved_simulation"] = {
            "physics_dt_s": cfg.sim.dt, "decimation": cfg.decimation,
            "solver_type": cfg.sim.physics.solver_type,
            "enable_external_forces_every_iteration": cfg.sim.physics.enable_external_forces_every_iteration,
            "solver_position_iterations": cfg.robot.spawn.articulation_props.solver_position_iteration_count,
            "solver_velocity_iterations": cfg.robot.spawn.articulation_props.solver_velocity_iteration_count,
            "self_collision_enabled": cfg.robot.spawn.articulation_props.enabled_self_collisions,
            "reset_motor_jitter_rad": cfg.reset_joint_jitter_rad,
        }

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._body_contact_sensors = {name: ContactSensor(cfg) for name, cfg in self.cfg.body_contact_sensors.items()}
        for name, sensor in self._body_contact_sensors.items():
            self.scene.sensors[f"contact_{name}"] = sensor
        self._feet_contact_sensors = [self._body_contact_sensors[f"{leg}_tibia"] for leg in contract.LEGS]
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light = sim_utils.DomeLightCfg(intensity=2000., color=(.75, .78, .82))
        light.func("/World/Light", light)

    def motor_state(self, name):
        return self.coordinates.gather(tensor(getattr(self._robot.data, name)))

    def motor_telemetry(self, name):
        value = tensor(getattr(self._motor_model, name))
        if value.shape != (self.num_envs, 18):
            raise ValueError(f"Unexpected motor telemetry shape for {name}: {value.shape}")
        return value.index_select(-1, self._motor_order)

    def closure_coordinate_error(self):
        return self.coordinates.closure_coordinate_error(tensor(self._robot.data.joint_pos))

    def _vector_in_command_frame(self, vectors):
        return body_to_navigation_frame(vectors)

    def _pre_physics_step(self, actions):
        self._actions, self._processed_actions, self._joint_target_slew_limited_fraction = self.coordinates.process_action(
            actions, self._processed_actions, step_dt=self.step_dt)
        self._target_schedule.begin(self._processed_actions)

    def _apply_action(self):
        self._robot.set_joint_position_target_index(target=self._target_schedule.step(), joint_ids=self.active_joint_ids)

    def _get_observations(self):
        data = self._robot.data
        result = observations(tensor(data.root_lin_vel_b), tensor(data.root_ang_vel_b), tensor(data.projected_gravity_b),
            self._commands, self.motor_state("joint_pos")-self.coordinates.default,
            self.motor_state("joint_vel"), self._actions, self.motor_telemetry("burst_headroom"))
        self._previous_actions.copy_(self._actions)
        return {"policy": result}

    def _contact_state(self):
        forces = torch.stack([tensor(sensor.data.force_matrix_w)[:, 0, 0] for sensor in self._feet_contact_sensors], 1)
        points = torch.stack([tensor(sensor.data.contact_pos_w)[:, 0, 0] for sensor in self._feet_contact_sensors], 1)
        positions = torch.stack([tensor(sensor.data.pos_w)[:, 0] for sensor in self._feet_contact_sensors], 1)
        quats = torch.stack([tensor(sensor.data.quat_w)[:, 0] for sensor in self._feet_contact_sensors], 1)
        finite_point = torch.isfinite(points).all(-1)
        loaded = torch.linalg.norm(forces, dim=-1) > 1.
        safe_points = torch.where(finite_point.unsqueeze(-1), points, positions)
        local = math_utils.quat_apply_inverse(quats, safe_points-positions)
        pads = loaded & finite_point & (local[:, :, 1] > self.cfg.distal_foot_min_y_m)
        shafts = loaded & finite_point & ~pads
        link_pos = tensor(self._robot.data.body_link_pos_w)[:, self._feet_body_ids]
        lin = tensor(self._robot.data.body_link_lin_vel_w)[:, self._feet_body_ids]
        ang = tensor(self._robot.data.body_link_ang_vel_w)[:, self._feet_body_ids]
        speed = torch.linalg.norm((lin + torch.cross(ang, safe_points-link_pos, dim=-1))[:, :, :2], dim=-1)
        bad = ((loaded & ~finite_point).any(-1) | ~torch.isfinite(forces).all(dim=(-1,-2))
               | ~torch.isfinite(positions).all(dim=(-1,-2)) | ~torch.isfinite(quats).all(dim=(-1,-2)))
        return pads, shafts, speed, bad

    def _get_foot_contact_state(self):
        return self._contact_state()[:3]

    def _nonfoot_contact_counts(self, shafts):
        fields = [tensor(sensor.data.net_forces_w)[:, 0] for name, sensor in self._body_contact_sensors.items()
                  if not name.endswith("_tibia")]
        forces = torch.stack(fields, 1)
        count = (torch.linalg.norm(forces, dim=-1) > 1.).sum(-1) + shafts.sum(-1)
        return count, ~torch.isfinite(forces).all(dim=(-1,-2))

    def _get_rewards(self):
        data = self._robot.data
        pads, shafts, speed, _ = self._contact_state()
        nonfoot, _ = self._nonfoot_contact_counts(shafts)
        components = reward_terms(command=self._commands,
            linear_navigation=body_to_navigation_frame(tensor(data.root_lin_vel_b)),
            angular_navigation=body_to_navigation_frame(tensor(data.root_ang_vel_b)),
            gravity_body=tensor(data.projected_gravity_b), active_torque=self.motor_state("applied_torque"),
            active_velocity=self.motor_state("joint_vel"), active_acceleration=self.motor_state("joint_acc"),
            active_position=self.motor_state("joint_pos"), soft_limits=self.coordinates.soft_limits,
            action=self._actions, previous_action=self._previous_actions,
            height=tensor(data.root_pos_w)[:, 2]-tensor(self._terrain.env_origins)[:, 2],
            nominal_height=self.cfg.nominal_height_m, nonfoot_contacts=nonfoot, support_count=pads.sum(-1),
            foot_slip=(speed.square()*pads).sum(-1), clipping_nm=self.motor_telemetry("clipping_nm"),
            overload_nm=self.motor_telemetry("continuous_overload_nm"))
        for name, value in components.items():
            if name not in self._reward_sums:
                self._reward_sums[name] = torch.zeros_like(value)
            self._reward_sums[name] += value * self.step_dt
        reward = torch.stack(list(components.values())).sum(0) * self.step_dt
        reward -= self.reset_terminated.float() * 5.
        # The completed transition retains its old command; the next policy
        # observation receives a newly sampled command after its hold expires.
        self._command_time_left_s -= self.step_dt
        due = torch.nonzero((self._command_time_left_s <= 0) & ~self.reset_buf).flatten()
        if len(due):
            self._sample_commands(due)
        return reward

    def _get_dones(self):
        data = self._robot.data
        _, shafts, _, bad_contact = self._contact_state()
        _, bad_other_contact = self._nonfoot_contact_counts(shafts)
        finite = torch.stack([torch.isfinite(tensor(getattr(data, name))).all(-1) for name in (
            "joint_pos", "joint_vel", "computed_torque", "applied_torque", "root_pos_w", "root_quat_w",
            "root_lin_vel_b", "root_ang_vel_b", "projected_gravity_b")]).all(0)
        base_force = torch.linalg.norm(tensor(self._body_contact_sensors["body"].data.net_forces_w)[:,0], dim=-1)
        height = tensor(data.root_pos_w)[:,2]-tensor(self._terrain.env_origins)[:,2]
        headroom = self.motor_telemetry("burst_headroom")
        reasons = {
            "base_contact": base_force > 5., "low_height": height < .055,
            "upside_down": tensor(data.projected_gravity_b)[:,2] > -.45,
            "nonfinite_state": ~finite, "invalid_contact": bad_contact | bad_other_contact,
            "invalid_motor_input": self.motor_telemetry("invalid_input").any(-1),
            "invalid_motor_budget": (~torch.isfinite(headroom) | (headroom < 0) | (headroom > 1)).any(-1),
            "closure_coordinate_error": self.closure_coordinate_error().abs().max(-1)[0] > self.cfg.closure_coordinate_termination_rad,
        }
        self.last_termination_reasons = reasons
        return torch.stack(list(reasons.values())).any(0), self.episode_length_buf >= self.max_episode_length-1

    def _sample_commands(self, env_ids):
        if self._standing_only:
            self._commands[env_ids] = 0.
        else:
            commands = torch.empty(len(env_ids), 3, device=self.device)
            for index, interval in enumerate((self.cfg.command_lin_vel_x_range_mps,
                    self.cfg.command_lin_vel_y_range_mps, self.cfg.command_yaw_rate_range_rad_s)):
                commands[:,index].uniform_(*interval)
            commands[torch.rand(len(env_ids), device=self.device) < self.cfg.stand_command_fraction] = 0.
            self._commands[env_ids] = commands
        self._command_time_left_s[env_ids] = self.cfg.command_hold_time_s

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        else:
            env_ids = tensor(env_ids).to(dtype=torch.int32)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self.extras["log"] = {f"Episode_Reward/{name}": values[env_ids].mean()/self.max_episode_length_s
                              for name, values in self._reward_sums.items()}
        for values in self._reward_sums.values():
            values[env_ids] = 0.
        self._actions[env_ids] = 0.
        self._previous_actions[env_ids] = 0.
        motors = self.coordinates.default.repeat(len(env_ids), 1)
        if self.cfg.reset_joint_jitter_rad:
            motors += torch.empty_like(motors).uniform_(-self.cfg.reset_joint_jitter_rad, self.cfg.reset_joint_jitter_rad)
            motors = motors.clamp(min=self.coordinates.soft_limits[:,0], max=self.coordinates.soft_limits[:,1])
        q, qd = self.coordinates.closed_reset(motors, torch.zeros_like(motors))
        self._processed_actions[env_ids] = motors
        self._target_schedule.reset(motors, env_ids=env_ids)
        self._joint_target_slew_limited_fraction[env_ids] = 0.
        pose = tensor(self._robot.data.default_root_pose)[env_ids].clone()
        velocity = tensor(self._robot.data.default_root_vel)[env_ids].clone()
        pose[:,:3] += tensor(self._terrain.env_origins)[env_ids]
        self._robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)
        self._robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=env_ids)
        self._robot.set_joint_position_target_index(target=motors, joint_ids=self.active_joint_ids, env_ids=env_ids)
        self._sample_commands(env_ids)
