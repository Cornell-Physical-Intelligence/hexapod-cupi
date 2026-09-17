"""Direct RL environment for the approved mass-corrected RS05 robot.

The control loop repeats the accepted prototype: one 50 Hz control step emits a
held joint target, eight 400 Hz substeps apply the named-damping servo through
the explicit actuator, and the reward and terminations read the post-step
state. Observations use the canonical per-leg joint order; the articulation
reports a different order, so the environment maps between the two by name and
refuses to run when the articulation disagrees with the declared order.

The optional diagnostic capture records every substep in the layout the frozen
scorer reads. Recording evidence admits nothing: the scorer recomputes each
channel and decides.
"""

from __future__ import annotations

import json
from pathlib import Path

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from ...actuators.rs05_paper_walk_model import MAX_APPLIED_NM, damping_vector, motor_effort
from . import math as task_math
from .capture import DiagnosticGeometry, Rs05DiagnosticCapture, capture_row
from .config import HexapodMkiiRs05FlatEnvCfg, GROUND_PLANE_COLLISION_PATH
from .contacts import classify_patches


def tensor(value):
    """Isaac Lab returns Warp arrays for some buffers; keep Torch tensors."""

    return value.torch if hasattr(value, "torch") else value


def native_numpy(value):
    value = tensor(value)
    return np.array(value.cpu().numpy() if hasattr(value, 'cpu') else value.numpy(), copy=True)


def _first_attribute(source, names, message):
    for name in names:
        value = getattr(source, name, None)
        if value is not None:
            return tensor(value)
    raise ValueError(message)


class HexapodMkiiRs05Env(DirectRLEnv):
    cfg: HexapodMkiiRs05FlatEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._verify_timing()
        self._bind_joint_order()
        self._verify_model()
        count, device = self.num_envs, self.device
        self._neutral = self._canonical_vector(self.cfg.robot.init_state.joint_pos)
        self._damping = damping_vector(self.cfg.canonical_joint_names, device=device)
        self._history = torch.zeros(count, task_math.HISTORY_LENGTH, task_math.PROPRIO_WIDTH, device=device)
        self._actions = torch.zeros(count, 18, device=device)
        self._previous_action = torch.zeros(count, 18, device=device)
        self._held = self._neutral.expand(count, -1).clone()
        self._previous_target = self._held.clone()
        self._target_articulation = self._held.index_select(-1, self._articulation_index)
        self._commands = torch.zeros(count, 3, device=device)
        self._command_time_left_s = torch.zeros(count, device=device)
        self._reward_sums = {}
        self.last_termination_reasons = {}
        self._reset_accumulators()
        self._capture = None
        self._capture_geometry = None
        self._pending_row = None
        self._previous_capture_position = None
        self._substep = 0
        self.runtime_manifest = self._runtime_manifest()

    # ---------------------------------------------------------------- scene

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._body_contact_sensors = {
            name: ContactSensor(sensor_cfg) for name, sensor_cfg in self.cfg.body_contact_sensors.items()
        }
        for name, sensor in self._body_contact_sensors.items():
            self.scene.sensors[f"contact_{name}"] = sensor
        self._leg_names = [name[: -len("_tibia")] for name in self.cfg.body_names if name.endswith("_tibia")]
        self._feet_contact_sensors = [self._body_contact_sensors[f"{leg}_tibia"] for leg in self._leg_names]
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.78, 0.82))
        light.func("/World/Light", light)

    # ------------------------------------------------------------ contracts

    def _verify_timing(self):
        if (self.cfg.sim.dt != task_math.PHYSICS_DT_S or self.cfg.decimation != task_math.DECIMATION
                or abs(self.step_dt - task_math.PHYSICS_DT_S * task_math.DECIMATION) > 1e-12):
            raise ValueError("The RS05 task runs 400 Hz physics with eight substeps per control step")
        if (self.cfg.action_scale != task_math.ACTION_SCALE_RAD
                or self.cfg.target_slew_rad_per_control != task_math.TARGET_SLEW_RAD):
            raise ValueError("The accepted action scale and target slew are fixed")
        if (self.cfg.observation_space != task_math.POLICY_OBSERVATION_WIDTH
                or self.cfg.state_space != task_math.CRITIC_OBSERVATION_WIDTH):
            raise ValueError("The declared observation widths differ from the accepted layout")

    def _bind_joint_order(self):
        names = list(self._robot.joint_names)
        if tuple(names) != tuple(self.cfg.expected_runtime_joint_names):
            raise ValueError(f"The articulation joint order differs from the declared order: {names}")
        canonical = list(self.cfg.canonical_joint_names)
        if sorted(canonical) != sorted(names):
            raise ValueError("The canonical joint names differ from the articulation joints")
        self._canonical_index = torch.tensor(
            [names.index(name) for name in canonical], device=self.device, dtype=torch.long
        )
        self._articulation_index = torch.argsort(self._canonical_index)
        bodies = list(self._robot.body_names)
        if sorted(bodies) != sorted(self.cfg.body_names):
            raise ValueError("The articulation bodies differ from the declared 19 bodies")
        self._body_names = bodies
        self._toe_body_index = [bodies.index(f"{leg}_tibia") for leg in self._leg_names]

    def _verify_model(self):
        limits = _first_attribute(
            self._robot.data, ("joint_pos_limits", "joint_limits"),
            "The articulation does not expose joint position limits",
        )
        expected = self._declared_limits()
        actual = limits.index_select(-2, self._canonical_index)
        if actual.shape[-2:] != (18, 2) or not torch.allclose(
            actual, expected.expand(self.num_envs, -1, -1), atol=2e-6, rtol=0
        ):
            raise ValueError("The simulated joint limits differ from the declared model limits")
        self._lower, self._upper = expected[:, 0].contiguous(), expected[:, 1].contiguous()
        motors = self._robot.actuators["legs"]
        motor_names = list(motors.joint_names)
        if len(motor_names) != 18 or set(motor_names) != set(self.cfg.canonical_joint_names):
            raise ValueError("The actuator group must hold exactly the 18 named RS05 motors")
        if set(self._robot.actuators) != {"legs"}:
            raise ValueError("The RS05 task drives one named motor group")
        self._motors = motors

    def _declared_limits(self):
        table = {name: (lower, upper) for name, lower, upper in self.cfg.joint_limits_rad}
        if len(table) != 18:
            raise ValueError("The declared joint limit table must hold the 18 named joints")
        return torch.tensor(
            [[table[name][0], table[name][1]] for name in self.cfg.canonical_joint_names],
            device=self.device, dtype=torch.float32,
        )

    def _canonical_vector(self, values_by_name):
        return torch.tensor(
            [values_by_name[name] for name in self.cfg.canonical_joint_names],
            device=self.device, dtype=torch.float32,
        )

    def _runtime_manifest(self):
        return {
            "task_id": "Isaac-Velocity-Flat-Hexapod-MKII-RS05-Direct-v0",
            "usd_path": self.cfg.robot.spawn.usd_path,
            "asset_usd_sha256": self.cfg.asset_usd_sha256,
            "physics_dt_s": self.cfg.sim.dt,
            "decimation": self.cfg.decimation,
            "solver_position_iterations": self.cfg.robot.spawn.articulation_props.solver_position_iteration_count,
            "solver_velocity_iterations": self.cfg.robot.spawn.articulation_props.solver_velocity_iteration_count,
            "self_collision_enabled": self.cfg.robot.spawn.articulation_props.enabled_self_collisions,
            "soft_joint_pos_limit_factor": self.cfg.robot.soft_joint_pos_limit_factor,
            "action_scale_rad": self.cfg.action_scale,
            "target_slew_rad_per_control": self.cfg.target_slew_rad_per_control,
            "canonical_joint_names": list(self.cfg.canonical_joint_names),
            "articulation_joint_names": list(self._robot.joint_names),
            "body_names": list(self._robot.body_names),
            "observation_width": task_math.POLICY_OBSERVATION_WIDTH,
            "critic_width": task_math.CRITIC_OBSERVATION_WIDTH,
            "physical_admission": False,
            "scope": "Source configuration record; it admits no model and qualifies no behavior.",
        }

    # ---------------------------------------------------------------- state

    def _canonical_joint_state(self):
        data = self._robot.data
        joint_pos = tensor(data.joint_pos).index_select(-1, self._canonical_index)
        joint_vel = tensor(data.joint_vel).index_select(-1, self._canonical_index)
        return joint_pos, joint_vel

    def _root_height(self):
        return tensor(self._robot.data.root_pos_w)[:, 2] - tensor(self._terrain.env_origins)[:, 2]

    def _reset_accumulators(self, env_ids=None):
        shape = (self.num_envs, 18)
        if env_ids is None:
            self._torque_square = torch.zeros(shape, device=self.device)
            self._requested_abs_max = torch.zeros(shape, device=self.device)
            self._applied_abs_max = torch.zeros(shape, device=self.device)
            self._saturation_count = torch.zeros(shape, device=self.device, dtype=torch.long)
        else:
            for buffer in (self._torque_square, self._requested_abs_max, self._applied_abs_max,
                           self._saturation_count):
                buffer[env_ids] = 0

    # ------------------------------------------------------------ RL hooks

    def _pre_physics_step(self, actions):
        action = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if action.shape != (self.num_envs, 18) or not bool(torch.isfinite(action).all()):
            raise ValueError("The policy action must be a finite tensor of shape (environments, 18)")
        self._actions = action.clamp(-1.0, 1.0)
        self._previous_target = self._held
        self._held = task_math.emitted_target(
            action, self._previous_target, self._neutral, self._lower, self._upper,
            self.cfg.action_scale, self.cfg.target_slew_rad_per_control,
        )
        self._target_articulation = self._held.index_select(-1, self._articulation_index)
        self._reset_accumulators()
        self._substep = 0

    def _apply_action(self):
        joint_pos, joint_vel = self._canonical_joint_state()
        requested, applied, ceiling = motor_effort(joint_pos, joint_vel, self._held, self._damping)
        self._torque_square += applied.square()
        self._requested_abs_max = torch.maximum(self._requested_abs_max, requested.abs())
        self._applied_abs_max = torch.maximum(self._applied_abs_max, applied.abs())
        self._saturation_count += (requested.abs() > MAX_APPLIED_NM).long()
        if self._capture is not None:
            self._open_capture_row(joint_pos, joint_vel, requested, applied, ceiling)
        self._robot.set_joint_position_target(self._target_articulation)
        self._substep += 1

    def _advance_control_state(self):
        """Fold the post-step state into the history before any reset runs."""

        if self._capture is not None:
            self._close_capture_row()
        data = self._robot.data
        joint_pos, joint_vel = self._canonical_joint_state()
        row = task_math.proprio_row(
            tensor(data.root_ang_vel_b), tensor(data.projected_gravity_b), joint_pos, joint_vel, self._neutral
        )
        self._history = task_math.advance_history(self._history, row)
        self._previous_action = task_math.executed_action_feature(
            self._held, self._neutral, self.cfg.action_scale
        ).clone()

    def _get_dones(self):
        self._advance_control_state()
        joint_pos, _ = self._canonical_joint_state()
        terminated, reasons = task_math.termination_flags(
            root_height=self._root_height(),
            gravity_body=tensor(self._robot.data.projected_gravity_b),
            joint_pos=joint_pos,
            lower=self._lower,
            upper=self._upper,
        )
        self.last_termination_reasons = reasons
        return terminated, self.episode_length_buf >= self.max_episode_length - 1

    def _get_rewards(self):
        data = self._robot.data
        components, total = task_math.reward_terms(
            commands=self._commands,
            linear_body=tensor(data.root_link_lin_vel_b),
            angular_body=tensor(data.root_ang_vel_b),
            gravity_body=tensor(data.projected_gravity_b),
            torque_square_sum=self._torque_square,
            target=self._held,
            previous_target=self._previous_target,
            terminated=self.reset_terminated,
            substeps=self.cfg.decimation,
            slew=self.cfg.target_slew_rad_per_control,
        )
        for name, value in components.items():
            if name not in self._reward_sums:
                self._reward_sums[name] = torch.zeros_like(value)
            self._reward_sums[name] += value
        self._command_time_left_s -= self.step_dt
        due = torch.nonzero((self._command_time_left_s <= 0) & ~self.reset_buf).flatten()
        if len(due):
            self._sample_commands(due)
        return total

    def _get_observations(self):
        policy = task_math.policy_observation(self._history, self._commands, self._previous_action)
        critic = task_math.critic_observation(policy, tensor(self._robot.data.root_link_lin_vel_b))
        return {"policy": policy, "critic": critic}

    def _sample_commands(self, env_ids):
        if self.cfg.standing_only:
            self._commands[env_ids] = 0.0
        else:
            commands = torch.empty(len(env_ids), 3, device=self.device)
            ranges = (self.cfg.command_forward_range_mps, self.cfg.command_left_range_mps,
                      self.cfg.command_yaw_rate_range_rad_s)
            for index, interval in enumerate(ranges):
                commands[:, index].uniform_(*interval)
            if self.cfg.stand_command_fraction:
                standing = torch.rand(len(env_ids), device=self.device) < self.cfg.stand_command_fraction
                commands[standing] = 0.0
            self._commands[env_ids] = commands
        self._command_time_left_s[env_ids] = self.cfg.command_hold_time_s

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        else:
            env_ids = tensor(env_ids).to(dtype=torch.int32)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        rows = len(env_ids)
        self.extras["log"] = {
            f"Episode_Reward/{name}": values[env_ids].mean()
            for name, values in self._reward_sums.items()
        }
        for values in self._reward_sums.values():
            values[env_ids] = 0.0
        pose = tensor(self._robot.data.default_root_pose)[env_ids].clone()
        pose[:, :3] += tensor(self._terrain.env_origins)[env_ids]
        velocity = torch.zeros_like(tensor(self._robot.data.default_root_vel)[env_ids])
        joint_pos = self._neutral.index_select(-1, self._articulation_index).expand(rows, -1).contiguous()
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)
        self._robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._held[env_ids] = self._neutral
        self._previous_target[env_ids] = self._neutral
        self._target_articulation = self._held.index_select(-1, self._articulation_index)
        self._actions[env_ids] = 0.0
        self._previous_action[env_ids] = 0.0
        self._reset_accumulators(env_ids)
        self._sample_commands(env_ids)
        # The released pose is upright with the stance joints at rest, so the
        # five history frames repeat that state.
        upright = torch.zeros(rows, 3, device=self.device)
        gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(rows, -1)
        row = task_math.proprio_row(
            upright, gravity, self._neutral.expand(rows, -1), torch.zeros(rows, 18, device=self.device),
            self._neutral,
        )
        self._history[env_ids] = row[:, None].expand(-1, task_math.HISTORY_LENGTH, -1)

    # -------------------------------------------------------------- capture

    def begin_diagnostic_capture(self, output, *, geometry_path, geometry_extrema_path, scope, strict=True):
        """Record every substep in the frozen scorer's layout.

        Call this after ``reset``. The capture holds the post-reset state as its
        starting point and aborts on the first failed row.
        """

        if self._capture is not None:
            raise RuntimeError("A diagnostic capture is already running")
        meta_path = Path(geometry_path)
        self._capture_geometry = DiagnosticGeometry.load(
            meta_path, geometry_extrema_path, self._body_names,
            expected_usd_sha256=self.cfg.asset_usd_sha256,
        )
        pattern = "/World/envs/env_*/Robot/*"
        view = self.sim.physics_sim_view
        self._diagnostic_contact_view = view.create_rigid_contact_view(
            pattern, [GROUND_PLANE_COLLISION_PATH], max_contact_data_count=1024*self.num_envs)
        roots = {path+'/Robot': index for index, path in enumerate(self.scene.env_prim_paths)}
        self._diagnostic_sensor_map = [(roots[path.rsplit('/', 1)[0]], path.rsplit('/', 1)[1])
            for path in self._diagnostic_contact_view.sensor_paths]
        expected = {(replica, body) for replica in range(self.num_envs) for body in self._body_names}
        if (set(self._diagnostic_sensor_map) != expected
                or len(self._diagnostic_sensor_map) != len(expected)
                or self._diagnostic_contact_view.filter_count != 1):
            raise ValueError('Diagnostic contact view differs from the complete body mapping')
        joint_pos, joint_vel = self._canonical_joint_state()
        root_pose = self._root_pose_xyzw()
        initial_state = {
            "counter_after": int(self._physics_step_count()),
            "post_reset": {
                "joint_position_rad": joint_pos.cpu().tolist(),
                "joint_velocity_rad_s": joint_vel.cpu().tolist(),
                "root_pose_xyzw": root_pose.cpu().tolist(),
            },
            "scope": "Recorded starting state of a reset-free diagnostic; it implies no admission.",
        }
        self._capture = Rs05DiagnosticCapture(
            output,
            joint_names=list(self.cfg.canonical_joint_names),
            body_names=list(self._body_names),
            root_paths=list(self.scene.env_prim_paths),
            neutral_joint_position_rad=self._neutral.cpu().tolist(),
            native_readback=self._native_readback(),
            initial_state=initial_state,
            physics_dt_s=self.cfg.sim.dt,
            strict=strict,
        )
        self._previous_capture_position = joint_pos.double().cpu().numpy()
        self._pending_row = None
        # DirectRLEnv writes articulation commands after _apply_action and
        # before sim.step. Read the native effort at that write boundary.
        self._capture_original_write = self._robot.write_data_to_sim
        self._robot.write_data_to_sim = self._write_capture_forces
        return self._capture

    def end_diagnostic_capture(self, *, scope):
        if self._capture is None:
            raise RuntimeError("No diagnostic capture is running")
        self._robot.write_data_to_sim = self._capture_original_write
        capture, self._capture = self._capture, None
        capture.close(scope=scope)
        self._pending_row = None
        return capture

    def _write_capture_forces(self):
        self._capture_original_write()
        if self._capture is not None and self._pending_row is not None:
            if 'native_input_pre_nm' in self._pending_row:
                raise RuntimeError('Repeated command write inside one captured substep')
            self._pending_row['native_input_pre_nm'] = native_numpy(
                self._robot.root_view.get_dof_actuation_forces())[:, self._canonical_index.cpu().numpy()]

    def _physics_step_count(self):
        """The simulator's own step counter, or the environment's substep count."""

        method = getattr(self.sim, "get_physics_step_count", None)
        return int(method()) if method is not None else int(self._sim_step_counter)

    def _root_pose_xyzw(self):
        position = tensor(self._robot.data.root_pos_w) - tensor(self._terrain.env_origins)
        quaternion = tensor(self._robot.data.root_quat_w)
        return torch.cat((position, quaternion[:, [1, 2, 3, 0]]), dim=-1)

    def _link_pose_xyzw(self):
        position = tensor(self._robot.data.body_link_pos_w) - tensor(self._terrain.env_origins)[:, None]
        quaternion = tensor(self._robot.data.body_link_quat_w)
        return torch.cat((position, quaternion[:, :, [1, 2, 3, 0]]), dim=-1)

    def _root_com_velocity(self):
        linear = _first_attribute(self._robot.data, ("root_com_lin_vel_w", "root_lin_vel_w"),
                                  "The articulation does not expose a world linear velocity")
        angular = _first_attribute(self._robot.data, ("root_com_ang_vel_w", "root_ang_vel_w"),
                                   "The articulation does not expose a world angular velocity")
        return torch.cat((linear, angular), dim=-1)

    def _native_readback(self):
        limits = _first_attribute(self._robot.data, ("joint_pos_limits", "joint_limits"),
                                  "The articulation does not expose joint position limits")
        velocity = _first_attribute(
            self._robot.data, ("joint_vel_limits", "joint_velocity_limits", "soft_joint_vel_limits"),
            "The articulation does not expose joint velocity limits",
        )
        return {
            "limits": limits.index_select(-2, self._canonical_index).cpu().tolist(),
            "native_max_velocity": velocity.index_select(-1, self._canonical_index).cpu().tolist(),
            "canonical_joint_names": list(self.cfg.canonical_joint_names),
            "articulation_joint_names": list(self._robot.joint_names),
            "body_names": list(self._body_names),
            "solver_iterations_requested": [
                self.cfg.robot.spawn.articulation_props.solver_position_iteration_count,
                self.cfg.robot.spawn.articulation_props.solver_velocity_iteration_count,
            ],
            "implicit_drive_and_armature_zero": True,
            "runtime_manifest": self.runtime_manifest,
        }

    def _open_capture_row(self, joint_pos, joint_vel, requested, applied, ceiling):
        self._close_capture_row()
        self._pending_row = {
            "pre_joint_position_rad": joint_pos.cpu().numpy(),
            "pre_joint_velocity_rad_s": joint_vel.cpu().numpy(),
            "joint_target_rad": self._held.cpu().numpy(),
            "computed_torque_nm": requested.cpu().numpy(),
            "applied_torque_nm": applied.cpu().numpy(),
            "effort_ceiling_nm": ceiling.cpu().numpy(),
            "substep_index": self._substep,
        }

    def _close_capture_row(self):
        if self._capture is None or self._pending_row is None:
            return
        pending, self._pending_row = self._pending_row, None
        joint_pos, joint_vel = self._canonical_joint_state()
        link_pose = self._link_pose_xyzw()
        link_pose_numpy = link_pose.cpu().numpy()
        contacts = self._contact_channels(link_pose_numpy)
        minimum, non_toe = self._capture_geometry.clearance(link_pose_numpy)
        applied_readback = native_numpy(self._robot.root_view.get_dof_actuation_forces())[:, self._canonical_index.cpu().numpy()]
        root_pose = self._root_pose_xyzw().cpu().numpy()
        terminated = (
            (root_pose[:, 2] < 0.055)
            | contacts["nonfoot_contact"]
            | (non_toe < -0.001)
        )
        row = capture_row(
            num_envs=self.num_envs,
            sequence=self._capture.count,
            substep_index=pending["substep_index"],
            explicit_counter=self._physics_step_count(),
            physics_dt_s=self.cfg.sim.dt,
            pre_joint_position_rad=pending["pre_joint_position_rad"],
            pre_joint_velocity_rad_s=pending["pre_joint_velocity_rad_s"],
            joint_position_rad=joint_pos.cpu().numpy(),
            joint_velocity_rad_s=joint_vel.cpu().numpy(),
            root_pose_xyzw=root_pose,
            link_pose_xyzw=link_pose_numpy,
            root_com_velocity=self._root_com_velocity().cpu().numpy(),
            joint_target_rad=pending["joint_target_rad"],
            computed_torque_nm=pending["computed_torque_nm"],
            applied_torque_nm=pending["applied_torque_nm"],
            effort_ceiling_nm=pending["effort_ceiling_nm"],
            native_input_pre_nm=pending["native_input_pre_nm"],
            native_input_post_nm=applied_readback,
            distal_contact=contacts["distal_contact"],
            distal_force_world_n=contacts["distal_force_world_n"],
            nonfoot_contact=contacts["nonfoot_contact"],
            nonfoot_force_world_n=contacts["nonfoot_force_world_n"],
            minimum_mesh_floor_m=minimum,
            minimum_non_toe_floor_m=non_toe,
            previous_joint_position_rad=self._previous_capture_position,
            terminated=terminated,
            # The truncation flag of the control step in progress, as it stood
            # when that step started. A capture runs inside one episode.
            truncated=self.reset_time_outs.cpu().numpy(),
        )
        self._previous_capture_position = row["joint_position_rad"].astype(float)
        row["command"] = self._commands.cpu().numpy().copy()
        self._capture.record(row, patches=contacts["patches"])

    def _contact_channels(self, link_pose):
        data = [native_numpy(value) for value in
                self._diagnostic_contact_view.get_contact_data(dt=self.cfg.sim.dt)]
        return classify_patches(data, self._diagnostic_sensor_map, link_pose,
            tensor(self._terrain.env_origins).cpu().numpy(), self._capture_geometry, self._leg_names)

    def write_runtime_manifest(self, path):
        Path(path).write_text(json.dumps(self.runtime_manifest, indent=2, allow_nan=False) + "\n")
