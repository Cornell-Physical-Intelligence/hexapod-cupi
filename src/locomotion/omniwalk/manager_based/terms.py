"""Manager terms for the walking task: native reads, the slew-limited action, the command bank, rewards and resets.

Every term reads the robot through one ``NativeRobot``, which keeps the raw PhysX reads and the arithmetic of
``omniwalk.env.IsaacEnv`` until native parity admits ``robot.data`` in their place.
"""

import json
from collections.abc import Callable
from pathlib import Path

import omni.physx
import torch
import warp as wp
from isaaclab.managers import ActionTerm, ActionTermCfg, CommandTerm, CommandTermCfg, RecorderTerm
from isaaclab.sim import CollisionPropertiesCfg, SpawnerCfg
from isaaclab.sim.schemas import define_collision_properties
from isaaclab.sim.utils import (
    bind_physics_material,
    get_current_stage,
    get_first_matching_child_prim,
)
from isaaclab.utils import configclass
from pxr import UsdPhysics

from locomotion.omniwalk.mdp.actions import emitted_target, executed_action_feature
from locomotion.omniwalk.mdp.commands import TaskConfig
from locomotion.omniwalk.mdp.frames import navigation
from locomotion.omniwalk.mdp.loads import SubstepLoads, control_telemetry
from locomotion.omniwalk.mdp.memory import CommandSchedule, RewardMemory
from locomotion.omniwalk.mdp.observations import proprio
from locomotion.omniwalk.mdp.rewards import REWARD_CONFIG
from locomotion.omniwalk.mdp.state import measured_state, toe_points
from locomotion.omniwalk.mdp.terminations import episode_timeout, physical_termination
from locomotion.robots.hexapod import BODY_NAMES, JOINT_NAMES, LEGS

ROBOT, CONTACT, ACTION, COMMAND = "robot", "floor_contact", "joint_targets", "velocity"
FLOOR = "/Ground"


def native_robot(env):
    """The environment's one ``NativeRobot``, built on first use after the simulation starts."""
    native = getattr(env, "_omniwalk_native", None)
    if native is None:
        native = env._omniwalk_native = NativeRobot(env)
    return native


class NativeRobot:
    """Raw PhysX reads of the robot and floor force, the canonical joint order and one control's loads.

    ``current`` holds the measured state after the latest control or reset. ``capture`` is None or a
    per-substep hook with the arguments of the Direct task's hook; it receives this object first, which
    lacks the ``OmniEnv`` surface that the evaluation captures read.
    """

    def __init__(self, env):
        model = env.cfg.robot_model
        if env._physics_handles_decimation:
            raise RuntimeError("The walking task requires eight separate PhysX steps")
        self.env, self.device, self.num_envs = env, env.device, env.num_envs
        self.output = None if model.output is None else Path(model.output)
        self.robot = env.scene[ROBOT]
        self.motor = self.robot.actuators["motor"]
        self.view = self.robot.root_view
        joints, bodies = (
            list(self.view.shared_metatype.dof_names),
            list(self.view.shared_metatype.link_names),
        )
        if self.view.count != env.num_envs or self.view.shared_metatype.fixed_base:
            raise ValueError("Wrong native floating articulation count")
        if set(joints) != set(JOINT_NAMES) or set(bodies) != set(BODY_NAMES):
            raise ValueError("Wrong native 19-body/18-joint names")
        roots = [path + "/Robot" for path in env.scene.env_prim_paths]
        if [str(x).rsplit("/", 1)[0] for x in self.view.prim_paths] != roots:
            raise ValueError("Native environment root order differs")

        def tensor(x):
            return torch.as_tensor(x, dtype=torch.float32, device=self.device)

        self.native_joint_names, self.native_body_names = joints, bodies
        self.joint_index = torch.tensor([joints.index(n) for n in JOINT_NAMES], device=self.device)
        self.inverse_joint_index = torch.argsort(self.joint_index)
        self.body_index = bodies.index("body")
        self.toe_index = torch.tensor(
            [bodies.index(leg + "_tibia") for leg in LEGS], device=self.device
        )
        self.other_index = [i for i, name in enumerate(bodies) if not name.endswith("_tibia")]
        named = {j["name"]: j for j in json.loads(Path(model.model_path).read_text())["joints"]}
        limits = tensor([[named[n]["lower"], named[n]["upper"]] for n in JOINT_NAMES])
        self.lower, self.upper = limits[:, 0], limits[:, 1]
        self.neutral = tensor(model.neutral)
        self.reset_height = model.reset_height_m
        self.origins = env.scene.env_origins.clone()
        self.toe_local = tensor(
            toe_points(json.loads(Path(model.geometry_path).read_text()), model.reference_metadata)
        )
        self.root_com_local = self.get("get_coms")[:, self.body_index, :3]
        self.contact = env.scene[CONTACT].contact_view
        paths = list(self.contact.sensor_paths)
        if len(paths) != 19 * env.num_envs or self.contact.filter_count != 1:
            raise ValueError("Native contact matrix layout differs")
        self.contact_index = torch.tensor(
            [paths.index(r + "/" + b) for r in roots for b in bodies], device=self.device
        )
        self.sensor_map = [(roots.index(p.rsplit("/", 1)[0]), p.rsplit("/", 1)[1]) for p in paths]
        action = env.cfg.actions.joint_targets
        self.memory = RewardMemory(
            env.num_envs,
            self.device,
            neutral=self.neutral,
            lower=self.lower,
            upper=self.upper,
            action_scale=action.scale,
            nominal_height=model.nominal_height_m,
        )
        self.capture = None
        self.substep = 0
        self.reward_step = self.telemetry_step = None
        self._subscribe_native_errors()
        self.current = self.read()

    def _subscribe_native_errors(self):
        self.native_errors = []

        def native_error(event):
            self.native_errors.append({"type": int(event.type), "payload": repr(event.payload)})
            self.save("native_errors.json", self.native_errors)

        self.save("native_errors.json", self.native_errors)
        self._error_subscription = (
            omni.physx.get_physx_interface()
            .get_error_event_stream()
            .create_subscription_to_pop(native_error)
        )

    def save(self, name, value):
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def get(self, getter):
        # PhysX exposes static properties such as limits on CPU even when
        # dynamic state lives on CUDA. Keep all environment tensors together.
        return wp.to_torch(getattr(self.view, getter)()).to(device=self.device).clone()

    def read(self):
        self.env.sim.physics_sim_view.update_articulations_kinematic()
        return measured_state(
            self.get("get_dof_positions")[:, self.joint_index],
            self.get("get_dof_velocities")[:, self.joint_index],
            self.get("get_root_transforms"),
            self.get("get_link_transforms"),
            self.get("get_root_velocities"),
            self.root_com_local,
            self.toe_index,
            self.toe_local,
        )

    def begin_control(self, action, target):
        if self.native_errors:
            raise RuntimeError("Native physics errors prevent stepping")
        self.action, self.target = action.clone(), target
        self.loads = SubstepLoads(target, self.toe_index, self.other_index)
        self.substep = 0
        self.before_counter = self.env.sim.get_physics_step_count()

    def collect_substep(self):
        """Accumulate one substep's torques and floor force after physics, call ``capture`` and close the control."""
        state = self.read()
        # PhysX keeps the submitted actuation force through the step, so this read equals the pre-step write.
        native_pre = (
            self.get("get_dof_actuation_forces")[:, self.joint_index]
            if self.capture is not None
            else None
        )
        names = self.motor.joint_names
        order = [names.index(name) for name in JOINT_NAMES]
        q, dq = self.motor.pre_position[:, order], self.motor.pre_velocity[:, order]
        requested, applied = (
            self.motor.computed_effort[:, order],
            self.motor.applied_effort[:, order],
        )
        ceiling = self.motor.ceiling[:, order]
        forces = wp.to_torch(self.contact.get_contact_force_matrix(self.env.physics_dt)).clone()
        if forces.shape != (19 * self.num_envs, 1, 3) or not bool(torch.isfinite(forces).all()):
            raise FloatingPointError("Native floor-force shape/finite check failed")
        forces = forces[self.contact_index, 0].reshape(self.num_envs, 19, 3)
        self.loads.add(requested, applied, forces)
        if self.capture is not None:
            self.capture(
                self,
                state,
                q,
                dq,
                requested,
                applied,
                ceiling,
                self.target,
                forces,
                self.substep - 1,
                native_pre,
            )
        if self.native_errors:
            raise RuntimeError("Native physics error during stepping")
        self.requested, self.applied, self.forces = requested, applied, forces
        if self.substep == self.env.cfg.decimation:
            if (
                self.env.sim.get_physics_step_count()
                != self.before_counter + self.env.cfg.decimation
            ):
                raise RuntimeError("Control did not advance exactly eight physics steps")
            self.current = state

    def ended(self):
        """Physical terminations and timeouts of the completed control, whether or not termination terms reset rows."""
        env = self.env
        return (
            physical_termination(self.current, self.lower, self.upper),
            episode_timeout(env.episode_length_buf, env.cfg.episode_length_s, env.step_dt),
        )

    def telemetry(self):
        """The completed control's telemetry under the command it held."""
        if self.telemetry_step != self.env.common_step_counter:
            terminated, truncated = self.ended()
            self._telemetry = control_telemetry(
                self.current,
                self.target,
                self.requested,
                self.applied,
                self.forces,
                self.loads,
                self.env.command_manager.get_command(COMMAND),
                self.action,
                terminated,
                truncated,
            )
            self.telemetry_step = self.env.common_step_counter
        return self._telemetry

    def write_reset(self, rows):
        rows = rows.to(torch.long)
        q = self.neutral[self.inverse_joint_index].expand(len(rows), -1).clone()
        zero = torch.zeros_like(q)
        pose = (
            torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=self.device)
            .expand(len(rows), -1)
            .clone()
        )
        pose[:, :3] = self.origins[rows]
        pose[:, 2] = self.reset_height
        ids = rows.to(torch.int32)
        self.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids)
        self.robot.write_root_com_velocity_to_sim_index(
            root_velocity=torch.zeros((len(rows), 6), device=self.device), env_ids=ids
        )
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=zero, env_ids=ids)
        self.robot.set_joint_position_target_index(target=q, env_ids=ids)
        self.robot.set_joint_velocity_target_index(target=zero, env_ids=ids)
        self.robot.set_joint_effort_target_index(target=zero, env_ids=ids)
        # Clear the selected native force rows without invoking all-row actuator compute.
        force = self.get("get_dof_actuation_forces")
        force[rows] = 0
        self.view.set_dof_actuation_forces(
            wp.from_torch(force.contiguous()),
            wp.array(rows.cpu().numpy(), dtype=wp.uint32, device=self.device),
        )
        self.current = self.read()


# Scene


def spawn_recorded_floor(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """The recorded 80 m two-triangle floor mesh, built by the stock terrain utility with default collider offsets."""
    import trimesh
    from isaaclab.terrains.utils import create_prim_from_mesh

    mesh = trimesh.Trimesh(
        vertices=[(-40.0, -40.0, 0.0), (40.0, -40.0, 0.0), (40.0, 40.0, 0.0), (-40.0, 40.0, 0.0)],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    create_prim_from_mesh(prim_path, mesh, physics_material=cfg.physics_material)
    stage = get_current_stage()
    UsdPhysics.MeshCollisionAPI.Apply(
        stage.GetPrimAtPath(prim_path + "/mesh")
    ).CreateApproximationAttr("none")
    return stage.GetPrimAtPath(prim_path)


@configclass
class RecordedFloorCfg(SpawnerCfg):
    func: Callable = spawn_recorded_floor
    physics_material: object = None
    contact_offset_m: float | None = None
    rest_offset_m: float | None = None


def bind_floor_material(env, env_ids):
    """Give every robot collider the floor's material and author floor offsets only when the floor declares them."""
    stage, floor = get_current_stage(), env.cfg.scene.floor.spawn
    collider = get_first_matching_child_prim(
        FLOOR, lambda prim: prim.HasAPI(UsdPhysics.CollisionAPI), stage=stage
    )
    material = get_first_matching_child_prim(
        FLOOR, lambda prim: prim.HasAPI(UsdPhysics.MaterialAPI), stage=stage
    )
    if collider is None or material is None:
        raise RuntimeError("Recorded floor has no collider or physics material")
    if floor.contact_offset_m is not None:
        define_collision_properties(
            str(collider.GetPath()),
            CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=floor.contact_offset_m,
                rest_offset=floor.rest_offset_m,
            ),
            stage=stage,
        )
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI) and "/Robot/" in str(prim.GetPath()):
            bind_physics_material(str(prim.GetPath()), str(material.GetPath()), stage=stage)


# Action


class SlewLimitedJointTargets(ActionTerm):
    """Hold joint targets that move at most ``slew`` radians per control toward the clamped policy request."""

    cfg: "SlewLimitedJointTargetsCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.native = native_robot(env)
        self._raw = torch.zeros(self.num_envs, 18, device=self.device)
        self.held = self.native.neutral.expand(self.num_envs, -1).clone()
        self.target = self.held.clone()

    @property
    def action_dim(self):
        return 18

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self.target

    def process_actions(self, actions):
        if actions.shape != (self.num_envs, 18) or not bool(torch.isfinite(actions).all()):
            raise ValueError("Policy action shape/finite check failed")
        self._raw[:] = actions
        native = self.native
        self.target = emitted_target(
            actions,
            self.held,
            native.neutral,
            native.lower,
            native.upper,
            self.cfg.scale,
            self.cfg.slew,
        )
        self.held = self.target
        native.begin_control(actions, self.target)

    def apply_actions(self):
        self._asset.set_joint_position_target_index(
            target=self.target[:, self.native.inverse_joint_index]
        )
        self.native.substep += 1

    def reset(self, env_ids=None):
        self.held[slice(None) if env_ids is None else env_ids] = self.native.neutral


@configclass
class SlewLimitedJointTargetsCfg(ActionTermCfg):
    class_type: type = SlewLimitedJointTargets
    asset_name: str = ROBOT
    scale: float = 0.35
    slew: float = 0.040


# Command


class BankCommand(CommandTerm):
    """Hold each bank command for 100 to 250 controls, with the recorded zero quota and turn draws.

    Reset rows draw after the rows whose hold expired, in the training task's order: at the end of the
    step's compute, or at the first read after a reset outside a step. With ``resample`` False the term
    keeps the commands that the caller writes into ``commands``, as evaluation does.
    """

    cfg: "BankCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.schedule = CommandSchedule(
            TaskConfig(seed=cfg.seed), cfg.turn_draw_fraction, self.device
        )
        self.commands = (
            torch.as_tensor(cfg.initial, dtype=torch.float32, device=self.device)
            .expand(self.num_envs, -1)
            .clone()
        )
        self.remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def command(self):
        self._draw_pending()
        return self.commands

    def reset(self, env_ids=None):
        rows = slice(None) if env_ids is None else env_ids
        self.command_counter[rows] = 0
        self.pending[rows] = self.cfg.resample
        return {}

    def compute(self, dt):
        if not self.cfg.resample:
            return
        self.remaining -= 1
        self._resample(((self.remaining <= 0) & ~self.pending).nonzero().flatten())
        self._draw_pending()

    def _draw_pending(self):
        rows = self.pending.nonzero().flatten()
        if len(rows):
            self.pending[rows] = False
            self._resample(rows)

    def _resample(self, env_ids):
        if len(env_ids):
            self.commands[env_ids], self.remaining[env_ids] = self.schedule.draw(len(env_ids))
            self.command_counter[env_ids] += 1

    def _resample_command(self, env_ids):
        raise NotImplementedError(
            "BankCommand draws commands and hold lengths together in _resample"
        )

    def _update_metrics(self):
        pass

    def _update_command(self):
        pass


@configclass
class BankCommandCfg(CommandTermCfg):
    class_type: type = BankCommand
    resampling_time_range: tuple = (2.0, 5.0)
    seed: int = 20260914
    turn_draw_fraction: float = REWARD_CONFIG.turn_draw_fraction
    resample: bool = True
    initial: tuple = (0.0, 0.0, 0.0)


# Observations


def proprio_frame(env):
    native = native_robot(env)
    return proprio(native.current, native.neutral)


def held_command(env):
    return env.command_manager.get_command(COMMAND)


def executed_action(env):
    term = env.action_manager.get_term(ACTION)
    return executed_action_feature(term.held, term.native.neutral, term.cfg.scale)


def linear_velocity_nav(env):
    return navigation(native_robot(env).current["linear"])


# Terminations


def physical_end(env):
    native = native_robot(env)
    return physical_termination(native.current, native.lower, native.upper)


def control_timeout(env, episode_seconds):
    return episode_timeout(env.episode_length_buf, episode_seconds, env.step_dt)


# Rewards


def walking_reward(env):
    """All components of the recorded walking reward for the control just completed, computed once per control."""
    native = native_robot(env)
    if native.reward_step != env.common_step_counter:
        command = env.command_manager.get_command(COMMAND)
        _, native.reward_components, _ = native.memory.reward(
            native.telemetry(), command, native.ended()[0], env.episode_length_buf
        )
        native.reward_step = env.common_step_counter
    return native.reward_components


def reward_component(env, name):
    """One component per second of control: the manager multiplies by the control period, so weight 1 restores it."""
    return walking_reward(env)[name] / env.step_dt


# Events


def reset_to_stance(env, env_ids):
    native_robot(env).write_reset(env_ids)


def restart_reward_memory(env, env_ids):
    native = native_robot(env)
    native.memory.restart(
        env_ids.to(torch.long),
        native.current,
        env.command_manager.get_command(COMMAND),
        native.neutral.expand(env.num_envs, -1),
    )


# Recorder


class SubstepLoads400Hz(RecorderTerm):
    """Fold each 400 Hz physics step into the control's loads; the episode data records nothing."""

    def record_post_physics_decimation_step(self):
        native_robot(self._env).collect_substep()
        return None, None
