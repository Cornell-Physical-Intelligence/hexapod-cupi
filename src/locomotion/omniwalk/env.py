"""Run the walking robot on the stock Isaac Lab Direct lifecycle and own every native PhysX read and write."""

import json
import math
from pathlib import Path

import omni.physx
import torch
import warp as wp
from isaaclab.envs import DirectRLEnv
from isaaclab.sim import CollisionPropertiesCfg, CuboidCfg, GroundPlaneCfg
from isaaclab.sim.schemas import define_collision_properties
from isaaclab.sim.utils import bind_physics_material, get_first_matching_child_prim
from isaaclab_physx.assets.articulation import Articulation
from isaaclab_physx.sim.spawners.materials import RigidBodyMaterialCfg
from pxr import PhysxSchema, UsdPhysics

from locomotion.omniwalk.env_cfg import make_config
from locomotion.omniwalk.mdp.actions import emitted_target, executed_action_feature
from locomotion.omniwalk.mdp.loads import SubstepLoads, control_telemetry
from locomotion.omniwalk.mdp.observations import observations, proprio
from locomotion.omniwalk.mdp.state import measured_state, toe_points
from locomotion.omniwalk.mdp.terminations import episode_timeout, physical_termination
from locomotion.robots.hexapod import BODY_NAMES, JOINT_NAMES, LEGS, USD_SHA256, verify_assets


class CaptureArticulation(Articulation):
    """Read submitted force before physics; capture state after physics."""

    def write_data_to_sim(self):
        super().write_data_to_sim()
        if self.env.control_active:
            self.env.record_submitted_force()

    def update(self, dt):
        super().update(dt)
        if self.env.control_active:
            self.env.collect_substep()


class IsaacEnv(DirectRLEnv):
    """Use stock decimation and reset hooks; hold the scene, native views, their layout guards and per-control loads.

    ``cfg`` is the stock DirectRLEnvCfg and ``config`` the EnvConfig. This class owns ``commands``,
    ``history``, ``held`` and ``previous_action`` and updates them in place. ``capture`` is None or a
    per-substep hook that OmniEnv binds before each step. State and property reads use raw PhysX
    views until native parity admits ``robot.data`` in their place.
    """

    def __init__(
        self, config, asset, model_path, geometry_path, output, *, reference_metadata=None
    ):
        self.config = config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.model = verify_assets(asset, model_path)
        self.geometry_path = Path(geometry_path)
        self.geometry_meta = json.loads(self.geometry_path.read_text())
        if self.geometry_meta.get("asset_robot_sha256") != USD_SHA256:
            raise ValueError("Diagnostic geometry belongs to another asset")
        self.reference_metadata = reference_metadata or {}
        self.torch = lambda x: torch.as_tensor(x, dtype=torch.float32, device=config.device)
        self.joint_names, self.body_names = list(JOINT_NAMES), list(BODY_NAMES)
        self.neutral = self.torch(
            self.reference_metadata.get("nominal_joint_position_rad", [0.0] * 18)
        )
        self.reset_height = float(
            self.reference_metadata.get("reset_root_height_m", config.reset_height_m)
        )
        if (
            self.neutral.shape != (18,)
            or not torch.isfinite(self.neutral).all()
            or not math.isfinite(self.reset_height)
        ):
            raise ValueError("Invalid named neutral reference")
        direct_cfg, self.robot_cfg = make_config(
            config, asset, zip(JOINT_NAMES, self.neutral.cpu().tolist()), self.reset_height, output
        )
        self.control_active = False
        super().__init__(direct_cfg)
        if self._physics_handles_decimation:
            raise RuntimeError("Direct candidate requires eight separate PhysX steps")
        self.motor = self.robot.actuators["motor"]
        self.scene_prim_path = direct_cfg.sim.physics_prim_path
        self.origins = self.scene.env_origins.clone()
        self.roots = [path + "/Robot" for path in self.scene.env_prim_paths]
        self.view = self.robot.root_view
        self.native_joint_names = list(self.view.shared_metatype.dof_names)
        self.native_body_names = list(self.view.shared_metatype.link_names)
        if self.view.count != config.num_envs or self.view.shared_metatype.fixed_base:
            raise ValueError("Wrong native floating articulation count")
        if set(self.native_joint_names) != set(JOINT_NAMES) or set(self.native_body_names) != set(
            BODY_NAMES
        ):
            raise ValueError("Wrong native 19-body/18-joint names")
        actual_roots = [str(x).rsplit("/", 1)[0] for x in self.view.prim_paths]
        if actual_roots != self.roots:
            raise ValueError("Native environment root order differs")
        self.joint_index = torch.tensor(
            [self.native_joint_names.index(n) for n in JOINT_NAMES], device=self.device
        )
        self.inverse_joint_index = torch.argsort(self.joint_index)
        self.body_index = self.native_body_names.index("body")
        self.toe_index = torch.tensor(
            [self.native_body_names.index(leg + "_tibia") for leg in LEGS], device=self.device
        )
        named = {j["name"]: j for j in self.model["joints"]}
        expected_limits = self.torch([[named[n]["lower"], named[n]["upper"]] for n in JOINT_NAMES])
        self.lower, self.upper = expected_limits[:, 0], expected_limits[:, 1]
        self.root_com_local = self.get("get_coms")[:, self.body_index, :3]
        self.contact = self.sim.physics_sim_view.create_rigid_contact_view(
            "/World/envs/env_*/Robot/*",
            [self.ground_collision_path],
            max_contact_data_count=1024 * config.num_envs,
        )
        paths = list(self.contact.sensor_paths)
        if len(paths) != 19 * config.num_envs or self.contact.filter_count != 1:
            raise ValueError("Native contact matrix layout differs")
        self.contact_index = torch.tensor(
            [paths.index(r + "/" + b) for r in self.roots for b in self.native_body_names],
            device=self.device,
        )
        self.sensor_map = [
            (self.roots.index(p.rsplit("/", 1)[0]), p.rsplit("/", 1)[1]) for p in paths
        ]
        self.toe_local = self.torch(toe_points(self.geometry_meta, self.reference_metadata))
        self.history = torch.zeros((config.num_envs, 5, 42), device=self.device)
        self.previous_action = torch.zeros((config.num_envs, 18), device=self.device)
        self.held = self.neutral.expand(config.num_envs, -1).clone()
        self.commands = (
            self.torch(
                [config.command_forward_mps, config.command_left_mps, config.command_yaw_rad_s]
            )
            .expand(config.num_envs, -1)
            .clone()
        )
        self.total_controls = 0
        self.telemetry = {}
        self.capture = None

    def _setup_scene(self):
        self.robot = CaptureArticulation(self.robot_cfg)
        self.robot.env = self
        self.scene.articulations["robot"] = self.robot
        self.scene.clone_environments(copy_from_source=True)
        stage = self.sim.stage
        material = RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        )
        if self.config.floor == "isaaclab_ground_plane":
            floor = GroundPlaneCfg(size=(80.0, 80.0), physics_material=material)
            floor.func("/Ground", floor)
        elif self.config.floor == "isaaclab_cuboid":
            floor = CuboidCfg(
                size=(80.0, 80.0, 0.1),
                collision_props=CollisionPropertiesCfg(collision_enabled=True),
                physics_material=material,
            )
            floor.func("/Ground", floor, translation=(0.0, 0.0, -0.05))
        else:
            import trimesh
            from isaaclab.terrains.utils import create_prim_from_mesh

            mesh = trimesh.Trimesh(
                vertices=[
                    (-40.0, -40.0, 0.0),
                    (40.0, -40.0, 0.0),
                    (40.0, 40.0, 0.0),
                    (-40.0, 40.0, 0.0),
                ],
                faces=[[0, 1, 2], [0, 2, 3]],
                process=False,
            )
            create_prim_from_mesh("/Ground", mesh, physics_material=material)
            UsdPhysics.MeshCollisionAPI.Apply(
                stage.GetPrimAtPath("/Ground/mesh")
            ).CreateApproximationAttr("none")
        root = stage.GetPrimAtPath("/Ground")
        collider = (
            root
            if root.HasAPI(UsdPhysics.CollisionAPI)
            else get_first_matching_child_prim(
                "/Ground", lambda prim: prim.HasAPI(UsdPhysics.CollisionAPI), stage=stage
            )
        )
        physics_material = get_first_matching_child_prim(
            "/Ground", lambda prim: prim.HasAPI(UsdPhysics.MaterialAPI), stage=stage
        )
        if collider is None or physics_material is None:
            raise RuntimeError("Stock floor has no collider or physics material")
        if self.config.authors_floor_offsets:
            define_collision_properties(
                str(collider.GetPath()),
                CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=self.config.floor_contact_offset_m,
                    rest_offset=self.config.floor_rest_offset_m,
                ),
                stage=stage,
            )
        self.ground_collision_path = str(collider.GetPath())
        self.scene.filter_collisions(global_prim_paths=["/Ground"])
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI) and "/Robot/" in str(prim.GetPath()):
                bind_physics_material(
                    str(prim.GetPath()), str(physics_material.GetPath()), stage=stage
                )
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and "/Robot/" in str(prim.GetPath()):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        self._subscribe_native_errors()

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

    def _pre_physics_step(self, actions):
        if self.native_errors:
            raise RuntimeError("Native physics errors prevent stepping")
        self.control_active = True
        self.action = actions.clone()
        self.target = emitted_target(
            actions,
            self.held,
            self.neutral,
            self.lower,
            self.upper,
            self.config.action_scale_rad,
            self.config.target_slew_rad,
        )
        other = [i for i, name in enumerate(self.native_body_names) if not name.endswith("_tibia")]
        self.loads = SubstepLoads(self.target, self.toe_index, other)
        self.substep = 0
        self.before_counter = self.sim.get_physics_step_count()

    def _apply_action(self):
        self.robot.set_joint_position_target_index(target=self.target[:, self.inverse_joint_index])
        self.substep += 1

    def record_submitted_force(self):
        """Read the force that the stock write submitted, before PhysX steps."""
        self.native_pre = (
            self.get("get_dof_actuation_forces")[:, self.joint_index]
            if self.capture is not None
            else None
        )

    def collect_substep(self):
        """Accumulate saturation, torque and floor force after a physics step, then call ``capture``."""
        state = self.read_state()
        native_pre = self.native_pre
        names = self.motor.joint_names
        order = [names.index(name) for name in JOINT_NAMES]
        q, dq = self.motor.pre_position[:, order], self.motor.pre_velocity[:, order]
        requested, applied = (
            self.motor.computed_effort[:, order],
            self.motor.applied_effort[:, order],
        )
        ceiling = self.motor.ceiling[:, order]
        forces = wp.to_torch(self.contact.get_contact_force_matrix(self.config.physics_dt)).clone()
        if forces.shape != (19 * self.num_envs, 1, 3) or not bool(torch.isfinite(forces).all()):
            raise FloatingPointError("Native floor-force shape/finite check failed")
        forces = forces[self.contact_index, 0].reshape(self.num_envs, 19, 3)
        self.loads.add(requested, applied, forces)
        if self.capture is not None:
            self.capture(
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

    def _get_dones(self):
        if self.sim.get_physics_step_count() != self.before_counter + 8 or self.substep != 8:
            raise RuntimeError("Control did not advance exactly eight physics steps")
        self.control_active = False
        target = self.target
        state = self.read_state()
        self.held.copy_(target)
        self.previous_action.copy_(
            executed_action_feature(target, self.neutral, self.config.action_scale_rad)
        )
        self.history.copy_(
            torch.cat((self.history[:, 1:], proprio(state, self.neutral)[:, None]), dim=1)
        )
        self.total_controls += 1
        terminated = physical_termination(state, self.lower, self.upper)
        truncated = episode_timeout(
            self.episode_length_buf, self.config.episode_seconds, self.config.control_dt
        )
        self.current = state
        self.telemetry = control_telemetry(
            state,
            target,
            self.requested,
            self.applied,
            self.forces,
            self.loads,
            self.commands,
            self.action,
            terminated,
            truncated,
        )
        # The task owns episodes, so the stock loop never resets a row by itself.
        zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return zeros, zeros.clone()

    def _get_rewards(self):
        # TrainingTask computes the walking reward after the native step returns.
        return torch.zeros(self.num_envs, device=self.device)

    def _get_observations(self):
        result = observations(self.history, self.commands, self.previous_action, self.current)
        return {"policy": result.pop("obs"), **result}

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.write_reset(env_ids)

    def reset_selected(self, env_ids):
        """Reset the given rows through the stock hook, then read state without a physics step."""
        if len(env_ids):
            self._reset_idx(env_ids)
        self.current = self.read_state()
        return self.current

    def write_reset(self, rows):
        rows = rows.to(torch.long)
        q = self.neutral[self.inverse_joint_index].expand(len(rows), -1).clone()
        zero = torch.zeros_like(q)
        pose = self.torch([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).expand(len(rows), -1).clone()
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
            self.native_array(force),
            wp.array(rows.cpu().numpy(), dtype=wp.uint32, device=self.device),
        )
        self.held[rows] = self.neutral
        self.previous_action[rows] = 0

    def save(self, name, value):
        (self.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def get(self, getter):
        # PhysX exposes static properties such as limits on CPU even when
        # dynamic state lives on CUDA. Keep all environment tensors together.
        return wp.to_torch(getattr(self.view, getter)()).to(device=self.device).clone()

    def native_array(self, tensor):
        return wp.from_torch(tensor.contiguous())

    def read_state(self):
        self.sim.physics_sim_view.update_articulations_kinematic()
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
