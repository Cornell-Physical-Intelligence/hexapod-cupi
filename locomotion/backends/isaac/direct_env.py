"""Opt-in stock DirectRLEnv candidate; native qualification remains pending.

Isaac Lab owns physics stepping and scene updates. The existing RewardTask owns
commands, rewards and selected resets. Its proximity and terminal-bootstrap
checks must run before a reset, so the stock loop sees zero done flags. The
adapter exposes the unchanged canonical done flags to that external task.
"""
from pathlib import Path
import json
import math
import numpy as np
import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation
from .config import BODY_NAMES, JOINT_NAMES, KD, LEGS, MASS_KG, verify_assets
from .direct_config import make_config, ISAACLAB_REVISION
from .direct_control import reset_rows
from .env import LocomotionEnv
from ...core.motor import emitted_target, executed_action_feature
from ...core.transforms import navigation


class _DirectPhysics(DirectRLEnv):
    """Use stock decimation and reset hooks behind the task API adapter."""

    def __init__(self, cfg, robot_cfg, owner):
        self.owner, self.robot_cfg = owner, robot_cfg
        super().__init__(cfg)
        if self._physics_handles_decimation:
            raise RuntimeError("Direct candidate requires eight separate PhysX steps")

    def _setup_scene(self):
        from pxr import UsdGeom, UsdPhysics, UsdShade, PhysxSchema
        self.robot = Articulation(self.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        self.scene.clone_environments(copy_from_source=True)
        stage = self.sim.stage
        floor = UsdGeom.Mesh.Define(stage, "/Ground")
        floor.CreatePointsAttr([(-40., -40., 0.), (40., -40., 0.), (40., 40., 0.), (-40., 40., 0.)])
        floor.CreateFaceVertexCountsAttr([3, 3])
        floor.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(floor.GetPrim()).CreateApproximationAttr("none")
        material = UsdShade.Material.Define(stage, "/PilotMaterial")
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr(1.)
        physics_material.CreateDynamicFrictionAttr(1.)
        physics_material.CreateRestitutionAttr(0.)
        physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        physx_material.CreateFrictionCombineModeAttr("multiply")
        physx_material.CreateRestitutionCombineModeAttr("multiply")
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and "/Robot/" in str(prim.GetPath()):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        self.owner.sim = self.sim
        self.owner._subscribe_native_errors()

    def _pre_physics_step(self, actions):
        self.owner._begin_control(actions)

    def _apply_action(self):
        if self.owner.substep:
            self.owner._collect_substep()
        self.robot.set_joint_position_target_index(target=self.owner.target[:, self.owner.inverse_joint_index])
        self.owner.substep += 1

    def _get_dones(self):
        self.owner._collect_substep()
        self.owner._finish_control()
        zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return zeros, zeros.clone()

    def _get_rewards(self):
        # RewardTask computes the selected frozen reward after the native return.
        return torch.zeros(self.num_envs, device=self.device)

    def _get_observations(self):
        result = self.owner._observations(self.owner.current)
        return {"policy": result.pop("obs"), **result}

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.owner._write_reset(env_ids)


class DirectLocomotionEnv(LocomotionEnv):
    """Keep the current task/capture API around one stock DirectRLEnv."""

    def __init__(self, cfg, asset, model_path, geometry_path, output, *, reference_metadata=None):
        import warp as wp
        self.cfg, self.device, self.num_envs = cfg, cfg.device, cfg.num_envs
        self.wp = wp
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.model = verify_assets(asset, model_path)
        self.geometry_path = Path(geometry_path)
        self.geometry_meta = json.loads(self.geometry_path.read_text())
        if self.geometry_meta.get("asset_robot_sha256") != "3be98420adc7923923757d6557d8492d5930f84128c1b1366b9c6db1bb03207c":
            raise ValueError("Diagnostic geometry belongs to another asset")
        self.reference_metadata = reference_metadata or {}
        self.torch = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        self.joint_names, self.body_names = list(JOINT_NAMES), list(BODY_NAMES)
        self.neutral = self.torch(self.reference_metadata.get("nominal_joint_position_rad", [0.] * 18))
        self.reset_height = float(self.reference_metadata.get("reset_root_height_m", cfg.reset_height_m))
        if self.neutral.shape != (18,) or not torch.isfinite(self.neutral).all() or not math.isfinite(self.reset_height):
            raise ValueError("Invalid named neutral reference")
        direct_cfg, robot_cfg = make_config(cfg, asset, zip(JOINT_NAMES, self.neutral.cpu().tolist()), self.reset_height, output)
        self.direct = _DirectPhysics(direct_cfg, robot_cfg, self)
        self.sim = self.direct.sim
        self.robot = self.direct.robot
        self.motor = self.robot.actuators["motor"]
        self.scene_prim_path = direct_cfg.sim.physics_prim_path
        self.origins = self.direct.scene.env_origins.clone()
        self.roots = [path + "/Robot" for path in self.direct.scene.env_prim_paths]
        self.view = self.robot.root_view
        self.native_joint_names = list(self.view.shared_metatype.dof_names)
        self.native_body_names = list(self.view.shared_metatype.link_names)
        if self.view.count != cfg.num_envs or self.view.shared_metatype.fixed_base:
            raise ValueError("Wrong native floating articulation count")
        if set(self.native_joint_names) != set(JOINT_NAMES) or set(self.native_body_names) != set(BODY_NAMES):
            raise ValueError("Wrong native 19-body/18-joint names")
        actual_roots = [str(x).rsplit("/", 1)[0] for x in self.view.prim_paths]
        if actual_roots != self.roots:
            raise ValueError("Native environment root order differs")
        self.joint_index = torch.tensor([self.native_joint_names.index(n) for n in JOINT_NAMES], device=self.device)
        self.inverse_joint_index = torch.argsort(self.joint_index)
        self.body_index = self.native_body_names.index("body")
        self.toe_index = torch.tensor([self.native_body_names.index(l + "_tibia") for l in LEGS], device=self.device)
        native_limits = self.get("get_dof_limits")[:, self.joint_index]
        named = {j["name"]: j for j in self.model["joints"]}
        expected_limits = self.torch([[named[n]["lower"], named[n]["upper"]] for n in JOINT_NAMES])
        if not torch.allclose(native_limits, expected_limits.expand(cfg.num_envs, -1, -1), atol=2e-6, rtol=0):
            raise ValueError("Actual native limits differ from the canonical model")
        self.lower, self.upper = expected_limits[:, 0], expected_limits[:, 1]
        if torch.any(self.neutral < self.lower) or torch.any(self.neutral > self.upper):
            raise ValueError("Reference neutral violates named canonical limits")
        masses = self.get("get_masses")
        named_links = {b["name"]: b for b in self.model["links"]}
        expected_masses = self.torch([named_links[n]["mass"] for n in self.native_body_names])
        if not torch.allclose(masses, expected_masses.expand(cfg.num_envs, -1), atol=1e-6, rtol=0):
            raise ValueError("Native body mass differs from canonical mass ledger")
        for getter in ("get_dof_stiffnesses", "get_dof_dampings", "get_dof_armatures"):
            if torch.any(self.get(getter) != 0):
                raise ValueError("Unexpected implicit drive or armature: " + getter)
        sdf = self.sim.physics_sim_view.create_sdf_shape_view("/World/envs/env_*/Robot/*/collisions/part_*", 1)
        if sdf.count != cfg.num_envs * 153 or not sdf.check():
            raise ValueError("Detailed collision shapes absent or changed")
        self.root_com_local = self.get("get_coms")[:, self.body_index, :3]
        self.kd = self.torch(KD)
        self.contact = self.sim.physics_sim_view.create_rigid_contact_view("/World/envs/env_*/Robot/*", ["/Ground"], max_contact_data_count=1024 * cfg.num_envs)
        paths = list(self.contact.sensor_paths)
        if len(paths) != 19 * cfg.num_envs or self.contact.filter_count != 1:
            raise ValueError("Native contact matrix layout differs")
        self.contact_index = torch.tensor([paths.index(r + "/" + b) for r in self.roots for b in self.native_body_names], device=self.device)
        self.sensor_map = [(self.roots.index(p.rsplit("/", 1)[0]), p.rsplit("/", 1)[1]) for p in paths]
        default_toes = []
        for leg in LEGS:
            shape = next(x for x in self.geometry_meta["shapes"] if x["body"] == leg + "_tibia")
            transform = np.asarray(shape["shape_to_link"])
            bounds = np.asarray(shape["cap_bounds_m"])
            point = np.array([bounds[1, 0], 0., 0., 1.])
            default_toes.append((transform @ point)[:3])
        self.toe_local = self.torch(np.asarray(self.reference_metadata.get("toe_local_points_m", default_toes)))
        if self.toe_local.shape != (6, 3) or not torch.isfinite(self.toe_local).all():
            raise ValueError("Invalid reference toe points")
        self.history = torch.zeros((cfg.num_envs, 5, 42), device=self.device)
        self.previous_action = torch.zeros((cfg.num_envs, 18), device=self.device)
        self.held = self.neutral.expand(cfg.num_envs, -1).clone()
        self.commands = self.torch([cfg.command_forward_mps, cfg.command_left_mps, cfg.command_yaw_rad_s]).expand(cfg.num_envs, -1).clone()
        self.episode_steps = torch.zeros(cfg.num_envs, dtype=torch.long, device=self.device)
        self.total_controls = 0
        self.telemetry = {}
        self._all_ids = wp.array(np.arange(cfg.num_envs), dtype=wp.uint32, device=self.device)
        self._rgb = None
        self._camera = None
        self.capture = None
        self.native_readback = {"native_joint_names": self.native_joint_names, "native_body_names": self.native_body_names,
            "canonical_joint_names": self.joint_names, "root_paths": self.roots, "mass_per_replica_kg": masses.sum(-1).cpu().tolist(),
            "masses": masses.cpu().tolist(), "limits": native_limits.cpu().tolist(), "native_max_velocity": self.get("get_dof_max_velocities")[:, self.joint_index].cpu().tolist(), "sdf_shapes": sdf.count,
            "solver_iterations_requested": [32, 0], "implicit_drive_and_armature_zero": True,
            "asset_mass_target_kg": MASS_KG, "config": cfg.declaration()}
        self.save("native_readback.json", self.native_readback)
        self.verify_native_recipe("after_sdk_reset")
        self.native_readback["backend"] = "direct_candidate"
        self.native_readback["isaaclab_revision_expected"] = ISAACLAB_REVISION
        self.native_readback["episode_owner"] = "external RewardTask; stock done flags are zero"
        self.native_readback["force_readback_phase"] = "after physics, before next stock scene write"
        self.save("native_readback.json", self.native_readback)
        self.reset()

    def _subscribe_native_errors(self):
        import omni.physx
        self.native_errors = []
        def native_error(event):
            self.native_errors.append({"type": int(event.type), "payload": repr(event.payload)})
            self.save("native_errors.json", self.native_errors)
        self.save("native_errors.json", self.native_errors)
        self._error_subscription = omni.physx.get_physx_interface().get_error_event_stream().create_subscription_to_pop(native_error)

    def reset(self, indices=None):
        rows = reset_rows(indices, self.num_envs, self.device)
        if self.native_errors:
            raise RuntimeError("Cannot reset across a native physics error")
        before = self.sim.get_physics_step_count()
        if len(rows):
            self.direct._reset_idx(rows.to(torch.int32))
        self.current = self._read()
        if self.sim.get_physics_step_count() != before:
            raise RuntimeError("Reset unexpectedly advanced physics")
        if len(rows):
            expected = self.neutral.expand(len(rows), -1)
            if not torch.allclose(self.current["q"][rows], expected, atol=2e-6, rtol=0):
                raise RuntimeError("Selected joint reset readback differs")
            expected_root = self.torch([0., 0., 0., 0., 0., 0., 1.]).expand(len(rows), -1).clone()
            expected_root[:, :3] = self.origins[rows]
            expected_root[:, 2] = self.reset_height
            if not torch.allclose(self.current["root"][rows], expected_root, atol=2e-6, rtol=0):
                raise RuntimeError("Selected root reset readback differs")
            self.history[rows] = self._proprio(self.current)[rows, None].expand(-1, 5, -1)
        return self._observations(self.current)

    def _write_reset(self, rows):
        rows = rows.to(torch.long)
        q = self.neutral[self.inverse_joint_index].expand(len(rows), -1).clone()
        zero = torch.zeros_like(q)
        pose = self.torch([0., 0., 0., 0., 0., 0., 1.]).expand(len(rows), -1).clone()
        pose[:, :3] = self.origins[rows]
        pose[:, 2] = self.reset_height
        ids = rows.to(torch.int32)
        self.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=torch.zeros((len(rows), 6), device=self.device), env_ids=ids)
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=zero, env_ids=ids)
        self.robot.set_joint_position_target_index(target=q, env_ids=ids)
        self.robot.set_joint_velocity_target_index(target=zero, env_ids=ids)
        self.robot.set_joint_effort_target_index(target=zero, env_ids=ids)
        # Clear the selected native force rows without invoking all-row actuator compute.
        force = self.get("get_dof_actuation_forces")
        force[rows] = 0
        self.view.set_dof_actuation_forces(self.native_array(force), self.wp.array(rows.cpu().numpy(), dtype=self.wp.uint32, device=self.device))
        self.held[rows] = self.neutral
        self.previous_action[rows] = 0
        self.episode_steps[rows] = 0

    def _begin_control(self, action):
        if self.native_errors:
            raise RuntimeError("Native physics errors prevent stepping")
        self.action = action.clone()
        self.target = emitted_target(action, self.held, self.neutral, self.lower, self.upper,
                                     self.cfg.action_scale_rad, self.cfg.target_slew_rad)
        self.saturation = torch.zeros_like(self.target, dtype=torch.long)
        self.torque_square = torch.zeros_like(self.target)
        self.requested_max = torch.zeros_like(self.target)
        self.applied_max = torch.zeros_like(self.target)
        self.other_force_max = torch.zeros(self.num_envs, device=self.device)
        self.tibia_min_force = torch.full((self.num_envs, 6), torch.inf, device=self.device)
        self.substep = 0
        self.before_counter = self.sim.get_physics_step_count()

    def _collect_substep(self):
        state = self._read()
        # The stock writer has submitted this substep's explicit effort. Read it
        # before the next write; PhysX retains the submitted force after stepping.
        native_pre = self.get("get_dof_actuation_forces")[:, self.joint_index] if self.capture is not None else None
        names = self.motor.joint_names
        order = [names.index(name) for name in JOINT_NAMES]
        q, dq = self.motor.pre_position[:, order], self.motor.pre_velocity[:, order]
        requested, applied = self.motor.computed_effort[:, order], self.motor.applied_effort[:, order]
        ceiling = self.motor.ceiling[:, order]
        forces = self.wp.to_torch(self.contact.get_contact_force_matrix(self.cfg.physics_dt)).clone()
        if forces.shape != (19 * self.num_envs, 1, 3) or not bool(torch.isfinite(forces).all()):
            raise FloatingPointError("Native floor-force shape/finite check failed")
        forces = forces[self.contact_index, 0].reshape(self.num_envs, 19, 3)
        norms = torch.linalg.vector_norm(forces, dim=-1)
        other = [i for i, name in enumerate(self.native_body_names) if not name.endswith("_tibia")]
        self.other_force_max = torch.maximum(self.other_force_max, norms[:, other].amax(-1))
        self.tibia_min_force = torch.minimum(self.tibia_min_force, norms[:, self.toe_index])
        self.saturation += requested.abs() > 1.6
        self.torque_square += applied.square()
        self.requested_max = torch.maximum(self.requested_max, requested.abs())
        self.applied_max = torch.maximum(self.applied_max, applied.abs())
        if self.capture is not None:
            self.capture(self, state, q, dq, requested, applied, ceiling, self.target, forces, self.substep - 1, native_pre)
        if self.native_errors:
            raise RuntimeError("Native physics error during stepping")
        self.requested, self.applied, self.forces = requested, applied, forces

    def _finish_control(self):
        if self.sim.get_physics_step_count() != self.before_counter + 8 or self.substep != 8:
            raise RuntimeError("Control did not advance exactly eight physics steps")
        target = self.target
        requested, applied = self.requested, self.applied
        forces = self.forces
        saturation, torque_square = self.saturation, self.torque_square
        requested_max, applied_max = self.requested_max, self.applied_max
        other_force_max, tibia_min_force = self.other_force_max, self.tibia_min_force
        action = self.action
        state = self._read()
        self.held = target
        self.previous_action = executed_action_feature(target, self.neutral, self.cfg.action_scale_rad).clone()
        self.history = torch.cat((self.history[:, 1:], self._proprio(state)[:, None]), dim=1)
        self.episode_steps += 1
        self.total_controls += 1
        velocity_nav = navigation(state["linear"])
        angular_nav = navigation(state["angular"])
        joint_violation = ((state["q"] < self.lower - 2e-6) | (state["q"] > self.upper + 2e-6)).any(-1)
        terminated = (state["root"][:, 2] < .045) | (state["gravity"][:, 2] > -math.cos(.85)) | joint_violation
        truncated = self.episode_steps >= round(self.cfg.episode_seconds / self.cfg.control_dt)
        self.current = state
        self.telemetry = {"root_pose_xyzw": state["root"].clone(), "linear_velocity_body": state["linear"].clone(),
            "linear_velocity_nav": velocity_nav.clone(), "angular_velocity_body": state["angular"].clone(),
            "joint_position_rad": state["q"].clone(), "joint_velocity_rad_s": state["dq"].clone(),
            "joint_target_rad": target.clone(), "computed_torque_nm": requested.clone(), "applied_torque_nm": applied.clone(),
            "saturation_count_400hz": saturation, "torque_square_sum_400hz": torque_square,
            "requested_torque_abs_max_400hz": requested_max, "applied_torque_abs_max_400hz": applied_max,
            "tibia_floor_force_world_n": forces[:, self.toe_index].clone(), "tibia_floor_force_min_norm_400hz": tibia_min_force,
            "other_body_force_max_400hz": other_force_max, "toe_xyz_world": state["toe_world"].clone(),
            "toe_xyz_body": state["toe_body"].clone(), "command": self.commands.clone(), "action": action.clone(),
            "terminated": terminated.clone(), "truncated": truncated.clone()}

    def step(self, action):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action.shape != (self.num_envs, 18) or not bool(torch.isfinite(action).all()):
            raise ValueError("Policy action shape/finite check failed")
        observations, _, _, _, _ = self.direct.step(action)
        return {"obs": observations.pop("policy"), **observations,
                "terminated": self.telemetry["terminated"], "truncated": self.telemetry["truncated"]}
