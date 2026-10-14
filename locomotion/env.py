"""Native GPU canonical hexapod environment for a separately qualified AMP pilot.

No historical task, checkpoint, source snapshot or admission is imported. All
physics uses the selected detailed USD; reference points do not replace meshes.
"""
from pathlib import Path
import json
import math
import numpy as np
import torch
from .env_config import BODY_NAMES, JOINT_NAMES, KD, LEGS, MASS_KG, verify_assets
from .amp import extract_features


from .core.transforms import rotate, inverse_rotate, navigation
from .core.motor import motor_force, emitted_target, executed_action_feature


class LocomotionEnv:
    num_actions = 18
    observation_width = 231
    critic_width = 234
    amp_width = 61

    def __init__(self, cfg, asset, model_path, geometry_path, output, *, reference_metadata=None):
        import warp as wp
        from isaaclab.sim import SimulationContext, SimulationCfg
        from isaaclab_physx.physics import PhysxCfg
        from pxr import UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Gf
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
        self.joint_names = list(JOINT_NAMES)
        self.body_names = list(BODY_NAMES)
        self.neutral = self.torch(self.reference_metadata.get("nominal_joint_position_rad", [0.] * 18))
        self.reset_height = float(self.reference_metadata.get("reset_root_height_m", cfg.reset_height_m))
        if self.neutral.shape != (18,) or not torch.isfinite(self.neutral).all():
            raise ValueError("Invalid named neutral reference")
        visualizers = []
        if cfg.render:
            from isaaclab_visualizers.kit import KitVisualizerCfg
            visualizers = [KitVisualizerCfg()]
        simulation_cfg = SimulationCfg(dt=cfg.physics_dt, gravity=(0., 0., -9.81), device=cfg.device,
            render_interval=cfg.decimation, physics=PhysxCfg(enable_external_forces_every_iteration=True),
            visualizer_cfgs=visualizers, log_dir=str(self.output / "isaac_logs"))
        self.sim = SimulationContext(simulation_cfg)
        self.scene_prim_path = simulation_cfg.physics_prim_path
        stage = self.sim.stage
        import omni.physx
        self.native_errors = []
        def native_error(event):
            self.native_errors.append({"type": int(event.type), "payload": repr(event.payload)})
            self.save("native_errors.json", self.native_errors)
        self.save("native_errors.json", self.native_errors)
        self._error_subscription = omni.physx.get_physx_interface().get_error_event_stream().create_subscription_to_pop(native_error)
        # A grid centred on the origin keeps every replica on the 80 m floor mesh up to 1024 robots; a grid
        # that starts at the origin leaves the 80 m mesh beyond 441 robots at 2 m spacing.
        side = math.ceil(math.sqrt(cfg.num_envs))
        half = (side - 1) / 2
        origins = [((i % side - half) * cfg.spacing_m, (i // side - half) * cfg.spacing_m, 0.) for i in range(cfg.num_envs)]
        self.origins = self.torch(origins)
        self.roots = [f"/Robot_{i:03d}" for i in range(cfg.num_envs)]
        for root, origin in zip(self.roots, origins):
            prim = stage.DefinePrim(root, "Xform")
            prim.GetReferences().AddReference(str((Path(asset) / "robot.usda").resolve()))
            UsdGeom.Xformable(prim).AddTranslateOp(opSuffix="allocation").Set(Gf.Vec3d(*origin))
            body = stage.GetPrimAtPath(root + "/body")
            articulation = PhysxSchema.PhysxArticulationAPI.Apply(body)
            articulation.CreateSolverPositionIterationCountAttr(32)
            articulation.CreateSolverVelocityIterationCountAttr(0)
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
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and str(prim.GetPath()).startswith("/Robot_"):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        self.sim.reset()
        if self.native_errors:
            raise RuntimeError("Native physics error during initialization")
        self.view = self.sim.physics_sim_view.create_articulation_view("/Robot_*/body")
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
        sdf = self.sim.physics_sim_view.create_sdf_shape_view("/Robot_*/*/collisions/part_*", 1)
        if sdf.count != cfg.num_envs * 153 or not sdf.check():
            raise ValueError("Detailed collision shapes absent or changed")
        self.root_com_local = self.get("get_coms")[:, self.body_index, :3]
        self.kd = self.torch(KD)
        self.contact = self.sim.physics_sim_view.create_rigid_contact_view("/Robot_*/*", ["/Ground"], max_contact_data_count=1024 * cfg.num_envs)
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
        self.reset()

    def verify_native_recipe(self, phase):
        """Read actual scene/articulation/material values, not only authored intent."""
        from pxr import PhysxSchema
        solver = {}
        for root in self.roots:
            prim = self.sim.stage.GetPrimAtPath(root + "/body")
            api = PhysxSchema.PhysxArticulationAPI(prim)
            values = [api.GetSolverPositionIterationCountAttr().Get(), api.GetSolverVelocityIterationCountAttr().Get()]
            if values != [32, 0]:
                raise ValueError("Actual solver attributes differ: " + root)
            if prim.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is not True:
                raise ValueError("Canonical self-collision setting differs")
            solver[root] = values
        scene = self.sim.stage.GetPrimAtPath(self.scene_prim_path)
        scene_values = {name: scene.GetAttribute(name).Get() for name in
                        ("physxScene:timeStepsPerSecond", "physxScene:enableExternalForcesEveryIteration")}
        if scene_values["physxScene:timeStepsPerSecond"] != 400 or scene_values["physxScene:enableExternalForcesEveryIteration"] is not True:
            raise ValueError("Actual simulation timing/external-force configuration differs")
        material = self.get("get_material_properties")
        offsets = self.get("get_contact_offsets")
        rests = self.get("get_rest_offsets")
        if material.shape != (self.num_envs, 153, 3) or not torch.allclose(material, self.torch([1., 1., 0.]).expand_as(material), atol=1e-7, rtol=0):
            raise ValueError("Actual native materials differ")
        if offsets.shape != (self.num_envs, 153) or not torch.allclose(offsets, torch.full_like(offsets, .001), atol=1e-9, rtol=0) or torch.any(rests != 0):
            raise ValueError("Actual native contact/rest offsets differ")
        if self.native_errors:
            raise RuntimeError("Native physics errors prevent acceptance")
        self.native_readback.setdefault("recipe_readbacks", {})[phase] = {
            "solver_attributes": solver, "scene_attributes": scene_values,
            "material_properties": material.cpu().tolist(), "contact_offsets": offsets.cpu().tolist(),
            "rest_offsets": rests.cpu().tolist(), "backend_iteration_introspection": False,
            "native_error_events": list(self.native_errors)}
        self.save("native_readback.json", self.native_readback)

    def save(self, name, value):
        (self.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def get(self, getter):
        # PhysX exposes static properties such as limits on CPU even when
        # dynamic state lives on CUDA. Keep all environment tensors together.
        return self.wp.to_torch(getattr(self.view, getter)()).to(device=self.device).clone()

    def native_array(self, tensor):
        return self.wp.from_torch(tensor.contiguous())

    def _read(self):
        self.sim.physics_sim_view.update_articulations_kinematic()
        q = self.get("get_dof_positions")[:, self.joint_index]
        dq = self.get("get_dof_velocities")[:, self.joint_index]
        root = self.get("get_root_transforms")
        link = self.get("get_link_transforms")
        velocity = self.get("get_root_velocities")
        for key, value in (("q", q), ("dq", dq), ("root", root), ("link", link), ("velocity", velocity)):
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError("Nonfinite native " + key)
        quaternion = root[:, 3:]
        angular = inverse_rotate(quaternion, velocity[:, 3:])
        com_offset_world = rotate(quaternion, self.root_com_local)
        origin_velocity = velocity[:, :3] - torch.cross(velocity[:, 3:], com_offset_world, dim=-1)
        linear = inverse_rotate(quaternion, origin_velocity)
        gravity = inverse_rotate(quaternion, self.torch([0., 0., -1.]).expand(self.num_envs, -1))
        toe_pose = link[:, self.toe_index]
        toe_world = toe_pose[:, :, :3] + rotate(toe_pose[:, :, 3:], self.toe_local.expand(self.num_envs, -1, -1))
        toe_body = inverse_rotate(quaternion[:, None].expand(-1, 6, -1), toe_world - root[:, None, :3])
        return {"q": q, "dq": dq, "root": root, "link": link, "root_velocity": velocity,
                "linear": linear, "angular": angular, "gravity": gravity,
                "toe_world": toe_world, "toe_body": toe_body}

    def _proprio(self, state):
        return torch.cat((navigation(state["angular"]), navigation(state["gravity"]),
                          state["q"] - self.neutral, state["dq"]), dim=-1)

    def _observations(self, state):
        obs = torch.cat((self.history.flatten(1), self.commands, self.previous_action), dim=-1)
        critic = torch.cat((obs, navigation(state["linear"])), dim=-1)
        result = {"obs": obs, "critic": critic}
        if self.cfg.record_motion_features:
            result["amp"] = extract_features(state)
        return result

    def reset(self, indices=None):
        if getattr(self, "native_errors", []):
            raise RuntimeError("Cannot reset across a native physics error")
        indices = torch.arange(self.num_envs, device=self.device) if indices is None else torch.as_tensor(indices, dtype=torch.long, device=self.device)
        if indices.ndim != 1 or bool(torch.any(indices < 0)) or bool(torch.any(indices >= self.num_envs)) or len(indices.unique()) != len(indices):
            raise ValueError("Invalid selected reset rows")
        if len(indices):
            before_counter = self.sim.get_physics_step_count()
            ids = self.wp.array(indices.cpu().numpy(), dtype=self.wp.uint32, device=self.device)
            root = self.get("get_root_transforms")
            root[indices, :3] = self.origins[indices]
            root[indices, 2] = self.reset_height
            root[indices, 3:] = self.torch([0., 0., 0., 1.])
            velocity = self.get("get_root_velocities")
            velocity[indices] = 0
            q = self.get("get_dof_positions")
            dq = self.get("get_dof_velocities")
            q[indices] = self.neutral[self.inverse_joint_index]
            dq[indices] = 0
            force = self.get("get_dof_actuation_forces")
            force[indices] = 0
            self.view.set_dof_actuation_forces(self.native_array(force), ids)
            self.view.set_root_transforms(self.native_array(root), ids)
            self.view.set_root_velocities(self.native_array(velocity), ids)
            self.view.set_dof_positions(self.native_array(q), ids)
            self.view.set_dof_velocities(self.native_array(dq), ids)
            self.view.set_dof_position_targets(self.native_array(q), ids)
            self.view.set_dof_velocity_targets(self.native_array(dq), ids)
            self.held[indices] = self.neutral
            self.previous_action[indices] = 0
            self.episode_steps[indices] = 0
            state = self._read()
            if self.sim.get_physics_step_count() != before_counter:
                raise RuntimeError("Reset unexpectedly advanced physics")
            if not torch.allclose(state["q"][indices], self.neutral.expand(len(indices), -1), atol=2e-6, rtol=0):
                raise RuntimeError("Selected joint reset readback differs")
            if not torch.allclose(state["root"][indices], root[indices], atol=2e-6, rtol=0):
                raise RuntimeError("Selected root reset readback differs")
            self.history[indices] = self._proprio(state)[indices, None].expand(-1, 5, -1)
        else:
            state = self._read()
        self.current = state
        return self._observations(state)

    def step(self, action):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action.shape != (self.num_envs, 18) or not bool(torch.isfinite(action).all()):
            raise ValueError("Policy action shape/finite check failed")
        target = emitted_target(action, self.held, self.neutral, self.lower, self.upper, self.cfg.action_scale_rad, self.cfg.target_slew_rad)
        saturation = torch.zeros_like(target, dtype=torch.long)
        torque_square = torch.zeros_like(target)
        requested_max = torch.zeros_like(target)
        applied_max = torch.zeros_like(target)
        other_force_max = torch.zeros(self.num_envs, device=self.device)
        tibia_min_force = torch.full((self.num_envs, 6), torch.inf, device=self.device)
        before_counter = self.sim.get_physics_step_count()
        for substep in range(self.cfg.decimation):
            q = self.get("get_dof_positions")[:, self.joint_index]
            dq = self.get("get_dof_velocities")[:, self.joint_index]
            if not bool(torch.isfinite(q).all() and torch.isfinite(dq).all()):
                raise FloatingPointError("Nonfinite pre-servo state")
            requested, applied, ceiling = motor_force(q, dq, target, self.kd)
            self.view.set_dof_actuation_forces(self.native_array(applied[:, self.inverse_joint_index]), self._all_ids)
            native_pre = self.get("get_dof_actuation_forces")[:, self.joint_index] if self.capture is not None else None
            self.sim.step(render=False)
            state = self._read()
            forces = self.wp.to_torch(self.contact.get_contact_force_matrix(self.cfg.physics_dt)).clone()
            if forces.shape != (19 * self.num_envs, 1, 3) or not bool(torch.isfinite(forces).all()):
                raise FloatingPointError("Native floor-force shape/finite check failed")
            forces = forces[self.contact_index, 0].reshape(self.num_envs, 19, 3)
            norms = torch.linalg.vector_norm(forces, dim=-1)
            other = [i for i, name in enumerate(self.native_body_names) if not name.endswith("_tibia")]
            other_force_max = torch.maximum(other_force_max, norms[:, other].amax(-1))
            tibia_min_force = torch.minimum(tibia_min_force, norms[:, self.toe_index])
            saturation += requested.abs() > 1.6
            torque_square += applied.square()
            requested_max = torch.maximum(requested_max, requested.abs())
            applied_max = torch.maximum(applied_max, applied.abs())
            if self.capture is not None:
                self.capture(self, state, q, dq, requested, applied, ceiling, target, forces, substep, native_pre)
            if self.native_errors:
                raise RuntimeError("Native physics error during stepping")
        if self.sim.get_physics_step_count() != before_counter + 8:
            raise RuntimeError("Control did not advance exactly eight physics steps")
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
        result = self._observations(state)
        result.update(terminated=terminated, truncated=truncated)
        return result



# Exact contact/clearance mathematics carried into this fresh namespace from the
# canonical source005 recipe; no frozen source module is imported at runtime.
from .evaluation.capture import (_diagnostic_rotation, _DiagnosticGeometry, _classify_patches,
    DiagnosticCapture, _diagnostic_servo, _quiet_metrics, score_diagnostic)
