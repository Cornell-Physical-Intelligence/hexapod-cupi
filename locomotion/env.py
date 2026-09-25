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


def rotate(quaternion, vector):
    q = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    t = 2 * torch.cross(q[..., :3], vector, dim=-1)
    return vector + q[..., 3:] * t + torch.cross(q[..., :3], t, dim=-1)


def inverse_rotate(quaternion, vector):
    q = quaternion.clone()
    q[..., :3] *= -1
    return rotate(q, vector)


def navigation(vector):
    return torch.stack((-vector[..., 1], vector[..., 0], vector[..., 2]), dim=-1)


def motor_force(q, dq, target, kd):
    """Canonical float32 PD plus the provisional 48V speed-dependent curve."""
    requested = 12.0 * (target - q) - kd * dq
    rpm = dq.to(torch.float64).abs() * 60.0 / (2.0 * math.pi)
    xp = torch.tensor([0., 70., 275., 340., 450., 477., 480.], dtype=torch.float64, device=q.device)
    yp = torch.tensor([5.5, 5.5, 4., 3., 1.6, .5, 0.], dtype=torch.float64, device=q.device)
    index = torch.searchsorted(xp, rpm.contiguous(), right=True).clamp(1, 6)
    ceiling = yp[index - 1] + (rpm - xp[index - 1]) * (yp[index] - yp[index - 1]) / (xp[index] - xp[index - 1])
    ceiling = torch.where(rpm >= 480., 0., ceiling).clamp(min=0., max=1.6).to(torch.float32)
    return requested, torch.maximum(torch.minimum(requested, ceiling), -ceiling), ceiling


def emitted_target(action, held, neutral, lower, upper, scale=.35, slew=.04):
    requested = (neutral + scale * action.clamp(-1, 1)).clamp(lower, upper)
    lo = torch.maximum(lower.to(torch.float64), held.to(torch.float64) - slew)
    hi = torch.minimum(upper.to(torch.float64), held.to(torch.float64) + slew)
    target = torch.maximum(torch.minimum(requested.to(torch.float64), hi), lo).to(torch.float32)
    target = torch.where(target.to(torch.float64) > hi, torch.nextafter(target, torch.full_like(target, -torch.inf)), target)
    target = torch.where(target.to(torch.float64) < lo, torch.nextafter(target, torch.full_like(target, torch.inf)), target)
    return target


def executed_action_feature(target, neutral, scale):
    """Expose the actual held-target state after joint clamp and slew limiting."""
    return (target - neutral) / scale


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
        side = math.ceil(math.sqrt(cfg.num_envs))
        origins = [(i % side * cfg.spacing_m, i // side * cfg.spacing_m, 0.) for i in range(cfg.num_envs)]
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
def _diagnostic_rotation(q):
    x, y, z, w = np.asarray(q, dtype=float)
    if not np.isfinite(q).all() or abs(x*x+y*y+z*z+w*w-1.) > 2e-5:
        raise ValueError("Quaternion is not normalized XYZW")
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

class _DiagnosticGeometry:
 def __init__(self,meta,clouds,body_names):
  self.meta=meta;self.clouds={k:np.asarray(v)for k,v in clouds.items()};self.names=body_names
  if set(body_names)!=set(meta['body_names']):raise ValueError('Geometry body mapping differs')
  self.shapes={x['body']:x for x in meta['shapes']}
 def cap(self,body,point_world,pose):
  s=self.shapes[body];T=np.asarray(s['shape_to_link']);R=_diagnostic_rotation(np.asarray(pose)[3:])
  local=(np.asarray(point_world)-pose[:3])@R
  point=(local-T[:3,3])@T[:3,:3]
  b=np.asarray(s['cap_bounds_m']);skin=s['contact_offset_m']
  ok=point[0]>=s['cap_lower_x_m']-1e-12 and point[0]<=b[1,0]+skin and np.all(point[1:]>=b[0,1:]-skin)and np.all(point[1:]<=b[1,1:]+skin)
  return bool(ok),point
 def clearance(self,poses):
  n=len(poses);allmin=np.full(n,np.inf);nonmin=np.full(n,np.inf)
  for e in range(n):
   for j,name in enumerate(self.names):
    p=poses[e,j];z=_diagnostic_rotation(p[3:])[2]
    allmin[e]=min(allmin[e],float((self.clouds['all__'+name]@z+p[2]).min()))
    nonmin[e]=min(nonmin[e],float((self.clouds['non_toe__'+name]@z+p[2]).min()))
  return allmin,nonmin

def _classify_patches(data,sensor_map,poses,geometry,num_envs):
 """Classify each actual native patch before force aggregation. No centroid proxy."""
 force,point,normal,sep,counts,starts=[np.asarray(x)for x in data]
 if counts.shape!=(19*num_envs,1)or starts.shape!=counts.shape:raise ValueError('Incomplete sensor/filter contact layout')
 if np.any(counts<0)or np.any(starts<0):raise ValueError('Negative contact indexing')
 feet=np.zeros((num_envs,6,3));other=np.zeros((num_envs,4,3));patches=[];used=set();cap_cache={}
 for i,(e,body)in enumerate(sensor_map):
  start,count=int(starts[i,0]),int(counts[i,0])
  if count==0:continue
  if start+count>len(force):raise ValueError('Contact buffer overflow/range')
  for k in range(start,start+count):
   if k in used:raise ValueError('Overlapping contact-buffer ranges')
   used.add(k);f=float(force[k,0]);p=point[k];n=normal[k];d=float(sep[k,0])
   if not np.isfinite(np.r_[f,p,n,d]).all():raise ValueError('Nonfinite used contact patch')
   inactive_zero_normal=bool(f==0. and np.all(n==0.) and d==0.)
   if abs(np.linalg.norm(n)-1)>1e-3 and not inactive_zero_normal:raise ValueError('Invalid patch normal')
   local=None
   if body.endswith('_tibia'):
    key=(e,body)
    if key not in cap_cache:
     s=geometry.shapes[body];pose=poses[e,geometry.names.index(body)]
     cap_cache[key]=(pose,np.asarray(s['shape_to_link']),_diagnostic_rotation(np.asarray(pose)[3:]),np.asarray(s['cap_bounds_m']),s['contact_offset_m'],s['cap_lower_x_m'])
    pose,T,R,b,skin,lower=cap_cache[key]
    local=(np.asarray(p)-pose[:3])@R
    local=(local-T[:3,3])@T[:3,:3]
    cap=bool(local[0]>=lower-1e-12 and local[0]<=b[1,0]+skin and np.all(local[1:]>=b[0,1:]-skin)and np.all(local[1:]<=b[1,1:]+skin))
    category='toe'if cap else'shaft'
   else:category='coxa'if body.endswith('_coxa')else'femur'if body.endswith('_femur')else'body'
   if category=='toe':feet[e,LEGS.index(body[:2])]+=f*n
   else:other[e,['body','coxa','femur','shaft'].index(category)]+=f*n
   patches.append({'buffer_index':k,'env':e,'body':body,'category':category,'normal_force_n':f,'point_world_m':p.tolist(),'normal_world':n.tolist(),'separation_m':d,'inactive_zero_normal':inactive_zero_normal,'shape_point_m':None if local is None else local.tolist()})
 if len(used)>=len(force):raise ValueError('Contact capacity exhausted; completeness is unknown')
 # Any nonfoot patch with >1N is retained even when vector cancellation would hide it.
 nonfoot=np.zeros(num_envs,dtype=bool);body_forces={}
 for p in patches:
  if p['category']!='toe':
   if abs(p['normal_force_n'])>1.:nonfoot[p['env']]=True
   key=(p['env'],p['body'])
   if key not in body_forces:body_forces[key]=np.zeros(3)
   body_forces[key]=body_forces[key]+p['normal_force_n']*np.asarray(p['normal_world'])
 for i,(e,body)in enumerate(sensor_map):
  body_force=body_forces.get((e,body))
  if body_force is not None and np.linalg.norm(body_force)>1.:nonfoot[e]=True
 return {'distal_contact':np.linalg.norm(feet,axis=-1)>1.,'distal_force_world':feet,'nonfoot_contact':nonfoot,'nonfoot_force_world':other,'patches':patches,'used_patch_count':len(used)}


class DiagnosticCapture:
    """Full 400Hz standing evidence, compatible with the unchanged numeric scorer.

    This slow, complete diagnostic is separate from compact learner telemetry.
    A native failure is retained and aborts; no rows are reset or discarded.
    """
    def __init__(self, env, output, geometry_extrema):
        self.env = env
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        with np.load(geometry_extrema, allow_pickle=False) as data:
            self.geometry = _DiagnosticGeometry(env.geometry_meta, {k: data[k] for k in data.files}, env.native_body_names)
        self.rows, self.controls, self.files = [], [], []
        self.count = 0
        self.failure = None
        self.counter = env.sim.get_physics_step_count()
        self.previous_q = env.current["q"].cpu().numpy().astype(float)
        self.contacts = (self.output / "contacts.jsonl").open("w")
        self.save("native_readback.json", env.native_readback)
        self.save("initial_reset.json", {"counter_after": self.counter,
            "post_reset": {"joint_position_rad": self.previous_q.tolist(),
                           "joint_velocity_rad_s": env.current["dq"].cpu().tolist(),
                           "root_pose_xyzw": env.current["root"].cpu().tolist()},
            "scope": "Recorded initial state for a reset-free diagnostic; no admission is implied."})
        env.capture = self

    def save(self, name, value):
        (self.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def __call__(self, env, state, q, dq, requested, applied, ceiling, target, forces, substep, native_pre):
        cpu = lambda value: value.detach().cpu().numpy().copy()
        row = {"pre_joint_position_rad": cpu(q), "pre_joint_velocity_rad_s": cpu(dq),
            "joint_position_rad": cpu(state["q"]), "joint_velocity_rad_s": cpu(state["dq"]),
            "root_pose_xyzw": cpu(state["root"]), "link_pose_xyzw": cpu(state["link"]),
            "root_com_velocity": cpu(state["root_velocity"]), "joint_target_rad": cpu(target),
            "computed_torque_nm": cpu(requested), "applied_torque_nm": cpu(applied),
            "effort_ceiling_nm": cpu(ceiling), "native_input_pre_nm": cpu(native_pre),
            "native_input_post_nm": cpu(env.get("get_dof_actuation_forces")[:, env.joint_index]),
            "sequence": np.array(self.count), "control_index": np.array(self.count // 8),
            "substep_index": np.array(substep), "explicit_counter": np.array(env.sim.get_physics_step_count()),
            "time_s": np.array((self.count + 1) * env.cfg.physics_dt),
            "contact_valid": np.ones(env.num_envs, bool), "interval_valid": np.ones(env.num_envs, bool),
            "truncated": np.zeros(env.num_envs, bool)}
        try:
            data = [np.array(x.numpy(), copy=True) for x in env.contact.get_contact_data(env.cfg.physics_dt)]
            contact = _classify_patches(data, env.sensor_map, row["link_pose_xyzw"], self.geometry, env.num_envs)
            minimum, non_toe = self.geometry.clearance(row["link_pose_xyzw"])
            row.update(distal_contact=contact["distal_contact"], distal_force_world_n=contact["distal_force_world"],
                nonfoot_contact=contact["nonfoot_contact"], nonfoot_force_world_n=contact["nonfoot_force_world"],
                minimum_mesh_floor_m=minimum, minimum_non_toe_floor_m=non_toe,
                interval_angle_rate_rad_s=(row["joint_position_rad"].astype(float)-self.previous_q)/env.cfg.physics_dt)
            limits = np.asarray(env.native_readback["limits"])
            max_velocity = np.asarray(env.native_readback["native_max_velocity"])
            joint_bad = np.any((row["joint_position_rad"] < limits[:, :, 0]-2e-6) | (row["joint_position_rad"] > limits[:, :, 1]+2e-6), axis=1)
            speed_bad = np.any(abs(row["joint_velocity_rad_s"]) > max_velocity+2e-6, axis=1)
            row["terminated"] = (row["root_pose_xyzw"][:, 2] < .055) | joint_bad | speed_bad | contact["nonfoot_contact"] | (non_toe < -.001)
            self.rows.append(row)
            if substep == 7:
                self.controls.append(row)
            self.contacts.write(json.dumps({"sequence": self.count, "explicit_counter": int(row["explicit_counter"]), "patches": contact["patches"]}, allow_nan=False) + "\n")
            self.count += 1
            self.previous_q = row["joint_position_rad"].astype(float)
            if int(row["explicit_counter"]) != self.counter + 1:
                raise RuntimeError("Diagnostic physics counter differs")
            self.counter += 1
            if not all(np.isfinite(value).all() for value in row.values()):
                raise ValueError("Nonfinite diagnostic native sample")
            if not np.array_equal(row["native_input_pre_nm"], row["applied_torque_nm"]):
                raise ValueError("Native external-force input differs")
            if np.max(abs(row["applied_torque_nm"])) > 1.60001 or row["terminated"].any():
                raise ValueError("Original diagnostic cap/plate/joint/nonfoot/clearance gate failed")
            if len(self.rows) >= 800:
                self.flush()
        except BaseException as error:
            self.failure = repr(error)
            self.save("failed_partial_step.json", {k: np.asarray(v).tolist() for k, v in row.items()})
            self.flush()
            raise

    def flush(self):
        if self.rows:
            name = f"substeps_{len(self.files):03d}.npz"
            np.savez_compressed(self.output / name, **{k: np.stack([r[k] for r in self.rows]) for k in self.rows[0]})
            self.files.append(name)
            self.rows = []
        self.contacts.flush()

    def close(self):
        self.env.capture = None
        self.flush()
        self.contacts.close()
        if self.controls:
            np.savez_compressed(self.output / "control_trace.npz", **{k: np.stack([r[k] for r in self.controls]) for k in self.controls[0]})
        self.save("session.json", {"steps": self.count, "controls": len(self.controls), "reset_count": 1,
            "failure": self.failure, "substep_files": self.files, "joint_names": self.env.joint_names,
            "body_names": self.env.native_body_names, "root_paths": self.env.roots,
            "all_rows_recorded": self.count == len(self.controls) * 8, "contact_evidence_complete": self.failure is None,
            "neutral_joint_position_rad": self.env.neutral.cpu().tolist(),
            "physical_admission": False, "scope": "Unchanged gate thresholds; scorer must independently recompute this new source/scene result."})


# Neutral-aware successor of the canonical diagnostic scorer. Numeric gate
# thresholds and arithmetic stay unchanged; targets bind the declared stance.
def _diagnostic_servo(q,dq,target,kp,kd):
 q,dq,target=[np.asarray(x,dtype=np.float32)for x in(q,dq,target)]
 if q.ndim!=2 or q.shape[1]!=18 or dq.shape!=q.shape or target.shape!=q.shape:raise ValueError('Wrong named batched servo shape')
 if any(not np.isfinite(x).all()for x in(q,dq,target,kp,kd)):raise ValueError('Nonfinite servo input')
 raw=kp*(target-q)-kd*dq
 ceiling=np.minimum(1.6,np.interp(abs(dq.astype(float))*60/(2*np.pi),np.array([0.,70.,275.,340.,450.,477.,480.]),np.array([5.5,5.5,4.,3.,1.6,.5,0.]),right=0.)).astype(np.float32)
 # Float32 representation of 1.6 is accepted only to the explicit1.60001 numeric gate.
 return raw.astype(np.float32),np.clip(raw,-ceiling,ceiling).astype(np.float32),ceiling


_QUIET_GATES = {
    "max_planar_excursion_m": .01,
    "max_heading_excursion_deg": 2.,
    "max_joint_velocity_rms_rad_s": .03,
    "max_joint_position_range_rad": .02,
    "max_target_step_abs_p95_rad_per_20ms": .002,
    "max_requested_torque_saturation_fraction": .005,
    "max_applied_torque_nm": 1.60001,
}

def _quiet_metrics(data, env_index, start_step, joint_names, dt):
    """Score one contiguous window without deleting failures or restarting time."""
    take = lambda key: data[key][start_step:, env_index]
    q = take("joint_position_rad")
    target = take("joint_target_rad")
    velocity = take("joint_velocity_rad_s")
    position = take("position_world_m")
    quat = take("quaternion_world_wxyz")
    if len(q) < 2:
        raise ValueError("Quiet window needs at least two samples")
    w, x, y, z = quat.T
    heading = np.unwrap(np.arctan2(-1 + 2*(x*x+z*z), 2*(w*z-x*y)))
    joint_rms = np.sqrt(np.mean(velocity**2, axis=0))
    qrange = np.ptp(q, axis=0)
    target_p95 = np.quantile(np.abs(np.diff(target, axis=0)) * .02 / dt, .95, axis=0)
    requested = take("computed_torque_nm")
    if not all(np.isfinite(value).all() for value in (q, target, velocity, position, quat, requested, take("applied_torque_nm"))):
        raise ValueError("Nonfinite quiet-review state")
    row = {
        "window_samples": len(q), "window_duration_s": len(q) * dt,
        "max_planar_excursion_m": float(np.linalg.norm(position[:, :2] - position[0, :2], axis=1).max()),
        "max_heading_excursion_deg": float(np.degrees(np.abs(heading - heading[0]).max())),
        "max_joint_velocity_rms_rad_s": float(joint_rms.max()),
        "max_joint_position_range_rad": float(qrange.max()),
        "max_target_step_abs_p95_rad_per_20ms": float(target_p95.max()),
        "max_requested_torque_saturation_fraction": float((np.abs(requested) > 1.6).mean(0).max()),
        "mean_requested_torque_saturation_fraction": float((np.abs(requested) > 1.6).mean()),
        "max_applied_torque_nm": float(np.abs(take("applied_torque_nm")).max()),
        "requested_torque_abs_max_nm": float(np.abs(requested).max()),
        # Any failure anywhere in the trial invalidates the result, even if the
        # robot is quiet after an automatic reset inside/before the scored window.
        "terminations": int(data["terminated"][:, env_index].sum()),
        "truncations": int(data["truncated"][:, env_index].sum()),
        "joints": {name: {"velocity_rms_rad_s": float(joint_rms[j]),
                          "position_range_rad": float(qrange[j]),
                          "target_step_abs_p95_rad_per_20ms": float(target_p95[j]),
                          "saturation_fraction": float((np.abs(requested[:, j]) > 1.6).mean())}
                   for j, name in enumerate(joint_names)},
    }
    row["failed_bounds"] = [key for key, bound in _QUIET_GATES.items() if row[key] > bound]
    row["pass"] = not row["failed_bounds"] and row["terminations"] == 0 and row["truncations"] == 0
    return row

def score_diagnostic(directory):
 read=lambda path:json.loads(Path(path).read_text());STEPS=8000;CONTROLS=1000;SETTLE=200;DT=.0025
 d=Path(directory);s=read(d/'session.json');names=s['joint_names'];n=len(s['root_paths'])
 with np.load(d/'control_trace.npz')as z:data={k:z[k]for k in z.files}
 if len(data['sequence'])!=CONTROLS:raise ValueError('Incomplete control trace')
 data['position_world_m']=data['root_pose_xyzw'][...,:3]
 data['quaternion_world_wxyz']=data['root_pose_xyzw'][...,[6,3,4,5]]
 result={'num_envs':n,'controls':len(data['sequence']),'substeps':0,'gates':_QUIET_GATES,'replicas':[],'scope':'Provisional simulation standing screen; no hardware calibration or walking admission.'}
 maxap=np.zeros(n);rawmax=np.zeros(n);sat=np.zeros((n,18),int);postcount=0;badcontact=np.zeros(n,int);nonfoot=np.zeros(n,int);clear=np.full(n,np.inf);height=np.full(n,np.inf);qdelta=np.zeros(n);counter=int(read(d/'initial_reset.json')['counter_after']);previous=None;previous_dq=None;control_end=[]
 native=read(d/'native_readback.json');limits=np.asarray(native['limits']);maxvel=np.asarray(native['native_max_velocity']);order=[JOINT_NAMES.index(x)for x in names];kp=np.full(18,12.,np.float32)[order];kd=np.asarray(KD,np.float32)[order]
 for f in s['substep_files']:
  with np.load(d/f)as z:r={k:z[k]for k in z.files}
  m=len(r['sequence']);start=result['substeps']
  if not np.array_equal(r['sequence'],np.arange(start,start+m)):raise ValueError('Raw sequence mismatch')
  if not np.array_equal(r['control_index'],np.arange(start,start+m)//8)or not np.array_equal(r['substep_index'],np.arange(start,start+m)%8):raise ValueError('Raw8-substep ordering mismatch')
  if counter is not None and r['explicit_counter'][0]!=counter+1 or np.any(np.diff(r['explicit_counter'])!=1):raise ValueError('Native counter recurrence mismatch')
  counter=int(r['explicit_counter'][-1]);result['substeps']+=m
  if not np.allclose(r['time_s'],(r['sequence']+1)*DT,atol=1e-12,rtol=0):raise ValueError('Raw timestamp differs')
  if not r['contact_valid'].all()or not r['interval_valid'].all():raise ValueError('Invalid raw observation history/contact')
  if not all(np.isfinite(v).all()for v in r.values()):raise ValueError('Nonfinite raw state')
  if np.any((r['joint_position_rad']<limits[None,:,:,0]-2e-6)|(r['joint_position_rad']>limits[None,:,:,1]+2e-6))or np.any(abs(r['joint_velocity_rad_s'])>maxvel[None]+2e-6):raise ValueError('Actual measured joint/rate bound exceeded')
  if not np.array_equal(r['native_input_pre_nm'],r['applied_torque_nm']):raise ValueError('Raw native external input mismatch')
  if np.any(r['joint_target_rad']!=np.asarray(s['neutral_joint_position_rad'],np.float32)):raise ValueError('Declared neutral target changed')
  q=r['joint_position_rad'].astype(float)
  if previous is None:
   reset=read(d/'initial_reset.json')['post_reset'];previous=np.asarray(reset['joint_position_rad']);previous_dq=np.asarray(reset['joint_velocity_rad_s'])
  if not np.array_equal(r['pre_joint_position_rad'],np.concatenate([previous[None],q[:-1]]))or not np.array_equal(r['pre_joint_velocity_rad_s'],np.concatenate([previous_dq[None],r['joint_velocity_rad_s'][:-1]])):raise ValueError('Servo input not previous native post-step state')
  a,b,c=_diagnostic_servo(r['pre_joint_position_rad'].reshape(-1,18),r['pre_joint_velocity_rad_s'].reshape(-1,18),r['joint_target_rad'].reshape(-1,18),kp,kd)
  for key,want in [('computed_torque_nm',a),('applied_torque_nm',b),('effort_ceiling_nm',c)]:
   if not np.array_equal(r[key],want.reshape(m,n,18)):raise ValueError('Raw provisional servo recurrence differs:'+key)
  dq=np.diff(np.concatenate([previous[None],q]),axis=0)/DT
  if not np.allclose(dq,r['interval_angle_rate_rad_s'],atol=1e-8,rtol=0):raise ValueError('Interval angle channel differs')
  previous=q[-1];previous_dq=r['joint_velocity_rad_s'][-1];qdelta=np.maximum(qdelta,np.max(abs(dq),axis=(0,2)))
  maxap=np.maximum(maxap,np.max(abs(r['applied_torque_nm']),axis=(0,2)));rawmax=np.maximum(rawmax,np.max(abs(r['computed_torque_nm']),axis=(0,2)))
  height=np.minimum(height,np.min(r['root_pose_xyzw'][...,2],axis=0))
  nonfoot+=r['nonfoot_contact'].sum(0);clear=np.minimum(clear,np.min(r['minimum_non_toe_floor_m'],axis=0))
  post=r['control_index']>=SETTLE;postcount+=int(post.sum())
  if post.any():
   sat+=(abs(r['computed_torque_nm'][post])>1.6).sum(0)
   badcontact+=(~r['distal_contact'][post].all(-1)).sum(0)
  end=r['substep_index']==7
  for index in np.flatnonzero(end):
   control=int(r['control_index'][index])
   for key in data:
    if key in r and not np.array_equal(data[key][control],r[key][index]):raise ValueError('Control endpoint differs:'+key)
 if result['substeps']!=STEPS or postcount!=(CONTROLS-SETTLE)*8:raise ValueError('Incomplete400Hz acquisition')
 for e in range(n):
  q=_quiet_metrics(data,e,SETTLE,names,.02)
  physical={'max_applied_all_substeps_nm':float(maxap[e]),'max_requested_all_substeps_nm':float(rawmax[e]),'max_requested_saturation_fraction_400hz':float((sat[e]/postcount).max()),'mean_requested_saturation_fraction_400hz':float(sat[e].sum()/(postcount*18)),
   'post_settle_missing_six_toe_substeps':int(badcontact[e]),'all_controlled_nonfoot_substeps':int(nonfoot[e]),'minimum_non_toe_mesh_floor_m':float(clear[e]),'minimum_plate_height_m':float(height[e]),'interval_angle_rate_abs_max_rad_s':float(qdelta[e])}
  failed=[]
  if maxap[e]>1.60001:failed.append('applied_cap')
  if (sat[e]/postcount).max()>.005 or sat[e].sum()/(postcount*18)>.005:failed.append('requested_saturation_400hz')
  if badcontact[e]:failed.append('six_toe_support')
  if nonfoot[e]:failed.append('nonfoot_contact')
  if clear[e]<-.001:failed.append('non_toe_floor_clearance')
  if height[e]<.055:failed.append('plate_height')
  if q['window_duration_s']<10:failed.append('quiet_duration')
  result['replicas'].append({'env':e,'quiet':q,'physical':physical,'failed_physical_bounds':failed,'pass':q['pass']and not failed})
 result['all_pass']=all(x['pass']for x in result['replicas']);return result
