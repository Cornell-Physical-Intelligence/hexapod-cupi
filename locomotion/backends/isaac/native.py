"""Stock Isaac Lab lifecycle and per-physics-step capture hooks."""
import torch
from isaaclab.envs import DirectRLEnv
from isaaclab_physx.assets.articulation import Articulation


class CaptureArticulation(Articulation):
    """Read submitted force before physics; capture state after physics."""

    def write_data_to_sim(self):
        super().write_data_to_sim()
        if self.owner._control_active:
            self.owner.native_pre = (self.owner.get("get_dof_actuation_forces")[:, self.owner.joint_index]
                                     if self.owner.capture is not None else None)

    def update(self, dt):
        super().update(dt)
        if self.owner._control_active:
            self.owner._collect_substep()


class NativeEnv(DirectRLEnv):
    """Use stock decimation and reset hooks behind the task API adapter."""

    def __init__(self, cfg, robot_cfg, owner):
        self.owner, self.robot_cfg = owner, robot_cfg
        super().__init__(cfg)
        if self._physics_handles_decimation:
            raise RuntimeError("Direct candidate requires eight separate PhysX steps")

    def _setup_scene(self):
        from pxr import UsdGeom, UsdPhysics, UsdShade, PhysxSchema
        self.robot = CaptureArticulation(self.robot_cfg)
        self.robot.owner = self.owner
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
        self.robot.set_joint_position_target_index(target=self.owner.target[:, self.owner.inverse_joint_index])
        self.owner.substep += 1

    def _get_dones(self):
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


