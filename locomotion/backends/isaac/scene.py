"""Build the unchanged robot grid, floor and contact-report recipe."""
from pathlib import Path
import math


def build_scene(stage, asset, cfg, tensor):
    from pxr import UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Gf
    # A grid centred on the origin keeps every replica on the 80 m floor mesh up to 1024 robots; a grid
    # that starts at the origin leaves the 80 m mesh beyond 441 robots at 2 m spacing.
    side = math.ceil(math.sqrt(cfg.num_envs))
    half = (side - 1) / 2
    origins = [((i % side - half) * cfg.spacing_m, (i // side - half) * cfg.spacing_m, 0.) for i in range(cfg.num_envs)]
    origins_tensor = tensor(origins)
    roots = [f"/Robot_{i:03d}" for i in range(cfg.num_envs)]
    for root, origin in zip(roots, origins):
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
    return origins_tensor, roots
