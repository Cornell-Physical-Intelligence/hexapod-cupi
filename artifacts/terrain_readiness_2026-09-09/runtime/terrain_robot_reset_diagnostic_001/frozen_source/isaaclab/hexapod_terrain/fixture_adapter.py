"""Single-environment import adapter for a terrain standing/contact smoke.

This intentionally has no training registration. Fixed-world-height flat rewards
must be replaced by terrain-relative objectives before a terrain PPO task is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import math
from pathlib import Path


@dataclass(frozen=True)
class MildTerrainSpec:
    catalog: Path
    fixture_id: str
    friction: float = 1.0
    contact_offset_m: float = .001
    start_body_yaw_rad: float = math.pi / 2  # Body -Y forward becomes course +X.
    speed_range_mps: tuple[float, float] = (.05, .12)
    yaw_rate_limit_rad_s: float = .20

    def __post_init__(self):
        values = (self.friction, self.contact_offset_m, self.start_body_yaw_rad,
                  *self.speed_range_mps, self.yaw_rate_limit_rad_s)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Terrain settings must be finite")
        if not (self.friction > 0 and 0 < self.contact_offset_m <= .002):
            raise ValueError("Invalid material/contact settings")
        if not (0 < self.speed_range_mps[0] <= self.speed_range_mps[1] <= .12):
            raise ValueError("Mild pilot speeds must lie in (0,0.12] m/s")
        if not 0 < self.yaw_rate_limit_rad_s <= .20:
            raise ValueError("Mild pilot yaw limit must lie in (0,0.20] rad/s")

    def load(self):
        # Put repository tools/ on PYTHONPATH alongside isaaclab/ in the launcher.
        from terrain_fixture_checks import audit_usd, load_catalog

        entry, usd, vertices, faces = load_catalog(self.catalog, [self.fixture_id])[0]
        if entry.get("curriculum") == "avoidance_only" or entry["family"] == "pit":
            raise ValueError("Negative obstacles belong to avoidance tests, not the mild walking curriculum")
        audit_usd(usd, vertices, faces)
        return entry, usd


def require_matching_admission(admission, asset_identity):
    """Reuse the full standing gate and its exact study-asset identity."""
    fields = ("variant", "urdf_sha256", "plan_sha256", "stance_index")
    if not admission.get("gate", {}).get("passed"):
        raise ValueError("A passed full flat standing admission is required")
    if any(k not in asset_identity or asset_identity[k] != admission.get(k) for k in fields):
        raise ValueError("Standing admission does not match the exact asset/stance/plan")
    if len(asset_identity["urdf_sha256"]) != 64 or len(asset_identity["plan_sha256"]) != 64:
        raise ValueError("Invalid admission identity")


def reference_fixture_mesh(stage, prim_path, usd_path):
    """Reference a root-Mesh fixture without authoring a stronger Xform type.

    The installed generic USD spawner authors an Xform at the reference site,
    overriding this fixture's root Mesh type while retaining its collision API.
    Author the matching Mesh type only in the new scene; keep the USD unchanged.
    """
    from pxr import UsdGeom
    from terrain_fixture_checks import collision_meshes

    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"Terrain prim already exists: {prim_path}")
    prim = UsdGeom.Mesh.Define(stage, prim_path).GetPrim()
    prim.GetReferences().AddReference(str(usd_path))
    return collision_meshes(stage, prim_path)[0]


def adapt_flat_cfg_for_fixture_smoke(cfg, spec, *, admission, asset_identity, purpose="standing_smoke"):
    """Copy an already admitted full-robot cfg; never change asset/action mapping.

    At present only a one-robot standing smoke is permitted. A future training
    adapter needs terrain-relative resets/rewards, atlas origins, boundaries,
    support queries, and separate teacher/student observations. Calling this
    function cannot silently launch the old flat reward on terrain.
    """
    if purpose != "standing_smoke":
        raise ValueError("Terrain PPO is not enabled by this smoke-only adapter")
    require_matching_admission(admission, asset_identity)
    entry, usd = spec.load()

    from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
    from terrain_fixture_checks import bind_fixture_material

    class ExplicitFixtureImporter(TerrainImporter):
        def import_usd(self, name, usd_path):
            from isaaclab.sim import SimulationContext

            prim_path = self.cfg.prim_path + "/" + name
            if prim_path in self.terrain_prim_paths:
                raise ValueError(f"Terrain already registered: {prim_path}")
            stage = SimulationContext.instance().stage
            prim = reference_fixture_mesh(stage, prim_path, usd_path)
            self.terrain_prim_paths.append(prim_path)
            bind_fixture_material(stage, prim, friction=spec.friction,
                                  contact_offset_m=spec.contact_offset_m)

    # Isaac configclass.copy() uses dataclasses.replace, which drops controller
    # fields attached by configure_omni after dataclass construction.
    result = deepcopy(cfg)
    result.scene.num_envs = 1
    result.events = None
    result.terrain = TerrainImporterCfg(class_type=ExplicitFixtureImporter,
        prim_path="/World/ground", terrain_type="usd", usd_path=str(usd),
        env_spacing=4.0, num_envs=1, collision_group=-1, debug_vis=False)
    # The source's default prim is itself the Mesh; there is no GroundPlane child.
    collider_path = "/World/ground/terrain"
    for sensor in (result.base_contact_sensor, result.coxa_contact_sensor,
                   *result.feet_contact_sensors, *result.femur_contact_sensors):
        if sensor.track_contact_points:
            sensor.filter_prim_paths_expr = [collider_path]
    x, y = entry["start_xy_m"]
    result.robot.init_state.pos = (x, y, result.robot.init_state.pos[2])
    half_yaw = spec.start_body_yaw_rad / 2
    result.robot.init_state.rot = (math.cos(half_yaw), 0., 0., math.sin(half_yaw))
    result.command_lin_vel_x_range_mps = (0., 0.)
    result.command_lin_vel_y_range_mps = (0., 0.)
    result.command_yaw_rate_range_rad_s = (0., 0.)
    result.stand_command_fraction = 1.0
    return result
