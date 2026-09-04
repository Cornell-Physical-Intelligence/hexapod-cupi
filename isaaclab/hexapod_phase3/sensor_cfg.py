"""Isaac Lab 3.0 / Isaac Sim 6.0 scene configuration for Phase 3 sensors.

Mount assumptions use the imported root rigid-body frame, with physical forward
along -Y, lateral along +X, and +Z up.  The virtual mounts do not add camera,
LiDAR, bracket, cable, or compute payload mass to the robot. Physical CAD
measurements must replace these offsets before sim-to-real use.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from pxr import PhysxSchema, Usd

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ImuCfg, MultiMeshRayCasterCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.sensors import SensorFrameCfg
from isaaclab.utils.configclass import configclass

from hexapod_rl.asset_cfg import HEXAPOD_CFG, ROOT_LINK_NAME

from .mid360_pattern import Mid360SurrogatePatternCfg
from .sensor_model import DEFAULT_PHASE3_SENSOR_MODEL_CFG


PHYSICS_DT_S = 1.0 / 200.0
GROUND_PRIM_PATH = "/World/Phase3Ground"
ROBOT_ROOT_RIGID_PRIM = "{ENV_REGEX_NS}/Robot/Geometry/" + ROOT_LINK_NAME
LIDAR_MOUNT_PRIM = f"{ROBOT_ROOT_RIGID_PRIM}/phase3_mid360"


@dataclass(frozen=True)
class Phase3HardwareAccounting:
    """Published sensor envelope, deliberately separate from robot physics."""

    baseline_body_mass_kg: float = 1.5
    baseline_robot_mass_kg: float = 6.3

    mid360_mass_kg: float = 0.265
    mid360_dimensions_m_lwh: tuple[float, float, float] = (0.065, 0.065, 0.060)
    mid360_average_power_w: float = 6.5
    mid360_cold_peak_power_w: float = 14.0
    mid360_horizontal_fov_deg: float = 360.0
    mid360_vertical_fov_deg: float = 59.0
    mid360_vertical_fov_range_deg: tuple[float, float] = (-7.0, 52.0)
    mid360_min_range_m: float = 0.10
    mid360_max_range_10_percent_reflectivity_m: float = 40.0
    mid360_max_range_80_percent_reflectivity_m: float = 70.0

    d455_mass_kg: float = 0.116
    d455_dimensions_m_ldh: tuple[float, float, float] = (0.124, 0.026, 0.029)
    d455_max_operating_mode_power_w: float = 3.46147
    d455_depth_fov_deg_hv: tuple[float, float] = (87.0, 58.0)
    d455_min_depth_m: float = 0.52
    d455_ideal_range_m: tuple[float, float] = (0.60, 6.0)

    @property
    def combined_sensor_mass_kg(self) -> float:
        return self.mid360_mass_kg + self.d455_mass_kg

    @property
    def projected_body_payload_mass_kg(self) -> float:
        return self.baseline_body_mass_kg + self.combined_sensor_mass_kg

    @property
    def projected_robot_mass_kg(self) -> float:
        return self.baseline_robot_mass_kg + self.combined_sensor_mass_kg

    def as_report(self) -> dict[str, object]:
        """Return JSON-ready accounting without implying physical application."""

        return {
            "configured_lidar": {
                "model": "Livox Mid-360",
                "classification": "near-hemispherical 360x59 degree FoV",
                "mass_kg": self.mid360_mass_kg,
                "dimensions_m_lwh": self.mid360_dimensions_m_lwh,
                "average_power_w": self.mid360_average_power_w,
                "cold_peak_power_w": self.mid360_cold_peak_power_w,
                "horizontal_fov_deg": self.mid360_horizontal_fov_deg,
                "vertical_fov_deg": self.mid360_vertical_fov_deg,
                "vertical_fov_range_deg": self.mid360_vertical_fov_range_deg,
                "min_range_m": self.mid360_min_range_m,
                "max_range_10_percent_reflectivity_m": (
                    self.mid360_max_range_10_percent_reflectivity_m
                ),
                "max_range_80_percent_reflectivity_m": (
                    self.mid360_max_range_80_percent_reflectivity_m
                ),
            },
            "depth_camera": {
                "model": "RealSense D455",
                "mass_kg": self.d455_mass_kg,
                "dimensions_m_ldh": self.d455_dimensions_m_ldh,
                "max_operating_mode_power_w": (
                    self.d455_max_operating_mode_power_w
                ),
                "depth_fov_deg_hv": self.d455_depth_fov_deg_hv,
                "min_depth_m": self.d455_min_depth_m,
                "ideal_range_m": self.d455_ideal_range_m,
            },
            "mass_budget": {
                "baseline_body_mass_kg": self.baseline_body_mass_kg,
                "baseline_robot_mass_kg": self.baseline_robot_mass_kg,
                "combined_sensor_mass_kg": self.combined_sensor_mass_kg,
                "projected_body_payload_mass_kg": (
                    self.projected_body_payload_mass_kg
                ),
                "projected_robot_mass_kg": self.projected_robot_mass_kg,
                "excludes": ["brackets", "cables", "compute"],
            },
            "identification_fork": {
                "unitree_l1_fov_deg_hv": (360.0, 90.0),
                "requires_physical_label_or_photo_confirmation": True,
            },
            "applied_to_articulation": False,
        }


PHASE3_HARDWARE_ACCOUNTING = Phase3HardwareAccounting()


@dataclass(frozen=True)
class Phase3MountAssumptions:
    """Unmeasured root-frame poses with per-field quaternion conventions."""

    # NVIDIA D455 rig: near the physical -Y nose, 45 mm above the root frame.
    camera_pos_b_m: tuple[float, float, float] = (0.0, -0.105, 0.045)
    camera_downward_pitch_deg: float = 20.0
    # Active rotation q_bodyX(+20 deg) * q_bodyZ(-90 deg): nominal sensor +X
    # maps to body (0, -cos(20 deg), -sin(20 deg)), retaining physical -Y
    # forward while looking down. Isaac Sim 6 consumes WXYZ quaternions here.
    camera_quat_b_wxyz: tuple[float, float, float, float] = (
        0.696364240320019,
        0.12278780396897285,
        0.12278780396897282,
        -0.6963642403200189,
    )
    # Mid-360: top-deck center, high enough to reduce leg occlusion.
    lidar_pos_b_m: tuple[float, float, float] = (0.0, 0.0, 0.075)
    lidar_quat_b_xyzw: tuple[float, float, float, float] = (
        0.0,
        0.0,
        -0.7071067811865476,
        0.7071067811865476,
    )
    # Mid-360 ICM40609 offset from the lidar origin is exactly
    # (+11, +23.29, -44.12) mm in the sensor frame. A -90-degree body yaw
    # maps that to (+23.29, -11, -44.12) mm before adding the lidar origin.
    imu_pos_b_m: tuple[float, float, float] = (0.02329, -0.011, 0.03088)
    imu_quat_b_xyzw: tuple[float, float, float, float] = (
        0.0,
        0.0,
        -0.7071067811865476,
        0.7071067811865476,
    )


PHASE3_MOUNTS = Phase3MountAssumptions()


@configclass
class Phase3ArticulationCfg(ArticulationCfg):
    """Articulation config that removes contact reporting baked into the USD."""

    def _post_spawn(self, stage: Any) -> None:
        super()._post_spawn(stage)
        author_prim_path = (
            self.spawn.spawn_path
            if self.spawn is not None and self.spawn.spawn_path is not None
            else self.prim_path
        )
        root = stage.GetPrimAtPath(author_prim_path)
        if not root.IsValid():
            raise RuntimeError(
                f"Cannot remove Phase 3 contact reporting; invalid robot prim {author_prim_path}."
            )
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                prim.RemoveAPI(PhysxSchema.PhysxContactReportAPI)


def _make_phase3_robot_cfg() -> Phase3ArticulationCfg:
    """Copy the locomotion asset without sharing mutable nested configs."""
    base_cfg = HEXAPOD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=HEXAPOD_CFG.spawn.replace(activate_contact_sensors=False),
    )
    values = {
        field.name: getattr(base_cfg, field.name)
        for field in fields(ArticulationCfg)
        if field.init
    }
    return Phase3ArticulationCfg(**values)


@configclass
class HexapodPhase3SensorSceneCfg(InteractiveSceneCfg):
    """One perception scene; foot/contact sensors are intentionally absent."""

    ground = AssetBaseCfg(
        prim_path=GROUND_PRIM_PATH,
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.9,
                restitution=0.0,
            )
        ),
    )
    # A per-environment wall at the physical -Y nose ensures both the camera
    # depth channel and horizontal LiDAR beams have a finite smoke-test target.
    calibration_wall = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Phase3CalibrationWall",
        spawn=sim_utils.CuboidCfg(
            size=(0.80, 0.08, 0.50),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.55, 0.85), roughness=0.8
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -1.0, 0.25)),
    )
    # Phase 1 enables contact reporting, and the converted USD currently also
    # contains baked contact-report schemas. The Phase 3-specific post-spawn
    # hook removes those APIs from the source prim before environments clone.
    robot: Phase3ArticulationCfg = _make_phase3_robot_cfg()

    # Isaac Lab v3.0.0-beta2.patch1 declares RayCasterCfg.spawn but does not
    # invoke it when InteractiveScene constructs a sensor.  Spawn the physical
    # attachment frame as an ordinary scene asset, then track that frame with a
    # zero-offset ray caster.  Besides following the robot correctly, this makes
    # lidar.data.pos_w the actual optical origin used for range calculation.
    lidar_mount = AssetBaseCfg(
        prim_path=LIDAR_MOUNT_PRIM,
        spawn=SensorFrameCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=PHASE3_MOUNTS.lidar_pos_b_m,
            rot=PHASE3_MOUNTS.lidar_quat_b_xyzw,
        ),
    )
    lidar = MultiMeshRayCasterCfg(
        prim_path=LIDAR_MOUNT_PRIM,
        spawn=None,
        update_period=DEFAULT_PHASE3_SENSOR_MODEL_CFG.lidar.transport.sample_period_s,
        offset=MultiMeshRayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        ray_alignment="base",
        pattern_cfg=Mid360SurrogatePatternCfg(),
        max_distance=DEFAULT_PHASE3_SENSOR_MODEL_CFG.lidar.max_range_m,
        mesh_prim_paths=[
            GROUND_PRIM_PATH,
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Phase3CalibrationWall",
                is_shared=True,
                track_mesh_transforms=True,
            ),
        ],
        reference_meshes=True,
        update_mesh_ids=False,
        debug_vis=False,
    )
    imu = ImuCfg(
        prim_path=ROBOT_ROOT_RIGID_PRIM,
        update_period=DEFAULT_PHASE3_SENSOR_MODEL_CFG.imu.transport.sample_period_s,
        offset=ImuCfg.OffsetCfg(
            pos=PHASE3_MOUNTS.imu_pos_b_m,
            rot=PHASE3_MOUNTS.imu_quat_b_xyzw,
        ),
        debug_vis=False,
    )

    light = AssetBaseCfg(
        prim_path="/World/Phase3Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=2500.0, color=(0.78, 0.80, 0.85)
        ),
    )


def make_phase3_sim_cfg(device: str = "cuda:0") -> SimulationCfg:
    """Build the 200 Hz PhysX configuration used by the sensor smoke test."""
    return SimulationCfg(
        device=device,
        dt=PHYSICS_DT_S,
        # Isaac Sim 6 RTX sensors use their authored tick rate. Rendering every
        # physics step lets its multitick scheduler produce the 30 Hz D455 stream.
        render_interval=1,
        physics=PhysxCfg(gpu_max_rigid_patch_count=2**20),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
