"""Isaac Sim 6.0 runtime wrapper for NVIDIA's RealSense D455 USD.

The NVIDIA-provided USD is authored and loaded through the modern
``isaacsim.sensors.experimental.rtx`` API.  It is intentionally external to
``InteractiveScene.sensors`` because Isaac Lab's ``CameraCfg`` does not expose
the D455 single-view stereo post-processing pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import warp as wp
from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaacsim.sensors.experimental.rtx import (
    CameraSensor,
    RtxCamera,
    SingleViewDepthCameraSensor,
)

from hexapod_rl.asset_cfg import ROOT_LINK_NAME


D455_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Sensors/RealSense/D455/rsd455.usd"
D455_RESOLUTION_HW = (480, 848)
D455_TICK_RATE_HZ = 30.0


@dataclass(frozen=True)
class D455TorchFrame:
    rgb: torch.Tensor
    depth_m: torch.Tensor


@dataclass(frozen=True)
class D455DepthModelParameters:
    baseline_mm: float
    focal_length_px: float
    sensor_size_px: float
    maximum_disparity_px: float
    confidence_threshold: float
    noise_downscale_px: float
    noise_mean_px: float
    noise_sigma_px: float
    minimum_distance_m: float
    maximum_distance_m: float
    post_processing_enabled: bool


class D455Rig:
    """One NVIDIA D455 asset, RGB runtime, and stereo-depth runtime."""

    def __init__(
        self,
        parent_prim_path: str,
        translation_b_m: tuple[float, float, float],
        orientation_b_wxyz: tuple[float, float, float, float],
    ):
        self.mount_prim_path = f"{parent_prim_path}/phase3_d455"
        self.asset_prim_path = f"{self.mount_prim_path}/nvidia_asset"
        self._create_mount(
            parent_prim_path,
            translation_b_m,
            orientation_b_wxyz,
        )
        # RtxCamera.create() is the Sim 6.0-supported path for camera/depth USD
        # assets. It discovers the embedded pseudo-depth Camera and template
        # RenderProduct used by SingleViewDepthCameraSensor.  Do not pass a pose
        # here: create() applies transforms to the discovered Camera prim rather
        # than to the referenced D455 assembly, which would break the authored
        # internal camera extrinsics.  The parent mount Xform moves the whole rig.
        self._depth_camera = RtxCamera.create(
            self.asset_prim_path,
            tick_rate=D455_TICK_RATE_HZ,
            reset_xform_op_properties=False,
            usd_path=D455_USD_PATH,
        )
        self._disable_embedded_physics()

        camera_paths = self._discover_camera_paths()
        depth_paths = [
            path for path in camera_paths if path.endswith("/Camera_Pseudo_Depth")
        ]
        color_paths = [
            path
            for path in camera_paths
            if path.endswith("/Camera_OmniVision_OV9782_Color")
        ]
        wrapped_depth_paths = [str(path) for path in self._depth_camera.paths]
        if len(depth_paths) != 1 or wrapped_depth_paths != depth_paths:
            raise RuntimeError(
                "RtxCamera.create() did not select the D455 pseudo-depth "
                f"camera; wrapped={wrapped_depth_paths}, discovered={camera_paths}"
            )
        self._validate_depth_template(depth_paths[0])
        if len(color_paths) != 1:
            raise RuntimeError(
                "D455 color camera path changed; discovered cameras: "
                f"{camera_paths}"
            )
        self._color_camera = RtxCamera(
            color_paths[0],
            tick_rate=D455_TICK_RATE_HZ,
            reset_xform_op_properties=False,
        )
        self._color_sensor = CameraSensor(
            self._color_camera,
            resolution=D455_RESOLUTION_HW,
            annotators=["rgb"],
        )
        self._depth_sensor = SingleViewDepthCameraSensor(
            self._depth_camera,
            resolution=D455_RESOLUTION_HW,
            annotators=["depth_sensor_distance"],
        )
        self._depth_sensor.set_enabled_post_processing(True)
        self.depth_model_parameters = self._read_depth_model_parameters()
        self._validate_depth_model_parameters()

    def _read_depth_model_parameters(self) -> D455DepthModelParameters:
        minimum_distance_m, maximum_distance_m = (
            self._depth_sensor.get_sensor_distance_cutoffs()
        )
        noise_mean_px, noise_sigma_px = self._depth_sensor.get_sensor_noise_parameters()
        return D455DepthModelParameters(
            baseline_mm=float(self._depth_sensor.get_sensor_baseline()),
            focal_length_px=float(self._depth_sensor.get_sensor_focal_length()),
            sensor_size_px=float(self._depth_sensor.get_sensor_size()),
            maximum_disparity_px=float(
                self._depth_sensor.get_sensor_maximum_disparity()
            ),
            confidence_threshold=float(
                self._depth_sensor.get_sensor_disparity_confidence()
            ),
            noise_downscale_px=float(
                self._depth_sensor.get_sensor_disparity_noise_downscale()
            ),
            noise_mean_px=float(noise_mean_px),
            noise_sigma_px=float(noise_sigma_px),
            minimum_distance_m=float(minimum_distance_m),
            maximum_distance_m=float(maximum_distance_m),
            post_processing_enabled=bool(
                self._depth_sensor.get_enabled_post_processing()
            ),
        )

    def _validate_depth_model_parameters(self) -> None:
        params = self.depth_model_parameters
        finite_values = (
            params.baseline_mm,
            params.focal_length_px,
            params.sensor_size_px,
            params.maximum_disparity_px,
            params.confidence_threshold,
            params.noise_downscale_px,
            params.noise_mean_px,
            params.noise_sigma_px,
            params.minimum_distance_m,
            params.maximum_distance_m,
        )
        if not all(math.isfinite(value) for value in finite_values):
            raise RuntimeError(f"D455 depth template contains non-finite values: {params}")
        if min(
            params.baseline_mm,
            params.focal_length_px,
            params.sensor_size_px,
            params.maximum_disparity_px,
            params.noise_downscale_px,
        ) <= 0.0:
            raise RuntimeError(f"D455 depth geometry is invalid: {params}")
        if params.noise_mean_px < 0.0 or params.noise_sigma_px < 0.0:
            raise RuntimeError(f"D455 disparity noise is invalid: {params}")
        if not 0.0 <= params.confidence_threshold <= 1.0:
            raise RuntimeError(f"D455 confidence threshold is invalid: {params}")
        if not 0.0 <= params.minimum_distance_m < params.maximum_distance_m:
            raise RuntimeError(f"D455 distance cutoffs are invalid: {params}")
        if not params.post_processing_enabled:
            raise RuntimeError("D455 stereo-depth post-processing is disabled.")

    def _create_mount(
        self,
        parent_prim_path: str,
        translation_b_m: tuple[float, float, float],
        orientation_b_wxyz: tuple[float, float, float, float],
    ) -> None:
        """Author a local Xform without modifying the referenced D455 asset."""
        if len(translation_b_m) != 3 or len(orientation_b_wxyz) != 4:
            raise ValueError("D455 mount pose must contain xyz and wxyz values.")
        quat_norm = math.sqrt(sum(component * component for component in orientation_b_wxyz))
        if not math.isclose(quat_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError(
                f"D455 mount quaternion must be unit length, got norm {quat_norm:.8f}."
            )

        stage = sim_utils.get_current_stage()
        parent = stage.GetPrimAtPath(parent_prim_path)
        if not parent.IsValid():
            raise RuntimeError(f"D455 parent prim does not exist: {parent_prim_path}")
        if stage.GetPrimAtPath(self.mount_prim_path).IsValid():
            raise RuntimeError(f"D455 mount already exists: {self.mount_prim_path}")

        mount = UsdGeom.Xform.Define(stage, self.mount_prim_path)
        xformable = UsdGeom.Xformable(mount.GetPrim())
        xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*translation_b_m)
        )
        w, x, y, z = orientation_b_wxyz
        xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(w, Gf.Vec3d(x, y, z))
        )

    def _discover_camera_paths(self) -> list[str]:
        stage = sim_utils.get_current_stage()
        root = stage.GetPrimAtPath(self.asset_prim_path)
        if not root.IsValid():
            raise RuntimeError(f"D455 asset did not appear at {self.asset_prim_path}.")
        return [
            str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.IsA(UsdGeom.Camera)
        ]

    def _validate_depth_template(self, depth_camera_path: str) -> None:
        """Fail if the authored stereo RenderProduct is missing or mismatched."""
        stage = sim_utils.get_current_stage()
        root = stage.GetPrimAtPath(self.asset_prim_path)
        for prim in Usd.PrimRange(root):
            if (
                prim.GetTypeName() == "RenderProduct"
                and prim.HasAPI("OmniSensorDepthSensorSingleViewAPI")
                and prim.HasRelationship("camera")
            ):
                targets = prim.GetRelationship("camera").GetTargets()
                if len(targets) == 1 and str(targets[0]) == depth_camera_path:
                    return
        raise RuntimeError(
            "D455 USD contains no single-view stereo template for "
            f"{depth_camera_path}."
        )

    def _disable_embedded_physics(self) -> None:
        # The NVIDIA asset ships as a free rigid body. Since it is nested under
        # the robot's rigid root rather than joined with a fixed joint, every
        # embedded rigid body and collider must be disabled to make the USD a
        # visual/sensor rig without silently changing Phase 3 contact geometry.
        stage = sim_utils.get_current_stage()
        root = stage.GetPrimAtPath(self.asset_prim_path)
        for prim in Usd.PrimRange(root):
            rigid_api = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
            if rigid_api:
                rigid_api.CreateRigidBodyEnabledAttr(False)
            collision_api = UsdPhysics.CollisionAPI.Get(stage, prim.GetPath())
            if collision_api:
                collision_api.CreateCollisionEnabledAttr(False)

    def get_torch_frame(self) -> D455TorchFrame | None:
        rgb_data, _ = self._color_sensor.get_data("rgb")
        depth_data, _ = self._depth_sensor.get_data("depth_sensor_distance")
        if rgb_data is None or depth_data is None:
            return None
        rgb = wp.to_torch(rgb_data)
        depth_m = wp.to_torch(depth_data)
        # The single-camera runtime normally omits a view dimension. Accept a
        # singleton view dimension as well so minor annotator layout changes do
        # not accidentally create a double batch dimension below.
        if rgb.ndim == 4 and rgb.shape[0] == 1:
            rgb = rgb[0]
        if depth_m.ndim == 4 and depth_m.shape[0] == 1:
            depth_m = depth_m[0]
        return D455TorchFrame(rgb=rgb, depth_m=depth_m)


class D455RigBatch:
    """Small batches of independent D455 render products."""

    def __init__(self, rigs: list[D455Rig]):
        if not rigs:
            raise ValueError("At least one D455 rig is required.")
        if any(
            rig.depth_model_parameters != rigs[0].depth_model_parameters
            for rig in rigs[1:]
        ):
            raise RuntimeError("D455 rigs loaded inconsistent depth templates.")
        self.rigs = rigs

    @property
    def depth_model_parameters(self) -> D455DepthModelParameters:
        return self.rigs[0].depth_model_parameters

    @classmethod
    def create_for_hexapod_scene(
        cls,
        num_envs: int,
        translation_b_m: tuple[float, float, float],
        orientation_b_wxyz: tuple[float, float, float, float],
    ) -> "D455RigBatch":
        return cls(
            [
                D455Rig(
                    parent_prim_path=f"/World/envs/env_{index}/Robot/Geometry/{ROOT_LINK_NAME}",
                    translation_b_m=translation_b_m,
                    orientation_b_wxyz=orientation_b_wxyz,
                )
                for index in range(num_envs)
            ]
        )

    def get_torch_frame(self) -> D455TorchFrame | None:
        frames = [rig.get_torch_frame() for rig in self.rigs]
        if any(frame is None for frame in frames):
            return None
        rgb = torch.stack([frame.rgb for frame in frames], dim=0)
        depth = torch.stack([frame.depth_m for frame in frames], dim=0)
        return D455TorchFrame(rgb=rgb, depth_m=depth)
