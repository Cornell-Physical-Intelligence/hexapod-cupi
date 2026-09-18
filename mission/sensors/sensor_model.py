"""Latency, noise, bias, and dropout models for Phase 3 sensor observations.

Isaac Lab's sensor configs model capture rate and ideal sensor geometry.  This
module deliberately corrupts *copies* of those ideal buffers and delays their
delivery.  The returned ``valid`` and ``age_s`` tensors let a policy distinguish
fresh samples from held samples after packet loss.

The implementation is independent of Isaac Sim and can be unit-tested with
ordinary PyTorch.  All public inputs use a leading environment dimension.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field

import torch

SENSOR_TRANSPORT_SCHEMA = "phase3_sensor_transport_v2"


def _finite_clock(name: str, value: float) -> float:
    """Simulation seconds within one epoch; reject invalid input before mutation."""
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite, non-negative simulation seconds.")
    return value


def _check_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


@dataclass(frozen=True)
class TransportModelCfg:
    """Capture cadence and communication behavior for one sensor stream."""

    sample_period_s: float
    latency_s: float
    latency_jitter_s: float
    frame_dropout_probability: float
    stale_after_s: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (
            self.sample_period_s, self.latency_s, self.latency_jitter_s,
            self.frame_dropout_probability, self.stale_after_s,
        )):
            raise ValueError("Transport parameters must all be finite.")
        if self.sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be positive.")
        if self.latency_s < 0.0 or self.latency_jitter_s < 0.0:
            raise ValueError("Latency and latency jitter must be non-negative.")
        if self.latency_jitter_s > self.latency_s:
            raise ValueError("Latency jitter cannot exceed nominal latency.")
        if self.stale_after_s < self.latency_s + self.latency_jitter_s:
            raise ValueError("stale_after_s must cover the worst configured latency.")
        _check_probability("frame_dropout_probability", self.frame_dropout_probability)


@dataclass(frozen=True)
class CameraModelCfg:
    """Nominal indoor RGB-D camera observation model."""

    transport: TransportModelCfg = field(
        default_factory=lambda: TransportModelCfg(
            sample_period_s=1.0 / 30.0,
            latency_s=0.066,
            latency_jitter_s=0.010,
            frame_dropout_probability=0.010,
            stale_after_s=0.200,
        )
    )
    rgb_noise_std: float = 2.0 / 255.0
    rgb_pixel_dropout_probability: float = 0.002
    # The NVIDIA D455 USD already models disparity, stereo noise, and
    # outliers. Extra additive depth noise defaults to zero to avoid counting it
    # twice; transport and pixel loss are still modeled here.
    depth_noise_base_m: float = 0.0
    depth_noise_per_m: float = 0.0
    depth_pixel_dropout_probability: float = 0.010
    min_depth_m: float = 0.52
    max_depth_m: float = 6.0

    def __post_init__(self) -> None:
        _check_probability(
            "rgb_pixel_dropout_probability", self.rgb_pixel_dropout_probability
        )
        _check_probability(
            "depth_pixel_dropout_probability", self.depth_pixel_dropout_probability
        )
        if self.rgb_noise_std < 0.0:
            raise ValueError("rgb_noise_std must be non-negative.")
        if self.depth_noise_base_m < 0.0 or self.depth_noise_per_m < 0.0:
            raise ValueError("Depth noise terms must be non-negative.")
        if not 0.0 < self.min_depth_m < self.max_depth_m:
            raise ValueError("Camera depth limits are invalid.")


@dataclass(frozen=True)
class LidarModelCfg:
    """Nominal indoor Livox Mid-360 observation model."""

    transport: TransportModelCfg = field(
        default_factory=lambda: TransportModelCfg(
            sample_period_s=0.100,
            latency_s=0.100,
            latency_jitter_s=0.020,
            frame_dropout_probability=0.005,
            stale_after_s=0.350,
        )
    )
    # Conservative constant approximation to the published one-sigma envelope:
    # <=2 cm at 10 m and <=3 cm at 0.2 m. Hardware data should replace this
    # worst-case indoor value with a range/reflectivity-dependent model.
    range_noise_base_m: float = 0.030
    range_noise_per_m: float = 0.0
    range_quantization_m: float = 0.001
    ray_dropout_probability: float = 0.010
    false_return_probability: float = 1.0e-4
    min_range_m: float = 0.10
    max_range_m: float = 40.0

    def __post_init__(self) -> None:
        _check_probability("ray_dropout_probability", self.ray_dropout_probability)
        _check_probability("false_return_probability", self.false_return_probability)
        if self.range_noise_base_m < 0.0 or self.range_noise_per_m < 0.0:
            raise ValueError("LiDAR range noise terms must be non-negative.")
        if self.range_quantization_m <= 0.0:
            raise ValueError("range_quantization_m must be positive.")
        if not 0.0 < self.min_range_m < self.max_range_m:
            raise ValueError("LiDAR range limits are invalid.")


@dataclass(frozen=True)
class ImuModelCfg:
    """Nominal MEMS IMU noise and bias model."""

    transport: TransportModelCfg = field(
        default_factory=lambda: TransportModelCfg(
            sample_period_s=0.005,
            latency_s=0.005,
            latency_jitter_s=0.002,
            frame_dropout_probability=0.001,
            stale_after_s=0.030,
        )
    )
    # ICM-40609-D white-noise densities (4.5 mdps/sqrt(Hz), 100 ug/sqrt(Hz))
    # converted to per-sample standard deviation assuming 100 Hz noise
    # bandwidth at the Mid-360's 200 Hz output rate.
    gyro_noise_std_rad_s: float = 7.9e-4
    accel_noise_std_m_s2: float = 9.81e-3
    gyro_bias_initial_std_rad_s: float = 0.010
    accel_bias_initial_std_m_s2: float = 0.080
    gyro_bias_random_walk_rad_s_sqrt_s: float = 2.0e-4
    accel_bias_random_walk_m_s2_sqrt_s: float = 2.0e-3

    def __post_init__(self) -> None:
        values = (
            self.gyro_noise_std_rad_s,
            self.accel_noise_std_m_s2,
            self.gyro_bias_initial_std_rad_s,
            self.accel_bias_initial_std_m_s2,
            self.gyro_bias_random_walk_rad_s_sqrt_s,
            self.accel_bias_random_walk_m_s2_sqrt_s,
        )
        if any(value < 0.0 for value in values):
            raise ValueError("IMU noise and bias parameters must be non-negative.")


@dataclass(frozen=True)
class Phase3SensorModelCfg:
    camera: CameraModelCfg = field(default_factory=CameraModelCfg)
    lidar: LidarModelCfg = field(default_factory=LidarModelCfg)
    imu: ImuModelCfg = field(default_factory=ImuModelCfg)


DEFAULT_PHASE3_SENSOR_MODEL_CFG = Phase3SensorModelCfg()


@dataclass(frozen=True)
class Phase3SensorFrame:
    """Latest delivered frame for one stream.

    ``values`` are read-only references owned by the model. ``valid`` is false
    during latency warm-up and once a held sample exceeds ``stale_after_s``.
    ``age_s`` is measured from physical capture time, not delivery time.
    """

    values: dict[str, torch.Tensor]
    valid: torch.Tensor
    age_s: torch.Tensor
    capture_time_s: torch.Tensor


@dataclass(order=True)
class _QueuedFrame:
    ready_time_s: float
    sequence: int
    capture_time_s: float = field(compare=False)
    values: dict[str, torch.Tensor] = field(compare=False)
    valid: torch.Tensor = field(compare=False)


class _TensorRandom:
    def __init__(self, seed: int):
        self._seed = int(seed)
        self._generators: dict[str, torch.Generator] = {}

    def _generator(self, device: torch.device) -> torch.Generator:
        key = str(device)
        if key not in self._generators:
            generator = torch.Generator(device=device)
            device_offset = sum((index + 1) * ord(char) for index, char in enumerate(key))
            generator.manual_seed(self._seed + device_offset)
            self._generators[key] = generator
        return self._generators[key]

    def uniform(self, shape: tuple[int, ...], like: torch.Tensor) -> torch.Tensor:
        return torch.rand(
            shape,
            dtype=torch.float32,
            device=like.device,
            generator=self._generator(like.device),
        )

    def normal(self, like: torch.Tensor) -> torch.Tensor:
        return torch.randn(
            like.shape,
            dtype=like.dtype,
            device=like.device,
            generator=self._generator(like.device),
        )


class _DelayedStream:
    def __init__(self, cfg: TransportModelCfg, seed: int):
        self.cfg = cfg
        self._jitter_rng = random.Random(seed)
        self._pending: list[_QueuedFrame] = []
        self._sequence = 0
        self._latest_values: dict[str, torch.Tensor] | None = None
        self._has_value: torch.Tensor | None = None
        self._capture_time_s: torch.Tensor | None = None
        # Float64 ordering is separate from the unchanged float32 public age
        # metadata. Distinct Python capture times must not become a timestamp tie
        # merely because their public float32 representations round together.
        self._order_time_s: torch.Tensor | None = None
        self._order_sequence: torch.Tensor | None = None
        self._last_capture_s: float | None = None
        self._last_read_s: float | None = None
        self._payload_layout: dict | None = None

    def _validate_capture_time(self, capture_time_s: float) -> float:
        capture_time_s = _finite_clock("capture_time_s", capture_time_s)
        if self._last_capture_s is not None and capture_time_s < self._last_capture_s:
            raise ValueError("A stream capture clock cannot rewind within an epoch.")
        return capture_time_s

    def _validate_read_time(self, now_s: float) -> float:
        now_s = _finite_clock("now_s", now_s)
        if self._last_read_s is not None and now_s < self._last_read_s:
            raise ValueError("A stream read clock cannot rewind within an epoch.")
        return now_s

    @torch.inference_mode(False)
    @torch.no_grad()
    def enqueue(
        self,
        values: dict[str, torch.Tensor],
        capture_time_s: float,
        valid: torch.Tensor,
    ) -> None:
        capture_time_s = self._validate_capture_time(capture_time_s)
        if not values:
            raise ValueError("A delayed sensor frame must contain at least one value.")
        batch_size = next(iter(values.values())).shape[0]
        if valid.shape != (batch_size,) or valid.dtype != torch.bool:
            raise ValueError("Frame validity must be a boolean tensor of shape (num_envs,).")
        if any(value.shape[0] != batch_size for value in values.values()):
            raise ValueError("All sensor values must share the leading environment dimension.")
        if any(value.device != valid.device for value in values.values()):
            raise ValueError("Sensor values and validity must reside on the same device.")
        layout = {name: (value.shape, value.dtype, value.device) for name, value in values.items()}
        if self._payload_layout is not None and layout != self._payload_layout:
            raise ValueError("A stream cannot change payload fields, shapes, dtypes or devices within an epoch.")

        latency_s = self.cfg.latency_s + self._jitter_rng.uniform(
            -self.cfg.latency_jitter_s, self.cfg.latency_jitter_s
        )
        queued = _QueuedFrame(
            ready_time_s=capture_time_s + max(0.0, latency_s),
            sequence=self._sequence,
            capture_time_s=float(capture_time_s),
            values={name: value.detach().clone() for name, value in values.items()},
            valid=valid.detach().clone(),
        )
        self._sequence += 1
        heapq.heappush(self._pending, queued)
        self._last_capture_s = capture_time_s
        self._payload_layout = layout

    @torch.inference_mode(False)
    @torch.no_grad()
    def read(self, now_s: float) -> Phase3SensorFrame | None:
        now_s = self._validate_read_time(now_s)
        self._last_read_s = now_s
        # Preserve the historical readiness tolerance, but never reveal a sample
        # captured after now even if it lies inside that numerical tolerance.
        deferred = []
        while self._pending and self._pending[0].ready_time_s <= now_s + 1.0e-9:
            queued = heapq.heappop(self._pending)
            if queued.capture_time_s > now_s:
                deferred.append(queued)
                continue
            if self._latest_values is None:
                self._latest_values = {
                    name: torch.zeros_like(value) for name, value in queued.values.items()
                }
                self._has_value = torch.zeros_like(queued.valid)
                self._capture_time_s = torch.full(
                    queued.valid.shape,
                    -torch.inf,
                    dtype=torch.float32,
                    device=queued.valid.device,
                )
                self._order_time_s = torch.full_like(self._capture_time_s, -torch.inf, dtype=torch.float64)
                self._order_sequence = torch.full_like(queued.valid, -1, dtype=torch.int64)
            if set(queued.values) != set(self._latest_values):
                raise RuntimeError("A sensor stream changed its payload fields after initialization.")
            take = queued.valid & (
                (queued.capture_time_s > self._order_time_s)
                | ((queued.capture_time_s == self._order_time_s) & (queued.sequence > self._order_sequence))
            )
            for name, value in queued.values.items():
                mask_shape = (value.shape[0],) + (1,) * (value.ndim - 1)
                self._latest_values[name] = torch.where(
                    take.reshape(mask_shape), value, self._latest_values[name]
                )
            self._has_value |= take
            self._capture_time_s = torch.where(
                take,
                torch.full_like(self._capture_time_s, queued.capture_time_s),
                self._capture_time_s,
            )
            self._order_time_s = torch.where(take, queued.capture_time_s, self._order_time_s)
            self._order_sequence = torch.where(take, queued.sequence, self._order_sequence)
        for queued in deferred:
            heapq.heappush(self._pending, queued)

        if self._latest_values is None:
            return None
        age_s = torch.where(
            self._has_value,
            torch.full_like(self._capture_time_s, float(now_s)) - self._capture_time_s,
            torch.full_like(self._capture_time_s, torch.inf),
        )
        valid = self._has_value & (age_s >= 0.0) & (age_s <= self.cfg.stale_after_s)
        return Phase3SensorFrame(
            values=self._latest_values,
            valid=valid,
            age_s=age_s,
            capture_time_s=self._capture_time_s,
        )

    @torch.inference_mode(False)
    @torch.no_grad()
    def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        if env_ids is None:
            self._pending.clear()
            self._latest_values = None
            self._has_value = None
            self._capture_time_s = None
            self._order_time_s = None
            self._order_sequence = None
            self._last_capture_s = None
            self._last_read_s = None
            self._payload_layout = None
            return
        if self._has_value is not None:
            device = self._has_value.device
        elif self._pending:
            # A partial reset can happen during latency warm-up, before any
            # frame is delivered.  Pending pre-reset samples must still be
            # invalidated or they will leak into the new episode later.
            device = self._pending[0].valid.device
        else:
            return
        indices = torch.as_tensor(env_ids, dtype=torch.long, device=device)
        for queued in self._pending:
            queued.valid[indices] = False
        if self._has_value is None:
            return
        for value in self._latest_values.values():
            value[indices] = 0
        self._has_value[indices] = False
        self._capture_time_s[indices] = -torch.inf
        self._order_time_s[indices] = -torch.inf
        self._order_sequence[indices] = -1


class Phase3SensorModel:
    """Post-process ideal Isaac Lab camera, ray-caster, and IMU buffers."""

    def __init__(
        self,
        cfg: Phase3SensorModelCfg = DEFAULT_PHASE3_SENSOR_MODEL_CFG,
        seed: int = 0,
    ):
        self.cfg = cfg
        self._random = _TensorRandom(seed)
        self._camera_stream = _DelayedStream(cfg.camera.transport, seed + 101)
        self._lidar_stream = _DelayedStream(cfg.lidar.transport, seed + 211)
        self._imu_stream = _DelayedStream(cfg.imu.transport, seed + 307)
        self._gyro_bias: torch.Tensor | None = None
        self._accel_bias: torch.Tensor | None = None
        self._imu_bias_initialized: torch.Tensor | None = None

    def _frame_valid(
        self, reference: torch.Tensor, dropout_probability: float
    ) -> torch.Tensor:
        return self._random.uniform((reference.shape[0],), reference) >= dropout_probability

    @torch.inference_mode(False)
    @torch.no_grad()
    def capture_camera(
        self,
        rgb: torch.Tensor,
        depth_m: torch.Tensor,
        capture_time_s: float,
    ) -> None:
        """Capture RGB and optical-ray distance from the D455 depth annotator."""
        capture_time_s = self._camera_stream._validate_capture_time(capture_time_s)
        if rgb.ndim != 4 or rgb.shape[-1] not in (3, 4):
            raise ValueError("RGB input must have shape (N, H, W, 3 or 4).")
        if depth_m.ndim not in (3, 4) or depth_m.shape[:3] != rgb.shape[:3]:
            raise ValueError("Depth input must have shape (N, H, W[, 1]).")
        if depth_m.ndim == 4 and depth_m.shape[-1] != 1:
            raise ValueError("Depth input must have exactly one channel.")
        if depth_m.ndim == 3:
            depth_m = depth_m.unsqueeze(-1)

        camera_cfg = self.cfg.camera
        rgb_float = rgb[..., :3].to(torch.float32)
        if rgb.dtype == torch.uint8:
            rgb_float = rgb_float / 255.0
        rgb_float = torch.clamp(
            rgb_float + self._random.normal(rgb_float) * camera_cfg.rgb_noise_std,
            0.0,
            1.0,
        )
        rgb_pixel_valid = self._random.uniform(rgb_float.shape[:-1], rgb_float) >= (
            camera_cfg.rgb_pixel_dropout_probability
        )
        rgb_float = torch.where(rgb_pixel_valid.unsqueeze(-1), rgb_float, 0.0)

        depth_float = depth_m.to(torch.float32)
        ideal_depth_valid = (
            torch.isfinite(depth_float)
            & (depth_float >= camera_cfg.min_depth_m)
            & (depth_float <= camera_cfg.max_depth_m)
        )
        safe_depth = torch.where(
            ideal_depth_valid,
            depth_float,
            torch.full_like(depth_float, camera_cfg.max_depth_m),
        )
        depth_sigma = (
            camera_cfg.depth_noise_base_m
            + camera_cfg.depth_noise_per_m * safe_depth
        )
        noisy_depth = torch.clamp(
            safe_depth + self._random.normal(safe_depth) * depth_sigma,
            camera_cfg.min_depth_m,
            camera_cfg.max_depth_m,
        )
        depth_pixel_valid = ideal_depth_valid & (
            self._random.uniform(depth_float.shape, depth_float)
            >= camera_cfg.depth_pixel_dropout_probability
        )
        noisy_depth = torch.where(
            depth_pixel_valid,
            noisy_depth,
            torch.full_like(noisy_depth, camera_cfg.max_depth_m),
        )
        frame_valid = self._frame_valid(
            rgb_float, camera_cfg.transport.frame_dropout_probability
        )
        self._camera_stream.enqueue(
            {
                "rgb": rgb_float,
                "depth_m": noisy_depth,
                "depth_valid": depth_pixel_valid,
            },
            capture_time_s,
            frame_valid,
        )

    @torch.inference_mode(False)
    @torch.no_grad()
    def capture_lidar(
        self,
        ray_hits_w: torch.Tensor,
        sensor_pos_w: torch.Tensor,
        capture_time_s: float,
    ) -> None:
        """Convert ideal world-frame ray hits to corrupted scalar ranges."""
        capture_time_s = self._lidar_stream._validate_capture_time(capture_time_s)
        if ray_hits_w.ndim != 3 or ray_hits_w.shape[-1] != 3:
            raise ValueError("LiDAR hits must have shape (N, num_rays, 3).")
        if sensor_pos_w.shape != (ray_hits_w.shape[0], 3):
            raise ValueError("LiDAR position must have shape (N, 3).")
        lidar_cfg = self.cfg.lidar
        finite_hit = torch.isfinite(ray_hits_w).all(dim=-1)
        safe_hits_w = torch.where(
            finite_hit.unsqueeze(-1), ray_hits_w, sensor_pos_w.unsqueeze(1)
        )
        ranges_m = torch.linalg.norm(safe_hits_w - sensor_pos_w.unsqueeze(1), dim=-1)
        ideal_hit_valid = (
            finite_hit
            & (ranges_m >= lidar_cfg.min_range_m)
            & (ranges_m <= lidar_cfg.max_range_m)
        )
        safe_ranges_m = torch.where(
            ideal_hit_valid,
            ranges_m,
            torch.full_like(ranges_m, lidar_cfg.max_range_m),
        )
        range_sigma = (
            lidar_cfg.range_noise_base_m
            + lidar_cfg.range_noise_per_m * safe_ranges_m
        )
        noisy_ranges_m = safe_ranges_m + self._random.normal(safe_ranges_m) * range_sigma
        noisy_ranges_m = (
            torch.round(noisy_ranges_m / lidar_cfg.range_quantization_m)
            * lidar_cfg.range_quantization_m
        )
        ray_valid = ideal_hit_valid & (
            self._random.uniform(ranges_m.shape, ranges_m)
            >= lidar_cfg.ray_dropout_probability
        )
        false_return = (~ideal_hit_valid) & (
            self._random.uniform(ranges_m.shape, ranges_m)
            < lidar_cfg.false_return_probability
        )
        false_ranges_m = lidar_cfg.min_range_m + self._random.uniform(
            ranges_m.shape, ranges_m
        ) * (lidar_cfg.max_range_m - lidar_cfg.min_range_m)
        noisy_ranges_m = torch.where(false_return, false_ranges_m, noisy_ranges_m)
        ray_valid |= false_return
        noisy_ranges_m = torch.where(
            ray_valid,
            torch.clamp(
                noisy_ranges_m, lidar_cfg.min_range_m, lidar_cfg.max_range_m
            ),
            torch.full_like(noisy_ranges_m, lidar_cfg.max_range_m),
        )
        frame_valid = self._frame_valid(
            ranges_m, lidar_cfg.transport.frame_dropout_probability
        )
        self._lidar_stream.enqueue(
            {"ranges_m": noisy_ranges_m, "hit_valid": ray_valid},
            capture_time_s,
            frame_valid,
        )

    def _initialize_imu_biases(self, reference: torch.Tensor) -> None:
        num_envs = reference.shape[0]
        expected_shape = (num_envs, 3)
        if self._gyro_bias is None or self._gyro_bias.shape != expected_shape:
            self._gyro_bias = torch.zeros_like(reference)
            self._accel_bias = torch.zeros_like(reference)
            self._imu_bias_initialized = torch.zeros(
                num_envs, dtype=torch.bool, device=reference.device
            )
        uninitialized = ~self._imu_bias_initialized
        if torch.any(uninitialized):
            gyro_initial = (
                self._random.normal(reference)
                * self.cfg.imu.gyro_bias_initial_std_rad_s
            )
            accel_initial = (
                self._random.normal(reference)
                * self.cfg.imu.accel_bias_initial_std_m_s2
            )
            self._gyro_bias = torch.where(
                uninitialized.unsqueeze(-1), gyro_initial, self._gyro_bias
            )
            self._accel_bias = torch.where(
                uninitialized.unsqueeze(-1), accel_initial, self._accel_bias
            )
            self._imu_bias_initialized |= uninitialized

    @torch.inference_mode(False)
    @torch.no_grad()
    def capture_imu(
        self,
        angular_velocity_rad_s: torch.Tensor,
        linear_acceleration_m_s2: torch.Tensor,
        capture_time_s: float,
        dt_s: float | None = None,
    ) -> None:
        """Capture ideal body-frame angular velocity and proper acceleration."""
        capture_time_s = self._imu_stream._validate_capture_time(capture_time_s)
        if angular_velocity_rad_s.ndim != 2 or angular_velocity_rad_s.shape[1] != 3:
            raise ValueError("IMU angular velocity must have shape (N, 3).")
        if linear_acceleration_m_s2.shape != angular_velocity_rad_s.shape:
            raise ValueError("IMU acceleration must have shape (N, 3).")
        if angular_velocity_rad_s.device != linear_acceleration_m_s2.device:
            raise ValueError("IMU inputs must reside on the same device.")
        imu_cfg = self.cfg.imu
        dt_s = imu_cfg.transport.sample_period_s if dt_s is None else float(dt_s)
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("IMU capture dt_s must be finite and positive.")

        gyro = angular_velocity_rad_s.to(torch.float32)
        accel = linear_acceleration_m_s2.to(torch.float32)
        self._initialize_imu_biases(gyro)
        sqrt_dt = math.sqrt(dt_s)
        self._gyro_bias += (
            self._random.normal(self._gyro_bias)
            * imu_cfg.gyro_bias_random_walk_rad_s_sqrt_s
            * sqrt_dt
        )
        self._accel_bias += (
            self._random.normal(self._accel_bias)
            * imu_cfg.accel_bias_random_walk_m_s2_sqrt_s
            * sqrt_dt
        )
        measured_gyro = (
            gyro
            + self._gyro_bias
            + self._random.normal(gyro) * imu_cfg.gyro_noise_std_rad_s
        )
        measured_accel = (
            accel
            + self._accel_bias
            + self._random.normal(accel) * imu_cfg.accel_noise_std_m_s2
        )
        frame_valid = self._frame_valid(
            gyro, imu_cfg.transport.frame_dropout_probability
        )
        self._imu_stream.enqueue(
            {
                "angular_velocity_rad_s": measured_gyro,
                "linear_acceleration_m_s2": measured_accel,
            },
            capture_time_s,
            frame_valid,
        )

    def read(self, now_s: float) -> dict[str, Phase3SensorFrame]:
        """Release all frames whose modeled transport delay has elapsed."""
        # Validate all clocks before any stream drains or advances its read time.
        for stream in (self._camera_stream, self._lidar_stream, self._imu_stream):
            stream._validate_read_time(now_s)
        outputs = {
            "camera": self._camera_stream.read(now_s),
            "lidar": self._lidar_stream.read(now_s),
            "imu": self._imu_stream.read(now_s),
        }
        return {name: frame for name, frame in outputs.items() if frame is not None}

    @torch.inference_mode(False)
    @torch.no_grad()
    def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        """Reset delayed frames and IMU biases globally or for selected environments."""
        self._camera_stream.reset(env_ids)
        self._lidar_stream.reset(env_ids)
        self._imu_stream.reset(env_ids)
        if self._imu_bias_initialized is None:
            return
        if env_ids is None:
            self._gyro_bias = None
            self._accel_bias = None
            self._imu_bias_initialized = None
        else:
            indices = torch.as_tensor(
                env_ids, dtype=torch.long, device=self._imu_bias_initialized.device
            )
            self._gyro_bias[indices] = 0.0
            self._accel_bias[indices] = 0.0
            self._imu_bias_initialized[indices] = False
