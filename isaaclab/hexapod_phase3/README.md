# Phase 3 sensor prototype

This README preserves the isolated sensor prototype and its original assumptions.
Use [STATUS.md](../../STATUS.md) for current jobs and admission outcomes, the
[living plan](../../docs/PLAN.md) for architecture, and the
[terrain/sensing plan](../../artifacts/project_review_2026-09-04/TERRAIN_AND_SENSING_PLAN_2026-09-09.md)
for the current sensor workstream. Terrain/perception preparation is authorized
in parallel with Stage 2; this prototype alone qualifies no terrain policy.

The newer [production physical model](../../docs/MKII_FOURBAR_TRAINING.md) has
31 bodies and 18 motor actions. The user-selected
[C study](../../artifacts/length_study_2026-09-09/README.md) is a separate
72.5/126 mm scaled-mock experiment with fixed coxa and transferred CAD physics.
Neither asset can inherit the old mock mounts, 6.3 kg budget or observation
assumptions without validation. Stage 2 remains incomplete, including smooth
all-bearing/path control, bounded motor demand and quiet zero-command standing.
The [forward Benchmark 1](../../artifacts/length_study_2026-09-09/benchmarks/benchmark_01_c_300/README.md)
remains immutable comparison evidence.

This package is isolated from `hexapod_rl`: it registers no Gym task and is not
imported by the Phase 1 or Phase 2 locomotion environments. It combines:

- Isaac Sim 6's NVIDIA-provided RealSense D455 USD and single-view stereo-depth
  postprocessor at 848 x 480 and 30 Hz;
- an Isaac Lab `MultiMeshRayCaster` surrogate for a Livox Mid-360 at 20,000
  rays per 10 Hz scan (200,000 points/s), 360-degree azimuth, and -7 to +52
  degree elevation (360 x 59 degrees, near-hemispherical rather than a true
  hemisphere);
- an Isaac Lab `ImuCfg` at 200 Hz, mounted at the published Mid-360 ICM40609
  offset; and
- pure-PyTorch transport, noise, bias, quantization, pixel/ray loss, and packet
  dropout models. Delayed outputs include validity and sample-age tensors.

No foot/contact sensor is instantiated, and the Phase 1 asset's PhysX contact-
reporting activation is explicitly disabled in this isolated scene.

## Provisional mounts

These are historical prototype transforms, not measured production or final C
brackets. The [new mount study](../../artifacts/sensor_mount_study_2026-09-09/README.md)
uses a hashed serial-CAD snapshot and finds substantial near-foot blind areas;
it also does not establish fit or visibility for the newer physical four-bar.

All poses are relative to the exact imported root rigid body. Physical forward
is body -Y, lateral is body +X, and up is body +Z. This physical convention is
intentional even where older locomotion configurations used root-frame +X as a
command axis.

| Sensor | Root-frame position (m) | Rotation | Status |
| --- | --- | --- | --- |
| D455 | `(0, -0.105, 0.045)` | WXYZ `(0.69636424, 0.12278780, 0.12278780, -0.69636424)` | Unmeasured physical -Y nose mount; optical +X points 20 degrees down from body -Y |
| Mid-360 origin | `(0, 0, 0.075)` | -90 degree yaw XYZW | Unmeasured top-deck mount; nominal sensor +X maps to body -Y |
| ICM40609 | `(0.02329, -0.011, 0.03088)` | -90 degree yaw XYZW | Published `(11, 23.29, -44.12)` mm sensor-frame offset rotated into the body frame; axes still require hardware verification |

The D455 is referenced below a dedicated mount Xform. This preserves all
authored internal camera extrinsics. Its embedded free rigid body and colliders
are disabled because this sensor-only prototype does not add payload dynamics.
The Mid-360 ray caster tracks a separate Xform at the listed LiDAR origin, so
its reported origin and the origin used to calculate scalar ranges are identical.

The D455 rotation is the active composition of a +20-degree body-X pitch and a
-90-degree body-Z yaw. Its nominal optical-forward vector is therefore
`(0, -0.939693, -0.342020)` in the body frame: it remains aligned with physical
forward (-Y) and looks downward. At the Stage2C 0.181 m deck target, the
provisional 45 mm camera offset places the optical origin near 0.226 m, so its
center ray reaches level ground about 0.62 m ahead. This is inside the D455's
published 0.6--6 m ideal range. The pose still requires physical CAD and
occlusion validation.

The nominal sensor +X above belongs to this authored USD mount convention. The
CPU depth adapter uses standard optical +Z forward, +X image-right and +Y
image-down. Preserve and test the full internal-camera transform when bridging
the two; do not reuse the mount quaternion as an optical-frame calibration.

## Published hardware envelope and mass accounting

These are manufacturer-published reference values, not inferred simulator
inertia. `PHASE3_HARDWARE_ACCOUNTING` also includes them in the smoke-test JSON.

| Device | Dimensions | Mass | Power | Field of view | Range |
| --- | --- | ---: | ---: | --- | --- |
| Livox Mid-360 | 65 x 65 x 60 mm | 0.265 kg | 6.5 W average; 14 W cold self-heating peak | 360 degrees horizontal; -7 to +52 degrees elevation (59 degrees vertical) | 0.10 m minimum; 40 m at 10% reflectivity; 70 m at 80% |
| RealSense D455 | 124 x 26 x 29 mm | 0.116 kg | 3.46147 W at the published maximum operating mode | 87 x 58 degrees depth | 0.52 m minimum at maximum resolution; 0.6--6 m ideal |

| Historical mock accounting point | Mass |
| --- | ---: |
| Existing sensor-free body | 1.500 kg |
| Mid-360 + D455 | 0.381 kg |
| Projected body payload | 1.881 kg |
| Existing complete robot | 6.300 kg |
| Projected complete robot | 6.681 kg |

All projected values exclude brackets, cables, and onboard compute. The code
reports this budget but intentionally does **not** apply the sensor masses or an
approximate inertia to the articulation. Phase1/2 physics remains unchanged;
measured completed hardware and CAD inertia are required before a dynamics
update.

The 6.300/6.681 kg arithmetic is retained only to explain this prototype. Reconcile
the current CAD/study mass, physical linkage, sensors, compute, brackets, cabling
and payload in the [payload ledger](../../artifacts/terrain_readiness_2026-09-09/payload_ledger.json)
before changing dynamics; do not double-count already included components.

### Sensor identity and calibration

The user confirmed ownership of a Livox Mid-360 and RealSense D455 on 9 September.
The Mid-360's 360 x 59-degree envelope differs from a Unitree L1 profile; model
names and assumptions remain explicit if alternatives are tested. The prototype's
legacy `requires_physical_label_or_photo_confirmation` field is historical and
must not override that confirmed inventory. Device-specific intrinsics, extrinsics,
clock alignment, noise and loaded timing still require measurement.

## Default non-idealities

| Stream | Rate | Delivery latency | Whole-frame loss | Additional model |
| --- | ---: | ---: | ---: | --- |
| D455 RGB-D | 30 Hz | 66 +/- 10 ms | 1% | RGB noise/loss and 1% depth-pixel loss; the authored stereo model supplies depth noise |
| Mid-360 | 10 Hz | 100 +/- 20 ms | 0.5% | Conservative 3 cm range sigma, 1 mm quantization, 1% ray loss, 0.01% false returns |
| ICM40609 | 200 Hz | 5 +/- 2 ms | 0.1% | TDK-density-derived white noise plus conservative bias and random walk assumptions |

The transport values and bias/random-walk values are conservative engineering
defaults, not measurements from this robot. They should become randomized
ranges after bagged hardware data is available.

## Smoke test

The command below documents the original stationary smoke scene. Before using it,
follow the current [operations runbook](../../docs/OPERATIONS.md) and shared GPU
queue, check STATUS, and validate compatibility with the selected asset and
container. It is not a launch instruction for an occupied Spark.

```bash
cd /home/orionh/IsaacLab
docker compose --env-file docker/.env.base -f docker/docker-compose.yaml \
  --profile base run --rm --name hexapod-phase3-sensor-smoke \
  -w /workspace/hexapod/isaaclab \
  -e PYTHONPATH=/workspace/hexapod/isaaclab \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v /home/orionh/HEXAPOD:/workspace/hexapod:rw \
  --entrypoint /workspace/isaaclab/_isaac_sim/python.sh \
  isaac-lab-base /workspace/hexapod/isaaclab/phase3_sensor_smoke.py \
  --kit_args="--/exts/omni.kit.telemetry/skipDeferredStartup=true --/app/extensions/excluded/4=omni.kit.telemetry" \
  --headless --enable_cameras --device cuda:0 --steps 300 --num_envs 1
```

The script does not train. It verifies the scene contains only LiDAR and IMU
Isaac Lab sensors, checks D455 RGB/depth dimensions and authored stereo-model
parameters, checks the LiDAR optical origin, waits through modeled
latencies, verifies finite/range-bounded data and monotonic model capture times, and
requires nonempty depth and LiDAR returns from a calibration wall/ground. It
prints a JSON report. Keep this D455 smoke test to at most four
environments because it creates separate RGB and depth render products per
environment.

The current user scope includes terrain/perception work in parallel. Runtime
fixtures, full-robot standing, terrain walking and sensor-based policy evaluation
are distinct gates; report their outcomes separately and schedule GPU work through
the existing ownership/queue protocol.

## CPU integration prepared alongside locomotion

[Perception replay](../../artifacts/perception_readiness_2026-09-09/README.md)
provides calibrated-frame depth unprojection, acquisition-time pose interpolation,
timestamped point-cloud input, robot-return masking and height/variance/age/observed
maps. Its step/pit tests are synthetic, with no real-sensor calibration, ROS node
or actor injection. Missing or stale terrain never becomes observed flat support.

The prototype's radial `depth_m` must be explicitly converted before using an
adapter that expects optical-axis Z. The [CPU terrain preparation](../../artifacts/terrain_readiness_2026-09-09/readiness.json)
and [mild curriculum](../../artifacts/terrain_readiness_2026-09-09/mild_curriculum_001/curriculum.json)
provide separate fixtures. Check current runtime results in STATUS; a successful
terrain-only fixture is not proof that the complete C robot stands or walks there.

## Known gaps before policy use

- The Mid-360 pattern is a deterministic low-discrepancy coverage surrogate.
  Livox does not publish its proprietary non-repetitive temporal trajectory, so
  the same directions repeat each frame. Rolling acquisition, intensity,
  material/reflectivity response, and multi-return behavior are not reproduced.
  The later [mock-mesh occlusion probe](../../robot/sensors/README.md#historical-isaac-sim-cross-check)
  adds tracked robot meshes; it does not qualify new-asset sensor visibility.
- The conservative 40 m simulated maximum corresponds to the published range at
  low reflectivity; the 0.1 m blind zone and 0.15-degree one-sigma angular
  precision are represented. Angular perturbations are clipped at three sigma
  and fixed per ray, not time-varying. Constant 3 cm range noise conservatively
  uses the published close-range value rather than modeling range/reflectivity.
- The supplied Mid-360 STEP and FOV STEP are source references only. They have
  not been converted to a collision-safe USD. Published LiDAR/camera mass is
  reported but not applied; brackets, cables, compute, completed assembly mass,
  and measured inertia remain unknown.
- Camera/LiDAR offsets, cable clearance, and body/leg occlusion require a CAD
  mount measurement. The evidence-backed D455 pitch is still provisional. The
  smoke test intentionally fails when it cannot see its calibration target.
- D455 RGB and depth are separate RTX render products; frame-level hardware
  synchronization is not modeled. `depth_m` is radial camera distance from the
  `depth_sensor_distance` annotator, not optical-axis Z depth. The authored
  single-view depth pipeline is still a renderer approximation to physical stereo.
- The experimental camera annotators do not supply a hardware-style sequence
  number through this wrapper. The smoke test verifies the configured 30 Hz
  polling schedule, but cannot prove that every poll produced a distinct RTX frame.
- Ray-cast targets include the smoke-test ground, wall, and nineteen tracked
  leaf meshes of the archived mock. A terrain task must replace/extend them
  with the selected asset's actual static and dynamic mesh targets, and define
  the observation encoder separately.
