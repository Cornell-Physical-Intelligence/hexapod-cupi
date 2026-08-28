# Autonomy architecture and layer contracts

Durable reference for how the autonomy system is layered, what each layer
consumes and produces, and which interfaces are frozen. `docs/TRAINING.md`
describes the locomotion task in detail; this file describes everything above
and beside it, and the seams between them. Milestones and ownership live in
`docs/ROADMAP.md`; the reasoning behind the load-bearing choices lives in
`docs/decisions/`.

## 1. Mission

The Hexapod MKII must accept an **area of interest** in unknown terrain (for
example a forest), possibly designated by an air unit, and navigate to and
explore that area autonomously, safely, and effectively, in places that are
difficult for a person to reach. Everything below is in service of that.

## 2. Three layers, three contracts

```text
             air unit / operator
                    |
                    |  AreaOfInterest (polygon + priority)         [contract C4]
                    v
   +---------------------------------------------+
   |  NAVIGATION / AUTONOMY  (hexapod_nav, ROS 2) |  global map + pose -> path,
   |  exploration, traversability, safety monitor |  exploration frontier, stop
   +---------------------------------------------+
                    |
                    |  VelocityCommand [vx, vy, yaw_rate] @ 50 Hz   [contract C1]
                    v
   +---------------------------------------------+
   |  LOCOMOTION POLICY  (hexapod_env -> runtime) |  proprioception + local
   |  trained in Isaac Lab, runs at 50 Hz         |  height scan -> 18 targets
   +---------------------------------------------+
                    ^
                    |  HeightScan (body-frame local elevation grid)  [contract C2]
                    |
   +---------------------------------------------+
   |  PERCEPTION  (hexapod_perception, ROS 2)     |  Mid-360 + IMU -> LIO pose,
   |  lidar-inertial odometry, elevation mapping, |  registered cloud, local
   |  global occupancy / traversability           |  elevation map, global map
   +---------------------------------------------+
                    |
                    |  Odometry + maps                                [contract C3]
                    v
             navigation layer (above)
```

The locomotion policy never sees a point cloud. Perception never emits joint
targets. Navigation never reaches below the velocity command. Each team can
build, test, and replace its layer without the others being present, because
each seam is a small, versioned data contract with a validator.

## 3. Why the policy consumes a height scan, not the lidar

This is decision `docs/decisions/0002-policy-consumes-height-scan.md`. The
short version:

1. **Geometry.** The Mid-360's field of view is 360 x 59 degrees, from -7 to
   +52 degrees elevation. At the Stage2C deck height of 0.185 m the nearest
   possible ground return is `h / tan(7 deg)`, about 1.7 m flush and further
   on any mast (`robot/sensors/README.md`, mount placement study). The sensor
   cannot see the robot's own footing. Footing is the job of a downward-facing
   depth sensor (the provisional D455) and of the elevation map that perception
   accumulates as the robot moves.
2. **Standard practice.** Learned rough-terrain locomotion on quadrupeds and
   hexapods conditions the policy on proprioception plus a small body-frame
   height grid sampled around the feet. In Isaac Lab that is a `RayCaster`
   with a `GridPatternCfg` against the terrain during training; on the robot
   it is the same grid read out of a perception elevation map. The policy does
   not need to know which sensor produced it.
3. **Decoupling.** The RL team can train on terrain today with a simulated
   height scan and no lidar model at all. The perception team can build the
   real pipeline on bench data with no policy at all. The integration point is
   one grid of floats with a fixed frame, size, and resolution.
4. **Sim-to-real.** A height grid is a far narrower distribution to randomize
   over (noise, dropout, latency, map lag) than a 200 kpoint/s scan pattern
   whose real temporal trajectory Livox does not publish.

The simulated lidar still has two jobs: an integration scene where the
perception pipeline runs against the raycaster surrogate on a walking robot,
and mount/occlusion validation. It is not a training input.

## 4. Contracts

Frozen means the same thing it means in `CLAUDE.md`: a change is a new version
beside the old one, never an edit. Every contract has a stdlib-only validator
in `packages/hexapod_core` (or a message definition plus a validator for the
ROS 2 side) and a contract test under `isaaclab/tests/`.

| Id | Contract | Producer -> consumer | Status |
| --- | --- | --- | --- |
| C1 | `VelocityCommand` — `[vx, vy, yaw_rate]`, validated against the trained envelope, 50 Hz | navigation -> policy | **Frozen** (`hexapod_core.command`, `hexapod_nav.producer.CommandProducer`) |
| O1 | Observation v1 — 66-dim proprioceptive layout | runtime -> policy | **Frozen** (`hexapod_core.observation`) |
| C2 | `HeightScan` / Observation v2 — body-frame local elevation grid appended to O1: grid extent, cell size, frame origin, height clip, missing-cell sentinel, max age | perception -> policy (sim: raycaster -> policy) | **To define in M2** — see `docs/ROADMAP.md` |
| C3 | Odometry and maps — `nav_msgs/Odometry` in `odom`, registered cloud, `grid_map_msgs` elevation layer in `odom`/`map`, occupancy or traversability layer | perception -> navigation | **To define in P1** |
| C4 | `AreaOfInterest` — polygon (WGS84 or local ENU with a datum), priority, optional no-go polygons, revision id | air unit / operator -> navigation | **To define in M4**; the transport (radio link, message format) is a hardware decision |
| A1 | Action interface — 18 joint offsets, clip, scale, slew | policy -> runtime -> motors | **Frozen** (`hexapod_core.action`, `hexapod_runtime.action_pipeline`) |

Rules that follow from the table:

- A locomotion checkpoint is tagged with the observation version it was trained
  against. An O1 checkpoint cannot be driven by a C2 producer and vice versa.
- Perception publishes C2 at policy rate with a timestamp; the runtime refuses
  a scan older than the contract's max age and falls back to a flat-ground
  scan plus a reduced command envelope. That fallback is part of the contract,
  not an afterthought.
- Navigation may only emit commands inside the envelope of the checkpoint
  actually loaded; the runtime rejects the rest (`VelocityCommand.validate`).

## 5. Physical robot and runtime topology (decisions pending)

```text
Livox Mid-360 --(Ethernet)--> companion computer --(CAN)--> 18 x RobStride RS05
RealSense D455 --(USB3)-----^        |
                                     +-- ROS 2 graph: driver, LIO, mapping, nav
                                     +-- hexapod_runtime: obs builder, policy
                                         backend, action pipeline, watchdog
air-unit link --(radio/WiFi)---------^
```

Open hardware decisions that block sim-to-real, in the order they bite:

1. Companion computer (GPU-capable, e.g. a Jetson-class board, if the elevation
   mapping and policy inference run on-robot) and its power budget next to the
   Mid-360's 6.5 W average / 14 W cold peak.
2. Mid-360 mount: the placement study's 75 mm upright mast is the current
   recommendation; confirm in Isaac Sim, then design the 100 x 100 mm, 3 mm
   metal plate Livox requires.
3. Measured masses, inertias, and the actuator-to-joint mapping on the
   assembled robot (`docs/TRAINING.md` §7).
4. Air-unit link and the C4 transport.

## 6. What the simulator is for, per layer

| Layer | Isaac Sim / Isaac Lab role |
| --- | --- |
| Locomotion | Training and formal screening (existing). Terrain curriculum plus a simulated height scan (M2). Domain randomization of everything in `docs/TRAINING.md` §7. |
| Perception | Integration scene only: the raycaster Mid-360 surrogate and the D455 twin on the walking robot, producing bags the perception stack must process to the same C2/C3 outputs it produces from hardware bags. Not a training input. |
| Navigation | Closed-loop scenario tests: forest-like scenes with an AOI, exploration coverage and safety metrics, policy in the loop through C1. |
| Hardware bring-up | Observation/action parity against the runtime (`isaaclab/tests/test_runtime_parity.py`) before any powered walking. |

## 7. Repository placement

```text
packages/hexapod_core/        contracts C1, O1, A1 today; C2 validator lands here
packages/hexapod_env/         locomotion task; terrain + height-scan sensor in M2
packages/hexapod_runtime/     on-robot policy runtime; C2 ingestion + fallback
packages/hexapod_nav/         CommandProducer seam; planners and exploration land here
ros2_ws/                      (to create) perception and navigation ROS 2 packages
                              - hexapod_perception: driver bring-up, LIO, mapping
                              - hexapod_msgs: C2/C3/C4 message definitions
                              - hexapod_bringup: launch files, parameter sets
robot/                        assets; the M0 rebuilt robot goes in a new directory
docs/decisions/               ADRs, numbered, never edited after acceptance
```

Python packages that are stdlib-only stay stdlib-only. ROS 2 code lives under
`ros2_ws/` so the training container and the bare-interpreter test suite never
need ROS installed.
