# Plan: architecture, milestones, decisions

One file for where the program is going and why. Rewritten in place; a plan
change edits the relevant section and adds a dated entry to §7. The mission
itself is `dar.md`; run-level state is `STATUS.md`; the locomotion task is
`docs/TRAINING.md`; the Spark procedure is `docs/OPERATIONS.md`.

Status: draft for team ratification (2026-08-27, revised 2026-09-04). The four
decisions in §7 are `proposed`; accepting them is the first team action.

## 1. Layers and contracts

```text
             operator map page (phone / laptop)
                    |
                    |  AreaOfInterest (drawn polygon, no-go, revision)  [C4]
                    v
   +---------------------------------------------+
   |  NAVIGATION  (hexapod_nav, ROS 2)            |  polygon + map + pose ->
   |  coverage sweep, traversability, geofence,   |  sweep path, velocity
   |  safety monitor, data-collection trigger     |  commands, record/stop
   +---------------------------------------------+
                    |
                    |  VelocityCommand [vx, vy, yaw_rate] @ 50 Hz        [C1]
                    v
   +---------------------------------------------+
   |  LOCOMOTION POLICY  (hexapod_env -> runtime) |  proprioception + local
   |  trained in Isaac Lab, runs at 50 Hz         |  height scan -> 18 targets
   +---------------------------------------------+
                    ^
                    |  HeightScan (body-frame local elevation grid)      [C2]
                    |
   +---------------------------------------------+
   |  PERCEPTION  (hexapod_perception, ROS 2)     |  Mid-360 + IMU + GPS ->
   |  lidar-inertial odometry, GPS fusion,        |  map-frame pose, registered
   |  elevation mapping, traversability           |  cloud, elevation map
   +---------------------------------------------+
                    |
                    |  Map-frame pose + maps                             [C3]
                    v
             navigation layer (above)
```

The policy never sees a point cloud. Perception never emits joint targets.
Navigation never reaches below the velocity command. Each layer is built and
tested without the others because every seam is a small versioned contract
with a stdlib validator in `packages/hexapod_core` (or a ROS 2 message plus a
validator) and a contract test under `isaaclab/tests/`. Frozen means a change
is a new version beside the old one, never an edit.

| Id | Contract | Producer -> consumer | Status |
| --- | --- | --- | --- |
| C1 | `VelocityCommand` `[vx, vy, yaw_rate]`, validated against the trained envelope, 50 Hz | navigation -> policy | Frozen (`hexapod_core.command`, `hexapod_nav.producer.CommandProducer`) |
| O1 | Observation v1, 66-dim proprioceptive layout | runtime -> policy | Frozen (`hexapod_core.observation`) |
| A1 | Action interface, 18 joint offsets, clip, scale, slew | policy -> runtime -> motors | Frozen (`hexapod_core.action`, `hexapod_runtime.action_pipeline`) |
| C2 | `HeightScan` / Observation v2: body-frame local elevation grid appended to O1 (extent, cell size, frame origin, height clip, missing-cell sentinel, max age) | perception -> policy (sim: raycaster -> policy) | To define in M2 |
| C3 | Pose and maps: `nav_msgs/Odometry` in `odom`, fused pose in the operator's `map` frame (GPS + LIO, or a start-anchored local frame), registered cloud, `grid_map_msgs` elevation layer, traversability layer | perception -> navigation | To define in P1 |
| C4 | `AreaOfInterest`: the operator's drawn polygon (WGS84 with GPS, or local frame with the datum at the robot's start), optional no-go polygons, sweep spacing, revision id; the same polygon is the geofence | operator map page -> navigation | To define first in N1; the phone-to-robot transport is a hardware decision |

Rules that follow: a checkpoint is tagged with the observation version it was
trained against, and an O1 checkpoint cannot be driven by a C2 producer.
Perception publishes C2 at policy rate with a timestamp; the runtime refuses a
scan older than the max age and falls back to a flat-ground scan plus a reduced
command envelope, and that fallback is exercised in training. Navigation emits
only commands inside the envelope of the checkpoint actually loaded;
`VelocityCommand.validate` rejects the rest.

## 2. Physical topology and open hardware decisions

```text
Livox Mid-360 --(Ethernet)--> companion computer --(CAN)--> 18 x RobStride RS05
RealSense D455 --(USB3)-----^        |
GPS receiver ---(serial/USB)^        +-- ROS 2 graph: drivers, LIO + GPS fusion,
data sensor ----(payload)---^        |   mapping, coverage nav, safety monitor
                                     +-- hexapod_runtime: obs builder, policy
                                         backend, action pipeline, watchdog
operator phone / laptop --(WiFi/radio)-^   map page: draw polygon, go, stop
```

Decisions that block sim-to-real, in the order they bite; the whole list must
fit the $6000 in `dar.md` §6:

1. Companion computer (Jetson-class if elevation mapping and inference run
   on-robot) and its power budget next to the Mid-360's 6.5 W average / 14 W
   cold peak.
2. Mid-360 mount: the placement study (`robot/sensors/README.md`) recommends a
   75 mm upright mast; confirm in Isaac Sim, then design the 100 x 100 mm,
   3 mm metal plate Livox requires.
3. Measured masses, inertias, and actuator-to-joint mapping on the assembled
   robot (`docs/TRAINING.md` §7).
4. GPS receiver class (RTK or not) from the position-accuracy blank in
   `dar.md` §2, and the operator link that carries C4.
5. The data sensor: mass, mount, power, and what "steady enough to record"
   means for it (`dar.md` §2).

## 3. What the simulator is for

| Layer | Isaac Sim / Isaac Lab role |
| --- | --- |
| Locomotion | Training and formal screening. Terrain curriculum plus a simulated height scan in M2. Domain randomization of everything `docs/TRAINING.md` §7 lists as missing. |
| Perception | Integration scene only: raycaster Mid-360 surrogate and D455 twin on the walking robot, producing bags the perception stack must turn into the same C2/C3 outputs it produces from hardware bags. Never a training input. |
| Navigation | Closed-loop scenario tests: a drawn polygon on flat ground and on the `dar.md` terrain class; sweep coverage, deck steadiness during the traverse, fence violations, policy in the loop through C1. |
| Hardware bring-up | Observation/action parity against the runtime (`isaaclab/tests/test_runtime_parity.py`) before any powered walking. |

## 4. Where new code goes

```text
packages/hexapod_core/     C2 validator lands beside C1, O1, A1
packages/hexapod_env/      terrain + height-scan sensor (M2)
packages/hexapod_runtime/  C2 ingestion + fallback
packages/hexapod_nav/      coverage planner, geofence, safety monitor behind CommandProducer
ros2_ws/                   (to create) hexapod_perception, hexapod_msgs (C2/C3/C4),
                           hexapod_nav_ros, hexapod_operator (map page), hexapod_bringup
```

Python packages that are stdlib-only stay stdlib-only. ROS 2 code lives under
`ros2_ws/` so the training container and the test suite never need ROS.

## 5. Workstreams

| Code | Workstream | Owns | First deliverable |
| --- | --- | --- | --- |
| A | Platform and asset | `robot/`, `tools/`, URDF/USD generation, mass/inertia measurement, hardware mapping | Asset v1 validated on the Spark (M0) |
| B | Locomotion (RL) | `packages/hexapod_env`, `hexapod_train`, `hexapod_eval`, curriculum, gates | Flat-ground omnidirectional policy on asset v1 (M1) |
| C | Perception | `ros2_ws/hexapod_perception`, Mid-360 and D455 bring-up, LIO, GPS fusion, elevation and global maps, C2/C3 | Bench LIO + elevation map from handheld data (P1) |
| D | Autonomy and navigation | `packages/hexapod_nav`, `ros2_ws/` coverage planning, operator map page, C4, geofence and safety monitor, data-collection trigger | Sim closed-loop polygon sweep through C1 (N1) |
| E | Runtime and hardware bring-up | `packages/hexapod_runtime`, CAN/RS05 driver, watchdog, E-stop, GPS and companion-computer integration, parity tests | Restrained-robot parity check (H1) |

One person can hold more than one workstream; none may be held by nobody.
Names go in the table when the team ratifies this file.

## 6. Milestones and gates

Durations are estimates for a part-time student team and exist to expose
ordering. Each milestone has a gate: the numeric or binary check that closes
it, written before the work starts and never edited to admit a result. Every
numeric threshold added after 2026-09-03 is derived from `dar.md` §2; until
that table is filled in, gates are written with named blanks, never guessed
numbers.

```text
M0 ──> M1 ──> M2 ──> M3 ──> M4
 │      │      ^      ^      ^
 └─ P1 ────────┘──────┘      │
        └─ N1 (sim) ─────────┘
```

P1 has no dependency on locomotion and starts on day one. N1 needs only the
`CommandProducer` seam and a simulated robot, so it starts against the Phase-0
policy on flat ground before M1 finishes. The demo ladder in `dar.md` §5 maps
onto these: rung 1 is M1, rung 2 is N1 in simulation, rung 3 is H1, rungs 4 to
6 are M4 indoor, field, and field with the data sensor.

### M0: Foundations (weeks 1-3)

Done as of 2026-09-04: asset v1 (`robot/hexapod_mkii_assy/`) merged under its
own task ID with a spec bound to the URDF by test; uv workspace and CI running
the suite and the manifest check on every PR.

- A: Mid-360 link and payload mass on asset v1 at the chosen mount; commit or
  hash the Onshape export the URDF was built from.
- A + B: USD generated on the Spark; `validate.py --asset mkii_v1` passes; the
  provisional runtime joint order confirmed from its `joint_names=` line
  (`docs/OPERATIONS.md` §10).
- C: `ros2_ws/` skeleton; Mid-360 powered on the bench through
  `livox_ros_driver2`; first bag recorded; sensor identity confirmed from the
  physical label (`robot/sensors/README.md`).
- All: Git LFS for `artifacts/`; PR template; one new member follows the
  README quickstart end to end.

Gate: `validate.py` passes on asset v1 with total mass within 5 % of the
measured assembled robot (or of the CAD value until the robot is weighed,
stated explicitly); every joint limit matches
`robot/hexapod_mkii_assy/joint_limits.json`; runtime joint order confirmed;
CI green.

### M1: Flat-ground omnidirectional locomotion on asset v1 (weeks 3-9)

- B: retrain from scratch on asset v1, stance sweep redone for the 8.26 kg mass
  distribution, then the joystick envelope (forward, reverse, lateral, yaw,
  transitions). Add the missing domain randomization from `docs/TRAINING.md`
  §7: control latency, encoder/IMU noise, actuator strength, mass, pushes.
- B: one training config, one intervention file, one label per attempt; first
  in-vivo validation of `hexctl` on the Spark.
- E: RS05 CAN driver and `hexapod_runtime` wired to it on a restrained robot;
  observation parity test against the sim runtime.

Gate: the Stage2C gates from `docs/TRAINING.md` §6, unchanged, at the
`0.040 rad / 20 ms` limiter, plus lateral and yaw commands at the same
thresholds, plus zero falls under a push of a stated magnitude. The
deck-stability composite is the primary grade and speed is secondary; when
`dar.md` §2 is filled in, the composite, tilt, and yaw thresholds are
re-derived by a written gate change. Hardware: restrained-robot parity within a
stated tolerance on every observation term.

### P1: Perception on the bench (weeks 1-8, parallel)

- C: Mid-360 through `livox_ros_driver2`; lidar-inertial odometry (FAST-LIO2 or
  Point-LIO, both support the Mid-360) on handheld walks; an elevation map
  (`elevation_mapping_cupy` or equivalent) in `odom`; a body-frame local grid
  extracted from it at 50 Hz, the first C2 producer.
- C: a bag corpus under a documented naming scheme with ground-truth loops.
- C: C3 message set defined and validated, including the map-frame pose.
- C + E: GPS receiver selected against the `dar.md` §2 position-accuracy
  blank, logged beside LIO on the same walks, fused into the C3 pose. Fallback
  if GPS is deferred: a local frame anchored at the robot's start.

Gate: closed-loop LIO drift below a stated fraction of path length over a
stated course; fused map-frame pose error within the `dar.md` §2 blank on a
surveyed loop; elevation map at 5 cm cells at 10 Hz; local grid at 50 Hz under
the C2 max age; the M2 sim integration bag replayed through the same launch
file produces C2/C3.

### M2: Rough terrain in simulation (weeks 9-17)

Entry: M1 gate; C2 drafted jointly by B and C.

- B + C: freeze C2 in `hexapod_core` as Observation v2.
- B: terrain curriculum (Isaac Lab terrain generator: slopes, steps, stairs,
  random rough, discrete obstacles) with a `RayCaster` grid height scan under
  new task IDs; scan noise, dropout, and lag randomized to the C2 envelope.
- B: screening scene matching the `dar.md` §2 terrain class.
- C: raycaster Mid-360 surrogate and D455 twin on asset v1 in an integration
  scene; bags recorded from a walking policy; perception stack runs on them.
- A: mount confirmed in Isaac Sim against the placement study; bracket designed.

Gate: success rate (no fall, reaches goal) at or above a stated threshold per
terrain class, difficulty, and seed set; the proprioceptive-only M1 policy
fails the same screen (proves the height scan is load-bearing); perception
produces C2 from the sim bag within the max age on the target compute.

### H1/M3: Sim-to-real walking (weeks 17-26)

Entry: M1 hardware parity; M2 gate; sensor mount installed.

- E: watchdog, E-stop, joint-limit and 1.6 N m enforcement on hardware; an
  acceptance checklist mirroring `docs/TRAINING.md` §7.
- B + E: flat-ground walking on hardware with the M1 policy, then rough terrain
  with the M2 policy fed by the P1 pipeline.
- C: extrinsics calibration Mid-360/IMU/D455/body; map quality outdoors.

Gate: stated distance walked without a fall on flat ground and on a course
matching the `dar.md` §2 terrain class; deck steadiness inside the `dar.md` §2
limits over that distance; motor temperature and current duty inside the RS05
envelope; recorded bags for every run.

### N1/M4: Coverage autonomy in simulation, then field (weeks 20-34)

The concept of operations in `dar.md` §1, end to end.

- D: C4 written down first, then the operator map page (draw, review, go, stop).
- D: coverage path planning over the polygon (boustrophedon sweep at the
  `dar.md` §2 spacing; the ROS 2 Nav2 coverage plugin or equivalent), with the
  traversability layer marking cells not to step on; the polygon doubles as a
  geofence.
- D: safety monitor: stop on position uncertainty, stale map, tilt beyond the
  `dar.md` §2 limit, fence violation, or lost operator link; hardware E-stop
  from E.
- D: data-collection trigger: record only while the deck is inside the
  steadiness limits, each record stamped with the C3 pose.
- D: closed loop in Isaac Sim through C1: Phase-0 policy on flat ground first
  (start now), then the M1 policy on asset v1, then the M2 policy on terrain.
- D + all: indoor sweep of a drawn polygon on flat floor (local frame), then the
  outdoor trial on the `dar.md` terrain with GPS, then with the data sensor.

Gate: in sim, sweep coverage at or above a stated fraction within a stated
time, zero falls, zero fence violations, and deck steadiness inside the
`dar.md` §2 limits for a stated fraction of the traverse, over stated seeds; in
the field, one complete sweep of an operator-drawn polygon with the safety
monitor never overridden by hand and a position-stamped data record delivered.

## 7. Decisions

Numbered, dated, never edited after acceptance; a reversal gets a new number
and links back. Code comments reference these as `ADR-000N`.

### ADR-0001 (proposed 2026-08-27): rebuild the training asset; retire the mock from training

Every checkpoint under `artifacts/` was trained on the Onshape mock
(`robot/hexapod_mkii_mock_assy/`): user-specified mass targets, estimated
inertias, visual-mesh collisions, no foot pads, self-collision off
(`docs/TRAINING.md` §7). No checkpoint passes the Stage2C gates and none is
hardware-ready, and the Stage2C loop was spending its effort on infrastructure
and on a yaw gate for a robot whose mass distribution was known to be wrong.

Decision: build the training asset from the CAD assembly with real per-part
inertials, primitive collisions, explicit foot pads, and a scripted USD;
register it under new task IDs; retrain from scratch; stop launching Stage2C
probes on the mock. Landed 2026-09-04 as `robot/hexapod_mkii_assy/` under
`Isaac-Velocity-Flat-Hexapod-MKII-V1-Direct-v0` (8.26 kg; the sensor payload
and self-collision validation remain M0 items).

Consequences: everything under `artifacts/` through
`phase2_recovery_stage2c_stable_forward` and `stability_stance` is the Phase-0
(mock) lineage. Nothing is deleted; hashes and the current-best row in
`artifacts/README.md` stand. The 66-dim observation and 18-dim action contracts
survive; stance, deck targets, and reward scales are re-derived for the new
mass distribution. The two open contradictions in `STATUS.md` close as moot when
this is accepted, since the asset they concern is retired from training. The
first weeks of RL work produce an asset a gait can transfer from, not a gait.

### ADR-0002 (proposed 2026-08-27): the policy consumes a height scan, never a point cloud

The Mid-360's field of view is 360 x 59 degrees, from -7 to +52 degrees
elevation. At a 0.185 m deck height the nearest possible ground return is
`h / tan(7 deg)`, about 1.7 m, and further on any mast (`robot/sensors/README.md`).
The sensor cannot see the robot's own footing, and Livox does not publish the
Mid-360's temporal scan trajectory, so a simulated pattern is a coverage
surrogate, not the sensor. Learned rough-terrain locomotion on legged robots
conditions the policy on proprioception plus a small body-frame height grid
sampled around the feet: in Isaac Lab a `RayCaster` with a `GridPatternCfg`,
on the robot the same grid read out of a perception elevation map.

Decision: define C2 as Observation v2 in `hexapod_core`. Train against a
raycaster grid on generated terrain with noise, dropout, and lag randomized
over the C2 envelope. On the robot, perception produces C2 from its elevation
map. The Mid-360 is a mapping, odometry, and obstacle sensor; the D455 (or a
successor downward depth sensor) and the accumulated map supply local terrain.
The simulated Mid-360 is for perception integration and mount validation only.

Consequences: RL and perception start in parallel with one grid message as the
integration point; the input distribution is narrow enough to randomize
honestly; the degraded-perception fallback is part of C2 and exercised in
training; a future sensor changes only the perception layer.

### ADR-0003 (proposed 2026-08-27): perception and navigation are ROS 2; the Python packages stay ROS-free

The Livox SDK 2 driver, the LIO packages that support the Mid-360, GPU
elevation mapping, and Nav2 are ROS 2 packages, while `packages/` must import
on a bare interpreter and inside the Isaac Lab container.

Decision: `hexapod_perception`, `hexapod_msgs` (C2, C3, C4), `hexapod_nav_ros`,
`hexapod_operator`, and `hexapod_bringup` live under `ros2_ws/` on one pinned
distribution chosen at M0. `packages/hexapod_*` never import ROS; the runtime's
C2 ingestion is a thin adapter in `ros2_ws/` that calls `hexapod_runtime`.

Consequences: two build systems (uv plus colcon); CI runs the Python suite
everywhere and the colcon build on a ROS-capable runner. The companion computer
runs ROS 2 and is chosen together with the distribution. Each contract validator
exists twice, in `hexapod_core` and as the message definition, kept equal by a
contract test.

### ADR-0004 (proposed 2026-09-03): the mission is steady coverage of an operator-drawn bounded area

The 2026-08-27 plan framed the mission as autonomous exploration of unknown
terrain, possibly designated by an air unit. The team lead's requirements slide
(`DAR.png`, transcribed in `dar.md`) is a coverage mission: the area is known
and bounded before the robot moves, the operator is a person with a map, and
the deliverable is a steady data-collection platform.

Decision: `dar.md` is the mission of record. Navigation is coverage path
planning bounded by the drawn polygon as a geofence; frontier exploration and
air-unit hand-off leave the plan. C4 is the operator's drawn polygon, written
down before the map page is built. Steadiness is the primary locomotion grade,
with thresholds set from `dar.md` §2 through a written gate change; speed is
secondary. GPS fused with lidar-inertial odometry joins perception and C3
carries the map-frame pose. Data collection is a deliverable: records are taken
only while the deck is inside the steadiness limits and are position-stamped.
ADRs 0001 to 0003 stand unchanged; they are how this mission is built.

Consequences: §1, §2, §5, and §6 above are written for this mission. The
exploration framing survives only in `docs/archive/`.
