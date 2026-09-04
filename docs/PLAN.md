# Plan: architecture, milestones, decisions

This file states the program direction. You rewrite it in place. A plan change
edits the affected section and adds a dated entry to §7. `dar.md` states the
mission. `STATUS.md` holds run-level state. `docs/TRAINING.md` defines the
locomotion task and `docs/OPERATIONS.md` gives the Spark procedure.

Status: draft for team ratification (2026-08-27, revised 2026-09-04). The four
decisions in §7 are `proposed`. The team accepts them as its first action.

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

The policy receives no point cloud. Perception emits no joint targets.
Navigation emits velocity commands and nothing below them. You build and test
each layer without the others, because each seam is a small versioned contract
with a stdlib validator in `packages/hexapod_core` (or a ROS 2 message plus a
validator) and a contract test under `isaaclab/tests/`. To change a frozen
contract, you add a new version beside the old one.

| Id | Contract | Producer -> consumer | Status |
| --- | --- | --- | --- |
| C1 | `VelocityCommand` `[vx, vy, yaw_rate]`, validated against the trained envelope, 50 Hz | navigation -> policy | Frozen (`hexapod_core.command`, `hexapod_nav.producer.CommandProducer`) |
| O1 | Observation v1, 66-dim proprioceptive layout | runtime -> policy | Frozen (`hexapod_core.observation`) |
| A1 | Action interface, 18 joint offsets, clip, scale, slew | policy -> runtime -> motors | Frozen (`hexapod_core.action`, `hexapod_runtime.action_pipeline`) |
| C2 | `HeightScan` / Observation v2: body-frame local elevation grid appended to O1 (extent, cell size, frame origin, height clip, missing-cell sentinel, max age) | perception -> policy (sim: raycaster -> policy) | To define in M2 |
| C3 | Pose and maps: `nav_msgs/Odometry` in `odom`, fused pose in the operator's `map` frame (GPS + LIO, or a start-anchored local frame), registered cloud, `grid_map_msgs` elevation layer, traversability layer | perception -> navigation | To define in P1 |
| C4 | `AreaOfInterest`: the operator's drawn polygon (WGS84 with GPS, or local frame with the datum at the robot's start), optional no-go polygons, sweep spacing, revision id. The same polygon is the geofence | operator map page -> navigation | To define first in N1. The phone-to-robot transport is a hardware decision |

Rules that follow from the table:

- You tag each checkpoint with the observation version it trained against. A
  C2 producer cannot drive an O1 checkpoint.
- Perception publishes C2 at policy rate with a timestamp. The runtime refuses
  a scan older than the max age. It then falls back to a flat-ground scan and
  a reduced command envelope, and training exercises that fallback.
- Navigation emits only commands inside the envelope of the loaded checkpoint.
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

The team must make these decisions before sim-to-real, in this order. The
total cost must fit the $6000 in `dar.md` §6.

1. Companion computer (Jetson-class if elevation mapping and inference run on
   the robot) and its power budget beside the Mid-360's 6.5 W average and
   14 W cold peak.
2. Mid-360 mount. The placement study (`robot/sensors/README.md`) recommends a
   75 mm upright mast. Confirm it in Isaac Sim, then design the 100 x 100 mm,
   3 mm metal plate Livox requires.
3. Measured masses, inertias, and actuator-to-joint mapping on the assembled
   robot (`docs/TRAINING.md` §7).
4. GPS receiver class (RTK or standard) from the position-accuracy blank in
   `dar.md` §2, and the operator link that carries C4.
5. The data sensor: mass, mount, power, and the deck-steadiness limit it needs
   for a valid record (`dar.md` §2).

## 3. Simulator role per layer

| Layer | Isaac Sim / Isaac Lab role |
| --- | --- |
| Locomotion | Training and formal screening. Terrain curriculum plus a simulated height scan in M2. Domain randomization of each item `docs/TRAINING.md` §7 lists as missing. |
| Perception | Integration scene only. A raycaster Mid-360 surrogate and a D455 twin ride the walking robot and produce bags. The perception stack must turn those bags into the same C2 and C3 outputs it produces from hardware bags. The simulated lidar is no training input. |
| Navigation | Closed-loop scenario tests: a drawn polygon on flat ground and on the `dar.md` terrain class. You measure sweep coverage, deck steadiness during the traverse, and fence violations, with the policy in the loop through C1. |
| Hardware bring-up | Observation and action parity against the runtime (`isaaclab/tests/test_runtime_parity.py`) before powered walking. |

## 4. Location of new code

```text
packages/hexapod_core/     C2 validator lands beside C1, O1, A1
packages/hexapod_env/      terrain + height-scan sensor (M2)
packages/hexapod_runtime/  C2 ingestion + fallback
packages/hexapod_nav/      coverage planner, geofence, safety monitor behind CommandProducer
ros2_ws/                   (to create) hexapod_perception, hexapod_msgs (C2/C3/C4),
                           hexapod_nav_ros, hexapod_operator (map page), hexapod_bringup
```

Python packages that are stdlib-only stay stdlib-only. ROS 2 code lives under
`ros2_ws/`, so the training container and the test suite do not need ROS.

## 5. Workstreams

| Code | Workstream | Owns | First deliverable |
| --- | --- | --- | --- |
| A | Platform and asset | `robot/`, `tools/`, URDF/USD generation, mass/inertia measurement, hardware mapping | Asset v1 validated on the Spark (M0) |
| B | Locomotion (RL) | `packages/hexapod_env`, `hexapod_train`, `hexapod_eval`, curriculum, gates | Flat-ground omnidirectional policy on asset v1 (M1) |
| C | Perception | `ros2_ws/hexapod_perception`, Mid-360 and D455 bring-up, LIO, GPS fusion, elevation and global maps, C2/C3 | Bench LIO + elevation map from handheld data (P1) |
| D | Autonomy and navigation | `packages/hexapod_nav`, `ros2_ws/` coverage planning, operator map page, C4, geofence and safety monitor, data-collection trigger | Sim closed-loop polygon sweep through C1 (N1) |
| E | Runtime and hardware bring-up | `packages/hexapod_runtime`, CAN/RS05 driver, watchdog, E-stop, GPS and companion-computer integration, parity tests | Restrained-robot parity check (H1) |

One person can hold more than one workstream. Each workstream has a named
owner. The team writes the names into the table when it ratifies this file.

## 6. Milestones and gates

Durations are estimates for a part-time student team and show the ordering.
Each milestone has a gate: the numeric or binary check that closes it. You
write the gate before the work starts and do not edit it to admit a result.
Each numeric threshold added after 2026-09-03 derives from `dar.md` §2. Until
the team fills that table, you write gates with named blanks and no guessed
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
onto these milestones: rung 1 is M1, rung 2 is N1 in simulation, rung 3 is
H1, and rungs 4 to 6 are M4 indoor, field, and field with the data sensor.

### M0: Foundations (weeks 1-3)

Done as of 2026-09-04: the team merged asset v1 (`robot/hexapod_mkii_assy/`)
under its own task ID with a spec bound to the URDF by test. The uv workspace
and CI run the suite and the manifest check on each PR.

- A: Mid-360 link and payload mass on asset v1 at the chosen mount. Commit or
  hash the Onshape export the URDF came from.
- A + B: generate the USD on the Spark, pass `validate.py --asset mkii_v1`,
  and confirm the provisional runtime joint order from its `joint_names=` line
  (`docs/OPERATIONS.md` §10).
- C: `ros2_ws/` skeleton. Power the Mid-360 on the bench through
  `livox_ros_driver2`, record the first bag, and confirm the sensor identity
  from the physical label (`robot/sensors/README.md`).
- All: Git LFS for `artifacts/`, a PR template, and one new member who
  follows the README quickstart end to end.

Gate: `validate.py` passes on asset v1 with total mass within 5 % of the
measured assembled robot. Until the team weighs the robot, the CAD value
stands in and the gate record says so. Each joint limit matches
`robot/hexapod_mkii_assy/joint_limits.json`. The runtime joint order is
confirmed and CI is green.

### M1: Flat-ground omnidirectional locomotion on asset v1 (weeks 3-9)

- B: retrain from scratch on asset v1. Redo the stance sweep for the 8.26 kg
  mass distribution, then train the joystick envelope (forward, reverse,
  lateral, yaw, transitions). Add the domain randomization `docs/TRAINING.md`
  §7 lists as missing: control latency, encoder/IMU noise, actuator strength,
  mass, pushes.
- B: one training config, one intervention file, one label per attempt. Run
  the first in-vivo validation of `hexctl` on the Spark.
- E: wire the RS05 CAN driver and `hexapod_runtime` to a restrained robot and
  run the observation parity test against the sim runtime.

Gate: the Stage2C gates from `docs/TRAINING.md` §6, unchanged, at the
`0.040 rad / 20 ms` limiter, plus lateral and yaw commands at the same
thresholds, plus zero falls under a push of a stated magnitude. The
deck-stability composite is the primary grade and speed is secondary. After
the team fills `dar.md` §2, a written gate change re-derives the composite,
tilt, and yaw thresholds. Hardware: restrained-robot parity within a stated
tolerance on each observation term.

### P1: Perception on the bench (weeks 1-8, parallel)

- C: run the Mid-360 through `livox_ros_driver2` and lidar-inertial odometry
  (FAST-LIO2 or Point-LIO, both support the Mid-360) on handheld walks. Build
  an elevation map (`elevation_mapping_cupy` or equivalent) in `odom` and
  extract a body-frame local grid from it at 50 Hz. That grid is the first
  C2 producer.
- C: a bag corpus under a documented naming scheme with ground-truth loops.
- C: define and validate the C3 message set, including the map-frame pose.
- C + E: select the GPS receiver against the `dar.md` §2 position-accuracy
  blank, log it beside LIO on the same walks, and fuse it into the C3 pose.
  If the team defers GPS, anchor a local frame at the robot's start.

Gate: closed-loop LIO drift below a stated fraction of path length over a
stated course. Fused map-frame pose error within the `dar.md` §2 blank on a
surveyed loop. Elevation map at 5 cm cells at 10 Hz and local grid at 50 Hz
under the C2 max age. The M2 sim integration bag, replayed through the same
launch file, produces C2 and C3.

### M2: Rough terrain in simulation (weeks 9-17)

Entry: M1 gate, and B and C have drafted C2 together.

- B + C: freeze C2 in `hexapod_core` as Observation v2.
- B: terrain curriculum (Isaac Lab terrain generator: slopes, steps, stairs,
  random rough, discrete obstacles) with a `RayCaster` grid height scan under
  new task IDs. Randomize scan noise, dropout, and lag over the C2 envelope.
- B: a screening scene that matches the `dar.md` §2 terrain class.
- C: a raycaster Mid-360 surrogate and a D455 twin on asset v1 in an
  integration scene. Record bags from a walking policy and run the perception
  stack on them.
- A: confirm the mount in Isaac Sim against the placement study and design the
  bracket.

Gate: success rate (no fall, reaches goal) at or above a stated threshold per
terrain class, difficulty, and seed set. The proprioceptive-only M1 policy
fails the same screen, which proves the policy needs the height scan.
Perception produces C2 from the sim bag within the max age on the target
compute.

### H1/M3: Sim-to-real walking (weeks 17-26)

Entry: M1 hardware parity, M2 gate, sensor mount installed.

- E: watchdog, E-stop, joint-limit and 1.6 N m enforcement on hardware. An
  acceptance checklist that mirrors `docs/TRAINING.md` §7.
- B + E: flat-ground walking on hardware with the M1 policy, then rough terrain
  with the M2 policy fed by the P1 pipeline.
- C: extrinsics calibration for Mid-360, IMU, D455, and body. Map quality
  outdoors.

Gate: a stated distance walked without a fall on flat ground and on a course
that matches the `dar.md` §2 terrain class. Deck steadiness inside the
`dar.md` §2 limits over that distance. Motor temperature and current duty
inside the RS05 envelope. Recorded bags for each run.

### N1/M4: Coverage autonomy in simulation, then field (weeks 20-34)

This milestone delivers the concept of operations in `dar.md` §1 end to end.

- D: write C4 first, then build the operator map page (draw, review, go,
  stop).
- D: coverage path planning over the polygon (boustrophedon sweep at the
  `dar.md` §2 spacing, with the ROS 2 Nav2 coverage plugin or equivalent).
  The traversability layer marks cells the robot must not step on. The
  polygon is also the geofence.
- D: safety monitor. Stop on position uncertainty, stale map, tilt beyond the
  `dar.md` §2 limit, fence violation, or lost operator link. E supplies the
  hardware E-stop.
- D: data-collection trigger. Record only while the deck is inside the
  steadiness limits and stamp each record with the C3 pose.
- D: closed loop in Isaac Sim through C1. Start now with the Phase-0 policy on
  flat ground, then use the M1 policy on asset v1, then the M2 policy on
  terrain.
- D + all: indoor sweep of a drawn polygon on a flat floor (local frame), then
  the outdoor trial on the `dar.md` terrain with GPS, then with the data
  sensor.

Gate, simulation: sweep coverage at or above a stated fraction within a stated
time, zero falls, zero fence violations, and deck steadiness inside the
`dar.md` §2 limits for a stated fraction of the traverse, over stated seeds.
Gate, field: one complete sweep of an operator-drawn polygon with no manual
override of the safety monitor, and a position-stamped data record delivered.

## 7. Decisions

Decisions are numbered and dated. After acceptance you do not edit one. A
reversal gets a new number and links back. Code comments reference these as
`ADR-000N`.

### ADR-0001 (proposed 2026-08-27): rebuild the training asset and retire the mock from training

Context: each checkpoint under `artifacts/` trained on the Onshape mock
(`robot/hexapod_mkii_mock_assy/`), which has user-specified mass targets,
estimated inertias, visual-mesh collisions, no foot pads, and self-collision
off (`docs/TRAINING.md` §7). No checkpoint passes the Stage2C gates and none
is hardware-ready. The Stage2C loop spent its effort on infrastructure and on
a yaw gate for a robot with a known-wrong mass distribution.

Decision: build the training asset from the CAD assembly with per-part
inertials, primitive collisions, explicit foot pads, and a scripted USD.
Register it under new task IDs, retrain from scratch, and stop launching
Stage2C probes on the mock. The asset landed on 2026-09-04 as
`robot/hexapod_mkii_assy/` under `Isaac-Velocity-Flat-Hexapod-MKII-V1-Direct-v0`
(8.26 kg). The sensor payload and self-collision validation remain M0 items.

Consequences: each artifact under `artifacts/` through
`phase2_recovery_stage2c_stable_forward` and `stability_stance` belongs to the
Phase-0 (mock) lineage. Nobody deletes them. The hashes and the current-best
row in `artifacts/README.md` stand. The 66-dim observation and 18-dim action
contracts survive. You re-derive stance, deck targets, and reward scales for
the new mass distribution. Acceptance closes the two open contradictions in
`STATUS.md` as moot, because they concern an asset retired from training. The
first weeks of RL work produce a transferable asset before they produce a gait.

### ADR-0002 (proposed 2026-08-27): the policy consumes a height scan

Context: the Mid-360's field of view is 360 x 59 degrees, from -7 to +52
degrees elevation. At a 0.185 m deck height the nearest possible ground return
is `h / tan(7 deg)`, about 1.7 m, and further on a mast
(`robot/sensors/README.md`). The sensor cannot see the robot's own footing.
Livox does not publish the Mid-360's temporal scan trajectory, so a simulated
pattern reproduces the coverage and no more. Learned rough-terrain locomotion
on legged robots conditions the policy on proprioception plus a small
body-frame height grid sampled around the feet. In Isaac Lab that is a
`RayCaster` with a `GridPatternCfg`. On the robot it is the same grid read out
of a perception elevation map.

Decision: define C2 as Observation v2 in `hexapod_core`. Train against a
raycaster grid on generated terrain, with noise, dropout, and lag randomized
over the C2 envelope. On the robot, perception produces C2 from its elevation
map. The Mid-360 serves mapping, odometry, and obstacle detection. The D455
(or a successor downward depth sensor) and the accumulated map supply local
terrain. The simulated Mid-360 serves perception integration and mount
validation. The policy receives no point cloud.

Consequences: RL and perception start in parallel with one grid message as the
integration point. The input distribution is narrow enough to randomize with
confidence. The degraded-perception fallback is part of C2 and training
exercises it. A future sensor changes only the perception layer.

### ADR-0003 (proposed 2026-08-27): perception and navigation are ROS 2 and the Python packages stay ROS-free

Context: the Livox SDK 2 driver, the LIO packages that support the Mid-360,
GPU elevation mapping, and Nav2 are ROS 2 packages. `packages/` must import on
a bare interpreter and inside the Isaac Lab container.

Decision: `hexapod_perception`, `hexapod_msgs` (C2, C3, C4), `hexapod_nav_ros`,
`hexapod_operator`, and `hexapod_bringup` live under `ros2_ws/` on one pinned
distribution the team chooses at M0. `packages/hexapod_*` import no ROS. The
runtime's C2 ingestion is a thin adapter in `ros2_ws/` that calls
`hexapod_runtime`.

Consequences: two build systems (uv plus colcon). CI runs the Python suite on
each runner and the colcon build on a ROS-capable runner. The companion
computer runs ROS 2, and the team chooses it together with the distribution.
Each contract validator exists twice, in `hexapod_core` and as the message
definition, and a contract test keeps them equal.

### ADR-0004 (proposed 2026-09-03): the mission is steady coverage of an operator-drawn bounded area

Context: the 2026-08-27 plan framed the mission as autonomous exploration of
unknown terrain, possibly designated by an air unit. The team lead's
requirements slide (`DAR.png`, transcribed in `dar.md`) describes a coverage
mission. The area is known and bounded before the robot moves, the operator is
a person with a map, and the deliverable is a steady data-collection platform.

Decision: `dar.md` is the mission of record. Navigation is coverage path
planning bounded by the drawn polygon as a geofence. Frontier exploration and
air-unit hand-off leave the plan. C4 is the operator's drawn polygon, and the
team writes it down before it builds the map page. Steadiness is the primary
locomotion grade, with thresholds set from `dar.md` §2 through a written gate
change. Speed is secondary. GPS fused with lidar-inertial odometry joins
perception, and C3 carries the map-frame pose. Data collection is a
deliverable: the robot records only while the deck is inside the steadiness
limits and stamps each record with its position. ADRs 0001 to 0003 stand
unchanged as the means to this mission.

Consequences: §1, §2, §5, and §6 above describe this mission. The exploration
framing remains only in `docs/archive/`.
