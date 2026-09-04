# ADR-0002: The locomotion policy consumes a body-frame height scan, never a raw point cloud

Status: proposed (2026-08-27)

## Context

The program's original plan was to mount the Livox Mid-360 on the robot in
simulation first and train the walking policy with it. The mount placement
study (`tools/lidar_placement_study.py`, `robot/sensors/README.md`) shows the
Mid-360's -7 degree lower field-of-view edge puts the nearest visible ground
about 1.7 m from a deck-mounted sensor at the Stage2C stance, and further on
any mast. The sensor cannot see the footing. Livox also does not publish the
Mid-360's temporal scan trajectory, so the simulated pattern is a coverage
surrogate, not the real sensor.

Learned rough-terrain locomotion on legged robots conditions the policy on
proprioception plus a local elevation grid around the feet, produced in
simulation by a ray-cast grid and on hardware by an elevation-mapping node fed
by whatever sensors are available.

## Decision

Define contract C2, a body-frame local height scan with fixed extent, cell
size, frame, clip, missing-cell sentinel, and maximum age, as Observation v2
in `hexapod_core`. The policy is trained against C2 produced by an Isaac Lab
`RayCaster` grid on generated terrain, with noise, dropout, and lag randomized
over the C2 envelope. On the robot, perception produces C2 from its elevation
map. The Mid-360 is a mapping, odometry, and obstacle sensor; the D455 (or a
successor downward depth sensor) and the accumulated map supply local terrain.

The simulated Mid-360 is used for perception-pipeline integration and mount
validation, not as a policy input.

## Consequences

- The RL and perception workstreams start in parallel with no dependency
  until C2 is frozen, and are integrated by a single grid message.
- The policy's input distribution is narrow enough to randomize honestly.
- A degraded-perception fallback (flat scan plus reduced command envelope) is
  part of C2 and must be exercised in training.
- If a future sensor (a second lidar, a downward-facing solid-state unit) is
  added, only the perception layer changes.
