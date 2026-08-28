# ADR-0003: Perception and navigation are ROS 2 packages; training and runtime Python stay ROS-free

Status: proposed (2026-08-27)

## Context

The Livox SDK 2 driver, the lidar-inertial odometry packages that support the
Mid-360 (FAST-LIO2, Point-LIO), GPU elevation mapping, and mature planning and
exploration stacks are all ROS 2 packages. The existing `packages/` are
deliberately importable on a bare interpreter and inside the Isaac Lab
container; the contract tests depend on that.

## Decision

Perception (`hexapod_perception`), message definitions (`hexapod_msgs` for C2,
C3, C4), and bring-up (`hexapod_bringup`) live under `ros2_ws/` as ROS 2
packages targeting one pinned distribution chosen at M0. `packages/hexapod_*`
never import ROS; the runtime's C2 ingestion is a thin adapter in `ros2_ws/`
that calls `hexapod_runtime`. The unit-test command in `CLAUDE.md` does not
require ROS.

## Consequences

- Two build systems in one repository (Python packages plus colcon). CI runs
  the Python suite everywhere and the colcon build on a ROS-capable runner.
- The companion computer runs ROS 2; its distribution and the Jetson/other
  board choice are made together at M0.
- Contract validators exist twice, once in `hexapod_core` (Python, stdlib) and
  once as the message definition; a contract test keeps them equal.
