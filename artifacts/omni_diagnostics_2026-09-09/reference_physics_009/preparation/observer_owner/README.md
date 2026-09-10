# Read-only joint-angle evidence at every physics update

This proposed observer successor records actual `robot.data.joint_pos` as `joint_position_rad` beside the existing reported joint velocity at each existing400Hz scene update and the initial snapshot. Runtime joint names are retained in the NPZ metadata, and the observer rejects any change in their order. The final angle and velocity sample must exactly equal the ordinary pre-reset control row.

It does not change a physics setting, target, solver step, reset, contact sensor or acceptance gate. It does not substitute finite differences for reported velocity. The existing eight-updates, simulation-counter, SDK-timestamp, initial/failure evidence and exact method-restoration checks remain. Original control-level telemetry remains independent. Source008 and its raw results are unchanged.

The installed articulation source, captured read-only with its SHA, defines `joint_pos` as the timestamped `get_dof_positions()` getter and `joint_vel` as `get_dof_velocities()`. This establishes the accessed SDK properties; it does not itself prove native buffer semantics or measurement fidelity. The new actual traces will permit direct joint-angle increments/rate comparison at400Hz. Current007/008 observer traces contain400Hz velocity but only50Hz joint angles, so they cannot supply this missing evidence retrospectively.

Eleven focused CPU tests pass, including the six existing count/timing/alias/torque/failure tests plus actual-property copied storage, initial and all-eight angle samples, named NPZ order, position/rate endpoint mismatch rejection, mid-run name-order rejection and initial nonfinite-angle preservation. The fake fixture verifies observer mechanics, not real physics. PPO independently reran all11 tests and reviewed the minimal diff without finding a concrete blocker.

Integration is a future separately frozen source with only `tools/physics_substeps.py` overlaid. No GPU job, task registration, training or production adoption is included here. Full initial/settling/settled values remain distinct, and hardware startup is unqualified.
