# hexapod_nav

This package owns the command-producer seam: the `CommandProducer` protocol and
the planar `Pose2D` it consumes (`producer.py`), and a deliberately minimal
`WaypointFollower` — heading error to a clamped yaw rate, distance to a clamped
forward speed, a stop radius that retires waypoints — that exists as a worked
example of satisfying that protocol (`waypoint.py`), not as a tuned planner. A
producer answers one question per control tick: given a pose and a time, what
`hexapod_core.command.VelocityCommand` should the policy receive? Real path
planning, obstacle avoidance, and mapping land here later and replace the
example wholesale; what they must not change is the three-scalar interface.
Don'ts: never import the simulator, `hexapod_env`, `hexapod_train`, or
`hexapod_runtime` — this package depends on `hexapod_core` alone and knows
nothing about joints, gaits, or torques; never emit a command that fails
`VelocityCommand.validate` against the envelope in force, since a command
outside the trained envelope asks a network for behavior it has never seen;
keep producers non-blocking and usable at the 50 Hz policy rate; and never
present the waypoint follower as a planner in docs or demos — it has no
obstacle awareness, no acceleration limits, and no state estimation, and saying
otherwise would be the beginning of someone trusting it.
