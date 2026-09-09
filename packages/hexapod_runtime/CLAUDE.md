# hexapod_runtime

This package owns the deployable policy runtime: the observation builder that
assembles the 66-vector from named sensor inputs in the exact order
`hexapod_core.observation` freezes (`observation_builder.py`), and the action
pipeline that turns a normalized 18-vector into absolute joint targets —
clip to [-1, 1], command-conditioned stand scale, scale-and-offset from the
default joint positions, soft-limit clamp, then the radians-per-20 ms slew
limit rescaled by the real step dt (`action_pipeline.py`). It is pure Python
over stdlib and `hexapod_core` only, single-robot rather than batched, and the
same code is meant to run in simulation and on the robot's companion computer.
Don'ts: never import Isaac Lab, `hexapod_env`, gymnasium, torch, or numpy from
here — the runtime has to import on hardware where none of them exist, and a
policy backend belongs above this package, not inside it; never re-derive or
"improve" a value that `hexapod_core` already freezes, and never let the two
action paths drift — parity with the training environment's own
`limit_processed_joint_target_slew` and
`apply_command_conditioned_stand_action_scale` is enforced by
`isaaclab/tests/test_runtime_parity.py`, and a divergence is a bug in this
package rather than a tuning decision; keep the slew history reset at every
episode or control-session boundary, since carrying a stale target across a
reset is exactly what the environment's history flag exists to prevent; and
never let a NaN or infinite value through the observation boundary silently.
