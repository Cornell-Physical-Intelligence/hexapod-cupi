# Single-robot placement diagnostic source

Prepared successor of canonical standing source005. Only `--num-envs 1` is accepted.
The required `--placement origin|xy14_4` is bound into the runtime identity,
authored before SDK warmup, and applied through the original physics-view reset.
Both arms retain the source root transform and append the same named translation
operation. The reset preserves the original float32 write and 2e-6 readback bound.

The robot, all153 SDF colliders, self-collision, floor, material, servo, timing,
32/0 solver and all physical/quiet scoring rules are unchanged. The live source005
floor/scene fields and body/joint order are compared with recorded single-robot
values. A read-only scene-attribute census adds information where old evidence
was incomplete; it does not claim independent backend solver introspection.

A passing report is only a diagnostic gate result. This source does not accept
32 robots, previous standing admission or PPO. Read the parent bundle's README
and OUTCOMES.json for the pre-registered interpretation and remaining limitations.
Historical notes, snapshots and test logs copied into this source retain their
original names and meanings. parents/source005 preserves the complete old source.
