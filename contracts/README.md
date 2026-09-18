# Shared contracts

You import `VelocityCommand` and `CommandEnvelope` from `contracts.command`,
and `Pose2D` from `contracts.pose`. Supply the permitted envelope when validating
a command. The coordinate helpers map forward to body -Y and left to body +X.
These modules use the standard library.

`release.COMPATIBILITY_KEYS` names the identity fields that the current PPO
loader checks. It supplies no new deployment schema or hardware qualification.
Historical observation layouts and motor adapters remain in the archive.
