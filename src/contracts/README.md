# Shared contracts

You import `VelocityCommand` and `CommandEnvelope` from `contracts.command`,
and `Pose2D` from `contracts.pose`. Supply the permitted envelope when validating
a command. The coordinate helpers map forward to body -Y and left to body +X.
These modules use the standard library.

`command.CommandSet` builds the finite bank from the `commands` mapping in
`configs/walking_requirements.json`. Navigation, training and evaluation share
that builder. `load_requirement_set` reads one named set and its byte hash;
the default path selects `source/configs` in a frozen pack or the repository's
`configs`. Call it before the control loop. An envelope filters the bank and
does not create commands between its entries.

`release.COMPATIBILITY_KEYS` names the identity fields that the current PPO
loader checks. `release.runtime_changes` compares the complete runtime manifest
for both the loader and the site's native-approval status. These checks supply
no new deployment schema or hardware qualification.
Historical observation layouts and motor adapters remain in the archive.
