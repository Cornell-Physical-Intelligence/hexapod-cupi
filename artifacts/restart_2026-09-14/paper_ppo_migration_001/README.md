# Explicit checkpoint migration

This one-off artifact migrates the actual train006 checkpoint from frozen
source017 learner v2 to a separately frozen v3 learner after root authorization.
It is not a production compatibility loader. No environment, fitting or native
allocation is performed.

The only intended changes are the schema/source identities, five explicitly
adopted Config fields, and a new zero-accepted-update streak initialized to zero.
All existing configuration, network/normalizer tensors, both Adam states, RNG
bytes, counters and last metrics remain exact. In particular, direct payload
serialization preserves the recorded CUDA RNG bytes; calling ordinary save on
this CPU host would replace those bytes with an empty CUDA-state list.

The completed proof must include both frozen strict loaders, cross-version
rejection without state mutation, and bitwise actor/velocity/critic/discriminator
and style-reward equality on the same three complete recorded native trials.
Those observations come from evaluation012 of a different BC checkpoint; they
test migration equivalence, not performance of the migrated PPO policy.

`BINDING_DISABLED.json` pins the existing inputs and leaves successor hashes and
authorization disabled. It is preserved as preparation evidence. Root supplies
the frozen source018 identity before a fresh authorized binding is written and
executed with `python -B migrate.py --bindings <binding>`. The artifact refuses
overwriting `run_001`, source additions and any changed bound input. No default
runtime checkpoint loader is weakened.

CPU proofs validate CPU generator restoration and preserve all serialized CUDA
state, but cannot execute CUDA generator restoration. Neither checkpoint holds
PhysX state: subsequent native continuation requires a fresh simulation reset
and matching admission. Stage 2 remains unqualified.
