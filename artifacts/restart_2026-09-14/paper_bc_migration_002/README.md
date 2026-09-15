# Explicit BCfit004 checkpoint migration002

**ROOT_ADOPTION_PENDING.** Root authorized this CPU migration and verification;
the resulting checkpoint still awaits a separate lead adoption record before
native use. This artifact neither fits a new policy nor changes a runtime loader.
The completed migration001 and all original checkpoint/source bytes remain intact.

Input: the actual BCfit004 checkpoint, SHA-256
`6b169d9a406b8a0d6208a3282f59111529b375461b2e86bc391dfb9fe0a2c8f9`.
Output: `run_001/checkpoint_update000000_migrated.pt`, SHA-256
`dfb6d3ecc6ff5fae30c77bbac9056875eec6f19c9cb129917f97542d3b68d52b`.

The unchanged old learner is frozen source017, SHA-256
`b0a3cf503a0c710239d0245ef24fb961d0d05f06feaf37fffee7b2c802caf77a`.
The reviewed v3 learner is frozen source018, SHA-256
`867e0859b6a66ae9226f5838fd71c553795bfce44e75832d45b2c79504c0518a`.
Both complete source manifests are verified before and after execution, including
rejection of unlisted or modified files. Neither frozen source is altered.

## Exact preserved state

Model and normalization tensors, AMP/discriminator tensors, both Adam payloads,
serialized RNG, prior identity, every existing Config field, counters and last
metrics are unchanged. The input has zero PPO updates/transitions/optimizer or
discriminator steps/episodes and 1,000 historical BC fitting steps; those values
remain exact. The two learning optimizer state dictionaries are empty in this
CPU-fitted input, and their complete payloads are still verified. Nonempty Adam
and synthetic CUDA RNG preservation are covered separately by the eight CPU
fixtures.

The only changes are the v2-to-v3 schema and learner-source identity, these five
already adopted configuration additions, and one new integer counter initialized
to zero:

- `rollback_kl_steps=true`
- `kl_backtrack_halvings=3`
- `kl_backtrack_factor=0.5`
- `freeze_actor_obs_normalizer=true`
- `zero_accepted_update_limit=3`
- Counter: `consecutive_zero_accepted_updates=0`

Direct payload serialization preserves the input's RNG structure. The actual
BCfit004 CUDA RNG list is empty, and it remains empty; there are no actual CUDA
generator bytes in this checkpoint to exercise on this CPU host. The verification
does not claim GPU generator restoration or bit-exact physics continuation.

## Verification and limits

The eight synthetic preservation/rejection fixtures pass. The real migration
then completes in 2.01 seconds. Both original strict loaders accept only their
own source/schema identities; cross-version loads reject without model,
optimizer, counter, metric or RNG mutation. CPU RNG restoration and ambient RNG
preservation pass. Actor mean, velocity estimate, critic value, action standard
deviation, discriminator output and style reward match bitwise on all 3,050
observations from the three complete actual evaluation012 traces. These are fixed
inference-equivalence inputs, not newly acquired performance evidence.

The original script was copied into this artifact only. Its two changes pin the
BCfit004 input hash and name the output for its true update000000 counter. It is
not imported by production. `PREPARATION.json` binds the parent script and tests;
`BINDING_AUTHORIZED.json`, `EXECUTION.json`, captured stdout/stderr and
`run_001/VERIFICATION.json` preserve the actual run. Output overwrite is refused.
`SHA256.json` seals all files here and excludes itself.

No new fit, PPO update, native transition or simulation occurred. PhysX state is
absent, and subsequent native diagnostics need a fresh reset, matching admission
and explicit root adoption. The planned cold-versus-settled BC startup comparison
has no result in this artifact. Full Stage 2 remains unqualified.
