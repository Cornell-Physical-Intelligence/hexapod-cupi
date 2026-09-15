# Independent PPO stability recommendation

Advisory only. Root must adopt an experiment before code or native work. This
review changes no checkpoint, dataset, learner, physics, or acceptance gate.
Velocity-supervised BC is separately reviewed by the paper-review agent.

## Evidence and actual consultation

The preserved request supplied the complete current learner, strict configuration
plumbing, final native train005 metrics and the failed native BC context. Actual
Claude Code requested `claude-fable-5-1 --effort max` with tools disabled and
completed successfully in session `812d1d51-ed83-4875-9e8c-9f879610ac12`.
`RECEIPT.json` preserves the complete reported model usage, including an auxiliary
Haiku entry; its role is not identified by the response. The main reported output
and reasoning are Fable 5.1. `REVIEW.md` is the unedited returned review.

The separate hash-bound `EVIDENCE.json` summarizes all 200 native updates. Logged
mean approximate KL exceeds 0.015 in every update; successive 50-update means
are 0.0514, 0.1687, 0.2291 and 0.2554. These are sampled KL estimates averaged
over pre-step minibatches. They do not measure final full-rollout KL or prove the
cause of failed gait. The complete deterministic direction trials also failed.

## Smallest candidate for root adoption

Use a **new, freshly fitted successor** with a new source/schema identity. Preserve
the existing default `learning_rate=3e-4`; add an explicitly optional
`target_kl=None` configuration field. For this candidate, declare
`learning_rate=1e-4` and `target_kl=0.02` at construction. A set target must be
positive and finite, with booleans rejected. Do not adapt LR automatically.

Store detached old Gaussian means and standard deviations from the actual
collection-time policy. Use its clamped standard deviation exactly. Freeze
normalizers through the update as before. Compute analytic
`KL(old || current)` for each stored observation, **sum across 18 actions and
average across rollout samples**. The diagonal-Gaussian expression is

`sum_j(log(new_std / old_std) + (old_std^2 + (old_mean - new_mean)^2) / (2 * new_std^2) - 0.5)`.

Check the initial reference for consistency, then recompute through the entire
normalizer/estimator/memory/policy/std path after each shared-model optimizer
step. When full-rollout mean KL exceeds 0.02, stop all further shared-model
steps in that update. This also stops auxiliary-estimator and critic steps;
checking only a cached policy head would miss estimator-induced action changes.
The crossing step remains accepted. This is an **early-stop safeguard with one
possible crossing step**, not a hard KL cap. No rollback is proposed; an isolated
Adam step can overshoot substantially, so record its actual size.

Keep the discriminator's existing 20-minibatch schedule and LR. Continue its
updates on the same stored transitions, permutations and expert-sampling calls
after shared-model steps stop. This preserves its planned update count and RNG
consumption for the same incoming rollout/RNG state. A later rollout can still
differ because the policy changed. Compute actor and discriminator metric
averages using their separate executed-batch counts.

Fable additionally recommends a discarded BC-only Adam optimizer. **I do not
include that additional intervention in the minimal KL/LR candidate.** The
current code carries shared Adam moments from BC; changing that is consequential
and is not justified by train005's KL evidence, since train005 used no BC. My
minimal proposal retains shared Adam and uses the newly declared `1e-4` rate for
both fresh fitting and subsequent PPO, with moments preserved and their step
counts recorded. That also changes BC optimization speed and must be evaluated
as part of the fresh fit. If root instead selects a separate BC rate or discarded
optimizer, both phase policies need explicit configuration, receipts and tests.
Never silently change LR or discard moments after claiming strict checkpoint
load. Do not import either existing BC1000 or train005 payload into the new schema.

## Required checks before any adoption reaches native execution

- Formula tests cover identical distributions, mean shift, changed variance,
  KL direction, and 18-dimension sum followed by sample mean. A local float64
  check against PyTorch's independent Normal KL agreed within 5.6e-17.
- A forced threshold crossing permits exactly the crossing shared-model step,
  then leaves every model/Adam parameter and moment unchanged for the rest of
  the update; discriminator updates still reach their declared count. Include
  an estimator-only perturbation to prove the whole actor path is monitored.
- Guard evaluation alone changes neither normalizer buffers nor RNG. Disabled
  mode retains the previous numeric training path. Report actual executed PPO
  and discriminator batches, initial/final/maximum analytic KL, largest observed
  KL increment, stop reason, LR and moments provenance. Keep existing sampled
  `approx_kl` clearly labeled; it is a different statistic.
- Nonfinite KL/reference inputs fail explicitly. Stale collection references
  fail before optimization. Handle zero executed batches as a clear failure,
  without dividing by zero or emitting NaN metrics.
- New-schema save/reload must reproduce model, normalizers, optimizer state,
  configuration, RNG and deterministic actions exactly. Old schema/source or
  changed target/LR must fail strict load. Evaluation restores the new checkpoint
  configuration and leaves statistics unchanged.

The current BC checkpoint has failed forward, quiet and stopping probes; this
review does not justify PPO on it. A newly fitted successor first needs useful
deterministic native behavior. If root later authorizes a stability diagnostic,
start with one saved PPO update and a matching native comparison before extending
the allocation. Preserve all failed results and the unchanged Stage 2 gates.

## Corrections and limits retained independently

Fable's statement that final KL is "likely above" the reported 0.2456 is not
established: KL need not rise monotonically and minibatches differ. Final KL is
unknown. Its Adam statements are only qualitative: approximate invariance to
uniform scaling does not hold generally with changing scales, existing moments,
epsilon and mixed objectives, and an individual parameter step is not guaranteed
to equal LR. No causal diagnosis follows from the gradient norms alone.

The 16-fold KL comparison between std 0.1 and 0.4 holds for equal-variance
Gaussians with the same mean displacement; it is not a measured comparison of
the trained policies. Likewise, a median of fewer than four accepted batches is
a diagnostic of frequent early stopping, not proof that LR alone is excessive.
The optimizer, loss mixture, variance and data can all affect drift.

Within-update KL excludes effective-policy changes caused by normalizer updates
between rollouts. It averages over stored states, not worst-case or unseen states,
and describes the raw Gaussian before the target limiter. It does not bound
closed-loop motion, jitter, torque demand or physical safety. The proposed 0.02
target and 1e-4 LR are engineering choices requiring measured comparison, not
certified thresholds or paper-prescribed acceptance gates.
