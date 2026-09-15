**Status labels:** [F]=measured fact, [H]=hypothesis, [P]=proposal, [A]=adopted prior decision, [NQ]=native qualification pending.

## 1. Verdict: do not continue unchanged

[F] Updates 151–200 accepted exactly one shared step per update; its full-rollout KL averaged 0.159 (8× target), max 0.498 (25×). With σ≈0.100, the diagonal-Gaussian formula makes KL 0.02 ↔ RMS action-mean shift 0.0047 per normalized dimension, 0.159 ↔ 0.0133, 0.498 ↔ 0.0235 (arithmetic, not a motion claim). [A] "crossing step retained, no rollback" therefore constrains nothing: the run is effectively one 1e-4 Adam step per update at a 20:1 D:shared ratio, critic explained variance negative in the last quarter. [F] The between-rollout normalizer change is a second, LR-independent policy change: 0.031 mean KL (last 50 updates, 128 current rows), 5.80 at update 2. [F, unattributed] 47% raw action clipping, 0.0011 m/s final commanded-direction speed, D at +0.996/−0.995 with style 0.0087. No causal mechanism is established for any of these. The conditional 1000-update recommendation is withdrawn.

## 2. Minimal recommendation [P]: two parts, one new source version (source018)

**Part A — genuine accepted-step bound = rollback + bounded same-gradient backtracking** (options b+c; c is not implementable without b):
- Before each shared Adam step, snapshot all model parameters and full Adam state (exp_avg, exp_avg_sq, step).
- After the step, compute the existing cumulative whole-actor analytic KL(old‖current) over all 3072 rows. Accept iff ≤ target_kl 0.02.
- On rejection: restore snapshot exactly, halve the LR, re-run optimizer.step() on the same stored (already clipped) gradients — no new forward/backward, no RNG. For fixed moments Adam's delta is exactly proportional to LR, so this is a line search along the Adam direction. Max 3 halvings (floor 1.25e-5). If still rejected, no further shared steps this update; D keeps all 20 steps and RNG draws unchanged.
- LR resets to base 1e-4 every update; no persistent adaptive LR (one explicit LR, no new state). [H] If first-step KL ∝ LR², 0.159–0.498 needs LR/2.8–LR/5, i.e. 2–3 halvings; log every rejected KL with its LR so the window tests this.
- Coupling: critic and estimator stay in the same step and are rolled back/halved with the actor. Deliberate: minimal and exactly testable. Critic starvation is monitored (explained variance) and would be a later explicit decision (parameter-group split), not this repair.
- Budget: ≤20 proposals per update (unchanged). Report accepted_steps, rejected_steps, attempts, lr_at_acceptance, kl_after_accepted_steps, kl_rejected. optimizer_steps keeps counting persisted steps (553 stays consistent).
- Zero accepted steps: never raise. Emit ppo_batches=0, PPO means as null, stop_reason="no_accepted_step"; train() stops via callback after 5 consecutive such updates and writes a receipt for root.
- D schedule: unchanged (20/update, same draws) for isolation; D saturation and the 20:1 ratio are logged as a separate open issue.

**Part B — freeze only the actor obs_normalizer at the migrated checkpoint** (config flag; mean/var/count buffers stay as loaded; critic and AMP normalizers keep updating). This removes the second measured KL source so the first window attributes all per-update policy change to accepted steps. Not claimed: that freezing revives BC-era behavior (BC weights were fitted to 3760-row stats now outweighed ~160×), or that normalizer drift caused the overshoots. normalizer_policy_kl reports 0 when frozen; a reporting-only shadow statistic is optional. No affine-invariance argument is relied on (±10 clamp, multiple input branches).

**Alternatives / tradeoffs:**
- Lower fixed LR or reactive adaptive LR (÷1.5/×1.5 on the next update): fewer lines, no restore logic; lags one update and still accepts overshoots whose first-step KL varied 10× (0.05–0.50) — never a bound; repeats the "early stop ≠ cap" ambiguity.
- Bounded/interpolated stats (shrink batch weight n until actor KL on actual rows ≤ bound): keeps adaptation, adds new count semantics and a second acceptance loop; defer until Stage 2 broader commands require it.
- Actor/critic optimizer split: better critic throughput; needs exact per-parameter Adam remap; defer.

## 3. Migration [P]: explicit one-off tool, not the runtime loader

- Inputs bound by hash: old checkpoint sha256 (eb2847…), old learner sha (b0a3cf…), old schema `canonical_paper_walk_ppo_amp_v2`, train006 config; new learner hash, new SCHEMA (v3), new Config (legacy fields identical; new fields: rollback on, retries 3, factor 0.5, freeze_actor_obs_normalizer on; defaults off = legacy).
- Copy unchanged: model state (incl. normalizer buffers), AMP state, both Adam state dicts (moments, steps), RNG, counters, metrics. LR param_group set explicitly (1e-4), before/after recorded. Nothing reset.
- Write a `migration` record: from/to sha256, schemas, configs, tool hash, changed fields, timestamp.
- Proof: instantiate the frozen source017 class and source018 class, load each checkpoint under its own strict loader, run one real observation batch through actor mean/velocity, critic value, D score → torch.equal on the same device; exact_equal on every tensor. New loader must reject the old checkpoint; old source must reject the new one; no compat branch in load().
- Branch point: eval013 (200) pending. Run the identical three deterministic probes on checkpoint_000100 (hash it independently first; [F] quarter 51–100 had 4.1 accepted steps, 0.4% raw clip, normalizer KL 0.004). Rule: branch from the latest checkpoint whose probes are not worse than BC on all three. If 200 and 100 both fail equally, tie-break to 100: [H] under a 0.02/update bound the mean moves ~0.005/dimension/update, so exiting a 47%-clipped regime from 200 would take order 100 updates if it must be exited — arithmetic, not a physics claim. Migration binds whichever hash root selects.

## 4. Required CPU tests (fixtures: seeded CPU env stub, num_envs 4, rollout_steps 3)

1. diagonal_gaussian_kl vs torch.distributions.kl_divergence summed, float64, unequal stds, tol 1e-10; closed form δ²/(2σ²).
2. Forced rejection (target 1e-12): parameters, log_std, all normalizer buffers, Adam exp_avg/exp_avg_sq/step exact_equal to pre-step; p.grad unchanged; retry delta at LR/2 = half the LR delta (rel 1e-6).
3. RNG (python/numpy/torch) identical across the retry loop; D and D-optimizer state after an update bitwise identical to the legacy path on the same rollout (D path is model-independent); D steps = 20.
4. Every accepted cumulative KL ≤ target on the actual rollout; every kl_rejected > target.
5. Freeze: obs_normalizer buffers exact across collect(); critic/AMP normalizers change; rejected steps never touch statistics.
6. Zero-accepted update: no exception, JSON with allow_nan=False succeeds, states unchanged, D 20 steps; consecutive-zero guard stops train() with a receipt.
7. Compatibility: flags off → model/optimizer/D/RNG states and shared metric values exactly equal the imported frozen source017 update.
8. Migration round trip, loader rejections, save/load with new fields and counters.

## 5. First native window [P → NQ]

20 updates (61,440 transitions, ~65 s at measured throughput) from the migrated branch, nothing else changed. Optimizer health: ≥1 accepted step in ≥90% of updates, no zero-step streak ≥3, all kl_final ≤ 0.02, normalizer KL 0, rejected-KL-vs-LR log reported against the LR² hypothesis, explained variance and grad norms reported, raw clip fraction/abs mean reported without a target. Task counters unchanged in definition (joint-limit terminations, saturation, zero-hold qdot RMS, commanded-direction speed). Separately: the three deterministic probes on the window-end checkpoint plus 400 Hz loads/support/quiet gates versus the branch baseline. Continue in 100-update blocks only if health holds and probes are not worse; otherwise root reconsiders (critic split, branch point). No additional BC fits. Raw Gaussian KL bounds neither physical motion nor unseen states; useful gait remains unshown.
