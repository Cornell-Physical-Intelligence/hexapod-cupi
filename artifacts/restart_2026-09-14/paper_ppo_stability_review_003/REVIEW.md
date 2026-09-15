**Scope.** Review only: authorizes no experiment; changes no train005/BC1000 checkpoint, dataset, or physical/numerical gate. Velocity-supervised BC is owned elsewhere.

**Current facts (supplied learner.py/metrics)**
- `approx_kl` is the mean of k3 over 20 minibatches, each taken before its own step; minibatch 1 is ≈0 by construction, so end-of-update KL is unmeasured and likely above 0.2456. `clip_fraction` 0.665 means most ratios were outside [0.8,1.2]. This evidences large within-update drift, not the cause of the missing gait (one seed; at std 0.1 a given mean shift yields 16× the KL of std 0.4).
- One Adam over all `model.parameters()`; the estimator output feeds the actor, so `estimator_loss` moves the action distribution. Adam is ≈invariant to uniform gradient scaling, so the global clip (pre-clip 8–12 vs 1) does not bound parameter steps; each Adam step moves parameters ≈LR.
- Normalizer stats change only in `_normalizers()` before collection; stored log_probs and the update share fixed stats; `act()` never updates them.
- `pretrain_bc` reuses the shared Adam and `learning_rate`, fits `obs_normalizer` once from the BC set, and never updates critic/`log_std`.
- Any source/Config change invalidates strict load of every existing checkpoint. Correct; keep.

**1. Guard and LR (proposal, successor only)**
- Config: `target_kl: float | None = None` (off unless set; `validate()` requires positive finite when set); the successor config sets `target_kl=0.02`, `learning_rate=1e-4`; `discriminator_learning_rate` unchanged. Both are Config fields set at construction (needs `train.py` plumbing) and part of checkpoint identity; no post-load LR edits. The lower shared LR also slows the critic (EV 0.54); accepted over adding per-group LRs.
- Definition: analytic KL(π_old‖π_new) of the diagonal Gaussian per stored state, summed over 18 dims (consistent with the summed log_prob in `ratio`), then averaged over all rollout samples. π_old is the collection-time policy whose `log_prob` is stored: record per-sample old means and the rollout's std snapshot (clamped as in `distribution()`). Old‖new is the direction k3 estimates.
- k3 vs analytic: k3 on stored actions is unbiased but action-sampled and noisy; analytic is exact per state, deterministic, RNG-free. Gate on analytic; keep `approx_kl` for continuity and log end-of-update full-rollout k3 alongside.
- Placement/scope: after every model optimizer step, `no_grad` recompute `model.distribution(data["obs"])` over the full stored rollout; if KL > target (no 1.5× factor), take no further model steps for the rest of this update (all remaining minibatches/epochs). Log `kl_initial` (before step 1; must be ≈0, else stale log_probs → raise), `kl_final`, largest single-step increment, `ppo_batches`, `early_stopped`.
- Claim strength: the crossing step stands, so the guarantee is "at most one Adam step past target", not a trust region; overshoot size is set by LR, hence the paired LR cut. Rollback is not needed for this claim and is not proposed. No adaptive LR, no optimizer split. Diagnostic only: median `ppo_batches` < `minibatches` means LR too high for the target.

**2. Shared optimizer / discriminator**
- The guard recomputes from raw stored obs through normalizer→estimator→memory→policy→`log_std`, so every shared-step effect on actions—including auxiliary estimator updates—is seen; critic changes cannot affect actions. Never gate on the policy head with cached estimator outputs.
- Early stop ends model steps only. The discriminator completes its full `epochs×minibatches` schedule on the same permutations (its loss uses stored features and expert samples, not current policy parameters), so `discriminator_steps` and RNG consumption are identical whether or not the guard fires; only `optimizer_steps` varies. Average each metric over its own executed batches.

**3. Metrics/tests (CPU stub env, before native work)**
- Zero KL: post-`collect()` analytic and k3 ≤1e-6.
- Excessive KL: tiny `target_kl` or synthetic bias shift → `ppo_batches=1`, `optimizer_steps`+1, `discriminator_steps`+epochs×minibatches, `early_stopped=True`.
- Formula: mean shift δ at equal σ gives 18·δ²/(2σ²); σ_new=2σ_old matches closed form; old‖new≠new‖old; per-dim vector sums to the scalar.
- k3/analytic agree within Monte-Carlo tolerance on a perturbed model.
- Invariance: obs/critic/amp normalizer buffers and `_rng()` exactly equal around a guard evaluation and around `act()`.
- No post-stop mutation: model/optimizer state_dicts at stop equal those at `update()` end; discriminator differs as designed.
- Finite: `kl_*` finite or raise; `allow_nan=False` passes; zero executed PPO batches raises, never NaN.
- Identity: SCHEMA bump; old checkpoints and a different `target_kl` fail `load` cleanly; new-schema train→save→fresh construct→strict `load`→exact deterministic actions, normalizers unchanged during evaluation.
- BC/Adam: declare `bc_learning_rate`/`bc_optimizer` in Config. Recommend `separate` (BC on a discarded Adam): assert PPO Adam state empty and `param_groups[0]["lr"]==learning_rate` before PPO, Adam `step==ppo_batches` after. If carried: assert `step==bc_steps`; document that β₂=0.999 retains BC gradient scale in the second moment for ~1000 PPO steps (~50 updates), so early effective steps are BC-scaled while `log_std`/critic hold none.

**4. Caveats**
- One seed, summary metrics, no final/per-step KL; the no-gait cause is unestablished.
- Between-rollout normalizer updates (largest early and right after BC's one-shot fit) alter the effective policy without parameter change; the guard cannot see them. An optional logged drift metric (same weights, old vs new stats) could; not a gate.
- KL is on raw Gaussians; the 0.040 rad/20 ms limiter and torque cap mean applied-target drift and jitter are not bounded by it (raw clip fraction 0 while requested saturation was nonzero).
- 0.02 and 1e-4 are engineering proposals (common PPO practice, 1/σ² sensitivity), not certified bounds; multi-seed comparison required.
- Physical acceptance and numerical qualification unchanged.

**Root must adopt before code/native work**
1. New source, SCHEMA, Config fields; successor constructed and fitted from scratch; no reuse of BC1000/train005 payloads.
2. KL definition, post-step placement, one-step-overshoot claim as the stated guarantee.
3. Discriminator-continues rule.
4. BC optimizer policy (`separate` recommended).
5. §3 tests passing; PPO stays unjustified until the successor's BC shows useful deterministic native behavior; native-run authorization remains root's separate decision.
