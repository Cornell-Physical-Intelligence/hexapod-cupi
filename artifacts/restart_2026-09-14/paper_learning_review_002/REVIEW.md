# Fable 5.1 learning audit — advisory only

**Boundaries.** No deterministic recording exists yet (evaluation 001–004 failed before controls), so nothing below is a walking verdict. Adopted so far, per root: the quiet-tail reward successor and the reporting-only learner corrections. Proposed here: one conditional contrast run; nothing is adopted, no checkpoint walks, Stage 2 is incomplete, numeric gates are untouched, and nothing implies hardware readiness. The paper (III-A/B, IV-B, V) supplies architecture and the AMP form; it mandates none of the choices discussed.

## 1. Observations (from supplied source and reported metrics)

- **Reward arithmetic (task.py).** The linear kernel exp(−e²/0.0009) has σ_v = 0.03 m/s. A motionless robot earns: zero command 1.0 + 0.3; 0.025 commands 0.50; 0.05 commands 0.062; arcs 0.17 + 0.17; yaw commands 1.0 + 0.11. Under the draw mix (15% zero, 20 equiprobable nonzero) that is ≈0.7 gross task reward per control before penalties. The reported update-135 mean of .258 is far below the *standing* ceiling, so the deficit needs no tracking explanation.
- **Noise versus limiter.** std .42 with scale .35 requests 0.147 rad of target noise per control against a 0.040 rad slew reach. By the Gaussian tail, requests exceed reach on roughly three-quarters of joint-controls, so applied targets random-walk at ±0.04 rad (2 rad/s) on every joint, zero holds included; only the sign of a request survives. The mean's influence on physics reduces to a slow drift term. This is what "saturation/jitter high" should look like mechanically.
- **Kernel–noise coupling.** Isotropic root-velocity jitter with per-axis RMS s scales the expected linear kernel by 1/(1+2s²/0.0009); s = 0.03 m/s costs two-thirds of the term even at perfect mean velocity. The per-class component logs in `task.status` can confirm whether the .258 deficit is kernel loss plus effort/height/tilt/contact rather than anything about the mean.
- **Quiet tail behaves as designed.** Slew-saturated jitter gives u_step = 400 → φ = 6.99 → 0.14 per control; 1 rad/s rates → φ ≈ 8 → 0.40; ≤≈0.5 on ≈15% of rows ≈ 0.08 mean. Task −14.7 → +.26 reflects removed dominance, not tracking.
- **Style ≈ .0166.** `style_reward` is exactly 0 for D ≤ −1, and the LSGAN loss drives policy scores to −1, so the prior channel is near-inert whatever the cause. The pilot is effectively task-kernel PPO.
- **std .40 → .42** is log_std +0.05 over 2700 Adam steps (≈2e−5/step against lr 3e−4): the entropy push (+.01/dim) is almost balanced by the surrogate. No collapse, no blow-up.
- **Terminations 4/23 040:** jittery standing, not instability.
- **Batch.** 768 samples/update over ≤32 held commands; classes are unevenly present per update. `Config.num_envs` defaults to 128; train_003 used 32.

## 2. Hypotheses (not findings)

- **H1 noise/limiter/kernel mismatch:** exploration is discarded by the limiter and converted into velocity jitter that the narrow kernel punishes; credit assignment is sign-coupled and noise-dominated.
- **H2 budget:** 153 600 transitions ≈ 51 robot-minutes; no trustworthy paper update-equivalent exists, but this is small for a 665 k-parameter actor-critic.
- **H3 standing attractor:** stillness already earns half the 0.025 kernel and all of yaw's linear term; locomotion attempts pay effort/target-motion/height/tilt/contact and risk −2; with AMP inert, PPO must discover a coordinated cycle by local search from a fixed point.

The recording discriminates: clean stand → H3 dominant; jitter/drift → H1; partial directional progress → H2.

## 3. Q1 — one experiment

*Extend samples.* The zero-change arm, but its only contrast is its own history; a flat result at 400 updates is still not asymptotic evidence. Its better form is `num_envs 128` (source default, admitted ≤128), which addresses per-update class coverage and critic fit but not H1. A continuation from a checkpoint is legitimate learner-state resumption (`load` re-resets environments), not exact rollout continuation.

*BC initialisation.* Not minimal: the post-BC checkpoint/probe path does not exist, the dataset lacks onset data (§4), and any PPO handoff at std .42 re-imposes the limiter random walk on the BC'd mean, making the outcome unreadable. At least two variables.

*Exploration std.* One `Config` field; no physics, reset, sampler or objective change; train_003 already exists as the matched control (same seed, source, budget).

**Preferred, conditional on the recording showing no commanded gait:** `train_004` = train_003 byte-identical except `initial_std = 0.1` (0.15 if root prefers a smaller step). At 0.1 the per-control request noise is 0.035 rad — the scale of the dataset's own maximum cyclic slew (0.0375) — and limiter saturation falls from ≈78% to ≈25%. Fresh weights/Adam, 32 envs, 200 updates, checkpoints at 50, ≤1800 s, same reporting learner, unchanged task.py and gates. The config change alters checkpoint identity, so no accidental cross-loading with train_003. Pilot adaptation, not a paper mandate.

Bounded scope:
- **CPU, no decision authority:** the D diagnostics of §6 on train_003's final discriminator; an optional BC MSE screen (train vs cycle 3 vs copy-previous baseline) as information only.
- **Native probes (root-dispatched):** deterministic mean, validated neutral, repeated measured history, 20 s each, identical set for train_003 final and train_004 at updates 100/200: zero, +0.025 forward, +0.05 forward, yaw +0.2, and one 0.05→0 switch at 10 s. Record signed projection, per-class kernel components, zero-hold joint-rate RMS, nonfoot events, saturation, |Δtarget| RMS, D scores of probe transitions, video.
- **Decision signals (pilot-level, not gates).** Manipulation check: rollout saturation fraction and mean |Δtarget| must fall, else the mechanism is not engaged. Primary: 0.05-linear-class `linear_tracking` mean sustained above the standing floor (≈0.06; ≥0.2 over the final 50 updates) *and* moving-class signed projection ≥0.01 m/s (≈25% of the replay's realized 0.042) *and* the deterministic 0.05 probe sustaining ≥5 s of signed progress without termination or nonfoot events → continue this arm; extension is then justified. Secondary: `action_std` trajectory (a climb back toward .4 confounds the contrast), approx_kl, clip_fraction, EV and return_std against train_003 at matched updates. Null: 0.05-class kernel at the floor with clean deterministic stands → H3; the next step is the onset/stop replay → BC path, not further std tuning.

## 4. Q2 — BC, if ever

- **Teacher forcing.** The fit minimises one-step MSE on expert states; closed-loop error compounds and no recovery data exists. Copy-previous-action reaches .00095 versus zero-action .064, so a low-MSE fit can be phase-blind; report MSE relative to that baseline on train and cycle 3 (selection-conditioned: a leakage-bounded sanity check, not generalisation).
- **Onset/stop.** Rows are steady cycle-2 under constant commands. From the validated neutral with a nonzero command, the observation (five identical frames, previous_action ≈ 0) is off-manifold and the copy-previous fixed point predicts a stand. A deterministic null from neutral cannot separate missing onset data, compounding error and phase-blindness — three confounds, hence uninterpretable. Cycle 1 cannot substitute unless the audit shows the observed command changed at gait onset; the audit lists onset as missing.
- **Coverage.** 21 commands, one cycle each, 120/60 imbalance, 6.45% zero, no switches, perturbations or alternative period.
- **Verdict.** The current file supports only a CPU screen. A native BC pilot first requires an onset/stop replay: IK rows from the validated neutral in which the observed command switches at the same control the gait starts and returns to zero at the control it stops, for at least the probe commands, under replay_001's row acceptance, hashing and reconstruction checks (previous_action = applied target, actions = (requested − neutral)/.35, no fabricated history). Native physics — root dispatch.
- **Evidence to proceed to PPO** (deterministic, from neutral, no phase/pose forcing; pilot thresholds, not gates): zero 20 s with no termination/nonfoot and all-joint rate RMS ≤0.05 rad/s; 0.05 forward with onset inside the hold and ≥10 s at ≥50% of the replay's 0.042 m/s, support ≥3, no termination; 0.05→0 stopping and holding quiet within ≈2 s; yaw ±0.2 sign-correct. Failed onset or stop means fix data or fit, not start PPO.
- **Provenance boundary.** BC ends in its own frozen checkpoint (bc_steps > 0, dataset SHA, normaliser state, seed); probes cite that SHA; the PPO run records `initialized_from`. `pretrain_bc` shares PPO's Adam, so PPO must re-create it or declare inherited moments; declare fresh critic and discriminator; declare that `collect()` blends BC normaliser statistics; the estimator is unsupervised during BC and starts velocity supervision at PPO, so log estimator loss at handoff. `log_std` is untouched by BC, so the PPO-phase std is an indispensable coupled decision — the reason to settle std alone first.

## 5. Q3 — exploration and RSI

Reference-state initialisation would be a **new adaptation** with a reset-admission consequence: initial states outside the validated neutral (joint, root, contact configuration) plus their measured history frames would need native admission before use. The inspected paper describes none; do not attribute it. Not recommended now.

std interaction: the limiter makes exploration bang-bang and state-independent, running through zero holds too, so the mean's quiet quality is never observed in training. Lowering std *may* help by letting applied targets track the mean, cutting kernel loss and incidental penalties, and possibly lifting policy transitions above D = −1. It *may not*: stillness also gets cheaper (toward H3); noise reduction cannot create a coherent cycle the mean lacks; the current mean may already stand cleanly; the entropy push may drift std back up. The matched contrast and per-class kernel resolve that ambiguity.

## 6. Q4 — discriminator

Score ≈ −1 is consistent with memorisation (≈280 passes over 1860 pairs by update 135 for a 651 k-parameter D) *and* with a genuine gap (policy transitions are jittery standing; the reference is a 0.55–0.69 rad/s tripod). Capacity is a fact, not a proven cause. Cycle 3 cannot decide: bit-identical actions, shared boundaries and period-identical states mean any D scores it as expert; it detects only exact-row overfit. CPU checks with the frozen D that do separate them: score real non-training rows (cycle-1 transient, env 31), a small-perturbation sensitivity sweep of expert pairs (diagnostic only, never data), and probe-transition scores of the deterministic mean versus stochastic rollouts. Mean transitions scoring materially higher than jittery ones would tie the inert style channel to H1.

## 7. Q5 — EV and optimisation

Returns moved from O(−10³) to O(+10) with the tail change, so old and new EV are incomparable. EV .109 at update 135 on noise-dominated returns is expected, not failure. Failure would show as approx_kl ≈ 0 with clip_fraction ≈ 0 under nonzero advantages, vanishing actor pre-clip or parameter-update norms, or flat EV/value loss over 100+ updates at stable return_std. The absolute 0.2 value clip against returns ≈ 27 is a design observation whose effect is measurable by logging the clipped-branch share (reporting-only), not a bound. The shared global clip (coefficient ≈1/11 at update 35) rescales both networks; Adam's scale invariance forbids a starvation inference; log the coefficient over time.

## 8. Evidence that would change this recommendation

- Recording shows partial directional progress → prefer extension (`num_envs 128`) over any std change.
- Per-class logs show the .258 deficit is contact/height/tilt, not kernel loss → investigate that physics before exploration.
- The mean's probe transitions already score above D = −1 → the AMP channel is live; revisit the discriminator schedule before std.
- An audited onset/stop replay exists → BC becomes a one-variable experiment behind its probe.
