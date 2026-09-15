# Review: first PPO pilot (source_004, update 37 interim)

## 1. Evidence vs hypothesis

**Established by the numbers plus the supplied code**

- No directional tracking. Signed projection ≈ 0 with unsigned drift 0.068 m/s (2–3× the command scale) is exploration drift. Note `_metrics` divides the projection sum by all rows, not `moving_command_rows`, so the signed figure is further diluted by zero and yaw-only rows.
- Exploration is essentially initial: log_std moved ln 0.40 → ln 0.4096 in ~740 Adam steps.
- The tracking kernel is flat where the policy lives: σ = 0.03 m/s, so 0.07 m/s drift against 0.025–0.05 commands yields exp(−5…−11) ≈ 0. Standing still would score 0.5 on 0.025 commands and 0.06 on 0.05.
- Quiet dominance is the only reward-structure-consistent explanation, though still unmeasured. Every other term is bounded: tilt ≤ 0.5, roll/pitch-rate, vertical, height, effort, target_motion sum to < 0.2 in plausible regimes, termination = 0 (no falls), nonfoot = 1 + 0.02·F. Without the two unbounded quiet quadratics, −14.4/step would require 400 Hz non-foot peak forces near 600 N routinely — implausible. Scale check: 1 rad/s during a zero hold gives quiet_joint_rate = −0.05·(1/0.03)² = −56/step; a 0.03 rad target step gives −0.02·(0.03/0.002)² = −4.5. The zero-hold time fraction is already in `status()` (zero_command_rows / environment_controls); read it out.

**Not established — actor-gradient suppression**

- `self.critic` is a parameter-disjoint MLP with its own normalizer; Adam is per-parameter, so a large value gradient cannot consume the actor's update. The only coupling is the global clip coefficient min(1, 1/‖g_all‖), which multiplies the actor gradient by a critic-dominated factor. Adam is invariant to a constant factor; a time-varying one adds multiplicative noise to actor steps but does not shrink them (ε = 1e-8 is far below any plausible scaled gradient). The all-parameter pre-clip `actor_grad_norm` cannot test this either way. `approx_kl` and `clip_fraction` are logged intact and are the direct actor-motion evidence; they were not supplied.
- What value_loss 5.5e4 does say: RMS error ≈ 330, the scale of a 24-step λ-return in a zero-hold env (≈ −60/step × ~8 effective steps ≈ −500) mixed with moving holds. Structurally: value clipping uses ε = 0.2 in raw return units; the max-form zeroes the gradient once a sample's value has moved > 0.2 toward target, so the critic advances ≤ ~0.2 per update toward targets of O(10²–10³). After 37 updates it has moved ≤ ~7 — near initialization by construction. The 200-update allocation cannot converge it at this reward scale.
- Consequence: advantages ≈ raw truncated returns, whose batch variance is set by which command an env holds, not by the sampled action. After normalization, quiet-hold samples all carry large negative advantage, moving samples small positive — a command indicator, not an action signal (SNR roughly 1:10–1:30). This, not gradient magnitude, is how the quiet quadratic dominates learning.

## 2. Preferred next intervention (only if the pilot fails and the decomposition confirms quiet terms carry ≥ half of |task reward|)

Replace both quiet quadratics with log1p of the same normalized squares; weights and scales unchanged:

```
quiet_joint_rate    = −0.05·quiet·mean_j log1p((q̇_j / 0.03)²)
quiet_target_motion = −0.02·quiet·mean_j log1p((Δu_j / 0.002)²)
```

Rationale: at the acceptance scale, log1p(x²) ≈ x², so the landscape at and below the gate is nearly unchanged (−0.035 vs −0.05 exactly at 0.03 rad/s; identical curvature below). At exploration scale the per-step magnitude falls −56 → −0.35 and −4.5 → −0.11; task reward becomes O(−1), comparable with tracking (+1) and style (+1); return scale drops ~30–50×, which incidentally brings the raw-unit value clip within reach (targets O(10) → ~50 updates) without touching learner bytes or the strict-load SHA.

Hard bound vs log1p: min((q̇/s)², B) is identical below √B·s and has zero gradient above, so a policy jittering above the bound pays a flat cost with no marginal incentive to quiet down; body-velocity tracking cannot see leg jitter in the air, so a jitter plateau is reachable. It adds a free parameter B. log1p is monotone everywhere with gradient ∝ 2x/(1+x²), never zero, no new parameter, quadratic where the gate lives. Per-joint log1p then mean keeps a single fast joint visible.

Reporting fixes (§3) change no optimization and go in immediately; the log1p change is the experiment. Do not bundle a learner change in the same run.

BC first? No. BC does not touch the critic cap; a BC'd tripod policy would still be hit by −56/step during zero holds unless the replay contains stops; and style reward 0.068 (near-total discriminator separation) means AMP is currently uninformative — BC might help that, but it is a second variable.

Switch condition: if the decomposition attributes most of −14.4 to non-foot forces or tilt, log1p is not the fix and the value-clip correction (§5) becomes the preferred single change.

## 3. Minimal measurements

Reporting: accumulate `components` in `_metrics` per command class (zero / 0.025 / 0.05 / yaw / arc), time-weighted; rename the discriminator's `policy_loss`; log actor-group and critic-group pre-clip norms plus clip coefficient; critic explained variance 1 − Var(R−V)/Var(R) and return mean/std; projection over moving rows; zero-hold joint-speed RMS (mean and worst joint); nonfoot event fraction.

Held-out probes on existing checkpoints (25, 50, …, final) and successor: deterministic mean action, each bank command plus zero, ≥ 20 s: signed projected speed, lateral/yaw error, zero-hold RMS, saturation, nonfoot events, falls. No thresholds invented.

Pre-registered diagnostic comparisons (decide the next experiment, not qualification): (a) EV at updates 50 and 100; (b) sign of projected speed on all 0.025 bearings; (c) projected speed improving over three consecutive checkpoints; (d) zero-hold RMS and (e) saturation not worsening; (f) discriminator score gap / style trend; (g) decomposition under log1p confirming the quiet terms no longer dominate.

## 4. BC

Separate, if used: `pretrain_bc` → checkpoint (bc_steps recorded) → native closed-loop probe of the deterministic BC policy per replayed command (realized q/q̇ RMS vs replay, signed speed, falls, saturation) → PPO as a new run with init checkpoint SHA recorded. Alignment: action[t] = (analytic target[t+1] − neutral)/0.35 in env joint order/sign (slew never binds: 0.0375 < 0.040); previous_action field = applied post-slew target normalized identically; command at 210:213 in navigation frame (forward = −native Y); five-frame stack with the same reset-repetition rule. Verify by regenerating replay observations through the env's own builder and asserting equality. `pretrain_bc` shares the PPO Adam, so MSE moments carry into PPO — record it, or reset optimizer state as a declared learner change. Open-loop MSE proves nothing closed loop.

Identity: a task.py change makes `task_definition.json` differ → new output_dir; learner load passes but critic/Adam are old-scale, so that is a warm start, not a resume — prefer fresh init. Any learner-byte change (including a new Config field, which also fails the config equality check) is correctly rejected by strict load; the only honest path is an explicit migration tool re-saving with `migrated_from_learner_sha256` and `exact_resume=false`.

## 5. Blockers

- Raw-unit value clip (ε = 0.2 vs returns O(10²–10³)) caps critic progress at ~0.2/update: a learnability blocker for this pilot, not a code bug. Fix (learner bytes): clip in units of running return std, or drop the clip.
- Metric-only: policy_loss overwrite; all-parameter grad norm; projection diluted by non-moving rows.
- Watch item, not a change now: tracking σ = 0.03 m/s is flat at exploration scale; if the log1p run converges to a frozen stance (0.5 tracking on 0.025 commands, ≈ 0 signed speed), that is the next hypothesis.
- Style 0.068 with 1260 expert transitions and a 1024/512 D is the memorization/domain-gap concern REVIEW_002 pre-registered; the held-out variant result was not supplied, so memorization vs real gap is undetermined.
- Saturation 2.95% during exploration exceeds the prior review's 2% replay screen; informational only, no gate applies during training.
