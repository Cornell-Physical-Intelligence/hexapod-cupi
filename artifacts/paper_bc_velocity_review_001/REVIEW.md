**Scope.** Advisory; nothing adopted; native physics, gates, and PPO KL/LR untouched. Adopted record: fit001, fit002 (708ab3c5…), evaluate009/010; pending: fit002 native contrast. Causal statements below are hypotheses.

**1. Objective and routing (proposal)**

Same 1000 steps / batch 512 / std .1 / seed / dataset002:

L = mean‖μ−a‖² (normalized actions, unchanged) + 1.0·mean‖v̂−v‖² (raw m/s; v = `velocity_targets_navigation_mps`, asserted finite [M,3], documented as [-s37, s36, s38], the critic[-3:] convention `update` already uses).

Coefficient 1 raw m/s is chosen because it is exactly PPO's term, not because it is well scaled: at ~.025 m/s targets it contributes ~6e-4 once the ~.8 m/s bias is gone. Any other unit or weight adds a second handoff discontinuity; scale-awareness belongs in diagnostics.

Routing: in `pretrain_bc` only, feed `velocity.detach()` to the policy concatenation; `actor()` and `update` stay byte-identical. Then ∂L_action/∂θ_est = 0 by construction, the estimator has one objective, and the policy learns an m/s-scaled input. Undetached routing should converge to nearly the same estimator (velocity gradients dwarf 1e-5-scale action gradients after the first steps) but forfeits attribution. Global `clip_grad_norm_` (max 1) still couples components through the shared clip factor; log pre-clip norm and clip activation.

No warm-up/freeze in the candidate: it adds a schedule the paper does not specify (IV-B is concurrent). An estimator-only fit (action term dropped) is a diagnostic bound only, never probed natively.

Source contract: new learner SHA; `bc_velocity_coefficient`, `bc_detach_velocity` in Config (hence checkpoint identity) and behavior_cloning.json with dataset SHA f016af1a…, target field/convention, baselines. Regression first: fresh source at coefficient 0 (term skipped), undetached, must reproduce fit002's model state_dict exactly (diagnostics consume no RNG). Only then is the candidate's delta objective-only.

**2. Sufficiency and validation**

The objective is faithful to IV-B where it matters: privileged velocity is a loss target, never actor input. It is not evidence: a constant-zero estimator scores RMSE ≈ .025 m/s, a 30× "improvement" over .839 carrying nothing. Report:

- Baselines: zero-velocity MSE, train-mean MSE, ridge 210→3 (linear recoverability bound), previous-action copy (fit001: 9.531e-4).
- Skill 1−MSE_est/MSE_baseline and Pearson r per axis (fit001's z, 1.492, was worst), per command (21 steady, 15 onset), per segment (steady/onset/zero-transient/zero-settled), train vs related cycle3 separately.
- Collapse: std(v̂)/std(v) per axis; positive skill vs zero but ≤0 vs mean baseline = trivial.
- Gradient components every 50 steps: ‖∂L_action/∂θ_policy‖, ‖∂L_action/∂θ_memory‖, ‖∂L_vel/∂θ_est‖, ‖∂L_action/∂θ_est‖ (must be 0), clip factor.
- Alignment: within contiguous replicas, score the fitted estimator against targets shifted −1/0/+1 controls; the minimum must sit at 0. Confirm obs[213:231] equals the prior row's action per replica.
- Expert self-consistency: dataset mean v per command vs commanded. If expert rows at command .05 do not themselves reach ~.05 m/s, propulsion is absent from the data, not the fit.

Nothing here qualifies walking.

**3. Handoff semantics (source-derived)**

The three native probes use the deterministic mean and depend only on weights + obs_normalizer; optimizer state is irrelevant to them. Decide optimizer semantics before PPO, not before probes.

At PPO start under current source:
- Adam: estimator/memory/policy carry step-1000 moments from 1e-5-scale losses; log_std and critic had grad None, so their state is fresh. If PPO's O(1) gradients dwarf stored second moments, the bias-corrected first step for stale parameters is roughly 2.5× a fresh Adam step. Recommend a dedicated BC optimizer over estimator/memory/policy (identical BC trajectory: Adam is per-parameter and critic/log_std never receive grad) so PPO's Adam starts clean; record it explicitly.
- obs_normalizer: BC count ≈3760; `collect()` calls `_normalizers()` before every rollout (128 rows, then 3072), diluting BC statistics to ~54% weight by the second rollout, ~37% by the third. Normalized history under BC'd weights shifts from update 1; log mean/var deltas.
- Estimator: with the candidate, PPO's estimator term starts ~1e-4 rather than ~.7, removing a recalibration of a policy input at update 1. Whether that transient contributed to train005's KL .2456/clip .665 is untested.
- Anchor strength: once calibrated, the raw-m/s term contributes ~1e-4 to the PPO loss while `actor()` routes policy gradients into the estimator undetached; drift back toward a latent during PPO is possible. Flagged; no PPO-side change proposed here.
- Critic, critic_normalizer, AMP normalizer, discriminator: fresh.

Earliest matched probes: the same forward 20s / quiet 20s / move 8s→stop 13s, same seeds and durations, fit002 (dataset-only) vs candidate (dataset+objective), identical metrics plus native per-axis estimator RMSE and per-leg support fractions versus the expert's. The other 13 probe cases and 96 Stage-2 cases remain outstanding.

**4. Prioritization, conditional on fit002's native contrast**

Fit the candidate now regardless: offline, ~4 s, no gate exposure, diagnostics stand alone; its native probes wait for fit002's result. Then:
- fit002 quiet improves, forward still ≈0: onset coverage exists but propulsion does not transfer. Check expert self-consistency; if the expert propels, treat as closed-loop shift — an on-policy question (PPO or genuine native corrective data), not calibration.
- fit002 quiet unchanged despite 640 zero rows: zero rows alone are not the fix; shift under the 5-frame closed loop is the leading, unproven hypothesis.
- move→stop: dataset002 has no walking-to-stop rows, so stop failure is expected and uninformative for either hypothesis. For BC-only qualification, genuine native deceleration transitions (commanded-to-zero IK targets physically replayed, accepted rows only) are required; relabeling is not a substitute. On a PPO path, the estimator handoff comes first and stop data is deferred.

Root owns selection.
