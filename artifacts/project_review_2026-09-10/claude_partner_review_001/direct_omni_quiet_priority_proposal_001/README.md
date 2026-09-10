# Proposed next bounded PPO experiment: quiet temporal priority

Run **one new 50-update, 1024 × 24 ablation from the same original checkpoint** with stronger temporal regularization only on exact-zero-command pairs. Compare it to the completed CAPS and curriculum pilots. Keep all physical limits, observations, schedule, optimizer settings and cold evaluation gates unchanged. This is a proposal for root review; no new training source or launch is admitted here.

## Why this experiment

The actual quiet rollouts show about 87.5% of joint target increments hitting the .04 rad limit. The current CAPS temporal loss decreases only 1.44% from the first ten to last ten updates, and neither pilot passes any quiet trial. That supports testing a stronger objective directed at this behavior. It does **not** prove that a longer unchanged run cannot work.

The new census also rules out a missing quiet mask: **68.34% of valid CAPS pairs are already at zero command**. It replays the exact command/reset pair rule over all 1024 recorded rows, including the initially invalid pair, and matches every update's reported valid-pair fraction to numerical precision. The current regularizer supplies gradients to both current and previous policy means; it does not differentiate through actual target clipping/slew. Thus limiter saturation does not itself zero this CAPS gradient. Its influence relative to the PPO actor gradient is unknown because those component norms were not recorded. Comparing scalar loss magnitudes cannot answer that question.

The policy observes five frames of raw previous actions, joint rates, positions and IMU data. Raw actions differ substantially from slew-limited targets. This feedback can represent a recurrent input-output pattern even with zero command, but the actual 315-element noisy observations and Jacobians were not saved. Neither an action-history instability nor observation-noise amplification is established as the cause. Do not silently replace those inputs: that changes the meaning of the preserved checkpoint and normalizer.

## Exact proposed objective

Retain the current pair mask: no terminal/reset crossing and identical consecutive command slices. Define a quiet pair only when **both** command slices are exactly zero. Let `d_i` be the current per-action mean squared difference of the two deterministic policy means.

```text
old temporal term = sum(valid_i * 0.1 * d_i) / max(sum(valid_i), 1)
new temporal term = sum(valid_i * (1.0 if quiet_i else 0.1) * d_i)
                    / max(sum(valid_i), 1)
spatial term      = unchanged 0.1 * spatial_mean_squared_difference
```

This raises each quiet pair's temporal coefficient tenfold while keeping the existing denominator and moving-pair coefficient. It does not introduce a separate quiet mean whose normalization would silently change the weighting again. Coefficient 1.0 is a deliberately testable choice, not an optimum or a convergence prediction. Keep both policy-mean passes differentiable; no action clamp, forced zero output, target smoothing filter, stopped-gradient teacher or physical gate change is introduced.

Use the original `1971b782…` checkpoint and the same initialization, std .1, 5e-5 initial/adaptive learning rate, 25% dedicated quiet schedule, episode handling and CAPS noise stream. Preserve 315/318 observations and the exact native003 finalizer. This creates a new explicitly named source/contract/guard successor. It is compared with the existing original-to-50 runs, not presented as resuming their optimizer state.

## Minimal instrumentation that makes this spend useful

Within the already collected PPO batches, record quiet/moving pair counts and separate unweighted temporal means for each update. Record every minibatch's KL and before/after rate. On the first minibatch of updates 1, 10, 25 and 50, record actor-gradient norms and cosine between the PPO actor term and the quiet temporal term before clipping, then the combined actor-gradient norm before/after clipping. Use `autograd.grad` without assigning parameter `.grad`, new random draws, optimizer steps or normalizer updates; retain the same graph for the actual update. These are diagnostics, not new acceptance conditions.

The meaningful CPU checks are the loss's exact parity at quiet weight .1, quiet-only coefficient scaling at 1.0, unchanged moving/spatial behavior, reset/command-change masks and normalizer/RNG/optimizer nonmutation by diagnostics. An actual two-update integration smoke precedes the matched pilot only if the new entry/contract requires it; do not repeat unrelated physics or observation studies. Root decides and owns dispatch.

## Decision after 50 updates

Preserve immutable decisions 10/25/50 and final strict reload. Use the same initial/final 12 constant cases and 48 moving-to-stop trials, with the existing 10-second measured quiet window. Inspect every replica and direction, requested/applied torque, nonfoot events and terminations. Keep raw SDK rates and interval-angle evidence separate.

A lower auxiliary loss is insufficient. The experiment is useful only if actual quiet target increments/saturation and pose drift improve without destroying commanded movement; formal quiet still requires the existing bounds in every required trial. If targets remain at .04 despite a materially reduced quiet policy-mean loss, inspect how the measured states and limiter map interact. If the gradient diagnostic shows the quiet term has little influence or is consistently opposed/clipped, it gives a concrete next adjustment. No automatic continuation is authorized by this proposal.

## Why not simply extend the budget now?

A bounded extension is scientifically valid: 50 updates is not an infeasibility result, and learning rates are not stuck. It would test whether the existing objective eventually escapes the oscillation. However, both completed pilots already deliver comparable quiet chatter and the aggregate CAPS change is small. A fresh 50-update objective ablation yields a more discriminating comparison at the same measured experience budget. Prior actual learning plus strict verification took about 97 seconds per 50-update pilot on Spark; this is a measurement, not an end-to-end ETA or guarantee for the new instrumentation.

If root chooses unchanged continuation instead, give it a separate identity, checkpoint/optimizer-resume contract and bounded decision endpoint. Do not label it the matched original-to-50 experiment or infer that more iterations are guaranteed to repair quiet.

## Evidence and scope

`pair_share.py` is NumPy-only and reads the frozen matched publication. `pair_share.json` contains all 50 per-update counts and receipt comparisons. `PRIOR_REVIEW.json` pins the separate frozen three-claim audit, which includes full learning-rate histories, actual quiet target analysis and resampling proof. No GPU, frozen source, tracked repository or checkpoint was changed. The project-site update proposal must be adopted by root if this design becomes a tracked experiment.
