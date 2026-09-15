# Independent learning audit and conditional recommendation

This is advice for root review after the actual deterministic recording. It adopts no experiment, modifies no source or dataset, and grants no Stage 2 or hardware acceptance.

## What was inspected

The accepted native replay's four NPZ hashes match its preserved report. Its 1860 training pairs select cycle 2, controls 260–319, from accepted environments 0–30. The third cycle is separate. Recorded joints/velocities match native endpoints exactly. Five-frame angular/joint histories reconstruct exactly; gravity differs by at most 1.9e-9 in the independent CPU calculation. Commands match observation columns 210:213. Previous-action features reflect actual held targets; normalized desired-action reconstruction differs from recorded target by at most 2.3e-8 rad. All arrays are finite; action range is −0.73244…+0.69968 and maximum actual target step is 0.03750977 rad.

All 21 commands have demonstrations. Training-cycle translation achieves 81.7–87.3% of commanded speed; pure yaw 73.5–73.8%. The 0.05 m/s forward command realizes 0.04182 m/s. Moving support is usually at least three toes, but one rightward sample has two. These remain replay feasibility results, not formal qualification.

Uniform sampling gives the first ten commands twice the representation of the remaining eleven. Zero-command examples are 6.45% of the data. There are no command transitions, walking-to-stop examples or independent-period examples in the selected dataset. A copy-previous-action baseline obtains MSE 0.000953 versus the zero predictor's 0.064274: low one-step imitation error can coexist with a frozen controller.

There is no direct heldout-file ingestion by PPO/AMP/BC and no identical full state/observation row across the two files. Nevertheless all target actions repeat; 31 boundary states are shared, the first 124 heldout observations contain training-history frames, and both cycles influence row acceptance. Cycle 3 is a same-trajectory sanity check conditioned on selection, not an independent validation set.

The retained raw cycle 1 does contain command-aligned onset after four seconds of settling. A separate CPU screen using the existing replay numbers finds coverage for 15/21 commands; six exceed the 2% requested-saturation screen during onset, including 22.5% for 0.05 m/s leftward. Do not append that cycle wholesale or silently change the accepted file. Even usable onset follows a settled state, whereas formal probes command from the original reset state; coverage does not establish reset-start generalization. No stopping motion was recorded.

## Completed corrected-reward run

The 200-update checkpoint SHA is `27c7ef7c15cc61faf90fdb10d636f2bbd7893f213cc14afdf45ec2656cfaed9d`. It contains 153600 native transitions. Cumulative moving-only signed speed is 0.000557 m/s, requested-torque saturation 5.139%, and terminations 92. The successive 50-update blocks have 2, 6, 17 and 67 terminations. Mean raw-action clipping increases from 2.26% to 19.28%; style falls from 0.1358 to 0.0139. The final task reward is 0.2409, EV 0.0989 and action std 0.4383. Scale correction removed the dominating quiet cost; it did not demonstrate directional learning. The last quarter deteriorated under stochastic rollout.

CPU inference with the exact final discriminator gives average scores +0.99699 on training replay and +0.99409 on cycle 3, with style near one on both. Exact-row memorization alone is weakened as an explanation; the same-period split cannot establish wider generalization. The actor's target MSE is approximately 0.173 on both cycles. These are offline diagnostics, not closed-loop tests.

## Conditional next experiment

Fable 5.1 at maximum effort completed a real tools-disabled review and same-session follow-up, session `82e1442c-ccd4-4c27-94bd-c234a1b41893`. Original responses and requested/model-reported identities are preserved. Its corrected recommendation is a fresh 32-environment, 200-update contrast changing only initial exploration std from 0.4 to 0.1. I consider that the smallest informative intervention **if the deterministic mean also makes large excursions, drifts or falls**. Preserve seed, rewards, physics, limiter, architecture and all gates; measure actual target-step occupancy separately from motor saturation, and compare direction, contact, quiet behavior, KL and late-fall trends. Lower variance is a hypothesis, not a walking guarantee. If it only produces clean standing, stop tuning variance and investigate gait acquisition.

**If the deterministic mean stands quietly but cannot start commanded motion**, a separately checkpointed BC pilot becomes preferable as a test of missing gait coordination. Use a new dataset identity with independently screened existing onset/settle rows and accepted steady rows; keep excluded rows/failures. Decide stop coverage explicitly. Fit alone cannot justify PPO: first probe the deterministic BC actor from the original reset, under the original command/stop/quiet timings, including signed movement and motor/contact readouts. If starts or stops fail, extend native demonstration coverage or diagnose the fit before PPO. BC normalizers, dataset SHA, seed, bc_steps and shared-Adam moments must be recorded. Existing pretrain_bc neither changes log_std nor explicitly supervises its velocity estimator; the native probe and later PPO handoff must acknowledge that. No implicit optimizer reset or source-check bypass is acceptable.

**If deterministic probes show partial correct motion and improving matched checkpoint behavior**, bounded continuation can be justified. It should remain 32 environments: only 1 and 32 have admission. The 200-update run is only 96 simulated seconds per replica; that is insufficient to claim asymptotic failure, but extending its worsening trajectory without a favorable native signal is weakly justified.

The paper describes trajectory-optimization priors, adversarial transition rewards and concurrent PPO; its inspected training section does not explicitly describe BC pretraining or reference-state initialization. Both would be declared adaptations. RSI additionally changes reset admission. [Primary paper, Sections III–IV](https://arxiv.org/html/2511.03167v1#S4).

## Corrections retained rather than hidden

The follow-up withdraws the false 128-environment admission claim, the assertion that all raw onset data are absent, an inappropriate stopping schedule and substitute quiet threshold, and overstrong EV/Adam/noise claims. Remaining caution: the analytic 78% slew estimate assumes the Gaussian is centered on the held target; final means have drifted, so actual telemetry must test it. Clipped-action growth is consistent with several mechanisms and does not prove that clipping necessarily biases PPO outward. Likewise clean deterministic standing does not causally rule out a training-noise problem. The proposed branches are engineering priorities, not identified causes.

Only root's review of the recording and these receipts can select the next run. Every original qualification gate remains unchanged.
