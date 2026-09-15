**Corrections to my review**

1. Withdraw "num_envs 128 as a zero-change extension." Only the 1- and 32-replica admissions exist; 128 is a source default, not admitted, and would need separate admission plus proximity/isolation readiness. Any extension arm stays at 32.

2. The "≈78% / every joint / only sign survives" figure was analytic: it assumed requests centred on the held target and ignored joint clamp and history. It is not measured slew occupancy, and inside the limiter the full request applies. The reported saturation .0514 is motor (torque-cap) occupancy, not target-slew occupancy, which remains unmeasured and should be logged from telemetry target deltas. Final raw |a| .69 with 25% clipped means requests are mean-dominated, superseding my noise-only picture.

3. Withdraw "cycle 1 cannot substitute" and "an onset replay is required first." The raw replay's 380 controls include the 4 s settle, a real command change at control 200 with coherent history, and cycle-1 onset; onset is absent only from the selected 1860-row file and can be screened on CPU. Only walking-to-stop was never recorded.

4. EV is invariant to consistent affine rescaling of returns and predictions. Old/new EVs are non-matched because objective and reward distribution changed, not because of magnitude, and ".109 expected" was unsupported. Adam neutralises only constant rescaling; the clip coefficient moved (≈1/11 at update 35, ≈1/22 at the end), so time-varying clipping can matter and starvation is not excluded in general. In this run the actor plainly moved (KL .053, PPO clip fraction .41): the observed mode is large noisy steps, not starvation.

5. My 10 s switch probe and ≤0.05 rad/s quiet figure are withdrawn. Root's 13-probe helper with the original 20/32 s quiet windows and 8 s move + 13 s stop timing is the instrument; no quiet or stopping gate is replaced.

**What the final evidence adds**

train_003 is a characterised trajectory, not a flat null: falls 2→6→17→67 by quarter, raw clipping .02→.19, style .136→.014, |a| .69, moving signed speed 5.6e−4 m/s, .025-class kernel .134 against a .50 standing floor. The frozen D accepts cycle 3 (+.994) and the actor's expert-observation MSE is .173, worse than zero-action .064: the policy is genuinely off-manifold and drifting to large amplitude; exact-row D memorisation is weakened (adjacent, not independent).

**Retained conditional step: fresh std .1 contrast, 32 envs, 200 updates**

Seed, source, budget and replicas matched. It is informative without walking because it contrasts trends train_003 already exhibits: (a) recurrence of per-quarter falls, amplitude drift and style decay would indict the optimisation regime rather than exploration noise, redirecting to BC or step size, not further std tuning; (b) the .025-class kernel rising toward .50 and moving signed speed leaving zero would confirm jitter loss; (c) KL and clip fraction: at std .1 the same mean step costs ≈16× the KL, so PPO clipping slows mean motion in action units — a consequence of the one variable, to be logged, not a second change. If the environment clamps |a| ≤ 1, outward samples inherit the boundary's advantage and bias the mean outward; clip growth alongside |a| .69 is consistent with that, and std .1 weakens the feedback for any mean. More updates is least informative: continuation extends a deteriorating trajectory, and a same-config restart tests only seed reproducibility.

**Video observation that would favour BC first**

If evaluate_006 shows the deterministic mean standing cleanly — quiet within the original windows, no large excursions, no falls, no progress — the mean sits at a standing attractor where noise is not its binding constraint; the missing ingredient is a gait prior, and BC first becomes the one-variable step, with the handoff std declared. Large excursions, tilt or falls in the video confirm drift and keep the std contrast first.

**BC-first boundary, if chosen**

Data are existing raw rows only — settle, cycle-1 onset from control 200, cycle 2 — screened by the same acceptance/reconstruction checks (CPU, no physics). Stopping has no demonstration: either PPO learns it from the existing quiet reward, or root dispatches a native stop replay on unchanged physics, judged by the existing 8 s move + 13 s stop case, with no new gates. Provenance as before: frozen post-BC checkpoint, probe citing its SHA, re-created or declared Adam, fresh critic/D, blended normaliser declared.
