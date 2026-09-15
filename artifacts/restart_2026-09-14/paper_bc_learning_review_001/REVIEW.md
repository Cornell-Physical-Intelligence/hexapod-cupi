**Measurements (supplied)**
- BC fit003/004: identical tensors, zero PPO. Onset request underfit on its own training inputs (0.0321 rad RMS vs 0.00173 steady); dataset first labels identical across replicas, consistent with fit shortfall rather than label conflict.
- After the 0.040 rad/20 ms limiter, cold first held-target disagreement is 0.00516 rad. Divergence appears later: by ~900 controls, six joint channels and 38.43% of held-target scalars leave dataset ranges; actor target 0.1001 rad from nearest same-command teacher; neighbor spread 0.01056 vs actor error 0.10086, so local averaging is not shown.
- Root height differs 7.04 mm, unobserved. Matched warm native outcome: unknown.
- PPO: ckpt220 stationary; ckpt320 completes probes but fails forward tracking and six-toe support. Discriminator saturated (−0.99953/+0.99924, style 0.001409); 38.58% raw-action clipping. Stable KL is not gait evidence.

**Hypotheses (unverified)**
- H1: onset underfit compounds through held-target feedback. Weakened by the tiny post-limiter first-step error; not excluded.
- H2: the unobserved initial-state difference seeds drift. Only the matched warm run tests this.
- H3: saturated AMP starves PPO of style gradient. Plausible; not linked to the observed gait failures.

**Comparison**
- (a) Refit only: no collection, no PPO. Cheapest; falsifiable by rerunning the identical frozen-normalizer diagnosis.
- (b) Prerequisite missing: no validated off-trajectory labeler; kNN labels not established. Needs new native collection. Not smallest.
- (c) PPO-scale cost; targets a measured saturation but no measured cause; the successor it would tune does not exist yet.

**Conditional smallest action**
- Warm matched run drifts like cold (similar out-of-range fraction, ~0.1 rad teacher distance): run (a), onset weighting pre-registered by Root, normalizer and velocity term unchanged. If onset error falls but drift persists, H1 is falsified and (b)'s prerequisite becomes the blocker.
- Warm matched run tracks, cold drifts: (a) not indicated — onset fit is not the difference. H2 leads; the deficiency is unlabeled off-trajectory states, i.e. (b)'s prerequisite (a validated expert for perturbed onsets). Record the gap; no BC/AMP change.
- Both branches: (c) deferred until a successor's cold/warm diagnosis is understood.

**Evidence before adoption (Root only)**
Frozen normalizer unchanged; onset training-input error no longer an order of magnitude above steady error, steady error not degraded; in both cold and warm native reruns, out-of-range fraction below 38.43% and teacher distance below 0.1001 (diagnostic comparisons, not gates); every mandatory gate evaluated unchanged; adoption requires forward tracking plus quiet/stopping/support passing existing gates. No phase input, scripted qualification, relaxed gates, or bypasses. Diagnostic improvement alone does not qualify Stage2.
