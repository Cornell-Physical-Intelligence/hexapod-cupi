# Actual quiet-priority pilot: training diagnostic review

The quiet objective is active and its conditional loss improves across the 50-update pilot. It remains smaller than PPO in all eight sampled actor gradients, with mixed local directions. These results do not establish physical quiet success or select a weight or budget change. The review covers the accepted training receipt; physical evaluation and complete terminal audit are outside this artifact.

## Mode-specific loss improves, beyond the smoke's mixture effect

The first five and last five updates are disjoint windows:

| Mean loss | Updates 1–5 | Updates 46–50 | Change |
| --- | ---: | ---: | ---: |
| Conditional quiet temporal | 0.313922 | 0.260734 | −16.94% |
| Conditional moving temporal | 0.325196 | 0.288273 | −11.35% |
| Aggregate temporal | 0.315808 | 0.270383 | −14.38% |
| Spatial | 0.001740 | 0.001410 | −18.97% |
| Total weighted CAPS | 0.258575 | 0.179628 | −30.53% |

Quiet conditional loss decreases across every consecutive five-update block: 0.313922, 0.304054, 0.296473, 0.290965, 0.286068, 0.280575, 0.274575, 0.270027, 0.265418, 0.260734. The last block is 1.76% below the preceding block. There is no plateau in these block means, but an extrapolated convergence time is unsupported.

The mean quiet share of valid pair presentations changes from 80.39% in the first five updates to 64.97% in the last five. Therefore the larger total weighted-loss improvement still mixes behavior and batch composition. Conditional quiet loss removes the quiet-versus-moving prevalence confound, but not changes in states, normalization or the on-policy distribution. These are raw policy-mean losses, not executed target-step, joint-rate or stop-gate measurements.

## Actual actor-gradient authority

| Update / minibatch | Quiet / PPO norm | PPO–quiet cosine | Quiet dot total gradient interval | Plain negative-gradient local effect on quiet loss |
| --- | ---: | ---: | ---: | --- |
| 1 / 1 | 14.71% | +0.1180 | +0.74195 to +0.74704 | decrease |
| 1 / 20 | 14.56% | −0.1290 | +0.04482 to +0.04962 | decrease |
| 10 / 1 | 8.89% | −0.0408 | +0.23098 to +0.25261 | decrease |
| 10 / 20 | 7.58% | −0.1942 | −0.73701 to −0.71649 | increase |
| 25 / 1 | 7.57% | +0.0322 | +0.34205 to +0.37173 | decrease |
| 25 / 20 | 7.49% | −0.1280 | −0.18890 to −0.16019 | increase |
| 50 / 1 | 6.60% | −0.0950 | −0.12325 to −0.09823 | increase |
| 50 / 20 | 5.95% | +0.0351 | +0.36318 to +0.38908 | decrease |

These are **weighted pre-Adam actor-gradient measurements**. All eight combined actor gradients are clipped to norm 1; actor and critic clipping is separate. Common clipping preserves component ratios and these dot-product signs. The bounds use recorded `Q·(P+Q)` and `|Q·(M+S)| ≤ ||Q|| (||M||+||S||)` for quiet, PPO, moving and spatial gradients. They are not realized Adam parameter updates. Three adverse local signs do not prove quiet learning failure, and five favorable signs do not guarantee success.

The pilot's initial quiet/PPO ratio is much larger than the smoke's 2.30/5.12%; the smoke has 32 replicas and the pilot has 1,024. The observed initial quiet gradient is similar while PPO's norm differs substantially. Batch size and trajectory composition prevent transferring the smoke's ratio as a fixed property of the objective. Within the pilot, quiet's relative norm declines toward 6%, while its loss keeps improving. The telemetry does not decompose Adam moments or measure a same-batch objective ablation.

## All 1,000 adaptive LR transitions replay correctly

Mean KL is 0.013909; 108/1,000 rows exceed the unchanged 0.02 LR-reduction trigger. LR ranges from 1e-5 to 7.59375e-5 and finishes near 1.5e-5. There are 186 rows at the floor within floating-point roundoff, versus 74 by literal equality. Thus the smoke's broad floor saturation does not describe the complete pilot. The first row has small nonzero KL and raises LR to 7.5e-5 before the first update; later observed KL spikes alone do not identify their cause.

The practical conclusion is to compare the completed policy's measured quiet/stop and tracking results before any next allocation. This training receipt supports continued diagnostic attention to quiet loss and actor authority, not an automatic longer run or a proven coefficient change. All existing physical and quality gates remain unchanged.

## Provenance and verification

- Training receipt: `tmp/direct_omni_quiet_pilot_training_observation_001/training_receipt.json`, SHA `c41560dbe34d29bbce40d1060bd1c8b0155cdcc27173dc2316806f8368ad9b11`.
- Source manifest: `ef962070bd770d03553eb824fb6caccada7d6a06ac26d2f180bad068f0a9cc62`.
- Final checkpoint bound by the accepted phase: `195593ca4d595184fefbc9161bad433f9fb9768d8fd16d514c7b755eee0a6173`. Checkpoint bytes were not fetched or reloaded by this review.
- Actual allocation: 1,024 replicas × 24 controls × 50 updates; 1,000 minibatches and eight declared sparse gradient rows. Learning receipt time is about 100.3447 seconds; this is not a general throughput benchmark.

The exact raw receipt text matches the supplied remote observation and accepted-training hash. `calculations.json` retains all 1,000 rows, the source/receipt bindings, eight gradient bounds, windows and hashes. Three focused independent tests check actual rows and reject missing late gradients, changed learning rates and corrupt mode counts. Earlier smoke evidence remains separately frozen; no source, checkpoint, GPU state, gate or tracked file was changed. Fable's final-only response and an independently checked disposition are preserved separately.
