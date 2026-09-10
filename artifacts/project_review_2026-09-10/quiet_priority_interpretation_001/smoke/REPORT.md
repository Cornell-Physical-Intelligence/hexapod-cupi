# Quiet-priority smoke: independent diagnostic interpretation

The quiet objective has measurable actor influence, but PPO dominates the two sampled gradients. The objective is connected and is the dominant regularizer; the smoke does not establish that it can overcome the inherited policy's quiet failures. The final quiet screen remains **0/48**. Two updates do not establish convergence or justify a coefficient or budget change.

## Actual weighted actor gradients

| Update / minibatch | Quiet / PPO norm | Quiet / spatial norm | Quiet / moving temporal norm | PPO–quiet cosine | Combined actor clipping |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / 1 | 2.301% | 943.8× | 190.2× | −0.01016 | 28.5122 → 1.0 |
| 1 / 20 | 5.124% | 967.0× | 168.3× | −0.06740 | 13.1277 → 1.0 |

These are gradients of the already weighted loss terms over the full actor parameter set. Quiet has nonzero authority relative to the other regularizers, while its local direction is nearly orthogonal to PPO with a small negative projection. That does not prove an irreconcilable objective conflict. PPO itself includes the standing-related rewards; its gradient is not a pure moving-command objective.

The pinned code clips actor and critic **separately**. Actor clipping applies a common factor of 0.03507 and 0.07617 to the combined actor vector; it preserves component ratios. The corresponding quiet-component norms after that common scaling are approximately 0.02301 and 0.05135. They are **not Adam parameter-update shares**: per-parameter moments, preconditioning and history are not decomposed in this telemetry. Full actor norms also pool policy-mean and distribution parameters; these rows cannot isolate the neural mean network's share.

A bounded directional check is possible without the missing vectors. Write `Q` for the quiet gradient and `G = P + Q + M + S` for the complete actor gradient. The recorded norms and cosine determine `Q·(P+Q)`; Cauchy–Schwarz bounds the remaining dot product by `||Q|| (||M||+||S||)`. This gives `Q·G` in **[0.23745, 0.24288]** at minibatch 1 and **[−0.14643, −0.14009]** at minibatch 20. Plain negative-gradient descent would therefore locally decrease quiet loss at the first sample and increase it at the last. Positive common clipping preserves that sign. These are local, pre-Adam bounds, not measured parameter-step effects or a convergence prediction.

## The aggregate loss decrease is not evidence of quieter behavior

| Quantity | Update 1 | Update 2 | Change |
| --- | ---: | ---: | ---: |
| Conditional quiet temporal mean | 0.309061 | 0.341619 | +10.53% |
| Conditional moving temporal mean | 0.352130 | 0.321577 | −8.68% |
| Aggregate temporal mean | 0.310170 | 0.337257 | +8.73% |
| Spatial mean | 0.002248 | 0.001651 | −26.58% |
| Total weighted CAPS | 0.303305 | 0.274139 | −9.62% |
| Quiet share of valid pair presentations | 97.70% | 78.13% | changed batch mix |

Quiet/moving terms share the denominator of **all valid temporal pairs**, with weights 1.0 and 0.1; spatial has weight 0.1. Pair presentations are 2,760 quiet / 65 moving in update 1 and 2,875 / 805 in update 2. These counts include repeated PPO epochs and are not unique physical samples. The changing mix confounds total weighted loss as a quiet-progress indicator. The two update losses also use different on-policy batches and evolving normalization/weights.

The Fable partnership identified a useful exact aggregate inversion, independently verified here: mean quiet contribution equals `(weighted − 0.1*temporal − 0.1*spatial)/0.9`. It decreases from 0.302293 to 0.266943; moving contribution rises from 0.000788 to 0.007031; spatial decreases from 0.000225 to 0.000165. Individual minibatch conditional losses remain unavailable. This resolves aggregate contributions but does not prove a unique causal explanation for their evolution.

## All 40 KL/LR transitions are consistent with the source

The desired KL is 0.01, with LR reduction above 0.02. Update 1 has 19/20 rows above that trigger, mean KL 0.09989 and peak 0.17613; update 2 has 16/20, mean 0.02752 and peak 0.03945. All 40 before/after learning rates independently reproduce the existing adaptive rule. The configured floor prevents further LR reduction; this implementation does not abort an epoch on KL overshoot. This is a limitation to watch, not a newly discovered implementation error or proof that the quiet term caused the overshoot.

The existing analyzer's 31 floor rows use exact equality. Counting floating-point equivalents gives **32/40**: update 2, minibatch 6 stores `1.0000000000000003e-5`. Preserve the raw row and both counting definitions. The remaining difference is only representation precision.

## Implementation review and limits

No concrete implementation blocker was found in the reviewed objective/diagnostic path. Four exact native004 CPU regressions passed independently: equal-weight parent equivalence; quiet shared-denominator and command masking; gradient measurement preserving gradients/model/RNG; and real RSL two-update instrumentation-on/off equivalence. Three independent receipt tests passed, including corrupted LR, missing sparse rows, incorrect counts and reordered rows. The sparse decomposition's summed norm matches the actual assigned actor norm at both recorded samples.

There are only **two gradient samples**, both from update 1; update 2 has none by the declared sampling schedule. They cannot characterize every minibatch, an Adam update or the later pilot. The regularizer compares raw deterministic policy means at adjacent observations, not executed target increments after action clipping/slew, stochastic action samples, actual joint rates or physical quiet gates. Those downstream quantities remain independently measured. No new GPU work, gate changes or source changes were performed.

The useful next evidence is the already planned pilot's update 10/25/50 samples, conditional losses and pair mix, all KL/LR rows, and matched physical quiet/stop results. Use those before selecting a new coefficient or training budget; this review proposes no automatic continuation or promotion.

## Exact evidence

- Actual campaign: `tmp/direct_omni_train_smoke003_terminal_001/run/campaign.json`, SHA `fc6e962d83920180c46445aee7fa5808caa7ea509f2f9455dfd72f304dbfb3b5`.
- Training receipt: `run/train/training_receipt.json`, SHA `b56c2ec33fd57b471e13db15215abcba2d8e161a8a48f516a2df668119db54b5`.
- Final checkpoint: `ffdb347eacea5d87172fdcd122eaa6836fc43ed1bd139f20493fdf3d9672b000`; bytes independently hashed locally. Strict reload is a producer result, not rerun by this diagnostic review.
- Original checkpoint: `1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8`.
- Executed source manifest: `ef962070bd770d03553eb824fb6caccada7d6a06ac26d2f180bad068f0a9cc62`; plan `eee25089a65ec112f5eeed9c70f6310ef019ca5f077618c736fcfaa859739fc2`.
- Native004 owner: `tmp/direct_omni_recovery_001/native_004`, 43 payloads, freeze `1ab678beeba6bf74978c210c4e38f13655e25728fe66e3007e3b1db5e19b6b20`.

`calculations_003.json` preserves all 40 rows, ratios, counts, dot-product bounds, recovered weighted contributions and repository-relative input hashes. Earlier calculations are retained before the dot-product and component extensions. All 52 local terminal payloads match the provided remote audit; all 43 native owner payloads and analyzer input hashes match. This is a local independent verification, not a new remote source/cleanup audit. Both Fable 5.1 final responses and a verified disposition are preserved; no internal thinking streams are retained. The partner's inherited-Adam suspicion is explicitly refuted by the raw initialization receipt (`optimizer: reset`, zero entries, `load_cfg.optimizer: false`). Its KL-cause and spatial-loss-cause claims are not adopted.
