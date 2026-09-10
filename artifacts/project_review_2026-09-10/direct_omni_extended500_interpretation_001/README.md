# Historical CAPS500: better tracking, quiet standing still fails

The historical C-study completed 500 PPO updates, but quiet standing remains **0/48 passed**, as at 50 updates. Planar error decreased in 11 of 12 command cases and median stop drift decreased from **50.42 to 34.27 mm**. Requested torque saturation increased and targets still repeatedly reach the playback slew cap. This is useful evidence of the budget response, not Stage 2 completion or a reason to launch further C-study training.

The user has now selected the **7.466088235 kg detailed direct-drive model** in `robot/active_model.json` for all future training. The next execution priority is a new model-bound native import/physics admission. No historical C checkpoint, stance, motor adapter, gain, torque limit or action contract is silently transferred.

## What is compared

This compares normal CAPS50 (1,228,800 transitions) with normal CAPS500 (12,288,000 transitions), independently started from the original checkpoint with fresh optimizer state. It is **not a matched-budget intervention**, multi-seed study, significance test or causal attribution. Source and host versions differ; declared physical, objective and evaluation settings match. The initial constant evaluation reports are exactly equal. The older native contract weights all temporal pairs by 0.1; the newer contract separately serializes an equal 0.1 quiet coefficient. The comparison preserves both raw selections rather than claiming their schemas are identical.

The frozen 500 analyzer's before/after table compares **original checkpoint → CAPS500** (`constant_baseline_kind=same_pilot_initial`). The table here compares **CAPS50 → CAPS500**. Both are valid, different baselines. Their regression lists should differ. In particular, the CAPS50 diagonal reset at 20.86 s belongs to the separate stop trial; constant trials have zero terminations in both reports.

| Quantity | CAPS50 | CAPS500 |
|---|---:|---:|
| Formal quiet passes | 0/48 | 0/48 |
| Median maximum quiet planar excursion | 50.422 mm | 34.272 mm |
| Median maximum quiet heading excursion | 7.852° | 6.245° |
| Median worst-joint SDK RMS | 1.27354 rad/s | 1.26290 rad/s |
| Worst adjacent-angle interval RMS | 1.50176 rad/s | 1.47821 rad/s |
| Median worst-joint position range | 0.31820 rad | 0.29662 rad |
| Worst-joint target-step p95 in every row | 0.04000002 rad/20 ms | 0.04000002 rad/20 ms |
| Mean requested saturation, all quiet joints/samples | 10.5454% | 11.1227% |
| Quiet requested peak | 6.9941 Nm | 5.8165 Nm |
| Entire 32 s stop-trial requested peak | 8.0824 Nm | 7.2553 Nm |
| Constant-trial requested peak | 7.1081 Nm | 6.7239 Nm |

The unchanged quiet bounds are 10 mm planar excursion, 2° heading excursion, 0.03 rad/s SDK RMS, 0.02 rad joint range, 0.002 rad target-step p95 per 20 ms, 0.5% worst-joint requested saturation and 1.60001 Nm applied torque. Requested and clamped applied torque are separate measurements. All trials fail the combined gate; this does **not** mean every individual bound fails.

## Reset windows remain failures

The 500-update worst score, **1.971535 m**, includes env20 terminating on base contact at **30.24 s**, inside the final quiet window. Its next recorded pose jumps **1.981822 m** through reset. It is not continuous walking drift and remains a failed trial. CAPS50 env20 and env44 reset at 18.82 and 20.86 s, both before quiet scoring.

| Descriptive subset | CAPS50 median / max | CAPS500 median / max |
|---|---:|---:|
| Entire 48-row formal window, resets retained | 50.422 / 75.108 mm | 34.272 / 1971.535 mm |
| No reset anywhere in each 32 s trial | 49.912 / 75.108 mm, n=46 | 34.154 / 54.581 mm, n=47 |
| Same 46 row IDs, excluding env20 and env44 in both | 49.912 / 75.108 mm | 34.272 / 54.581 mm |

Subset descriptions do not replace the 48-row gate. Both actual and requested commands are exactly zero throughout the original 500-sample quiet window. Its first and last samples are 22.02 and 32.00 s: 9.98 s endpoint span, with the original scorer's declared 10 s window retained.

## Directional comparison

The full 12-case table is [DIRECTION_TABLE.md](DIRECTION_TABLE.md), with exact unrounded values, commands and additional SDK/position-derived metrics in [comparison.json](comparison.json). Each constant case aggregates four replicas. Its adjacent-angle trace covers only one replica per case; it is not a four-replica angle-rate audit.

Planar error decreases in 11/12 cases; arc-right increases by 0.001904 m/s. Yaw error decreases in 6/12 and increases in 6/12. Requested saturation increases in 9/12, with the equal-case mean increasing from 10.1188% to 10.4685%. The largest planar reduction is forward-fast, 0.010813 m/s; its yaw error increases by 0.018885 rad/s. These mixed outcomes do not establish all-direction quality.

## Independent rate and action evidence

`compare.py` independently reproduces all 48 rows' adjacent-angle RMS from each raw stop trace. Only the interval following a pre-reset terminal row is excluded from this diagnostic derivative; the reset itself remains a failure. SDK RMS and adjacent-angle RMS are kept separate. The independent float64 SDK reduction matches the original float32 scorer within 2e-6 rad/s; this is a numerical replay tolerance, not a physical gate change.

The copied [independent jitter review](jitter_review/README.md) retains all seven frozen payloads and its original manifest. A fresh replay reproduced its report byte-for-byte. Across the 47 entirely reset-free quiet rows, raw action, executed target and actual joint angle share a **4.4 Hz** aggregate peak. Raw action has 75.89% of spectral energy in 2–5 Hz, executed target 64.19%, and actual angle 54.37%. Only 1.60%, 1.40% and 0.29% lie in 20–25 Hz. A dominant one-control 25 Hz alternation is not supported.

The same subset has 87.21% of adjacent target increments at the cap, 35.53% sign reversals among consecutive nonzero increments, and 18.03% raw actions outside normalized clipping bounds. The recorded clip and slew recurrences replay exactly. These are closed-loop descriptions, not identification of a single cause. Normalized energy fractions across different signals do not prove absolute attenuation.

The historical actor receives previous **clipped normalized action**, not explicit executed target/limiter state. That is an interface concern for the new policy contract, not permission to patch a historical actor. Earlier original-policy controls found little benefit from removing observation noise, worse saturation with an 80 ms filter, and quiet active-PD zero-action behavior. Those different-checkpoint, different-limiter controls do not admit the new model, but argue against blindly repeating a generic noise/filter sweep.

## Decision and independent partnership

The next bounded step is the canonical model's native import/SDF cooking/identity inspection, with no inferred gains or learning. Native frames, actual joint order, inertias and collisions must be read back before separately reviewed actuator and support-stance admission. Suspended holds and bounded excitation become useful only after their drive/envelope settings are justified. The CAD zero pose is not automatically a valid ground-support stance. The old 4.4 Hz observation is a diagnostic clue, not a prescribed new-model excitation.

Once a new runtime is admitted, record raw actor inputs, commands, requested and executed targets, actual q and SDK rates, limiter state, reset boundaries and requested/applied torque with synchronized physics timing. Establish policy-free quiet behavior before attributing a learned-policy failure to budget or objectives. Any new action/history architecture gets an explicit version and independent review. More C training is outside the current model directive.

One tools-disabled **Claude Fable 5.1 / maximum effort** consultation completed successfully. Its final response and exact session/model metadata are preserved; no internal thinking or JSONL event stream was collected. The CLI usage metadata also reports an auxiliary Haiku request; the substantive requested Fable model is separately recorded, with no manual fallback. [FABLE_DISPOSITION.json](FABLE_DISPOSITION.json) accepts the admission-first recommendation while correcting baseline/termination conflation, the claim that every bound failed, a categorical assertion that sensing floors were never measured, and overly strong causal or budget-invariance wording. The original response is preserved unchanged.

## Reproduction and publication

From the repository root, using fresh output paths:

```sh
python3 -B <this-bundle>/compare.py --repo-root . --output <fresh-comparison.json>
.venv/bin/python -B <this-bundle>/replay_jitter.py --repo-root . --output <fresh-jitter-directory>
```

The jitter portability adapter rebinds only the old script's fixed checkout-root assignment after verifying its source hash; all analysis code and the copied owner files remain unchanged. [REPRODUCTION.json](REPRODUCTION.json) records exact report parity. Inputs are pinned by repository-relative path and SHA-256 rather than copying large traces or checkpoints. `CONTEXT_INPUTS_SHA256.json` records the policy/model decision context as read; future legitimate document revisions are not silently assumed identical.

Initial comparison setup errors are retained as `INITIAL_*.txt`: old-schema fields, manifest filename, explicit quiet-weight serialization, reduction precision, and the pre-reset interval rule were corrected in this new analysis. They are not altered raw data, relaxed gates or failed physical experiments.

This directory contains CPU interpretation only. Root owns publication and all runtime actions. Any tracked adoption must follow `docs/PROJECT_SITE.md`: a bounded append-only site update, relevant STATUS/PLAN/model context and validated site build. Historical walking evidence, current omni work, terrain/perception preparation and the final survey mission remain separate milestones.
