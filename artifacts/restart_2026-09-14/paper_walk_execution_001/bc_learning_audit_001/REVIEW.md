# Independent BC implementation and demonstration audit

The audit found no observation/action timing, named-joint target, velocity-label, or saved-normalizer mismatch in the 3,760 selected native demonstration rows. It did find a concentrated fitting weakness: the 23 first controls of moving-command onsets contribute **50.77% of all squared requested-target error**, although they are only **0.612% of the dataset**. This is a measurement on actual teacher inputs. It does not establish why an autonomous policy subsequently stops moving or prove a remedy.

This artifact is CPU-only and does not change source, checkpoint, dataset, physics, or acceptance gates. It did not read the new paired cold/warm results. Source hashes, numerical results, and the repeatable verifier are in `SOURCE_PINS.json`, `RESULT.json`, and `audit.py`. No fitting or native execution occurred. Stage 2 remains unqualified.

## Verified implementation and data relationships

The current learner's `ActorCritic`, `RunningMeanStd`, and `pretrain_bc` ASTs match the frozen source017 functions used for BC fit004. The current environment, configuration, and replay source hashes match the actual replay declaration. Thus this review covers the existing BC path despite later PPO stability changes.

All 3,760 provenance rows are distinct actual `(raw_control, environment)` pairs. All seven recorded fields match the raw replay exactly. The five proprioceptive history samples match the corresponding pre-hold history; angular velocity, joint position, and joint velocity reconstruct exactly from the native states. Commands are current for the same impending hold. The pre-hold AMP state equals the preceding native endpoint, and the next joint position/velocity equal the endpoint after eight physics steps. All eight native held targets match each recorded control target.

The previous-action feature contains the preceding **actual limited target**, normalized by 0.35 rad, with at most `5.96e-8` float error. Desired-target reconstruction differs by at most `2.24e-8` rad. An independently written NumPy implementation of action clipping, named joint limits, the 0.04 rad slew envelope, and directed float rounding reproduces **all 67,680 actual emitted joint targets bitwise**. Desired versus emitted targets are deliberately different when the slew limit applies; the labels have not accidentally shifted by one control.

The velocity labels exactly equal native pre-hold body-origin velocity in navigation coordinates, `[-body_y, body_x, body_z]`. They are not command velocities. No reference future velocity is inserted into the actor. Gravity histories are copied recorded observations here; the original pinned reconstruction independently verified subsequent gravity from native quaternion to `1.86e-9`. Initial gravity cannot be independently recovered from AMP61 alone because that representation omits orientation.

Recomputing one full-dataset normalizer update reproduces every saved mean, variance, and count buffer bitwise (`count=3760.0001`). The actor consumes raw observations and normalizes them once. BC velocity detachment and ordinary inference produce identical forward outputs on all rows; detachment changes gradients only. Forty rows have at least one input clamped by the existing ±10 standardization bound, but **none of the 23 moving-onset first-control inputs clamp**. Clamp saturation therefore does not explain the measured error on those particular teacher inputs.

## The small average conceals a localized error

Errors below are RMS joint-target errors after multiplying normalized actions by 0.35. The limited-target calculation uses the teacher's actual previous held target and unchanged limiter; it is an offline comparison, not a new physics rollout. Subsets overlap where explicitly indicated.

| Teacher-input subset | Rows | Requested-target RMS | Share of total squared request error | Post-slew target RMS |
| --- | ---: | ---: | ---: | ---: |
| All selected data | 3,760 | 2.929 mrad | 100% | 2.232 mrad |
| Original steady cycle | 1,860 | 1.980 mrad | 22.60% | 1.979 mrad |
| Moving onset, first control | 23 | 26.681 mrad | 50.77% | 12.376 mrad |
| Moving onset, first five controls | 115 | 13.154 mrad | 61.70% | 7.453 mrad |
| Moving onset, remaining 55 controls | 1,265 | 1.849 mrad | 13.41% | 1.849 mrad |
| Forward 0.05 m/s, first control | 2 | 32.004 mrad | 6.35% | 6.746 mrad |

The estimator's RMS velocity error on the 23 first moving controls is only **0.000643 m/s**, versus 0.005843 m/s across all rows. This distinguishes the observed actor target underfit on those inputs from an obvious badly scaled velocity estimate there. It does not show that estimator errors are irrelevant later in closed loop.

The independent nearest-demo review already established identical requested onset labels for the two forward replicas and a cold initial actor input near those teacher states. Its reported **5.16 mrad** initial limited-target discrepancy is for the *actual cold initial observation*, whereas **6.75 mrad** above is the aggregate prediction error on the *two teacher onset inputs*. They are different measurements. That review also documents later closed-loop pose/target divergence. No new nearest-neighbor claim or target-averaging explanation is inferred here.

## What the training objective and data do not establish

The actor is feedforward. Its “memory” is an MLP over five 42-element samples spanning 80 ms, not a persistent recurrent state. The actor also receives current command and previous held target; its velocity estimator receives history alone. Explicit phase, contact, root height, and native linear velocity are absent from actor input. Joint positions and rates can encode gait progress, so absence of a clock is not by itself an implementation error or proof of ambiguous labels.

BC minimizes an equally weighted mean squared requested-action error on independently sampled teacher rows, plus the separately supervised velocity loss. It does not optimize the native trajectory that results from its own errors, post-slew action error, or a long-horizon limit cycle. A low pooled supervised error is therefore insufficient evidence of autonomous gait stability. The present measurement localizes where pooled error hides a large discrepancy; it does not prove that changing sampling alone restores walking.

The dataset has no positive walking command at raw reset. All moving onsets occur after the original four-second neutral hold. It includes cycle1 onsets for only 15 of 21 command identities; missing onset identities are 10, 11, 12, 15, 16, and 20. All 21 commands remain represented in steady data. The two zero-prefix replicas add 400 settling/standing rows, but there are **no moving-to-zero command transitions** in the raw replay. There are no teacher recovery actions from states reached by a drifting learned policy. Cycle3 remains excluded from fitting, but it is a subsequent cycle of the same trajectory, not an independent randomized validation condition. These are coverage limits, not corrupt labels.

## Prospective use of these findings

The already-adopted paired cold/warm native diagnostic should determine which comparison is informative next. If a fresh onset-weighted BC contrast is adopted, preserve the dataset, physics, normalization, velocity supervision, optimizer configuration, and exact checkpoint provenance while declaring the changed sampling/weighting. Report onset and steady errors separately, including actual emitted-target error, then rerun the same native windows. Better teacher-input fitting would be an intermediate result, not a walking pass. A failure after better onset fitting would not logically exclude an onset contribution.

If errors instead require corrective teacher actions away from the demonstrated path, those actions need a valid new teacher/data source and native feasibility evidence. Nearest-neighbor target copying is not validated recovery supervision. This audit supports neither an AMP reward change nor a claim that the current gates should be weakened. The concentration results were obtained after the separate Fable consultation completed; they are independent supplemental evidence, not claims that were reviewed in that consultation.
