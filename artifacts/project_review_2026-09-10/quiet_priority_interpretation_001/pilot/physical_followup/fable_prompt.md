Urgent bounded next-experiment decision with NEW actual physical counterevidence. Use Fable5.1 MAX; return final-only at most350words, no tools/edits/internal thinking stream. This is experimental prioritization, not a GPU authorization, qualification or theorem. Root owns allocation and existing gates.

The just-completed quiet-priority50 pilot has0/48 quiet, target-step p95still0.04rad/20ms, actualSDK joint RMS~1.27–1.29rad/s. forward_fast quiet excursion1.870m withtrialreset; diagonalalsoreset. Trackingmostlysmallimprovementsvsoriginal butrequested saturation regresses9/12 scenarios. Trainingpeakrequested11.865Nm vs1.6applied. Comparepreserved normalCAPS50: worstquietexcursion~75.1mm, also0/48 quiet andtargetp95.04/SDKRMS~1.28; priorCAPS50 also has forward_fast/diagonaltrialresets, so do NOTcallitreset-free. No statementthatquietcoefficientcausedregression withoutmatchedcounterfactual. Bothfreshfromoriginal1971 checkpoint withresetAdam,std.1,315/318,dynamicsunchanged. NormalCAPS usesquietweight.1,moving.1,spatial.1; quietpriority uses1.0,.1,.1. Native004 diagnosis instrumentation isCPU on/offequivalent butdifferent executedsourceidentity fromnormalCAPS native003.

Choose best NEXT bounded experiment among: (A) longer fresh-original samequietpriority objective; (B) normalCAPS500 budgetbaseline fromoriginal withmatchedstand/constant/stopdiagnostics andintermediateweights; (C) strongerquietgradientweight or architecturechange. Be explicitwhichquestion chosenexperiment answers, whatitdoesNOTisolate, why currentloss/gradient/physicalevidence supportspriority ratherthanprovingimpossibility. Do not requireopen-endedarchitectureworkorperfectcausalidentificationbeforeany usefulrun. Rootneedspracticaldecisionquicklyanddoesnotinferinfeasibilityfrom50. Discussfixedtotalbudgetandoneusefulmidpointphysicalcheckifvaluable; no relaxationofgates.

Independent provisional preference:B suppliesa budgetresponse baselinefor the existingbranchwithsmallerquietexcursions; no automaticpromotionandnotmatchedsamebudgetquietweightablation. Strongerweightis not yetprovedtargeteffectivebypreAdamnorms. Correct or disagreewiththispreferenceif evidence warrants.

Your prior training-only final:
**Verdict.** Quiet loss fell steadily on-policy across the pilot, but the eight sampled pre-Adam gradients do not show consistent quiet actor authority, and no weight-versus-budget decision is yet evidenced. No physical quiet pass is inferred. Terminal audit and scores lie outside this data.

**Sampled gradients.** Weighted components before the common actor clip and before Adam. P is PPO actor, Q quiet, M moving temporal, G the total.

| Update/mb | Q/P norm | Q/M norm | cos(P,Q) | Q·G sign |
|---|---|---|---|---|
| 1/1 | 0.147 | 196 | +0.12 | + |
| 1/20 | 0.146 | 210 | −0.13 | + |
| 10/1 | 0.089 | 43 | −0.04 | + |
| 10/20 | 0.076 | 47 | −0.19 | − |
| 25/1 | 0.076 | 17 | +0.03 | + |
| 25/20 | 0.075 | 17 | −0.13 | − |
| 50/1 | 0.066 | 20 | −0.10 | − |
| 50/20 | 0.059 | 19 | +0.04 | + |

- **Relative authority more than halved.** PPO actor norm nearly doubled while quiet norm eased a quarter. The clip was active in every row, so quiet's clipped contribution shrank in proportion. The opening ratio differs from the smoke ratios, and the pilot governs.
- **Quiet leads within CAPS.** It dominates moving and spatial throughout. The quiet-to-moving ratio fell as moving pairs grew, matching the ten-to-one weight times share ratio.
- **Cosines are small and mixed,** near-orthogonal, with no systematic alignment or conflict.
- **Dot sign is inconsistent.** Five rows positive, three negative. A negative row means plain descent locally raises quiet loss because the larger PPO component wins. The flip threshold shrank from about −0.15 to −0.06 cosine, so quiet's margin narrowed. Eight rows cannot establish a trend in sign. None of this is Adam's per-parameter update.

**On-policy losses.**

| Quantity | Update 1 to 50 | First five to last five |
|---|---|---|
| Quiet temporal mean | −14.3% | −16.9% |
| Moving temporal mean | −11.6% | −11.4% |
| Weighted total | −38.9% | −30.5% |
| Quiet share of valid pairs | 96% to 65% | 80% to 65% |

- **Weighted total is share-confounded.** Most of the recovered weighted quiet decline is the quiet share falling, as in the smoke run.
- **Quiet mean rose, then fell.** It rose at update 2, echoing the smoke rise, then declined almost monotonically with no plateau at update 50. Moving followed a shallower path. The differential is modest for a ten-to-one weight.
- **Batch change is not attribution.** Pair mix shifted in steps and visited states moved with the policy. Lower batch loss shows the current policy is smoother where it now operates. Attributing that to the quiet term needs a matched lower- or zero-weight baseline, absent here.

**Optimizer and integrity.** Adam started fresh, and iteration 1847 is not inherited state. LR sat at or near its floor for many rows and ended near it. Exact-floor, round-off-floor, and KL-overshoot rows are distinct counts and must not be conflated. KL shape supports no causal claim. The analyzer replayed receipt, hashes, and all LR transitions. Checkpoint bytes and reload were not re-verified here.

**Weight versus budget.** Not evidenced. Gradient authority is declining while loss decline continues without plateau. Neither series is causal, and near-floor Adam steps decouple them further. A decision needs a matched baseline and Adam-level update projections. No continuation or gate change follows.

Actualquietpriorityphysicalreport:
# Direct PPO comparison

The bounded pilot completed. Physical changes and regressions remain separate from optimizer loss and sparse gradients.

Acquisition completion is not physical acceptance. Every result retains the original 1.6 Nm cap and quiet bounds; no threshold or statistical significance is invented.

Training completed 50 updates, 1200 controls × 1024 replicas (1,228,800 transitions). Wrapper time 100.34 s; 12,246 transitions/s. Strict reload reported True.

Training events: 8 terminations, 1234 timeouts, 366 nonfoot environment-steps. Requested peak 11.865 Nm; applied peak 1.600000 Nm.

Receipt-only evidence: producer reports 50 updates, complete=True, strict reload=True. Raw/checkpoint verification remains separate.

Optimizer diagnostics (measurement, not a qualification):

Retained 50 update rows / 1000 minibatches; 8 sparse gradient rows.
Minibatch LR range 1e-05–7.59375e-05; 74 retained rows at the existing 1e-5 floor. Mean recorded KL 0.013909.

| Loss | Initial-window mean | Final-window mean |
|---|---:|---:|
| caps_temporal | 0.311455 | 0.272595 |
| caps_quiet_temporal_mean | 0.308988 | 0.263076 |
| caps_moving_temporal_mean | 0.321672 | 0.290167 |
| caps_spatial | 0.00165036 | 0.00141406 |
| caps_weighted | 0.251828 | 0.180992 |

Windows contain 10 disjoint updates each; exact IDs and all minibatch LR/KL values are in report.json.
Conditional quiet/moving means use their own pair counts. These optimizer presentations repeat PPO epochs; they are not unique environment steps.

| Update / minibatch | PPO actor norm | Quiet norm | Moving norm | Spatial norm | PPO–quiet cosine | Combined before → after clip |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 1 | 4.3688 | 0.64278 | 0.00327946 | 0.000675904 | 0.117987 | 4.491 → 1 |
| 1 / 20 | 4.42372 | 0.643943 | 0.00306621 | 0.00066227 | -0.12899 | 4.38749 → 1 |
| 10 / 1 | 7.51938 | 0.668578 | 0.01572 | 0.000457962 | -0.0408181 | 7.52286 → 1 |
| 10 / 20 | 8.99682 | 0.682332 | 0.0145875 | 0.000451011 | -0.194228 | 8.88747 → 1 |
| 25 / 1 | 6.61007 | 0.500327 | 0.0292341 | 0.00042904 | 0.0322222 | 6.64917 → 1 |
| 25 / 20 | 6.62253 | 0.495864 | 0.0285224 | 0.000430985 | -0.128028 | 6.57581 → 1 |
| 50 / 1 | 7.59676 | 0.501035 | 0.0245388 | 0.000423261 | -0.0950477 | 7.56518 → 1 |
| 50 / 20 | 8.17823 | 0.486376 | 0.0262084 | 0.000413183 | 0.035088 | 8.21155 → 1 |

Optimizer diagnostics only. Aggregate temporal losses are not quiet target-step p95; sparse gradients do not describe every minibatch or the Adam parameter update.

Constant directions use four-replica summaries. Joint-rate traces cover one replica per direction; those trace values are never represented as all four.

| Direction | Planar error before → after (m/s) | Yaw error before → after (rad/s) | Requested saturation before → after | Requested peak after (Nm) | Δterm |
|---|---:|---:|---:|---:|---:|
| stand | 0.0379 → 0.0296 | 0.0765 → 0.0741 | 10.70% → 11.16% | 6.287 | +0 |
| forward | 0.0406 → 0.0371 | 0.0950 → 0.0928 | 10.96% → 10.78% | 6.482 | +0 |
| reverse | 0.0354 → 0.0346 | 0.0981 → 0.0979 | 9.71% → 10.01% | 6.603 | +0 |
| left | 0.0354 → 0.0344 | 0.0888 → 0.0889 | 9.52% → 9.43% | 6.248 | +0 |
| right | 0.0347 → 0.0307 | 0.0848 → 0.0834 | 10.04% → 10.49% | 5.998 | +0 |
| forward_fast | 0.0622 → 0.0555 | 0.1395 → 0.1446 | 8.91% → 9.01% | 6.071 | +0 |
| turn_left | 0.0317 → 0.0282 | 0.0867 → 0.0867 | 9.96% → 10.01% | 6.301 | +0 |
| turn_right | 0.0401 → 0.0324 | 0.0800 → 0.0765 | 11.00% → 11.60% | 11.829 | +0 |
| arc_left | 0.0420 → 0.0371 | 0.1009 → 0.0971 | 11.41% → 11.96% | 6.517 | +0 |
| arc_right | 0.0362 → 0.0339 | 0.1048 → 0.1023 | 10.55% → 10.46% | 6.112 | +0 |
| strafe_arc | 0.0330 → 0.0316 | 0.1099 → 0.1098 | 9.72% → 9.77% | 5.893 | +0 |
| diagonal | 0.0355 → 0.0356 | 0.0943 → 0.0934 | 10.51% → 10.54% | 7.382 | +0 |

Observed regressions (any positive difference, not a significance test):
- planar_error_mps: diagonal.
- yaw_error_rad_s: left, forward_fast, turn_left.
- torque_saturation_fraction: stand, reverse, right, forward_fast, turn_left, turn_right, arc_left, strafe_arc, diagonal.

Final stop/quiet: 0/48 replicas passed the unchanged ten-second quiet checks.

| Direction | Passing replicas | Failed bounds across replicas | Worst raw joint RMS (rad/s) | Worst drift (mm) |
|---|---:|---|---:|---:|
| stand | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2738 | 63.66 |
| forward | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2872 | 73.94 |
| reverse | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2721 | 66.67 |
| left | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2936 | 84.04 |
| right | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2918 | 89.89 |
| forward_fast | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms, trial reset | 1.2701 | 1870.23 |
| turn_left | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2784 | 75.72 |
| turn_right | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2932 | 82.07 |
| arc_left | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2946 | 66.77 |
| arc_right | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2912 | 68.78 |
| strafe_arc | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2847 | 57.23 |
| diagonal | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms, trial reset | 1.2883 | 59.72 |

All 48 stop replicas, named-joint values and before/after deltas are in report.json. Raw SDK rates and adjacent-angle interval averages remain separate; neither is substituted into the other’s score.

Next decision: Compare the matched constant and stop evidence with the preserved pilots. Losses, gradient alignment and learning-rate trends alone cannot promote or extend training.

Input paths and SHA256 values are recorded in INPUTS_SHA256.json and rechecked after analysis. No source, checkpoint, input or GPU job is modified.


PriorCAPS50physicalreport:
# Direct PPO comparison

The bounded pilot completed. The tables show physical changes and regressions separately from optimizer loss.

Acquisition completion is not physical acceptance. Every result retains the original 1.6 Nm cap and quiet bounds; no threshold or statistical significance is invented.

Training completed 50 updates, 1200 controls × 1024 replicas (1,228,800 transitions). Wrapper time 96.79 s; 12,696 transitions/s. Strict reload reported True.

Training events: 11 terminations, 1231 timeouts, 369 nonfoot environment-steps. Requested peak 39.409 Nm; applied peak 1.600000 Nm.

Constant directions use four-replica summaries. Joint-rate traces cover one replica per direction; those trace values are never represented as all four.

| Direction | Planar error before → after (m/s) | Yaw error before → after (rad/s) | Requested saturation before → after | Requested peak after (Nm) | Δterm |
|---|---:|---:|---:|---:|---:|
| stand | 0.0379 → 0.0307 | 0.0765 → 0.0739 | 10.70% → 10.66% | 6.674 | +0 |
| forward | 0.0406 → 0.0363 | 0.0950 → 0.0909 | 10.96% → 10.93% | 6.692 | +0 |
| reverse | 0.0354 → 0.0348 | 0.0981 → 0.0902 | 9.71% → 9.76% | 6.393 | +0 |
| left | 0.0354 → 0.0349 | 0.0888 → 0.0884 | 9.52% → 9.49% | 6.574 | +0 |
| right | 0.0347 → 0.0321 | 0.0848 → 0.0829 | 10.04% → 10.03% | 5.401 | +0 |
| forward_fast | 0.0622 → 0.0607 | 0.1395 → 0.1352 | 8.91% → 8.53% | 5.724 | +0 |
| turn_left | 0.0317 → 0.0299 | 0.0867 → 0.0837 | 9.96% → 9.52% | 6.487 | +0 |
| turn_right | 0.0401 → 0.0334 | 0.0800 → 0.0821 | 11.00% → 10.94% | 6.408 | +0 |
| arc_left | 0.0420 → 0.0371 | 0.1009 → 0.0986 | 11.41% → 11.09% | 6.651 | +0 |
| arc_right | 0.0362 → 0.0331 | 0.1048 → 0.1015 | 10.55% → 10.07% | 7.108 | +0 |
| strafe_arc | 0.0330 → 0.0307 | 0.1099 → 0.1096 | 9.72% → 9.72% | 6.885 | +0 |
| diagonal | 0.0355 → 0.0360 | 0.0943 → 0.0921 | 10.51% → 10.69% | 6.237 | +0 |

Observed regressions (any positive difference, not a significance test):
- planar_error_mps: diagonal.
- yaw_error_rad_s: turn_right.
- torque_saturation_fraction: reverse, diagonal.

Final stop/quiet: 0/48 replicas passed the unchanged ten-second quiet checks.

| Direction | Passing replicas | Failed bounds across replicas | Worst raw joint RMS (rad/s) | Worst drift (mm) |
|---|---:|---|---:|---:|
| stand | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2873 | 72.42 |
| forward | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2904 | 59.64 |
| reverse | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2833 | 59.29 |
| left | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2908 | 45.93 |
| right | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2768 | 75.11 |
| forward_fast | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms, trial reset | 1.2793 | 63.92 |
| turn_left | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2843 | 55.17 |
| turn_right | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2935 | 66.06 |
| arc_left | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2814 | 67.85 |
| arc_right | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2766 | 74.57 |
| strafe_arc | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2870 | 51.05 |
| diagonal | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms, trial reset | 1.2899 | 65.61 |

All 48 stop replicas, named-joint values and before/after deltas are in report.json. Raw SDK rates and adjacent-angle interval averages remain separate; neither is substituted into the other’s score.

Next decision: Compare against the other matched branch and inspect all direction/quiet failures before any continuation. No loss-only or aggregate-only promotion.

Input paths and SHA256 values are recorded in INPUTS_SHA256.json and rechecked after analysis. No source, checkpoint, input or GPU job is modified.
