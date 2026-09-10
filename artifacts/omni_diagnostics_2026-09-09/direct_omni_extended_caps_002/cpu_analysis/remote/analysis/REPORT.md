# Direct PPO comparison

The bounded allocation completed. Physical changes and regressions remain separate from optimizer loss and sparse gradients.

Acquisition completion is not physical acceptance. Every result retains the original 1.6 Nm cap and quiet bounds; no threshold or statistical significance is invented.

Training completed 500 updates, 12000 controls × 1024 replicas (12,288,000 transitions). Wrapper time 1033.24 s; 11,893 transitions/s. Strict reload reported True.

Training events: 683 terminations, 12129 timeouts, 13549 nonfoot environment-steps. Requested peak 39.707 Nm; applied peak 1.600000 Nm.

Receipt-only evidence: producer reports 500 updates, complete=True, strict reload=True. Raw/checkpoint verification remains separate.

Optimizer diagnostics (measurement, not a qualification):

Retained 500 update rows / 10000 minibatches; 14 sparse gradient rows.
Minibatch LR range 1e-05–7.59375e-05; 1278 retained rows at the existing 1e-5 floor. Mean recorded KL 0.0132849.

| Loss | Initial-window mean | Final-window mean |
|---|---:|---:|
| caps_temporal | 0.316994 | 0.24156 |
| caps_quiet_temporal_mean | 0.315038 | 0.235273 |
| caps_moving_temporal_mean | 0.325355 | 0.254237 |
| caps_spatial | 0.00167116 | 0.00126961 |
| caps_weighted | 0.0318665 | 0.024283 |

Windows contain 10 disjoint updates each; exact IDs and all minibatch LR/KL values are in report.json.
Conditional quiet/moving means use their own pair counts. These optimizer presentations repeat PPO epochs; they are not unique environment steps.

| Update / minibatch | PPO actor norm | Quiet norm | Moving norm | Spatial norm | PPO–quiet cosine | Combined before → after clip |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 1 | 4.3688 | 0.064278 | 0.00327946 | 0.000675904 | 0.117987 | 4.37727 → 1 |
| 1 / 20 | 4.12138 | 0.0646027 | 0.00307183 | 0.000664294 | 0.0021745 | 4.12217 → 1 |
| 10 / 1 | 7.31101 | 0.0688592 | 0.0162601 | 0.000469962 | 0.0755356 | 7.31845 → 1 |
| 10 / 20 | 7.32448 | 0.0699111 | 0.0148594 | 0.000467735 | -0.113014 | 7.31538 → 1 |
| 25 / 1 | 8.39001 | 0.0515578 | 0.0306256 | 0.000453691 | 0.0335108 | 8.39433 → 1 |
| 25 / 20 | 5.41185 | 0.0530641 | 0.029629 | 0.00046037 | -0.0535604 | 5.40832 → 1 |
| 50 / 1 | 9.81241 | 0.0564308 | 0.0264312 | 0.000467315 | 0.0333417 | 9.81662 → 1 |
| 50 / 20 | 6.37873 | 0.0543302 | 0.0280276 | 0.000457609 | 0.0195479 | 6.3813 → 1 |
| 100 / 1 | 7.96628 | 0.0558514 | 0.0247789 | 0.000448809 | 0.0506095 | 7.9703 → 1 |
| 100 / 20 | 6.96712 | 0.0556984 | 0.0245945 | 0.000453713 | -0.118528 | 6.95817 → 1 |
| 250 / 1 | 8.03077 | 0.0500505 | 0.0248658 | 0.000418602 | -0.108322 | 8.02283 → 1 |
| 250 / 20 | 5.8902 | 0.0498949 | 0.0254532 | 0.000417106 | 0.0120648 | 5.89219 → 1 |
| 500 / 1 | 7.71731 | 0.0455389 | 0.0225276 | 0.000350207 | -0.0174272 | 7.71726 → 1 |
| 500 / 20 | 7.30991 | 0.046487 | 0.0218235 | 0.000359505 | -0.20059 | 7.29631 → 1 |

Optimizer diagnostics only. Aggregate temporal losses are not quiet target-step p95; sparse gradients do not describe every minibatch or the Adam parameter update.

Ordinary model_*.pt files are not read by this curated analyzer. Count/cadence is producer and host receipt evidence; verify all remote files with the separate terminal inventory.

Extended decision milestones: 1, 10, 25, 50, 100, 250, 500. completed declared cadence and seven local decisions verified; ordinary autosaves require separate full remote inventory

Constant directions use four-replica summaries. Joint-rate traces cover one replica per direction; those trace values are never represented as all four.

| Direction | Planar error before → after (m/s) | Yaw error before → after (rad/s) | Requested saturation before → after | Requested peak after (Nm) | Δterm |
|---|---:|---:|---:|---:|---:|
| stand | 0.0379 → 0.0233 | 0.0765 → 0.0619 | 10.70% → 11.10% | 5.185 | +0 |
| forward | 0.0406 → 0.0349 | 0.0950 → 0.1026 | 10.96% → 10.34% | 6.724 | +0 |
| reverse | 0.0354 → 0.0302 | 0.0981 → 0.0883 | 9.71% → 9.99% | 5.217 | +0 |
| left | 0.0354 → 0.0256 | 0.0888 → 0.0847 | 9.52% → 10.09% | 5.803 | +0 |
| right | 0.0347 → 0.0297 | 0.0848 → 0.0826 | 10.04% → 10.76% | 6.328 | +0 |
| forward_fast | 0.0622 → 0.0499 | 0.1395 → 0.1541 | 8.91% → 9.41% | 5.826 | +0 |
| turn_left | 0.0317 → 0.0236 | 0.0867 → 0.0845 | 9.96% → 10.35% | 5.895 | +0 |
| turn_right | 0.0401 → 0.0262 | 0.0800 → 0.0757 | 11.00% → 11.75% | 5.898 | +0 |
| arc_left | 0.0420 → 0.0334 | 0.1009 → 0.1021 | 11.41% → 10.11% | 6.500 | +0 |
| arc_right | 0.0362 → 0.0350 | 0.1048 → 0.1156 | 10.55% → 10.78% | 6.342 | +0 |
| strafe_arc | 0.0330 → 0.0264 | 0.1099 → 0.1226 | 9.72% → 9.64% | 5.834 | +0 |
| diagonal | 0.0355 → 0.0297 | 0.0943 → 0.0893 | 10.51% → 11.31% | 5.667 | +0 |

Observed regressions (any positive difference, not a significance test):
- planar_error_mps: none.
- yaw_error_rad_s: forward, forward_fast, arc_left, arc_right, strafe_arc.
- torque_saturation_fraction: stand, reverse, left, right, forward_fast, turn_left, turn_right, arc_right, diagonal.

Final stop/quiet: 0/48 replicas passed the unchanged ten-second quiet checks.

| Direction | Passing replicas | Failed bounds across replicas | Worst raw joint RMS (rad/s) | Worst drift (mm) |
|---|---:|---|---:|---:|
| stand | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2626 | 34.15 |
| forward | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2695 | 34.39 |
| reverse | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2779 | 37.50 |
| left | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2894 | 53.65 |
| right | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2743 | 54.58 |
| forward_fast | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms, trial reset | 1.2654 | 1971.54 |
| turn_left | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2733 | 37.32 |
| turn_right | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2761 | 44.09 |
| arc_left | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2792 | 48.41 |
| arc_right | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2706 | 31.90 |
| strafe_arc | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2718 | 50.06 |
| diagonal | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2872 | 49.41 |

All 48 stop replicas, named-joint values and before/after deltas are in report.json. Raw SDK rates and adjacent-angle interval averages remain separate; neither is substituted into the other’s score.

Next decision: Compare the matched constant and stop evidence with the preserved pilots. Losses, gradient alignment and learning-rate trends alone cannot promote or extend training.

Input paths and SHA256 values are recorded in INPUTS_SHA256.json and rechecked after analysis. No source, checkpoint, input or GPU job is modified.
