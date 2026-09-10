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
