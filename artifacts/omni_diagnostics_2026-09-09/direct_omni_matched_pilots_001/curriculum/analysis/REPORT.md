# Direct PPO comparison

The bounded pilot completed. The tables show physical changes and regressions separately from optimizer loss.

Acquisition completion is not physical acceptance. Every result retains the original 1.6 Nm cap and quiet bounds; no threshold or statistical significance is invented.

Training completed 50 updates, 1200 controls × 1024 replicas (1,228,800 transitions). Wrapper time 96.95 s; 12,674 transitions/s. Strict reload reported True.

Training events: 7 terminations, 1234 timeouts, 364 nonfoot environment-steps. Requested peak 12.324 Nm; applied peak 1.600000 Nm.

Constant directions use four-replica summaries. Joint-rate traces cover one replica per direction; those trace values are never represented as all four.

| Direction | Planar error before → after (m/s) | Yaw error before → after (rad/s) | Requested saturation before → after | Requested peak after (Nm) | Δterm |
|---|---:|---:|---:|---:|---:|
| stand | 0.0379 → 0.0309 | 0.0765 → 0.0736 | 10.70% → 10.27% | 6.207 | +0 |
| forward | 0.0406 → 0.0367 | 0.0950 → 0.0915 | 10.96% → 10.43% | 5.540 | +0 |
| reverse | 0.0354 → 0.0366 | 0.0981 → 0.0918 | 9.71% → 9.77% | 6.191 | +0 |
| left | 0.0354 → 0.0346 | 0.0888 → 0.0867 | 9.52% → 9.65% | 7.832 | +0 |
| right | 0.0347 → 0.0321 | 0.0848 → 0.0848 | 10.04% → 9.83% | 5.362 | +0 |
| forward_fast | 0.0622 → 0.0599 | 0.1395 → 0.1556 | 8.91% → 8.23% | 6.016 | +0 |
| turn_left | 0.0317 → 0.0281 | 0.0867 → 0.0813 | 9.96% → 9.63% | 6.017 | +0 |
| turn_right | 0.0401 → 0.0342 | 0.0800 → 0.0811 | 11.00% → 11.11% | 6.799 | +0 |
| arc_left | 0.0420 → 0.0377 | 0.1009 → 0.1000 | 11.41% → 11.18% | 6.516 | +0 |
| arc_right | 0.0362 → 0.0346 | 0.1048 → 0.0997 | 10.55% → 9.93% | 9.236 | +0 |
| strafe_arc | 0.0330 → 0.0321 | 0.1099 → 0.1080 | 9.72% → 9.83% | 5.355 | +0 |
| diagonal | 0.0355 → 0.0359 | 0.0943 → 0.0923 | 10.51% → 10.53% | 8.045 | +0 |

Observed regressions (any positive difference, not a significance test):
- planar_error_mps: reverse, diagonal.
- yaw_error_rad_s: right, forward_fast, turn_right.
- torque_saturation_fraction: reverse, left, turn_right, strafe_arc, diagonal.

Final stop/quiet: 0/48 replicas passed the unchanged ten-second quiet checks.

| Direction | Passing replicas | Failed bounds across replicas | Worst raw joint RMS (rad/s) | Worst drift (mm) |
|---|---:|---|---:|---:|
| stand | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2778 | 67.20 |
| forward | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2792 | 84.19 |
| reverse | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2851 | 61.46 |
| left | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2598 | 77.76 |
| right | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2846 | 67.26 |
| forward_fast | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms, trial reset | 1.2910 | 1887.43 |
| turn_left | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.3004 | 49.82 |
| turn_right | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2924 | 65.68 |
| arc_left | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2917 | 58.92 |
| arc_right | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms, trial reset | 1.2808 | 77.17 |
| strafe_arc | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2749 | 59.22 |
| diagonal | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2684 | 64.78 |

All 48 stop replicas, named-joint values and before/after deltas are in report.json. Raw SDK rates and adjacent-angle interval averages remain separate; neither is substituted into the other’s score.

Next decision: Compare against the other matched branch and inspect all direction/quiet failures before any continuation. No loss-only or aggregate-only promotion.

Input paths and SHA256 values are recorded in INPUTS_SHA256.json and rechecked after analysis. No source, checkpoint, input or GPU job is modified.
