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
