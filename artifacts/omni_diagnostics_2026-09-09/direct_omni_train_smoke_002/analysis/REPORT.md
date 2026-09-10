# Direct PPO comparison

The two-update integration smoke completed. It establishes execution/reload evidence, not convergence or smooth walking.

Acquisition completion is not physical acceptance. Every result retains the original 1.6 Nm cap and quiet bounds; no threshold or statistical significance is invented.

Training completed 2 updates, 48 controls × 32 replicas (1,536 transitions). Wrapper time 3.49 s; 440 transitions/s. Strict reload reported True.

Training events: 0 terminations, 1 timeouts, 0 nonfoot environment-steps. Requested peak 8.476 Nm; applied peak 1.600000 Nm.

Constant directions use four-replica summaries. Joint-rate traces cover one replica per direction; those trace values are never represented as all four.

| Direction | Planar error before → after (m/s) | Yaw error before → after (rad/s) | Requested saturation before → after | Requested peak after (Nm) | Δterm |
|---|---:|---:|---:|---:|---:|
| stand | 0.0379 → 0.0385 | 0.0765 → 0.0751 | 10.70% → 10.62% | 6.661 | +0 |
| forward | 0.0406 → 0.0417 | 0.0950 → 0.0914 | 10.96% → 11.25% | 6.386 | +0 |
| reverse | 0.0354 → 0.0361 | 0.0981 → 0.0952 | 9.71% → 9.64% | 6.667 | +0 |
| left | 0.0354 → 0.0352 | 0.0888 → 0.0856 | 9.52% → 9.36% | 6.708 | +0 |
| right | 0.0347 → 0.0345 | 0.0848 → 0.0832 | 10.04% → 9.74% | 5.955 | +0 |
| forward_fast | 0.0622 → 0.0610 | 0.1395 → 0.1340 | 8.91% → 8.86% | 6.320 | +0 |
| turn_left | 0.0317 → 0.0321 | 0.0867 → 0.0894 | 9.96% → 9.99% | 6.733 | +0 |
| turn_right | 0.0401 → 0.0398 | 0.0800 → 0.0797 | 11.00% → 10.82% | 7.098 | +0 |
| arc_left | 0.0420 → 0.0426 | 0.1009 → 0.1006 | 11.41% → 11.19% | 6.862 | +0 |
| arc_right | 0.0362 → 0.0355 | 0.1048 → 0.1056 | 10.55% → 10.64% | 6.059 | +0 |
| strafe_arc | 0.0330 → 0.0321 | 0.1099 → 0.1084 | 9.72% → 9.82% | 7.612 | +0 |
| diagonal | 0.0355 → 0.0370 | 0.0943 → 0.0967 | 10.51% → 10.61% | 7.968 | +0 |

Observed regressions (any positive difference, not a significance test):
- planar_error_mps: stand, forward, reverse, turn_left, arc_left, diagonal.
- yaw_error_rad_s: turn_left, arc_right, diagonal.
- torque_saturation_fraction: forward, turn_left, arc_right, strafe_arc, diagonal.

Final stop/quiet: 0/48 replicas passed the unchanged ten-second quiet checks.

| Direction | Passing replicas | Failed bounds across replicas | Worst raw joint RMS (rad/s) | Worst drift (mm) |
|---|---:|---|---:|---:|
| stand | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2929 | 197.92 |
| forward | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.3296 | 199.28 |
| reverse | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.3075 | 190.37 |
| left | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2892 | 179.11 |
| right | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.3078 | 182.06 |
| forward_fast | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.3058 | 213.40 |
| turn_left | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.3118 | 196.54 |
| turn_right | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.3037 | 196.75 |
| arc_left | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.3017 | 186.48 |
| arc_right | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms, trial reset | 1.3075 | 201.18 |
| strafe_arc | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms, trial reset | 1.2944 | 1211.84 |
| diagonal | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms, trial reset | 1.3026 | 1211.09 |

All 48 stop replicas, named-joint values and before/after deltas are in report.json. Raw SDK rates and adjacent-angle interval averages remain separate; neither is substituted into the other’s score.

No historical moving-to-stop baseline exists for the cold constant run. The final stop result is an absolute screen, not an improvement claim.

Next decision: Review every directional and quiet regression before separately dispatching the matched curriculum/CAPS pilots. Existing failures remain unqualified.

Input paths and SHA256 values are recorded in INPUTS_SHA256.json and rechecked after analysis. No source, checkpoint, input or GPU job is modified.
