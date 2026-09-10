# Direct PPO comparison

The two-update integration smoke completed. It establishes execution/reload evidence, not convergence or smooth walking.

Acquisition completion is not physical acceptance. Every result retains the original 1.6 Nm cap and quiet bounds; no threshold or statistical significance is invented.

Training completed 2 updates, 48 controls × 32 replicas (1,536 transitions). Wrapper time 3.64 s; 421 transitions/s. Strict reload reported True.

Training events: 0 terminations, 1 timeouts, 0 nonfoot environment-steps. Requested peak 8.476 Nm; applied peak 1.600000 Nm.

Receipt-only evidence: producer reports 2 updates, complete=True, strict reload=True. Raw/checkpoint verification remains separate.

Optimizer diagnostics (measurement, not a qualification):

Retained 2 update rows / 40 minibatches; 2 sparse gradient rows.
Minibatch LR range 1e-05–5e-05; 31 retained rows at the existing 1e-5 floor. Mean recorded KL 0.0637065.

| Loss | Initial-window mean | Final-window mean |
|---|---:|---:|
| caps_temporal | 0.31017 | 0.337257 |
| caps_quiet_temporal_mean | 0.309061 | 0.341619 |
| caps_moving_temporal_mean | 0.35213 | 0.321577 |
| caps_spatial | 0.00224834 | 0.00165079 |
| caps_weighted | 0.303305 | 0.274139 |

Windows contain 1 disjoint updates each; exact IDs and all minibatch LR/KL values are in report.json.
Conditional quiet/moving means use their own pair counts. These optimizer presentations repeat PPO epochs; they are not unique environment steps.

| Update / minibatch | PPO actor norm | Quiet norm | Moving norm | Spatial norm | PPO–quiet cosine | Combined before → after clip |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 1 | 28.5113 | 0.655923 | 0.00344793 | 0.000695008 | -0.0101635 | 28.5122 → 1 |
| 1 / 20 | 13.156 | 0.674167 | 0.00400468 | 0.000697141 | -0.0673969 | 13.1277 → 1 |

Optimizer diagnostics only. Aggregate temporal losses are not quiet target-step p95; sparse gradients do not describe every minibatch or the Adam parameter update.

Constant directions use four-replica summaries. Joint-rate traces cover one replica per direction; those trace values are never represented as all four.

| Direction | Planar error before → after (m/s) | Yaw error before → after (rad/s) | Requested saturation before → after | Requested peak after (Nm) | Δterm |
|---|---:|---:|---:|---:|---:|
| stand | 0.0379 → 0.0362 | 0.0765 → 0.0738 | 10.70% → 10.63% | 6.647 | +0 |
| forward | 0.0406 → 0.0417 | 0.0950 → 0.0956 | 10.96% → 10.97% | 6.932 | +0 |
| reverse | 0.0354 → 0.0362 | 0.0981 → 0.0968 | 9.71% → 9.56% | 7.143 | +0 |
| left | 0.0354 → 0.0362 | 0.0888 → 0.0835 | 9.52% → 9.44% | 6.205 | +0 |
| right | 0.0347 → 0.0343 | 0.0848 → 0.0858 | 10.04% → 9.97% | 5.554 | +0 |
| forward_fast | 0.0622 → 0.0623 | 0.1395 → 0.1341 | 8.91% → 8.71% | 6.097 | +0 |
| turn_left | 0.0317 → 0.0311 | 0.0867 → 0.0845 | 9.96% → 9.96% | 6.441 | +0 |
| turn_right | 0.0401 → 0.0390 | 0.0800 → 0.0771 | 11.00% → 11.11% | 6.712 | +0 |
| arc_left | 0.0420 → 0.0420 | 0.1009 → 0.1007 | 11.41% → 11.51% | 6.611 | +0 |
| arc_right | 0.0362 → 0.0356 | 0.1048 → 0.1064 | 10.55% → 10.61% | 6.368 | +0 |
| strafe_arc | 0.0330 → 0.0332 | 0.1099 → 0.1094 | 9.72% → 9.89% | 5.805 | +0 |
| diagonal | 0.0355 → 0.0365 | 0.0943 → 0.0934 | 10.51% → 10.76% | 7.401 | +0 |

Observed regressions (any positive difference, not a significance test):
- planar_error_mps: forward, reverse, left, forward_fast, arc_left, strafe_arc, diagonal.
- yaw_error_rad_s: forward, right, arc_right.
- torque_saturation_fraction: forward, turn_left, turn_right, arc_left, arc_right, strafe_arc, diagonal.

Final stop/quiet: 0/48 replicas passed the unchanged ten-second quiet checks.

| Direction | Passing replicas | Failed bounds across replicas | Worst raw joint RMS (rad/s) | Worst drift (mm) |
|---|---:|---|---:|---:|
| stand | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.3073 | 194.47 |
| forward | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.3015 | 194.91 |
| reverse | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.3189 | 177.18 |
| left | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.3203 | 205.34 |
| right | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2951 | 219.17 |
| forward_fast | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.3121 | 210.19 |
| turn_left | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.3096 | 205.68 |
| turn_right | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.2859 | 188.70 |
| arc_left | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.3045 | 195.84 |
| arc_right | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms, trial reset | 1.2971 | 1037.50 |
| strafe_arc | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms | 1.3160 | 175.37 |
| diagonal | 0/4 | max_heading_excursion_deg, max_joint_position_range_rad, max_joint_velocity_rms_rad_s, max_planar_excursion_m, max_requested_torque_saturation_fraction, max_target_step_abs_p95_rad_per_20ms, trial reset | 1.3142 | 1217.93 |

All 48 stop replicas, named-joint values and before/after deltas are in report.json. Raw SDK rates and adjacent-angle interval averages remain separate; neither is substituted into the other’s score.

No historical moving-to-stop baseline exists for the cold constant run. The final stop result is an absolute screen, not an improvement claim.

Next decision: Review every directional and quiet failure before separately dispatching the bounded quiet-priority pilot. Existing failures remain unqualified.

Input paths and SHA256 values are recorded in INPUTS_SHA256.json and rechecked after analysis. No source, checkpoint, input or GPU job is modified.
