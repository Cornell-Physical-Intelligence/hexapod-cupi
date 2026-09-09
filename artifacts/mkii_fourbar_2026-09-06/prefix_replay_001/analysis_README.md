# Individual-prefix replay analysis

Offline diagnostic evidence; no training or hardware admission.

Exact runtime match: **True**. Exact ordered32 reset positions: **True**.

| Motor | Replay minimum (rad) | Campaign008 minimum (rad) | Difference | Worst env |
|---|---:|---:|---:|---:|
| lf_coxa_yaw | 0.0393294916 | 0.0393294953 | -3.725e-09 | 13 |
| lm_coxa_yaw | 0.0377403945 | 0.0377404019 | -7.451e-09 | 2 |
| lr_coxa_yaw | 0.0381187797 | 0.0381187834 | -3.725e-09 | 24 |
| rf_coxa_yaw | 0.0382845476 | 0.0382845439 | 3.725e-09 | 20 |
| rm_coxa_yaw | 0.0377407148 | 0.0377407148 | 0 | 13 |
| rr_coxa_yaw | 0.0399744585 | 0.0399744585 | 0 | 2 |
| lf_femur_pitch | 0.028377831 | 0.0283778459 | -1.49e-08 | 4 |
| lm_femur_pitch | 0.0270124972 | 0.0270125121 | -1.49e-08 | 28 |
| lr_femur_pitch | 0.0280028433 | 0.0280028433 | 0 | 14 |
| rf_femur_pitch | 0.0281901807 | 0.0281901509 | 2.98e-08 | 29 |
| rm_femur_pitch | 0.0254345685 | 0.0254345685 | 0 | 8 |
| rr_femur_pitch | 0.0281733125 | 0.0281732678 | 4.47e-08 | 25 |
| lf_tibia_lever_pivot | 0.00706881285 | 0.00706875324 | 5.96e-08 | 31 |
| lm_tibia_lever_pivot | -0.00587320328 | -0.00587320328 | 0 | 0 |
| lr_tibia_lever_pivot | -0.190767914 | -0.190767854 | -5.96e-08 | 6 |

## Chronological detailed events

| Physics sample | Control | Env | Event | Value |
|---:|---:|---:|---|---:|
| 35200 | 2200 | 0 | peak_cached_minus_native_pre_speed_rad_s (lf_coxa_yaw) | 0 |
| 35200 | 2200 | 0 | peak_raw_minus_recorded_P_D_feedforward_nm (lf_coxa_yaw) | 0 |
| 36079 | 2254 | 24 | first_raw_applied_difference_above_10uNm (lf_femur_pitch) | 1 |
| 36082 | 2255 | 24 | first_raw_demand_above_vendor_peak (lf_femur_pitch) | 1 |
| 36089 | 2255 | 24 | peak_applied_torque_nm (lf_femur_pitch) | -5.5 |
| 36226 | 2264 | 9 | first_pre_speed_above_no_load_with_opposing_demand (lf_femur_pitch) | 1 |
| 36246 | 2265 | 9 | first_zero_feet_above_1N (all_feet) | 1 |
| 36264 | 2266 | 9 | peak_proportional_term_nm (lf_tibia_lever_pivot) | -70.2000427 |
| 36459 | 2278 | 16 | peak_raw_demand_nm (lf_tibia_lever_pivot) | -85.2076111 |
| 36459 | 2278 | 16 | peak_raw_minus_applied_abs_nm (lf_tibia_lever_pivot) | 85.2054367 |
| 36535 | 2283 | 0 | peak_C_pin_gap_m (lf) | 0.000213820953 |
| 36535 | 2283 | 0 | peak_C_pin_relative_speed_m_s (lf) | 7.05141354 |
| 36535 | 2283 | 0 | peak_interval_average_minus_post_instantaneous_speed_rad_s (lf_tibia_pitch) | -273.350389 |
| 36535 | 2283 | 0 | peak_passive_velocity_residual_rad_s (lf_tibia_pitch) | 221.326508 |
| 36535 | 2283 | 0 | peak_post_step_speed_rad_s (lf_femur_pitch) | 80.600853 |
| 36536 | 2283 | 0 | peak_derivative_term_nm (lf_femur_pitch) | -24.1802578 |
| 36536 | 2283 | 0 | peak_pre_step_speed_rad_s (lf_femur_pitch) | 80.600853 |

## Full-prefix control-boundary events

| Control | Env | Event | Value |
|---:|---:|---|---:|
| 0 | 0 | first_zero_feet_above_1N (all_feet) | 1 |
| 2254 | 24 | first_raw_applied_difference_above_10uNm (lf_femur_pitch) | 1 |
| 2264 | 9 | first_raw_demand_above_vendor_peak (lf_femur_pitch) | 1 |
| 2265 | 9 | first_endpoint_speed_above_no_load (lf_tibia_lever_pivot) | 1 |
| 2455 | 17 | peak_raw_demand_nm (lr_tibia_lever_pivot) | -84.2972641 |
| 2475 | 23 | peak_endpoint_speed_rad_s (lr_femur_pitch) | 70.0443726 |
| 2475 | 23 | peak_passive_endpoint_velocity_residual_rad_s (lr_tibia_pitch) | 102.561661 |

## Interpretation limits

Campaign008 has only global response minima, not per-control traces. No exact earliest cross-run divergence can be recovered.

- Only controls2200..2499 have detailed800Hz traces; earlier detailed onset is unobserved.
- Events are timestamped independently; peaks do not imply coincidence or a unique cause.
- q finite difference is an interval-average velocity, not the reported instantaneous endpoint velocity.
- Control telemetry aligns post-control qdot with the preceding final-substep torque; only detailed pre_qd aligns exactly with the PD demand input.
- 480rpm is the model's vendor no-load speed point, not an established hard braking-speed capability.
- Opposing raw demand is braking intent; applied torque and the provisional symmetric envelope determine delivered braking.
- 1N foot support and10uNm clipping selectors label events only; this analyzer grants no physical admission.

## Input identity

- replay_report: `f10ff4c90d6ff7b7f8ebf0165877b6a3aa4b7cfbe4379e1e8322b10e28349cf1` — `/home/orionh/HEXAPOD_runs/mkii_motion_recovery_v1/prefix_replay_001/hexapod-fourbar-diagnose-20260906T032844Z-d73ff362/report.json`
- campaign008_report: `c57a2f565d11b62213cca043aea4b26bdaa18d56cc2ad71848cf322793de751e` — `/home/orionh/HEXAPOD_runs/mkii_coordination_control_v1/campaigns/fourbar-campaign-20260906T021507Z-9d668815/nominal/hexapod-fourbar-validate-20260906T021740Z-96e3036c/report.json`
- kinematics: `0ab3acc43d4c6c77c93a996858b8f928a1f157a533c8967c5ac84b77222f7f61` — `/home/orionh/HEXAPOD_runs/mkii_motion_recovery_v1/source/configs/mkii_fourbar_v3_kinematics.json`

Every NPZ name, hash, sample interval and column layout is verified; the JSON records every file hash and exact event neighborhood.
