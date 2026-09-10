# Matched 50-update direct PPO pilots

Both pilots completed 50 updates and strict reload. Both failed 48/48 final quiet trials. Stage 2 remains incomplete.

| Quantity | Curriculum | CAPS |
|---|---:|---:|
| Actual transitions | 1228800 | 1228800 |
| Learn plus final verification (s) | 96.9526140911039 | 96.78856236021966 |
| Training terminations | 7 | 11 |
| Training nonfoot environment-steps | 364 | 369 |
| Final quiet passing replicas | 0 | 0 |

| Final quiet metric, 48 replicas | Existing bound | Curriculum median / worst | CAPS median / worst |
|---|---:|---:|---:|
| max_planar_excursion_m | 0.01 | 0.0519917 / 1.88743 | 0.0504218 / 0.075108 |
| max_joint_velocity_rms_rad_s | 0.03 | 1.26592 / 1.3004 | 1.27354 / 1.29346 |
| max_target_step_abs_p95_rad_per_20ms | 0.002 | 0.04 / 0.04 | 0.04 / 0.04 |
| max_requested_torque_saturation_fraction | 0.005 | 0.219 / 0.25 | 0.222 / 0.264 |

These summaries do not hide per-replica failures: all 48 paired stop rows and all 12 directional changes are in report.json. Initial constant and initial stop evidence compare exactly between branches. Applied torque remains capped; requested saturation still fails. Raw SDK joint rates and interval-angle evidence remain separate.

CAPS is the proposed preview checkpoint because its worst measured final stop excursion is 75.1 mm versus 1.887 m for curriculum. Neither is qualified: even CAPS exceeds 10 mm drift, 0.03 rad/s raw joint RMS, .002 rad target-step and .5% requested-saturation bounds. A preview must identify 50 updates, 1× playback and Stage 2 incomplete.

Neither branch passed quiet. Preserve both; inspect source audit and per-step raw target/observation response before selecting a changed smoothness intervention. No loss-only or median-only promotion and no automatic unchanged continuation.

Latest checkpoints:

- curriculum: `32ce29dbd8639ee03fd2ce6d216da82919a65d3ba1e2ce50eb836d69580625de` (50 updates).
- caps: `ca0545760f81b8b8fb0cafde4b1ef89bfa5d8765a8b43b578e451f648a954415` (50 updates).
