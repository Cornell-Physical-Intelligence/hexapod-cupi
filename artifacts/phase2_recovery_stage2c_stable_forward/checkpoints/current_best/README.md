# Current Stage 2C checkpoint

This directory preserves the strongest formally screened forward-walking policy
as of 2026-08-25. It is a candidate for continued simulation research, not a
hardware-ready policy.

- Checkpoint: `model_2.pt`
- SHA-256: `a66a1c83e5b0675ade73d05538ae57d81451568ab45d9e8bcfd2edc67bb2e69a`
- Training run: `2026-08-25_23-44-14_accel_scale4_yaw160_trackguard_20260825T234500Z_seed86`
- Training host path: `/home/orionh/HEXAPOD/isaaclab/logs/rsl_rl/hexapod_robstride_phase2_recovery_stage2c_stable_forward_direct/2026-08-25_23-44-14_accel_scale4_yaw160_trackguard_20260825T234500Z_seed86/model_2.pt`
- Evaluation control setting: processed joint-target slew limit `0.040 rad / 20 ms`
- Evaluator seed: `60`, nominal dynamics, four environments spaced 8 m apart

Formal four-command measurements:

| Command | Achieved forward | Yaw RMSE | Normalized deck composite | Worst joint duty | Falls |
| --- | ---: | ---: | ---: | ---: | ---: |
| stand | 0.0003 m/s | 0.0002 rad/s | 0.013 | 0.000 | 0 |
| 0.16 m/s | 0.1599 m/s | 0.1070 rad/s | 0.765 | 0.206 | 0 |
| 0.20 m/s | 0.1962 m/s | 0.1117 rad/s | 0.862 | 0.238 | 0 |
| 0.30 m/s | 0.2420 m/s | 0.1378 rad/s | 1.032 | 0.223 | 0 |

The checkpoint meets the forward-speed target at 0.30 m/s and has zero falls,
safe peak/burst torque, and safe per-joint duty. It does **not** yet pass Stage 2C:
yaw RMSE must be at most 0.08 rad/s, and the moving deck composite must be at most
1.0. The associated full report is under
`../../probes/accel_probe14_contact_yaw_moment1_slew040_20260826T022500Z/evaluation/`;
that experiment re-screened this parent alongside eight rejected children.

`params/env.yaml` and `params/agent.yaml` are the resolved training configuration
captured by Isaac Lab/RSL-RL. Load the checkpoint with the matching task and policy
architecture; do not infer configuration from the filename alone.
