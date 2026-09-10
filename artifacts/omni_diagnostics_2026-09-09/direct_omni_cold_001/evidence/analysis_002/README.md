# Cold direct-PPO diagnostic comparison

A completed diagnostic is not Stage 2 qualification or training admission. The original checkpoint is unchanged. The historical 0.03 and fresh 0.04 rad/20 ms profiles are labeled separately.

## historical_diagnostic003

Each row includes all four replicas. Requested and applied torque are different quantities. Post-settle statistics exclude each episode’s first two seconds; all-window and reset statistics remain in report.json and the original diagnostic.

| Scenario | Terminations | Mean forward / left (m/s) | Mean yaw (rad/s) | Planar error (m/s) | Requested max (N m) | Saturated samples |
|---|---:|---:|---:|---:|---:|---:|
| stand | 0 | 0.0040 / 0.0157 | -0.0121 | 0.0337 | 5.732 | 5.75% |
| forward | 0 | 0.0737 / 0.0211 | -0.0171 | 0.0458 | 5.705 | 5.91% |
| reverse | 0 | -0.0722 / 0.0019 | 0.0048 | 0.0365 | 5.274 | 5.65% |
| left | 0 | 0.0059 / 0.0753 | -0.0329 | 0.0403 | 5.703 | 5.29% |
| right | 0 | 0.0002 / -0.0667 | 0.0206 | 0.0405 | 4.966 | 6.22% |
| forward_fast | 0 | 0.1158 / 0.0191 | -0.0706 | 0.0893 | 4.236 | 5.64% |
| turn_left | 0 | 0.0055 / 0.0113 | 0.1573 | 0.0317 | 5.325 | 5.74% |
| turn_right | 0 | -0.0033 / 0.0199 | -0.1799 | 0.0342 | 5.676 | 6.30% |
| arc_left | 0 | 0.0568 / 0.0194 | 0.0868 | 0.0581 | 5.306 | 7.64% |
| arc_right | 0 | 0.0750 / 0.0136 | -0.1279 | 0.0380 | 5.203 | 5.38% |
| strafe_arc | 0 | 0.0021 / 0.0786 | 0.1072 | 0.0387 | 5.064 | 5.34% |
| diagonal | 0 | 0.0421 / 0.0617 | -0.0205 | 0.0433 | 4.842 | 6.39% |

## cold_formal004

Each row includes all four replicas. Requested and applied torque are different quantities. Post-settle statistics exclude each episode’s first two seconds; all-window and reset statistics remain in report.json and the original diagnostic.

| Scenario | Terminations | Mean forward / left (m/s) | Mean yaw (rad/s) | Planar error (m/s) | Requested max (N m) | Saturated samples |
|---|---:|---:|---:|---:|---:|---:|
| stand | 0 | 0.0009 / 0.0148 | -0.0057 | 0.0379 | 6.335 | 10.70% |
| forward | 0 | 0.0879 / 0.0150 | 0.0014 | 0.0406 | 6.331 | 10.96% |
| reverse | 0 | -0.0875 / 0.0055 | -0.0015 | 0.0354 | 6.398 | 9.71% |
| left | 0 | 0.0053 / 0.0946 | -0.0354 | 0.0354 | 5.776 | 9.52% |
| right | 0 | 0.0013 / -0.0868 | 0.0111 | 0.0347 | 6.307 | 10.04% |
| forward_fast | 0 | 0.1538 / 0.0299 | -0.0568 | 0.0622 | 6.264 | 8.91% |
| turn_left | 0 | 0.0069 / 0.0086 | 0.1909 | 0.0317 | 6.966 | 9.96% |
| turn_right | 0 | -0.0061 / 0.0181 | -0.1976 | 0.0401 | 6.835 | 11.00% |
| arc_left | 0 | 0.0873 / 0.0173 | 0.1335 | 0.0420 | 6.638 | 11.41% |
| arc_right | 0 | 0.0927 / 0.0131 | -0.1526 | 0.0362 | 7.054 | 10.55% |
| strafe_arc | 0 | 0.0050 / 0.0954 | 0.1185 | 0.0330 | 6.124 | 9.72% |
| diagonal | 0 | 0.0537 / 0.0724 | -0.0154 | 0.0355 | 5.928 | 10.51% |

There is no moving-to-stop transition in these constant-command cases; stopping behavior is unmeasured.
The saved 50 Hz trace contains one replica per scenario. Raw reported joint rates and angle differences over one control interval remain separate. No source/input, cleanup, restoration, hardware, substep-torque, or all-direction qualification claim follows from this summary.
