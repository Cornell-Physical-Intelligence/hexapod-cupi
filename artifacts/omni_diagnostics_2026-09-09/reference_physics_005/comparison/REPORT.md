# Actual 005: mixed solver comparison, no promotion

All 32 original standing checks passed, but the unchanged quiet-hold review passed **30/32**, compared with 32/32 in source 004. Source 005 is therefore **not promoted**. Its single physical change was enabling external forces on every TGS position iteration; TGS remained 16 position / 4 velocity iterations. No wave or PPO ran.

The comparison uses all 32 matching environments over control boundaries 200–1000 (16 seconds). For both link and COM frames, all four reported integrators improved 26 environments and worsened six. Every source 005 discrepancy is below the unchanged 5 mm diagnostic threshold. The worst link discrepancy is 2.935324 mm with 400 Hz trapezoid integration and 2.929059 mm with the original 50 Hz last-substep integration, compared with 9.705556 / 9.862308 mm in source 004. Both full 32-row arrays and all quadratures remain in `matched_comparison.json`; no favorable subset is selected.

The quiet failure is specific: environments 6 and 29, joint `revolute_2_1`, have reported velocity RMS 0.03507035 and 0.03435949 rad/s, exceeding the unchanged 0.03 bound. Their actual sampled joint-position ranges are only 0.00056219 and 0.00092697 rad. Reported velocity integrated over the sampled 15.98-second interval is +0.56039/+0.54904 rad, while net position changes are −0.000527/−0.000915 rad. This is consistent with a persistent reported joint-velocity bias. The 50 Hz position differences cannot exclude unobserved fast motion and do not replace the failed raw-velocity criterion.

All other unchanged quiet bounds pass. Worst planar excursion is 0.269666 mm, heading excursion 0.010396 degrees, joint position range 0.00448132 rad, and target-step p95 is zero. No reset, truncation, post-settle saturation or nonfoot contact occurred. The settled 400 Hz maximum requested torque is 1.48506594 Nm; the full-run applied maximum is 1.600000024 Nm.

Startup remains separate and unqualified: the initial pre-physics actuator buffer reads 63.7035446 Nm (environment 14, `revolute_2_6`) with the applied value clipped to −1.600000024 Nm. The largest actually observed post-update settling demand is 3.9618220 Nm at 40 ms (environment 8, `revolute_1_4`). These are inside the existing settling exclusion; they must not be omitted from an all-time torque or hardware-startup claim.

The source-bound option was read back as true on the actual physics scene; stabilization remained false and solver TGS. The remote audit verifies 20 raw payloads, 926 source files and 550 asset files unchanged, the owned container absent and pause 031 restored. This is a completed diagnostic with a failed separate quiet qualification, not a successful walking result. Historical older-actor testing of this option remains negative and unchanged.

Root authorized another separately versioned standing-only discriminator, source 006: retain this true option and change only velocity iterations 4→1. It must retain original quiet scoring, actual iteration readback, all 32 environments and all raw measurements. No combined walking trial is admitted by this result.

Reproduce the comparison without a simulator or GPU:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B compare.py --baseline-run /path/to/reference_physics_results_004/run --candidate-run /path/to/reference_physics_results_005/run --output /fresh/comparison
```

The frozen original quiet function is extracted unchanged from `frozen_quiet_scorer.py`. Its exact 64 rows, matched integration comparisons and failure details were reproduced. The underlying float32 simulation data is accumulated in float64 for independent integration; insignificant last digits can differ from runtime float32 reductions. The complete original runtime reports and their verdicts remain authoritative artifacts and are not rewritten.
