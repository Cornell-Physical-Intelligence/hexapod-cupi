# Stage 2C causal-probe record

These directories preserve the resolved configurations, evaluator payloads,
analysis, checksums, and best child from each recent causal probe. A "best
child" is the highest-ranked checkpoint from that probe; it is not necessarily
safe or promoted. The immutable current best remains:

`2026-08-25_23-44-14_accel_scale4_yaw160_trackguard_20260825T234500Z_seed86/model_2.pt`

SHA-256: `a66a1c83e5b0675ade73d05538ae57d81451568ab45d9e8bcfd2edc67bb2e69a`

At the common 0.040 rad/20 ms playback limiter and formal seed 60 screen, that
parent has zero falls, achieves 0.242 m/s at the 0.30 m/s command, has yaw-rate
RMSE 0.107/0.112/0.138 rad/s at 0.16/0.20/0.30 m/s, a worst moving deck score
of 1.032, worst-joint duty 0.238, 0.08 s burst, and 2.691 N*m peak demand.

## Recent results

| Probe | Isolated intervention | Screen | Best child result | Decision |
| --- | --- | --- | --- | --- |
| 15 | Full ground-wrench yaw moment `scale=-1`, `reference=2 N*m` | Diagnostic only: 6 s, 275 post-warmup samples | `model_2`: yaw 0.108/0.111/0.143, deck 1.045, peak 3.544 N*m | Reject; parent ranked first |
| 16 | Joint-torque slew `-0.01 -> -0.05` | Formal probe screen: 10 s, 475 samples | `model_10`: yaw 0.092/0.101/0.132, deck 1.043, burst 0.12 s, peak 3.629 N*m | Reject; high-speed yaw gain was too small and safety/stability regressed |
| 17 | Foot-slip penalty `-0.5 -> -3.0` | Formal probe screen: 10 s, 475 samples | `model_8`: yaw 0.096/0.103/0.138, deck 1.027, worst-joint duty 0.248 | Reject; yaw unchanged and motor duty regressed |
| 18 | Full ground-wrench yaw moment `scale=-2`, `reference=2 N*m` | Formal probe screen: 10 s, 475 samples | `model_8`: yaw 0.097/0.102/0.133, deck 1.027, burst 0.12 s | Reject; no gate-clearing tradeoff |
| 19 | Yaw-rate slew `scale=-4`, `reference=0.040 rad/s/step` | No valid training run | None; Kit stalled before AppReady and the fail-closed attempt was preserved | Invalid attempt; no policy evidence |
| 20 | Matched zero-intervention continuation, seed 99 | Formal probe screen: 10 s, 475 samples | `model_8`: yaw 0.098/0.102/0.134, deck 1.036, peak 3.344 N*m | Control only; reproduced Probe18 within about 1%, so the full ground-yaw term was not causal |
| 21 calibration A | Bilateral longitudinal contact moment `scale=-1`, `reference=0.50 N*m`, LR 0 | One 12,288 x 24 no-learning rollout | Raw mean/p50 1.654/1.800 N*m; bounded cost mean/p50 0.603/0.694; friction 5.127 N | Signal is live; recalibrate reference to measured p50 |
| 21 calibration B | Same no-learning collection, `reference=1.80 N*m` | One 12,288 x 24 no-learning rollout | Raw mean/p50 1.654/1.800 N*m; bounded cost mean/p50 0.355/0.400; friction 5.127 N | Pass; use `ref=1.80` for the causal training arm |
| 21 full attempt 0 | Bilateral longitudinal contact moment `scale=-2`, `reference=1.80 N*m`, seed 99 | No valid training run | Pre-AppReady timeout; no run/checkpoint; cleanup parser safely refused an incorrectly parsed identity | Invalid infrastructure attempt; artifact preserved and parser fixed |
| 21 full retry 1 | Same exact causal arm | No valid training run | Pre-AppReady stall; interrupted when an unrelated NSVA job claimed the GPU; fixed exact-ID cleanup succeeded | Invalid infrastructure attempt; arm remains scientifically untested |

Probe20's first `20260826T152600Z` launch timed out before AppReady. Its
container inspection, empty-GPU evidence, launcher metadata, and logs are kept
separately from the successful `retry1` evidence; it produced no candidate.

The first p50-telemetry calibration attempt (`20260826T170241Z`) exposed a
reset-batch indexing bug and produced no checkpoint. The fail-closed launcher
marked it invalid and preserved its traceback. The corrected retry and the
calibrated-reference collection both completed successfully.

The first full Probe21 attempt (`20260826T172645Z`) reached the 420-second
watchdog before AppReady. Docker emitted literal `\t` text from the inspect
template, so the launcher could not split the identity fields and conservatively
reported `identity-mismatch`; the exact known stalled container was then stopped
without deleting its evidence. The launchers now use a `|` delimiter and
`docker stop --timeout 30`. Retry 1 (`20260826T174200Z`) again remained before
AppReady, and was signaled only after the unrelated `/root/nsva_dl.sh` workload
spawned `validate_nsva.py` onto the GPU. It created no run or checkpoint; the
fixed cleanup validated its immutable container ID, stopped it, and proved it
absent. These are infrastructure failures, not negative policy results.

All formal probe screens use deterministic nominal playback at commands
`(0, 0, 0)`, `(0.16, 0, 0)`, `(0.20, 0, 0)`, and `(0.30, 0, 0)`, seed 60,
25 warmup steps, 500 requested steps, and the explicit 0.040 rad/20 ms final
target limiter. For the sharded screens recorded here, the independently
repeated parent reports and wrapper action-processing records are compared for
exact parsed-JSON equality before a fail-closed merger accepts the exact parent
plus `model_0.pt` through `model_11.pt` membership and ordering contract.

The acceptance goals remain yaw-rate RMSE at or below 0.080 rad/s at every
moving command, moving normalized deck composite at or below 1.000, zero
falls/timeouts, 0.30 m/s achieved forward velocity at least 0.240 m/s, and the
RS05 duty/burst/peak safety limits. None of these probe children replaces the
current-best checkpoint.
