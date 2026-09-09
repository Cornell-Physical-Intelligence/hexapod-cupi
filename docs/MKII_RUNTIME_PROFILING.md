# MKII runtime profiling plan

The prior logs show material solver-related cost, but do not separate native
physics execution from Python, contact-query or tensor-kernel overhead. Obtain
actual PPO throughput after the first admitted scratch run before changing the
runtime or forecasting a 512-environment training duration. GPU utilization
alone provides neither that decomposition nor a reliable training ETA.

## Measured standing throughput

These two runs used the same source identity
`6d81d3117af6d39d703b68ecbeb6a410806a0f6931d0f059863e4081769e25c7`,
32 environments, the PD030 controller, 800 Hz physics and 50 Hz control.
The measurements span the flushed `FOURBAR_STANDING step=1/600` and
`step=600/600` markers: **599 completed control intervals**, excluding startup.

| Solver position/velocity iterations | First marker, UTC 2026-09-05 | Last marker, UTC 2026-09-05 | Elapsed | Per control interval | Per batched physics step |
| --- | --- | --- | ---: | ---: | ---: |
| 64/1 | 19:58:23.181790941 | 20:03:12.222987214 | 289.041196273 s | 0.482540 s | 30.159 ms |
| 128/1 | 19:49:00.988433178 | 19:56:34.375743703 | 453.387310525 s | 0.756907 s | 47.307 ms |

Exact repository-relative log paths:

```text
artifacts/mkii_fourbar_2026-09-05/pd030_standing_comparison/nominal/hexapod-fourbar-validate-20260905T195746Z-14ead46a/container.log
artifacts/mkii_fourbar_2026-09-05/pd030_standing_comparison/refined/hexapod-fourbar-validate-20260905T194825Z-e5e2e098/container.log
```

The refined run took approximately **1.569 times** as long per control interval.
The arithmetic and sensor code were the same, but the trajectories and contacts
also differed. This is evidence that solver-related work matters, not a measured
percentage attributable to the solver. These historical 64/1 and 128/1 results
do not predict the later 16-velocity-pass recipe, new velocity telemetry,
different environment counts, or PPO learning. Their physics-step figures are
wall time for an entire 32-environment batch, not per-environment execution time.

## What the current code establishes

`PhysicalMetrics.capture()` runs after every `scene.update()` physics substep.
It evaluates link poses, pin closure, primitive clearance, all 31 body-contact
sensors, motor bounds, finite state and velocity residuals. The capture path
contains many small tensor operations and Python loops over named joints and
sensors; it also constructs and uploads an unchanged foot mask every substep.
Those are candidates for measurement, not established bottlenecks.

`PhysicalMetrics.drain()` already batches device-to-host readback once per
16-substep control interval. Its current 16 × 18 float64 scalar array is about
2.3 KB. The `.cpu().tolist()` call also waits for preceding GPU work; timing
that call alone would incorrectly attribute queued physics/kernel execution to
data-transfer or Python cost. Native sensor/backend activity is not separately
timed by the existing reports.

## Next measurement, after admitted scratch PPO

Preserve the admitted source, asset, solver recipe and checkpoint identities.
First inspect successive `progress.json` records: `iterations_completed` is
local to the process, `last_iteration` is absolute, `collect_seconds` and
`learn_seconds` describe the latest iteration, and `elapsed_seconds` includes
the preceding logging/hash work. With 24 rollout steps per iteration, measured
collection throughput is `num_envs × 24 / collect_seconds`. Differences between
successive elapsed values and collection-plus-learning time expose additional
logging/hash/checkpoint overhead, including periodic saves carried into the
next interval. Separate warmup, startup, final checkpoint verification and
post-checkpoint inference from sustained learning throughput.

If that run shows collection dominates, use one separately identified, bounded
CPU/CUDA timeline recording of the same rollout to distinguish actuator and
`scene.write_data_to_sim`, native `sim.step`, `scene.update`/contact queries,
metric capture, metric drain, and policy/PPO work. Preserve every operation;
do not benchmark by removing guards or inserting a synchronization at every
substep. Host timing alone measures enqueue/wait behavior and cannot reliably
partition GPU execution. Record the profiling overhead and compare with the
unchanged run before interpreting the result.

Only then consider caching static masks/contact order or replacing the six
pin-frame and twelve passive-relation Python loops with batched tensor math.
Independent equality tests must compare every metric on identical inputs,
including nonfinite and boundary cases, followed by a controlled live comparison.
Any changed reduction order needs explicit numerical review. All **16 substep
captures, all sensors, existing float32/float64 precision, sample counts,
physical parameters, thresholds and policy-boundary rejection cadence must
remain**. No optimization is implemented by this note; runtime changes require
a new source identity and appropriate qualification before admission.
