# Measured bottlenecks: profiler v2, attempt 001

**Simulation stepping is the largest measured cost; physical metric capture
is second.** The native CUDA trace confirms substantial articulation constraint
work. A frame-rate throttle is not supported as the main explanation.

The profile passed, the 32-environment short standing validator passed all 100
controls and 1,600 physics captures, and the supervisor removed the exact owned
container. This is the frozen **800 Hz, 64/16** baseline, source identity
`d6fbd6e9b5cf499dd8603613d6977ff226f443fb15f149bbe320fc11e15d5b54`;
it does not admit PPO or the new 1600 Hz source.

The remotely hashed originals and their measured reductions are recorded in
[measured_summary.json](measured_summary.json) and
[cuda_trace_summary.json](cuda_trace_summary.json). Root retrieves original
artifacts separately. The 82,890,837-byte trace's SHA-256 was independently
recomputed as `52d0cd58363f1c538468b161c5e44075066627e45ef7630b2c68f785b82b41fa`.

## Host timing, with profiling overhead exposed

Unprofiled control medians were **528.1 ms before** and **552.9 ms after** the
trace. The two traced controls averaged **648.3 ms** each, 17.3–22.8% slower
than those medians. Profiler start took 1.191 s and stop 0.594 s, both excluded
from control timing. Two samples cannot establish tail latency or a PPO ETA.

| Disjoint region | Host ms per traced control | Share of traced control wall time |
| --- | ---: | ---: |
| Simulation step, including native physics completion | 357.54 | 55.15% |
| Every physical metric capture, including lazy state/contact acquisition | 194.89 | 30.06% |
| Scene writes, including explicit actuator/budget processing | 35.29 | 5.44% |
| Original scene updates, including sensor timestamps | 25.09 | 3.87% |
| Dones plus rewards | 27.95 | 4.31% |
| Target scheduling | 4.31 | 0.66% |
| Metric drain, including any queued GPU wait | 0.23 | 0.04% |

These are inclusive **host** durations, not synchronized per-region GPU
execution times. Unlisted observations and loop/framework work account for the
small remainder. The actuator budget occupies 25.82 ms/control inside scene
writes, and all sensor acquisitions total 96.73 ms/control inside captures and
post-control work. Those nested values must not be added to the table again.

The capture region's measured exclusive host time after subtracting instrumented
lazy getters/acquisition is 107.79 ms/control. It still contains uninstrumented
Torch/native arithmetic and its dispatch/blocking costs; it is not a clean
single arithmetic-kernel measurement. The final drain is plainly not the main
cost in this run, and moving/removing its physical check is unjustified.

## Actual native solver kernels dominate observed GPU activity

The trace contains native PhysX/Warp kernels in addition to Torch operations,
despite those kernels often not appearing as GPU children of the CPU region in
Torch's printed summary. Inspecting only `hexapod/sim/step`'s reported CUDA-total
column would incorrectly miss most of this work.

| Observed kernel | Calls over two controls | Summed GPU duration |
| --- | ---: | ---: |
| `artiSolveInternalConstraintsTGS1T` | 2,560 | 221.90 ms |
| `artiSolveInternalTendonAndMimicJointConstraints1T` | 2,560 | 175.30 ms |
| `stepArticulation1TTGS` | 2,048 | 74.85 ms |
| `artiApplyTgsSubstepForces` | 2,048 | 54.27 ms |

These four kernels total **526.33 ms**, about 40.6% of the traced host interval
and 82.5% of the summed observed GPU activity. Their sum is about 73.6% of the
inclusive host simulation-step time. These fractions show real solver work;
they are not an exact exclusive attribution of all host waiting.

Observed kernel/memcpy/memset spans cover a union of **637.52 ms** within the
approximately 1,295.90 ms CPU control envelope; summed GPU durations are
638.29 ms. Stream 23 alone contains 542.57 ms of observed activity. This union
is not hardware occupancy/utilization and does not prove complete stream
coverage. `doAggPairCollisions` adds 31.60 ms over the two controls, well below
the four articulation kernels. No collision-filter or physical-setting change
is justified by that number alone.

## Contact acquisition counts are confirmed, but caching is a modest opportunity

Per ordinary control, the trace measures exactly:

- **496** sensor timestamp updates: 31 sensors × 16 physics steps.
- **595** sensor acquisitions and native net-force fetches: the 496 mandatory
  metric reads plus 99 done/reward accesses.
- **144 each** foot force-matrix, pose, and contact-data fetches, matching the
  repeated access path predicted from the SDK/source inspection.

The 595 acquisitions cost 96.73 ms/control inclusive. Removing all of that
would erase required sampling and is not an admissible optimization. Reusing
already acquired data only within the same physics timestamp can remove the
extra done/reward work while keeping every sensor's physical-step sample.
This is a smaller opportunity than the 55% simulation-step region; no speedup
is established until the change is measured with identical settings.

## Headless app-loop/throttle check

The profiled SDK source hashes match the inspected source copies:
`simulation_context.py` is
`819536a201b525774b4bc04d6bd9a9e3fc11397961b4881e9083eb4502ee958a`, and
`physx_manager.py` is
`79852394dafa48aeb5e9cdb39072dde27c9ab35f6b587a17ce931318ab44fd4a`.

`SimulationContext.step` calls `wait_for_playing`, then the physics-manager
step; rendering is conditional. The manager steps native `simulate` and
`fetch_results` directly. Its `wait_for_playing` returns immediately while the
timeline is playing and calls `app.update` only while paused. The profiler
recorded **zero simulation render calls**. No normal per-step sleep or app
frame-rate update appears in this headless code path.

The running application's resolved `rateLimitEnabled`/`maxfps` settings were
not captured, so their values are unknown. The native step/simulate/fetch
methods were read-only to the wrapper; separate host timing of those calls is
unavailable. Neither limitation erases the actual solver GPU kernels measured
above. There is no evidence-based reason to change frame-rate settings or run
another throttle probe before the prepared 1600 Hz campaign.

## Recommended next action

Continue the prepared 1600 Hz qualification/campaign with the reviewed metrics
optimization. Keep every physical sample, the 20 ms policy, motor behavior,
constraint settings, and acceptance bounds intact. The solver's iteration cost
is real; lowering it merely for speed would require a separately qualified
physical recipe and must not be slipped into this release.

Use the already planned scratch/full phases to measure throughput at 64 and
512 environments. More parallel environments may amortize launch overhead and
improve solver throughput, but this 32-environment trace does not establish
that speedup or predict full-job duration. If further software optimization is
needed, first remove repeated same-timestamp sensor reads and inspect the
remaining capture arithmetic, with exact data/reset semantics and a matched
profile afterward. Do not delay the now-prepared physical candidate for an
unmeasured frame-rate or sensor redesign.
