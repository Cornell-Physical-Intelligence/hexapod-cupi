# Device observation and history: wave004 / residual002

This is a CPU-verified, device-resident observation prototype, explicitly bound
to the frozen wave004 batched reference and residual002 controller. The actor
width is **838**, and the critic width is **841**. These replace no existing
checkpoint contract: 233, 495/498, 525/528 and 740/743 are incompatible, and the
prototype rejects every checkpoint load. It has no learner, physical admission
or production deployment claim. Wave005 has different flight semantics and an
unloading mode; it needs a separately versioned reference and observation schema.

## Executable interface

`DeviceTelemetry(env, named_layout, toe_points_local)` verifies the frozen C
URDF, exact 18 joints, 19 bodies, six named tibia sensors and all index maps at
construction. `capture` receives explicit device tensors for sample time,
pre-reset terminations/truncations, contact validity and contact age. It reads
the same installed SDK properties as the frozen NumPy telemetry oracle, without
per-environment loops or CPU tensor transfers. Unknown contact points retain a
raw NaN channel and an explicit validity mask; the encoded sentinel is zero,
never an invented foot anchor. Force-history NaNs fail their row rather than
silently becoming a false contact classification. Returned tensors are copies,
so later simulation updates cannot rewrite previous evidence.

The caller must supply an actual contact freshness contract and capture before
reset, using the original done predicate once. This adapter does not guess an
Isaac timestamp API or implement that hook. CPU fake-SDK parity is not proof of
installed Isaac device integration; a short real smoke remains required.

`ObservationBuilder(joint_names, num_envs, device)` binds float64 state and exact
source hashes once. Call `reset(selected_bool, fresh_episode_int64, time_s)`.
`build(packet)` accepts the measured batch, complete `BatchWave004.output()`,
residual002 output, next requested forward/left/yaw twist, world-up vector,
explicit episode and sample indices, and privileged raw simulator twist.
See `fixtures.py` for an executable synthetic example and `SPEC.json` for every
field and range. Malformed shape/dtype/source raises; dynamic numerical failures
invalidate only the affected row, emit NaN observations and require a fresh
selected reset. Duplicate identical samples are idempotent; changed duplicates,
stale samples, skipped steps and old episode IDs cannot append or reuse history.

All persistent command-filter, desired pose, original vertical swing, separate
horizontal polynomial, provisional landing, anchor/preload, support counters,
stop/quiet, precision and bounded residual P/V states are represented. World
points, vectors and polynomial coefficients are expressed relative to the
measured body frame. A simultaneous rigid change of the complete world frame
preserves the encoding; this does not qualify non-flat reference operation.
The history consists of five 63-value frames, five valid bits and episode age,
followed by the 517-value current block. Initial history is zero with false
flags; only the present frame is valid. Source/dynamic validity guards are
separate from learned features. Exact excluded diagnostic fields and rationale
are listed in the spec.

## Measurement provenance and velocity limitation

The packet explicitly declares `simulator_raw_instrumented`,
`synthetic_fixture`, or `future_estimator_unqualified`. Body/toe/contact state
comes from instrumentation; reference state is known ideal controller state.
These are not silently labeled deployable proprioception. Critic privileged
twist is the raw simulator-reported quantity, not established physical truth.
No observation noise is secretly sampled; a caller-supplied noisy measurement
is part of that exact sample and cannot be resampled on duplicate reads.

The observed source007 angular-rate inconsistency remains unresolved. Raw SDK
joint rates stay in the five-frame sensor history and in returned diagnostics.
An additional channel records **50 Hz interval-average joint-position rate**,
with explicit 0.02 s interval and validity. It is zero/invalid in the actor on
reset and NaN/invalid in auxiliary outputs when no valid interval exists. It
may alias substep motion and is not an instantaneous velocity replacement. No
quiet-standing metric, torque gate or simulation property is changed here.

## Reproduce

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s tmp/reference_policy_observation_004_001 -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  tmp/reference_policy_observation_004_001/benchmark.py
```

Sixteen tests cover the full 450-control mixed stance/swing/landing/stop sequence,
reversed joint order, inference-mode allocation, partial reset and failed-row
isolation, same-world-frame rotation/translation, hidden-state observability,
source/shape/freshness/NaN rejection, acceleration consistency, checkpoint
rejection, fake-SDK parity against the independent NumPy reader, unknown contact
points, raw biased velocity preservation and absence of dynamic host transfers.

The local one-thread CPU median at 128 replicas is about 0.61 ms for validated
packing and 2.21 ms for observation plus a new history update. Timing excludes
physics, reference state-machine execution and PPO; it is not a Spark/GPU
throughput estimate. `TIMING_REPORT.json` records all 1/32/128 measurements.
