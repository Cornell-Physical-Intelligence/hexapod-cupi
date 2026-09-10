# Batched measured-contact reference: wave004 parity prototype

This successor ports the frozen wave004 reference state machine to batched Torch
tensors. It has **no physics launcher, actor, training admission or checkpoint
compatibility**. Actual useful walking remains unqualified. The original scalar
sources, geometry kernel, actual traces and main checkout remain unchanged.

The exact scalar controller is
`a32b22eba03ff6fdd6b87d612c96173c519c7d2b212cb367a9b1f8f28fcd697d`,
from owner freeze
`dedd5a2deb21703fe1d8713fcc2c276339a58db3f14b9ab85fb65a3f2ee7e834`.
`source_contract.json` hashes the copied scalar oracle, actual004 trace and
unchanged predecessor kernel. Static source checks happen at construction;
dynamic numerical and contact checks still run each control step.

## Executable scope

`BatchWave004(joint_names_runtime, num_envs, device='cpu')` accepts a complete
typed batch of measured snapshots. The joint order is mapped by name. Call
`reset(snapshot, selected_bool_mask, fresh_episode_int64)` explicitly, then
`step(snapshot, requested_forward_left_yaw, dt=.02)`.

The input layout is declared by `SNAP_FLOAT` and `SNAP_BOOL`. Reset additionally
needs actual soft joint limits and executed target velocity. Floating inputs are
float64; original body-position storage precision is a separate explicit boolean
`position_is_float32`, set when packing the simulator's actual float32 snapshot.
The offline `pack` helper in the tests shows the contract; it is not a future
runtime adapter or evidence of zero-copy Isaac integration.

The implementation preserves:

- The original vertical swing clock, separate 0.80-duration horizontal quintic,
  original endpoint and serialized polynomial coefficients. Landing starts with
  the current combined P/V/A and retains the original swing for endpoint audit.
- Actual distal-contact/point validity, five-foot support and projected COM
  margin, body tracking, planted-foot drift, two-sample flight, measured lift,
  apex/descent, landing endpoint/excursion limits, provisional contact loss and
  post-blend three-sample contact confirmation. The batch hull calculation uses
  candidate support edges and is checked against the scalar SciPy hull margin.
- Command derating/filter state, no new liftoff after a stop request, finite
  reference quiet hold, named reset preload, actual joint-limit intersection,
  0.02 rad residual margin and the original discrete reference P/V/A budgets.
- Per-row failure latching, finite diagnostic state, selected-row reset and
  fresh episode identity. Invalid reference targets are NaN with a false mask;
  they must never be executed. Other replicas keep their own state and time.

The current contract is the **formal .04 rad per20ms total profile**, with
1.75 rad/s and6 rad/s² reserved for this reference. It does not repurpose the
.03 diagnostic profile as a hardware speed limit. Physical torque/contact/reset
acceptance remains in the separate screen; this controller does not replace the
1.6 N·m admission gate or the residual execution path.

## Precision behavior discovered by actual-trace parity

The scalar reference copies the reset body's NumPy position dtype. Synthetic
fixtures start with float64 position, while the actual simulator reports float32.
Its in-place position integration therefore rounds each actual step to float32.
The same rounding occurs in every one of the198 future-prediction substeps.
Casting only the final summed displacement is not equivalent.

This port preserves that behavior with the observable `position_float32` state.
The command-filter predictions are evaluated analytically as a tensor sequence,
but a fixed198-sample scan preserves the exact position rounding recurrence for
all replicas. There is no per-environment loop or dynamic `.cpu()`, `.numpy()`,
`.item()` or `.tolist()` call. Removing the scan or changing storage precision
requires a separately versioned semantic comparison; neither was silently done.

## Verification

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tmp/reference_tensor_wave_004_001 -p test_batch_wave.py
```

Ten tests pass. The largest parity test runs790 controls across six asynchronous
rows with forward/reverse/strafe/yaw/arc commands, standing, stops and a selected
reset while other rows continue. It checks target P/V/A, phase, counters, command
filter, desired pose and both anchor sets against the frozen scalar controller.
Separate tests cover original/XY clocks, provisional landing, repeated contact
gaps, confirmed support, no stop liftoffs, mixed reset precision, preload and
inference-mode lifecycle, failure isolation and malformed/stale inputs.

The complete actual004 measured prefix is also replayed through both algorithms.
They agree and both reject the RF landing at source sample624 (12.50 s), code10:
the unchanged12mm original-endpoint test fails. This is a **counterfactual replay
of measured inputs**, not physical execution of the new horizontal timing.
The replay cannot establish improved walking or convert that failure into a pass.
The raw actual004 trace SHA is
`0b18528e8d537cce7c3591d407970bf81348fa34b09e95080d30c2b6c7c813ee`.

## Local CPU timing

One complete step from matched mixed stance/swing/landing states, float64 Torch,
one local arm64 CPU thread, median of seven repetitions:

| Replicas | Frozen scalar | Batched prototype | Ratio |
| ---: | ---: | ---: | ---: |
| 1 | 0.747 ms | 4.121 ms | 0.18× |
| 32 | 22.666 ms | 4.876 ms | 4.65× |
| 128 | 91.566 ms | 7.617 ms | 12.02× |

The one-row prototype is slower because it computes masked transition work and
the fixed prediction scan for every row, even when no liftoff is scheduled.
This explicit cost is preferable to a hidden serial per-replica loop, but it is
not optimized away or ignored. `TIMING_REPORT.json` and the cProfile files expose
the measured functions and source hashes.

Static construction, cloning, input packing/transfers, observation assembly,
physics and PPO are excluded. Batched output includes full typed state clones;
the scalar includes its normal dictionary output. These are local CPU timings,
not Spark/GPU throughput, rollout estimates or a training completion ETA.

## Remaining integration boundary

`STATE_SCHEMA.json` records the complete typed reference state and provenance.
It is **not an actor observation schema**. The old233-value geometry block cannot
represent the new horizontal coefficients/clock, and neither233 nor the old740
encoder can silently replace a complete new actor contract. A successor encoder
must include all phase/contact/landing/stop/precision state, executable residual
P/V, sensor history and valid/reset flags in body-relative coordinates, with
explicit simulator-truth versus future-estimator provenance.

Before any training-scale allocation: independently review this port, construct
that complete encoder, remove per-row host packing in an explicit runtime adapter,
and run a small device/throughput smoke. The scalar actual-trace and asynchronous
tests remain the oracle. Physical standing/walking, torque and quiet-state gates
remain independent prerequisites. No old checkpoint may be resumed into this
new reference/action/observation lineage.
