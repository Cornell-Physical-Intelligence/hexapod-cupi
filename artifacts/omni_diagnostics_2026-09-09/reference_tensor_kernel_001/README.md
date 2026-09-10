# Batched geometry and partial reference-state kernel

This is a CPU-tested optimization boundary for the C serial study. It is **not**
a batched gait controller, a replacement observation schema, a physical admission,
or authorization to allocate 1,024 PPO environments. No main, production runtime,
frozen reference, Spark source, or GPU job was changed.

The existing scalar controller performs many tiny Torch/NumPy operations per
replica. The preceding profile in `../reference_batch_profile_001/README.md`
identified FK, IK, COM and dictionary validation/state conversion as dominant
costs. This successor preserves the scalar formulas and makes their replica
dimension a tensor dimension.

## Implemented boundary

- `tensor_kernel.py`: named FK/Jacobian/transforms, checked IK, exact mass-weighted
  COM, original quintic swing/landing sampling, and executable reference-knot
  P/V history with per-replica bounds, failure latching and explicit partial reset.
- `state_geometry.py`: a **233-value partial geometric state block** containing
  executable/residual P/V, anchors, preload, desired/measured frame geometry,
  both original-swing and landing polynomials, and initial joint preload.
  Its 215 fields shared with the frozen encoder are numerically compared before
  that encoder's final float32 cast. The remaining 18 values expose the stored
  initial joint preload. This block is not the actor's full 740-value input.
- `fixtures.py`: offline scalar fixture construction only. It deliberately uses
  dictionaries and CPU conversions and is excluded from kernel timings. A future
  vectorized controller must produce the structure of arrays directly on device.

The kernel accepts one explicit articulation joint order and maps it by name to
the URDF chains. Tests cover canonical/reversed order and all 18 cyclic shifts.
No order is inferred from array position. Actual soft limits are intersected with
the frozen geometry limits and retain the complete 0.02 rad residual margin.

All numeric dynamic checks return per-replica masks; malformed shapes, dtypes,
devices, names and static source identities raise. Invalid reference output is
NaN, while finite clipped IK coordinates remain labelled `q_checked` diagnostics.
Never send a diagnostic coordinate to the motor path without checking validity.
One failed row cannot contaminate or advance another row. Failure latches until a
fresh increasing episode counter and a stationary executed target pass reset.
A rejected selected reset disables that row; it cannot keep using stale history.

## Source and rate bindings

`source_contract.json` hashes every copied oracle/data input. Construction checks
these files once; each dynamic call still checks shape/type/device and its numeric
invariants. The copied source files are byte-identical to their frozen parents.

| Binding | Reference source | Lift |
| --- | --- | --- |
| `wave002_5mm` | `ef92745daf4c3bf544b6c6f9f65e18f19745d9e84811c8888a464312f864a56b` | 5 mm |
| `wave003_7mm` | `2f9c6e5ef5119b299a5e2983bfceff666fd9b3ff312c1156ac909221c01fde8d` | 7 mm |

These bindings are separate. The new wave004 horizontal-time parameterization
has additional state and is deliberately **not accepted** under either binding.
It needs a successor kernel/observation contract and new parity evidence.

The common `formal_004` profile reserves 1.75 rad/s for reference motion plus
0.25 rad/s for the residual, giving the existing 0.04 rad per 20 ms total budget.
The separately labelled `diagnostic_003` profile reserves 1.25 + 0.25 rad/s,
giving 0.03 rad per 20 ms. The latter is a diagnostic intervention, not a measured
motor speed limit. Neither profile changes the 6 rad/s² reference acceleration
budget or torque gates. Their results must not be pooled.

Float64 is required. This preserves the scalar oracle's numerical behavior;
float32 and CUDA/Spark performance remain untested. The encoder nominal follows
the admitted plan's float32 runtime values promoted to float64, while geometric
`q0` retains the original reference angles. A parity test caught this distinction.

## Verification and timings

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tmp/reference_tensor_kernel_001 -p test_tensor_kernel.py
```

Twelve tests pass. They include mixed stance/swing/landing parity against both
scalar references, exact COM, limb-isolated unreachable/limit/nonfinite masks,
original and landing polynomial P/V/A, same-frame rotation/translation covariance,
233-block state validity, source/lift incompatibility, stale time and partial reset,
quiet target preservation, distinct .03/.04 feasibility and immutable source names.
An AST check complements the numeric tests: the dynamic kernel contains no
`.cpu()`, `.numpy()`, `.item()`, `.tolist()`, or replica-count loop.

Local arm64 CPU, Torch float64, one thread; seven timed repetitions:

| Replicas | Frozen scalar FK+IK+COM | Batched FK+IK+COM | New 233 block, serial | New 233 block, batched |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.696 ms | 0.572 ms | 0.509 ms | 0.489 ms |
| 32 | 21.384 ms | 0.914 ms | 15.984 ms | 0.549 ms |
| 128 | 85.810 ms | 1.971 ms | 61.347 ms | 0.695 ms |
| 1024 | 692.552 ms | 13.669 ms | 500.459 ms | 2.404 ms |

The geometry baseline calls the original frozen scalar functions. The state-block
baseline calls this **new partial block** one replica at a time; it does not compare
a partial block against the old complete actor. Timings include dynamic masks but
exclude construction, fixture conversion, transfers, the contact state machine,
physics and PPO. These are not Spark/GPU predictions or training ETAs.

`TIMING_REPORT.json` contains samples, versions and timed source hashes. To rerun,
copy the bundle to a new directory and remove only the copied timing report before
executing `benchmark.py`; the original report is immutable.

## Next integration boundary

The next coherent independent step is a batched, stateful contact/reference update
that emits this tensor structure without per-replica dictionary/NumPy round trips.
It must preserve measured five-foot support, support polygon/COM margin, actual
body tracking, anchored-foot drift, flight/descent counters, landing/contact-gap
acceptance, command filtering, stop logic, failure latches and reset identity.
The `ReferenceKnots.advance(..., upstream_valid)` mask must carry all of these
checks; this geometry helper does not implement them or make them optional.

Then the remaining mode/counter/contact/stop/history observation blocks must be
ported and compared to the complete frozen encoder, including history-valid flags
and no stale/reset leakage. The existing 233 block cannot substitute for them.
Wave004's horizontal polynomial/time fraction also requires explicit extension.
Only after complete parity and device-resident integration should a small Isaac
throughput smoke be considered. Useful physical walking remains a separate gate.
