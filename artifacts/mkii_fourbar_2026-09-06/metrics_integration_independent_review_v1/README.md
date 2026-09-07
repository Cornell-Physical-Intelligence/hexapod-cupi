# Independent review of the proposed metrics optimization

**No runtime correctness defect found in the selected conservative patch.** An
independent CPU execution of its complete-row comparator passed eight tests and
800 paired 18-field physical-metric rows with identical bytes and zero measured
field differences. The exact installed scripted SDK quaternion functions were
used, with independent expected XYZW identity and positive 90-degree Z rotation
checks. No CUDA context, Isaac Sim instance, native sensor or GPU job was started
by this review. This is numerical/software evidence, not physical admission or
a performance measurement.

The reviewed candidate is
`../runtime_efficiency_integration_v1/proposed_validate_mkii_fourbar.py.txt`,
SHA-256 `811726701377ec9cc55672ac5df0f417bcb22eb112258bda259feed177becdb3`.
Its baseline is
`../runtime_efficiency_integration_v1/baseline_validate_mkii_fourbar.py.txt`,
SHA-256 `2007a7d84d9c51888a67df8bec7292924b2e57411d9312e6e5607c713617725c`.
The earlier fully batched proposal, SHA-256
`d3c162b2465ba30cd0c940619420407e8ee718bcf5c22010593773200d20a836`,
is a different candidate and is not approved by this review.

## Semantics inspected

- All 31 separate contact sensor `.data` properties remain read in dictionary
  order on every capture. Only their force norms are batched. The environment
  already requires each sensor's `body_names` to equal its single named body,
  so stacking the uniform `[N, 1, 3]` force tensors preserves the body dimension.
  This changes arithmetic scheduling around acquisition; it does not merge,
  cache, skip or rebind native sensor views.
- Closure positions and axes retain the original per-frame `einsum` input
  shapes, subtraction, norm and final stack/max order. Pin-velocity telemetry
  reuses exactly those computed offsets instead of recalculating them. World
  angular cross offset, world linear velocity and `v_C1 - v_C0` retain their
  physical meaning. COM data is not substituted for link-origin data.
- Passive indices are resolved from actual joint names. The fixed physical
  multipliers are exactly representable signs; the cached float32 tensor does
  not introduce rounding for this contract. Pin velocity remains `[N, 6, 3]`
  and passive residuals `[N, 12]`. Their squared-sum populations, float64 sum
  dtype and reduction layout/order remain unchanged.
- Only immutable indices are cached. Live pose, velocity, contact and motor
  telemetry remain sampled. Clearance, row order, finite checks, all acceptance
  thresholds, drainage and the actual training guard are unchanged. AST scope
  testing limits executable changes to `PhysicalMetrics.__init__` and
  `PhysicalMetrics.capture`.
- The scene-update hook still captures every physical update. The training
  guard still drains/checks the complete control interval before automatic
  reset can erase an early fault. A transient first-substep closure fault that
  disappears on the second substep is retained and rejected before reset.

## Independent execution

Command, from this worktree (the output was subsequently copied here):

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python \
  artifacts/mkii_fourbar_2026-09-06/runtime_efficiency_integration_v1/compare_complete_metrics.py \
  --source-dir /Users/andreboufama/Documents/CUPI/HEXAPOD/tmp/mkii_simulation_integrity \
  --device cpu \
  --report /tmp/hexapod_independent_native_complete_rows_8117267.json
```

`native_cpu_comparison.json` records the full test log, source identity and
hashes. It reports 9.601 seconds for the tests, all eight passing, no CUDA
initialization and unchanged functional source identity. This duration includes
synthetic fixture/testing work and is not an optimization speedup measurement.

The frozen validator copies import the surrounding current core contract:
**physics timestep 0.000625 s, decimation 32**, with 20 ms controls. They are not
self-contained copies of the older 800 Hz contract. The review caught and had
corrected obsolete 16-substep test expectations, including the need to test
both `D - 1` and `D + 1` captures rather than two undersampled intervals.

Coverage includes 32 and 512 synthetic environments, three randomized seeds,
independently permuted body/joint/sensor orders, complete float64 rows and
drained windows, threshold-neighbor float32 values at 0.1 mm closure and 1 N
contact, NaN and positive/negative infinity in contacts, repeated windows,
partial resets, incomplete/excess captures, and actual training-guard behavior.
Native quaternion provenance is the installed SDK math source SHA-256
`4ae6fe19d9b18c2747bf6546dc72f75cfe497ee2ba6046e95fabbeb33d958d33`.
The comparator verifies the copied function bodies/decorators against that
source and checks known rotations independently. An additional independent
extraction of only `matrix_from_quat` also produced exactly the expected
identity/Z-rotation matrices and equal `q`/`-q` matrices on CPU.

## Remaining evidence boundary

The same selected candidate still requires its complete-row CUDA comparison;
CPU equality does not certify different backend kernels. Synthetic sensors
establish read order/count but do not execute native PhysX retrieval or prove
its stream behavior. After numerical checks, an instrumented short physical
validation must retain sensor isolation and every-substep monitoring while
measuring actual cost. Neither the isolated arithmetic result nor the timing
of these tests admits PPO, alters physical acceptance thresholds, or establishes
that the robot remains stable at 1600 Hz.

No runtime source, URDF, USD, collision geometry or launcher was changed by
this review.
