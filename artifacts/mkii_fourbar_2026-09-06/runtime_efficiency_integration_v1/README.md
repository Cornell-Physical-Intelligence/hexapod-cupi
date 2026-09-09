# PhysicalMetrics conservative batching candidate

This is an unapplied, reviewable patch for `isaaclab/validate_mkii_fourbar.py`.
The selected file is `proposed_validate_mkii_fourbar.py.txt`, SHA-256
`811726701377ec9cc55672ac5df0f417bcb22eb112258bda259feed177becdb3`.
Its exact baseline is `baseline_validate_mkii_fourbar.py.txt`, SHA-256
`2007a7d84d9c51888a67df8bec7292924b2e57411d9312e6e5607c713617725c`.
`metrics_batching.patch` applies to that baseline; an isolated `git apply --check`
and apply reproduced the selected candidate byte for byte. Runtime files were
not changed. No release manifest or GPU job was created by this artifact task.

## What the patch preserves

Only `PhysicalMetrics.__init__` and `capture` change. Name-resolved cached indices
batch the twelve passive velocity relations. The six closure pairs retain each
original per-frame `einsum` shape, point/axis norm, and stacked maximum order.
Their exact offsets are reused for pin-velocity telemetry, avoiding the changed
matrix multiplication accumulation seen in the earlier fully batched prototype.
Pin velocities retain the N×6×3 layout; passive residuals retain N×12, so the
existing float64 squared-sum reduction order is unchanged.

All 31 native contact sensor `.data` properties are acquired once in the existing
dictionary order. Only the resulting vector norms are batched. Runtime sensor
initialization already requires each sensor to expose exactly its named body;
this is the uniform N×1×3 force shape required by stacking. Sensor binding,
update periods, and lazy acquisition are not changed. The 171 primitive clearance
calculation, all 18 row fields, drains, reporting, grading, and training guards are
unchanged. No samples are cached across time: every physics substep remains
captured, and the guard still drains and rejects before automatic reset.

The earlier `proposed_full_batch_validate_mkii_fourbar.py.txt` and
`metrics_full_batching.patch` are preserved proposals, **not the selected patch**.
The separate Spark arithmetic-subset comparison reported small float32 point and
pin-velocity differences at 512 rows for that strategy. The selected conservative
candidate deliberately retains the original closure contractions instead. Neither
the previous CPU dispatch counts nor the subset timings establish the speed of
this complete candidate or of Isaac simulation.

## Verification and limits

`sdk_cpu_comparison_002.json` reports 8 tests and 800 complete 18-field row
comparisons at 32 and 512 environments: every row is byte-equal, each field's
maximum absolute difference is zero, drained windows and guard outcomes match,
and no CUDA context was initialized. The source timing was 0.000625 seconds per
physics step with decimation 32; its functional identity was
`823b3b2172e80242187454ba02d7370a3263e75b7ffe939b5cf507d0c77a59d1`
and was unchanged from start to finish. The direct focused unittest suite also
passes all 8 tests after the module-cleanup fix.

Coverage includes independently permuted 31-body, 30-joint and sensor orders;
randomized CAD pin frames/poses; float32 neighbours of the 0.0001 m closure and
1 N contact thresholds; NaN/+Inf/−Inf force samples; repeated captures and
startup/settled/driven drains; partial resets; D−1/D+1 sample rejection; and an
early closure fault that must reject before reset. The exact threshold-neighbour
fixture uses identity/zero-offset frames to isolate comparison semantics. Actual
CAD offsets are covered by randomized complete-row comparisons and unchanged
per-frame contraction code; no claim is made that these finite cases prove all
possible states or replace physical admission.

The standalone comparator uses the exact installed Spark SDK's scripted XYZW
`matrix_from_quat`, plus the exact `quat_from_matrix` and its helper to construct
CAD-FK fixtures. It checks identity [0,0,0,1], positive 90-degree Z rotation, and
q versus −q against independent expected matrices before comparing rows.
The SDK source and extracted function bodies/decorators are verified against
`sdk_math_provenance.json`. This avoids the repository's older WXYZ test stub;
that stub is used only when running the test file directly without the wrapper.
No native Isaac initialization, contact retrieval, or PhysX stepping occurs in
these synthetic tests. Complete-row CUDA evidence is still required before
integration, followed by fresh physical qualification of the new source.

`sdk_cpu_comparison_001.json` is preserved, with its exact test source in
`test_complete_metrics_002.py.txt`. The first test source remains in
`test_complete_metrics_001.py.txt`. The current test restores only its own
`isaaclab.utils.math` module key, preserving unrelated modules lazily imported by
Torch and avoiding repeated C++ operator registration during fixture teardown.
Reports capture the actual imported physics timestep, decimation, full functional
source identity, backend versions, comparison bytes and SDK provenance. The
copied validator source alone is not a self-contained timing snapshot: it imports
core constants from the explicitly selected external source.

## Standalone staging

Stage these seven sibling files unchanged. The wrapper embeds its CAD-FK fixture
and does not need any `isaaclab/tests` modules in the frozen source archive.
The external source must supply its normal runtime, core, tools, URDF and configs.

| File | SHA-256 |
| --- | --- |
| `compare_complete_metrics.py` | `7fe26439b66645929c1ebbb5e87c8eedb6d510f2461ca9ced2449ef6adb53ca4` |
| `test_complete_metrics.py` | `dd2e6d06f20fb664d9680f6298f48cf37eb8c00d2e9a892f364bcb60ae41264d` |
| `baseline_validate_mkii_fourbar.py.txt` | `2007a7d84d9c51888a67df8bec7292924b2e57411d9312e6e5607c713617725c` |
| `proposed_validate_mkii_fourbar.py.txt` | `811726701377ec9cc55672ac5df0f417bcb22eb112258bda259feed177becdb3` |
| `installed_isaaclab_math.py.txt` | `4ae6fe19d9b18c2747bf6546dc72f75cfe497ee2ba6046e95fabbeb33d958d33` |
| `installed_matrix_from_quat.py` | `48eca4847b0ac648711b9fd61fab554315c488265385bcbdbee4f0be3f017380` |
| `sdk_math_provenance.json` | `083188f4f1d29fb4f0ae84e9ff79ba3a80454ae9ad095c684ed4370da4a61c28` |

CPU is the default; this command cannot automatically initialize CUDA:

```sh
python /absolute/bundle/compare_complete_metrics.py \
  --source-dir /absolute/frozen-source \
  --report /absolute/new-cpu-report.json
```

The same complete-row comparison runs on CUDA only when explicitly selected:

```sh
python /absolute/bundle/compare_complete_metrics.py \
  --source-dir /absolute/frozen-source \
  --baseline /absolute/bundle/baseline_validate_mkii_fourbar.py.txt \
  --candidate /absolute/bundle/proposed_validate_mkii_fourbar.py.txt \
  --device cuda:0 \
  --report /absolute/new-cuda-report.json
```

The report parent must already exist; existing reports cannot be overwritten.
CUDA synchronization brackets suite timing, but this diagnostic's wall time is
not a simulation performance benchmark. The comparator does not dispatch remote
jobs or create/clear shared reservations; the operator must run it within the
existing approved bounded GPU coordination flow. No fallback to CPU occurs if
explicit CUDA is unavailable. It fails if source bytes change during comparison.
