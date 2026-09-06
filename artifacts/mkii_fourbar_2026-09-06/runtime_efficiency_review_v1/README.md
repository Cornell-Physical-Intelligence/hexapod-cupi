# Physical-metrics efficiency review

Proposal only: no runtime, controller, sensor, asset, source contract, or gate was
edited. The executable candidate is the `proposed` function in
`cpu_batch_probe.py`. It operates on synthetic tensors and defaults to CPU. No
Spark profiling, CUDA benchmark, PhysX session, or robot training was performed.

**First candidate: batch the arithmetic inside each physical sample.**
`PhysicalMetrics.capture` (`isaaclab/validate_mkii_fourbar.py:174`) computes the
same 12 transformed pin offsets twice: once for point closure and once for pin
velocity. Its six-loop implementation performs 36 small matrix contractions;
12 passive velocity relations and 31 body-force norms also run through Python
loops. Cache the frame/body/joint index tensors once, using the actual runtime
names, then gather tensors with shapes `[N,6,2,3,3]` and `[N,12]`. Two batched
contractions produce the offsets and axes, with offsets reused for both pose
and velocity. Stack the 31 existing force tensors before their norm, preserving
the individual body's reduction dimension and order. The script implements
exactly this proposal; it does not combine sensor views. The immutable foot
mask at line 200 can also be constructed once, after validating sensor order.

Keep this function inside **every** existing `scene.update` callback. It batches
robots and linkage frames, never time. Keep all 16 substep rows, report fields,
float32 geometry, float64 squared sums, clearance calculations, finite checks,
and acceptance comparisons unchanged. In particular, retain `drain`'s sample
count check and existing accumulation order. The current drain already performs
one `.cpu().tolist()` per policy interval, not one transfer per substep. Deferring
it across policy intervals would delay the guard past automatic resets and is
not this proposal.

**Second candidate: reuse contact results within the completed transition.**
`env.py:174–201` computes foot classification/slip and nonfoot force counts.
Both `_get_dones` and `_get_rewards` call these functions. The captured installed
Isaac Lab `DirectRLEnv.step` calls dones and rewards consecutively, with no
physics step or reset between them; resets occur afterward (captured source
lines 467–474). A transition-local immutable contact snapshot can supply both.
Invalidate it at the next physical step and on every reset, including partial
resets and external reset paths. Preserve command sampling and reward/done
ordering. This would remove the duplicate Python/sensor-property accesses and
arithmetic, but has not been implemented or timed here. It affects 50 Hz
post-step work, so the 800 Hz metrics batch is the first candidate.

## Measured CPU evidence

`cpu_batch_probe_007.json` is the final external-script check; the preceding
numbered reports and matching `probe_source_NNN.py.txt` snapshots preserve the
incremental review. The final script was copied outside the checkout and used
`--source-dir` to read the same source without modifying it. Reports record
source/probe hashes, PyTorch/Python versions, dtype, device, random seeds, and
explicit permuted body/joint indices.

Forty randomized cases cover 32 and 512 rows, float32 and float64, actual CAD pin
frames, and independently permuted body/joint order. All tested finite outputs
match byte-for-byte; maximum float32 ULP difference is zero. A separate synthetic
zero-offset fixture tests the immediate float32 neighbours of `0.0001 m` and
`1 N`; threshold decisions match. NaN/Inf force classifications also match.
This establishes sampled CPU equivalence, not equivalence for every possible
state or for a different device/backend.

Representative single-thread Mac ARM64 / PyTorch 2.14 CPU medians from report
007 (seven repeats of 100 calls; arithmetic subset only):

| Environments | Current arithmetic | Batched arithmetic |
| --- | ---: | ---: |
| 32 | 0.540 ms | 0.068 ms |
| 512 | 1.260 ms | 0.538 ms |

The same subset has **793 → 58 ATen dispatches** per sample, including metadata
operations. Matrix multiply dispatches fall 36 → 2, vector norms 43 → 3, and
cross products 12 → 1. These are CPU-dispatch counts, **not measured CUDA kernel
counts**. Python dispatch and small tensor operations are plausible GPU overhead
sources, but these ratios cannot predict overall Spark training speed.

The benchmark excludes quaternion conversion, primitive clearance, motor-budget
updates, native sensor acquisition, remaining metric reductions, CPU drain,
rewards, and PhysX. In particular, all 31 contact-sensor objects still require
per-sample access. Their actual native acquisition cost and cache behaviour
were not measured; a sensor-view merge would require new binding/isolation
proofs and is not proposed during an active run. The RS05 model already batches
all 18 motors and retains its budget once per physical step; do not change its
numerical evaluation order as a performance shortcut.

## Reproduction and next gate

CPU execution, with a new output path:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python cpu_batch_probe.py \
  --source-dir /ABSOLUTE/FROZEN_SOURCE --report /ABSOLUTE/NEW_CPU_REPORT.json
```

Only when compute is authorized and the active job has finished, explicitly
select CUDA to test the actual Spark tensor backend:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python cpu_batch_probe.py \
  --source-dir /ABSOLUTE/FROZEN_SOURCE --device cuda:0 \
  --report /ABSOLUTE/NEW_CUDA_REPORT.json
```

CUDA timing synchronizes before and after each timed block; comparisons and
index setup run outside timed blocks. The script never starts a simulator,
claims GPU ownership, edits shared coordination, or starts this command itself.
It refuses report overwrite and preserves a failed comparison report. CUDA
equivalence and timing remain unmeasured. If adopted, the integrated metrics
code needs a new source identity, full-row comparison against the old capture,
every-substep/reset coverage tests, and fresh physical qualification. Any
floating-point or threshold difference must be explained before integration;
the current physics gates remain unchanged.
