# Source005 standing32: independent channel analysis preparation

The actual standing32_005 acquisition completed 8,000 steps, but only 10/32 robots passed the combined/physical gates and 24/32 passed quiet. Twenty-two robots missed six-toe support at 1–8 post-settle substeps. This preparation retains that original rejection and keeps PPO disabled. It is a CPU analyzer, not another native experiment or an acceptance-rule change.

The exact audited invocation is `c5397bd5a8dd4640899ad3dec2c30f47`; audit SHA is `8834ba3d90b713266f4a2dc270e07a7977ca154376862148fac8329147bad4cb`. Source005 is the unchanged 109-file `c87f2d340c935cd947c7aed3f3dcd7c9a9965b6e19d6726546abb560b27ab131` with solver32/0. The audit inventories all 39 original files, 5,019,294,627 bytes. Only explicitly consumed originals are opened and hashed by this analyzer, before and after analysis; unconsumed output hashes remain referenced to the exact full terminal audit. No multiGB raw download or duplicate is required.

## Measurements

The first bounded chunk pass identifies every post-settle missing-foot event and replays the exact >1 N resultant support flags and original event counts. It computes SDK and actual pre/post angle-increment statistics over all 6,400 post-settle 400 Hz rows, retaining the original report's 50 Hz quiet scores separately. An angle mismatch or source-equivalent toe aggregate discrepancy rejects with exact counter, environment, component and values.

For every event, the analyzer captures the preceding, current and following step, plus the minimal extra derivative context. It retains named joint positions, native SDK rates, actual angle increments, applied torques, tibia poses and velocities, every floor-body force vector, exact raw classified patches and the original patch-derived toe vectors. The floor matrix is indexed by exact `contact_view.sensor_paths`, not articulation body order. Link velocities, masses and local COM offsets use the recorded articulation names. Each source toe product is rounded to FP32 before ordered FP64 accumulation, preserving the arithmetic proven in analyzer002; no tolerance is added.

The floor matrix and detailed patches are separate getter copies after the same physics step, but can share a contact backend. The floor matrix includes all contacts on a tibia, so it cannot qualify a toe by itself. Disagreement is reported in vectors and norms, never automatically classified as reporting loss. Raw patch count, inactive slots, force sign, exact zero patterns and every original patch record near events remain available for interpretation.

The complete contact stream is hashed and all 8,000 row clocks/counters verified. Every recorded used buffer index is counted and checked against capacity; global count distributions and event-row counts are retained. Original native starts/counts arrays and global solver buffer occupancy were not exported. Counts reconstructed from preserved patches cannot prove the completeness of those unobserved buffers or rule out upstream reporting loss.

Whole-robot COM velocity and position use all 19 observed link masses, local COM offsets, link poses and link COM velocities. Backward differences give native-momentum acceleration and a diagnostic force balance with gravity; pose second differences are emitted where the neighboring context exists. The body-link acceleration is also retained. These derived values do not identify a solver cause: native velocity may carry the same bias, position differencing amplifies FP32 grid quantization, and unexported constraints can affect the comparison.

The source003 comparison is an exact selected-field extract from the published actual analyzer002 result `194e926651c3ac8f3c0aff6ebaef44cdb4ca39053b75a9898b51fe8259055870`. It contrasts per-environment support/quiet outcomes and SDK versus angle-rate statistics. Source003 has no new floor/link channels, and changing solver settings/instrumentation and batch state prevents a controlled causal inference from that comparison.

## Root execution

Run on Spark's host CPU with the standard library. The script imports no NumPy, simulator, Docker or GPU API, starts no subprocess and writes no files itself. Root redirects stdout/stderr to a fresh analysis directory outside the original run:

```sh
/usr/bin/python3 -B -S /path/to/analyzer/analyze_remote.py \
  --run /home/orionh/HEXAPOD_runs/canonical_direct_20260910/native_standing32_005 \
  --source /home/orionh/HEXAPOD_runs/canonical_direct_20260910/standing_source_005 \
  --audit /path/to/analyzer/inputs/audit.json
```

The runtime uses the byte-identical existing stdlib NPZ reader with a 32 MiB per-member bound, one numeric chunk at a time and a 32 MiB JSONL row bound. It stores only event-neighborhood details, compact global counts and one chunk's chosen arrays. Input paths are explicit; no remote state is altered. The original 1 N support, SDK quiet, torque, geometry, servo and physical gates remain unchanged.

## Validation and publication

Eight focused tests pass, including full synthetic 8,000-row analysis with an actual missing-foot event and intentionally disagreeing floor channel, shuffled sensor order, source-exact FP32 multiplication, inactive raw records, duplicate-index rejection, COM rotation and exact current native6 numeric-channel replay. The standalone `-B -S --help` import passes. These are CPU tests; no actual standing32 analysis result is claimed until root executes this frozen program and verifies its output.

The exact published source003 summary is hash-pinned; `inputs/SOURCE005_FREEZE_SHA256.json` and the prior analyzer freeze retain provenance. Root owns eventual publication and the required central record/Markdown under `docs/PROJECT_SITE.md`. This tmp-only preparation changes no tracked source, native session, admission or GPU allocation.
