# Batch128 standing preparation

Prepared locally for `diagnostic_batch128_001`; no native128 run or admission is
claimed. `admission_001` and all earlier inputs remain unchanged. The one-use
`prepare.py` verifies the frozen source016 tree, all 15 original pinned inputs,
and exact local counterparts before executing only the stdlib CPU preflight.
`cpu_preflight.json` records the actual successful command and returned identity;
it neither imports Isaac nor creates the proposed native output directory.

The binding changes the earlier batch32 recipe only in replica count, existing
frozen source identity and fresh output path. `root_review_complete=false` is
intentional. Source016's `env.py` and `env_config.py` match admitted source002
byte for byte. Its model, stance, geometry and physics configuration match the
actual one002 state and admission001; this permits reuse of one002 after a fresh
successful batch128. Full source identity still belongs to each actual attempt.

## Root dispatch

1. Review `binding.draft.json`. If accepted, serialize a fresh
   `bindings/batch128_001.json` with only `root_review_complete` changed to `true`,
   using two-space JSON indentation and one final newline. Its expected SHA-256
   is `40ca999173fc3ce93f40a43c8268c5ddd4fd450f8950e8c5bf0883845b98e7c8`.
   Keep this draft and every published binding unchanged.
2. Confirm previous allocation cleanup, the current exclusive reservation,
   producer descendants, both GPU locks and available GPU/host resources. Verify
   that the fresh output path does not exist. Root alone performs these steps.
3. Upload the reviewed binding to the exact remote path in `root_dispatch.json`.
   Reuse the already frozen remote source016; no source or guard edit is needed.
4. Run `host_translated_cpu_preflight_argv`, then `guard_preflight_argv` from
   `root_dispatch.json`, preserving both actual receipts. The former uses host
   asset paths, not container `/asset` paths. The latter verifies the complete
   pinned source, all remote inputs, supervisor and current reservation/resources.
5. Dispatch `native_dispatch_argv` under the existing owned lifecycle. The bound
   maximum is 1800 seconds. Preserve AppReady, all failed prefixes, terminal
   state, cleanup and post-exit resource/reservation receipts.

The controlled recipe is exactly 1000 controls, eight physics substeps each:
20 seconds at 50/400 Hz, with four seconds settling and 16 seconds scored quiet.
The 60-second episode timeout is unchanged from diagnostic batch32; it does not
extend the recording. Zero actions hold the declared walking neutral
`[0, -0.30, 0.40]` per leg through the existing 0.040 rad limiter and explicit
motor recurrence. No PPO, video or reset inside the trial is requested.

## Native result and raw verification required

All 128 replicas must pass; averages or a passing subset cannot admit the batch.
Recompute `score_diagnostic` from the exact source016 replay against every raw
chunk and compare the entire report with the native report. Verify complete
8000-step and 1000-control ordering/counters/timestamps, 128 named rows, native
joint/body order, finite state, exact preceding-state PD input, speed-dependent
torque cap and native actuation input, endpoint equality, one initial reset and
no truncations/terminations. Retain and audit all 8000 contact packets; no missing,
overlapping or exhausted contact-buffer ranges may be treated as complete.

Confirm native readbacks: 128 floating articulations, 2432 bodies, 2304 DOFs,
19584 detailed SDF shapes, 2432 body-floor sensor rows with one filter, exact
per-body masses (7.466088235 kg total each), canonical named limits and speed
limits, zero implicit drive/armature, authored/read-back solver attributes 32/0,
400 Hz external-force scene settings, 1 mm contact offsets, zero rest offsets,
friction/restitution 1/1/0, self-collision enabled and no native error events.
Verify both initial and terminal recipe readbacks and current source/input
hashes. Solver backend iteration introspection remains unavailable and must not
be relabeled as verified. Floor geometry is the unchanged source-declared 80 m
two-triangle mesh with its y=x seam; no new floor runtime readback is claimed by
this preparation.

Original standing gates remain unchanged: quiet planar excursion ≤0.01 m,
heading excursion ≤2 degrees, worst joint RMS velocity ≤0.03 rad/s, joint range
≤0.02 rad, target-step p95 ≤0.002 rad/20 ms and requested torque saturation
≤0.005. The 400 Hz audit also requires per-joint and mean post-settle requested
saturation ≤0.005, applied torque ≤1.60001 Nm, zero post-settle six-toe support
loss, zero classified nonfoot-contact events, non-toe floor clearance ≥−0.001 m,
plate height ≥0.055 m and native joint/rate bounds within the existing 2e−6
tolerance. No gate is removed because the batch is larger.

## Replica separation and finite floor

The diagnostic does not install `TrainingTask`'s moving-robot proximity guard.
Therefore independently inspect the recorded initial reset plus every one of
the 8000 native root samples, not just the 50 Hz endpoints or nominal grid.
Confirm exact `/Robot_000`…`/Robot_127` ordering and initial 2 m grid placement
(12 columns, x=0…22 m, y=0…20 m) within the existing reset tolerance.

`layout_geometry.json` derives a conservative all-pose body-centered radius
0.4708823300239754 m from each joint-path translation length plus the farthest
exact mesh vertex and contact skin. At each sample compute all 8128 unordered
root pairs: center distance minus twice this radius. Report the minimum, pair,
time and any non-positive gap; do not claim replica independence from spacing
alone. Also report the straight relative-root segment minimum between successive
400 Hz samples, including initial reset→first step. This sampled/swept audit is
not continuous collision detection or a proof of independent solver behavior.
The existing training guard's 0.20 m extra buffer can be reported separately; it
does not replace or redefine the original standing gates.

For each root sample, report `40 - abs(x) - radius` and
`40 - abs(y) - radius`. Both must stay positive to establish that the complete
conservative horizontal envelope remains inside the finite floor. Initial
planned sphere gap is 1.0582353399520492 m and minimum floor-edge margin is
17.529117669976024 m. These are geometry calculations, not native128 observations.
Different grid locations also cross the triangle seam; neither a pass nor a
failure alone attributes an effect to floating-point distance.

The existing batch32 first 800-substep chunk occupies 43,936,000 bytes when
expanded. Four-times scaling estimates 175,744,000 bytes per batch128 chunk,
excluding copies, contact JSON and native solver/GPU memory. Stream raw chunks
for analysis; native128 memory capacity and throughput remain unmeasured.

## Future admission002

Only after complete native acceptance and independent verification should root
create a new admission002. Retain one002's original hashes and substitute the
new actual batch128 state/report paths, hashes and `num_envs=128`; never alter
admission001. Write separate host and container path variants and verify each
using `require_admission` with the intended 128-replica configuration. Container
paths must use `/standing_one` and `/standing_batch` read-only mounts; host
preflight must use the actual remote paths. A subsequent 128-replica learner is
a fresh allocation: strict checkpoint configuration rejects ordinary resume of
a 32-replica checkpoint. Any weight transfer requires its own explicit reviewed
lineage. Successful standing would not qualify walking or Stage2.
