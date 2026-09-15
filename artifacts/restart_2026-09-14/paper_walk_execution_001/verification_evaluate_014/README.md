# Independent native evaluation014 verification

`RESULT_002.json` completes the independent CPU audit of all three actual
checkpoint220 trials: 24,400 native physics steps and 627,468 saved contact
patches. Forward tracking fails; the 20-second quiet trial and post-command stop
window pass every original numeric and native check. This is scoped diagnostic
progress, not Stage 2 qualification. All 96 required full-suite cases and ten
unselected learning probes remain missing; the 32-second long-quiet case was not
run here.

`BINDING_REVIEW.json` and `verify_binding.py` verify source018, checkpoint
`e608b454abbed24827810dbb91acf6c17fca677447e49e68cd6fc7e981c2de2a`,
the exact 19-body/18-joint/153-shape robot, unchanged physics relative to013,
readback/lifecycle, contact-completeness audit and owned-container cleanup.
All input hashes remain unchanged. The source018 runtime freeze is
`0baf8869effbee16253927363d41e315b43be27444493ac75e648801c7923c3a`.

## Recomputed native evidence

The artifact-only verifier follows the preceding013 method. It recomputes the
unchanged scorer, float32 PD demand, speed-dependent motor ceiling and actual
native torque input at all substeps, pre/post joint-state continuity and every
control endpoint. Every saved contact point is transformed through its recorded
body and shape frames, reclassified against the exact distal cap, and aggregated
with the native float32 force-vector arithmetic. The 1 N support predicate and
nonfoot classification reproduce exactly, including zero-force samples.

All three acquisitions are complete: 1,000/1,000/1,050 controls and
8,000/8,000/8,400 physics steps. No joint-position or SDK velocity bound violation,
termination, truncation, internal reset, active nonfoot event or applied-torque
cap breach occurs. Peak SDK joint speeds are 20.5015, 8.85218 and 20.5015 rad/s,
below the unchanged 50.2655 rad/s native bound. All applied torque stays within
the existing 1.60001 Nm numerical allowance. Recorded non-toe clearance and plate
height stay within their original bounds. The audit checks those saved clearance
channels; it does not transform every mesh vertex again.

Whole-trial requested torque peaks are 3.37568, 1.89924 and 3.37568 Nm, with actual
applied torque capped. All fixed scoring-window requested-saturation fractions
are zero. Whole-trial six-toe support is missing at 134, 360 and 164 substeps,
respectively; these startup and transient samples remain retained. Only the
existing quiet windows require continuous six-foot support. No threshold or
window is changed to obtain either pass.

## Same fixed windows, checkpoint200 to checkpoint220

Forward uses two seconds of settling and 18 seconds of measurement. Quiet uses
four seconds of settling and 16 seconds of measurement. Stop commands forward
for eight seconds, then zero, with the original two-second settling allowance
and 11-second quiet measurement window.

| Metric |013 |014 |
|---|---:|---:|
| Forward signed mean COM velocity, commanded +0.05 m/s |−0.0000157 |+0.00000831 |
| Forward planar error, limit 0.025 m/s |0.0500159 |0.0499917 |
| Quiet joint RMS, limit 0.03 rad/s |0.558420 |0.0113285 |
| Quiet joint range, limit 0.02 rad |0.110639 |0.000271439 |
| Quiet target-step p95, limit 0.002 rad/20 ms |0.0367862 |0.000111312 |
| Quiet missing-six-toe count, limit zero / 6,400 samples |6,400 |0 |
| Stop joint RMS, limit 0.03 rad/s |0.00596294 |0.00661340 |
| Stop joint range, limit 0.02 rad |0.00142009 |0.00582314 |
| Stop target-step p95, limit 0.002 rad/20 ms |0.0000921964 |0.0000922769 |
| Stop missing-six-toe count, limit zero / 4,400 samples |3 |0 |

Quiet now maintains six-foot support throughout its scored16 seconds, with the
smallest recorded per-toe force norm at least5.989 N. Stop maintains support
throughout its scored11 seconds, with the smallest per-toe norm at least5.668 N.
The forward scoring window has two missing-support samples; moving feet are
allowed, and its original failed bound remains planar tracking alone.

This is a passed quiet-recovery screen after the forward command. The robot did
not execute the requested forward walking beforehand, so the pass cannot be
called a demonstrated stop from successful walking. Forward remains effectively
stationary. The fixed-window raw action clipping fractions are61.11%,66.67%
and61.11%: passing quiet measurements does not establish healthy action range or
useful gait. Mean root heights in the quiet/stop windows are0.068424/0.069932 m;
this records an actual posture change, not a proven learning cause.

## Preserved cross-runtime arithmetic finding

`verify.py` and `RESULT.json` retain the initial strict metric-comparison failure.
Local NumPy2.5.2 and the recorded native runtime produce slightly different
percentile arithmetic. The successor `verify_002.py` explicitly reconstructs
float32 rank/interpolation for target motion and descriptive vertical-velocity
percentiles, matching their recorded values exactly. The quiet target percentile
difference was1.46e−11 rad; the stop target difference was7.28e−12 rad. Original
gate statuses were identical before and after this artifact-only comparison.

One stop-heading scalar is not bit-exact across runtimes: the native report has
0.0011543019209 degrees, local float32 scoring gives0.0011474717176 degrees and
an independent float64 `math.atan2` recomputation gives0.0011522648005 degrees.
The native/local difference is6.83e−6 degrees. All three values independently
pass the original2-degree gate. No compatibility tolerance replaces that gate,
no native scalar is overwritten, and this artifact does not claim bitwise
reproduction of that heading metric. All other recomputed metrics match after
the explicitly recorded percentile interpolation. Every gate verdict matches.

These files modify no maintained scorer, runtime, source freeze, policy, raw
trace, registry or qualification requirement. The root-owned analyzer and video
review remain separate. `SHA256.json` seals this directory and excludes itself.
