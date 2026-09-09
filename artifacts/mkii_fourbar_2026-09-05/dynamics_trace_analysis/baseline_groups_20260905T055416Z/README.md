# Baseline simultaneous-motor diagnostic, 2026-09-05

The original revolute-v3 mechanism stayed inside the closure and motor bounds in
this diagnostic. Its physical gate failed because measured foot support briefly
disappeared during the simultaneous negative tibia-lever command. This does not
reproduce campaign 004's 0.345 mm closure error / 33.8 N·m raw demand, and it does
not admit training.

The unchanged primary [report.json](report.json) is from Spark run
`hexapod-fourbar-diagnose-20260905T055416Z-c071dfca`. It completed 200 standing +
700 driven control steps in 8 environments: 14,400 physics samples per environment,
at 1.25 ms / 16 substeps, TGS 64 position / 1 velocity iteration, external forces
each iteration. The driven sequence was six coxa together ±0.04 rad, six femurs
together ±0.04 rad, six levers together ±0.04 rad (100 steps per sign), then
100 steps at zero offset. It did not include the preceding 1,800 individual-motor
steps used by the full qualification sequence.

| Window | Samples per environment | Peak raw / applied torque | Largest closure gap | Minimum feet above 1 N |
|---|---:|---:|---:|---:|
| Startup | 640 | 1.7104 N·m | 8.851 µm | 0 |
| Settled standing | 2,560 | 0.6657 N·m | 1.306 µm | 6 |
| Driven | 11,200 | 5.1633 N·m | 29.266 µm | 0 |

The driven passive-coordinate error was 0.0017324 rad and the axis chord error
was below 9.10e-7. No non-foot contact or motor-envelope excess was reported.
The original 100 µm closure / 0.005 rad passive bounds remain unchanged.

## Reversal event

Sample indices below are zero-based. The negative lever segment begins at
sample 11,200. Ten environment-samples at five distinct physics times had no
foot force greater than 1 N, during the second control step. Their sample
intervals lie 20.00–28.75 ms after that segment began. Each contiguous per-environment
interval lasted one or two samples (1.25–2.5 ms).

Eight of these environment-samples had exactly zero reported force on all six
feet; the other two had maxima 0.193 N and 0.262 N. Changing the diagnostic count
threshold to 0.5 N or 2 N still gives ten. This threshold comparison does not
change the acceptance gate. The body was moving downward at 0.0645–0.0865 m/s
at these samples.

At sample 11,220, environment 2, the right-middle foot reported
`[0.0000430, 0, 320.1805] N`, essentially vertical. The same sample has the
maximum right-middle closure gap, 29.266 µm, with hinge-local components
`[-7.855, 28.193, -0.000442] µm`. Force multiplied by the 1.25 ms sample duration
is 0.400226 N·s in world +Z. This is an integral of the simulator's reported
force over one sample, not an independently measured physical impulse.

One sample later, right-middle femur demand was −5.1633 N·m:
P = −0.4520, D = −4.7113, feedforward = 0 N·m. Its pre-step velocity was
7.8521 rad/s and its actual position increment divided by dt was 7.5248 rad/s.
Applied torque equals demand and is inside its instantaneous 5.4635 N·m limit.
These observations show a contact transient followed by substantial damping
demand; concurrence alone does not establish causation.

The old trace contains link velocities but no body poses or foot positions.
It therefore cannot distinguish exact geometric flight from zero reported force
at pads very near the ground. Tibia link-origin velocity is not foot contact-point
velocity. Body poses and terrain origins have been added to the next diagnostic
format to resolve this limitation; they are not retroactively inferred here.

## Feedback and actuator checks

Cached/direct pre-step positions and velocities match exactly across 3,456,000
joint-value samples each. The largest `P + D + feedforward − raw demand`
residual is 2.3842e-7 N·m, with no residual outside the analyzer's float32 rounding
allowance. This trace provides no evidence of a stale joint-state cache or an
incorrect PD sum.

Readbacks show zero solver drive stiffness/damping on all 30 coordinates,
0.0007 kg·m² armature on all 18 motors, and zero armature on 12 passive coordinates.
The explicit motor model reports Kp = 30 and Kd ≈ 0.6; implicit/Newton actuator
flags are false. The scene reports external-force updates every iteration.
These are simulation configuration checks, not measured hardware parameters.

## Evidence and reproduction

[summary.json](summary.json) contains all segment/global metrics, named event
snapshots, readbacks, support-loss intervals and original trace hashes.
[support.json](support.json) adds force/velocity observations around the reversal.
The analysis was CPU-only using `/opt/wx/venv-models/bin/python` (NumPy 2.5.2).
All eight original NPZ hashes, contiguous sample ranges, shapes, named columns and
finite values were checked. Each original NPZ has 618 columns; no primary report
was fabricated or modified. Local [SHA256SUMS](SHA256SUMS) pins the saved files.

Original report and NPZ directory:
`/home/orionh/HEXAPOD_runs/mkii_fourbar_diagnostics_v1/runs/hexapod-fourbar-diagnose-20260905T055416Z-c071dfca/`.
The roughly 163 MB NPZ set remains there. The full analysis with ±8-sample event
windows is retained separately at
`/home/orionh/HEXAPOD_runs/mkii_fourbar_diagnostics_v1/analysis/baseline_groups_20260905T055416Z/analysis.json`;
its SHA-256 is recorded in `summary.json`.

From this repository, with the original report and adjacent NPZ files available:

```sh
python3 artifacts/mkii_fourbar_2026-09-05/dynamics_trace_analysis/analyze.py \
  /path/to/report.json --out /new/path/analysis.json --window 8
python3 artifacts/mkii_fourbar_2026-09-05/dynamics_trace_analysis/baseline_groups_20260905T055416Z/inspect_support.py \
  /path/to/report.json --out /new/path/support.json
```

The analyzer SHA-256 is
`da8323ccd488135ed40e4d52efc4f8f8256301163df7bcc7ada51965029729fa`.
The original report SHA-256 is
`01106818cc9c16fd65b287ae099753123e7c117fd202e652390a6dfeab42d146`.

Different motion history, 8 versus 32 environments, and instrumentation prevent
attributing the difference from campaign 004 to one cause. This result is for
revolute-v3 and makes no claim about the D6 candidate, full qualification, PPO,
rough terrain or hardware.
