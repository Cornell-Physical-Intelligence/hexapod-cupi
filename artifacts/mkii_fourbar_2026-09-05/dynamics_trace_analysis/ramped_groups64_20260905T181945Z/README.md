# Ramped targets, 64/1 group diagnostic

This diagnostic completed with no physical-gate errors. Every group direction
passes, and the saved force-input targets confirm delivery over all 16 physics
updates. The diagnostic deliberately retains `pass: false` and does not grant
training admission; the full qualification is a separate run.

Run `hexapod-fourbar-diagnose-20260905T181945Z-ec78eb1e` used source `9cd8d4c`,
contract `c53071afdf320f6a9d6f91de09ddc6102de74df6f45a203e3a8166a8686548ca`,
the physical mimic v5 mechanism, TGS 64 position / 1 velocity iteration and
external forces each iteration. Eight environments completed 200 standing +
700 driven control steps: 14,400 physics updates per environment, at 1.25 ms
and 16 updates per 50 Hz control step.

| Measurement | Observed |
|---|---:|
| Minimum loaded feet after settling | 4 during motion; 6 in settled standing |
| Maximum C-pin gap | 25.0733 µm |
| Maximum passive-coordinate error | 0.000321329 rad |
| Maximum raw / applied torque | 4.29035 / 4.29035 N·m |
| Non-foot contacts / resets / motor-envelope excess | 0 / 0 / 0 |
| Minimum positive-minus-negative group response | +0.0123708 rad, right-rear lever |
| Minimum left-middle lever group response | +0.0250446 rad |

All 18 group responses exceed the unchanged +0.005 rad threshold. Each response
uses mean post-step position at the last physics update of control steps 90–99
for each command sign, then positive minus negative separately per environment.
The smallest coxa, femur and lever responses are +0.0605442, +0.0537496 and
+0.0123708 rad respectively. This group-only history omits the full
qualification's preceding individual-motor sequence.

## Actual command delivery

The offline check examined 2,073,600 target values across 7,200
environment/control intervals. For every interval, the processed endpoint remains
constant and the delivered position follows fractions 1/16 through 16/16 from
the previously delivered target. Every final target equals its endpoint exactly.
The largest difference from a float64 reconstruction of that line is
2.98023e-8 rad. The largest substep increment is 0.002500057 rad and the largest
control-endpoint increment is 0.040000021 rad, within float32 rounding of the
0.0025 / 0.04 rad limits. Velocity targets and feedforward torques are exactly zero.

For example, the first left-front lever reversal interval starts at
−0.537979364 rad and has endpoint −0.577979386 rad:

| Substep | Delivered target | Processed endpoint |
|---|---:|---:|
| 1 | −0.540479362 | −0.577979386 |
| 8 | −0.557979345 | −0.577979386 |
| 16 | −0.577979386 | −0.577979386 |

The next interval continues to −0.617979407 rad in another 16 increments.
[summary.json](summary.json) includes all 16 delivered values for these examples,
per-segment checks, named peak states, all per-environment group responses and
actuator readbacks. The nonzero delivered-target/endpoint difference is expected
during interpolation.

Cached and direct pre-step joint states match exactly across 3,456,000 values
each for position and velocity. The largest PD reconstruction residual is
1.19209e-7 N·m. These trace checks verify the simulated command filter and feedback
path; they do not establish hardware motor behavior or rough-terrain performance.

## Preserved evidence

[report.json](report.json), [supervisor.json](supervisor.json) and
[source.SHA256SUMS](source.SHA256SUMS) are unchanged copies from the run directory.
The source list's digest matches the supervisor metadata; all 239 contract-file
hashes match entries in that 1,150-file snapshot. The archive itself remains
remote, with its original metadata copied in the summary. Its checksum was not
independently recomputed by this analysis.

All eight NPZ hashes, shapes, named columns, finite values and contiguous sample
ranges were verified using `../analyze.py`. Large NPZ files remain at
`/home/orionh/HEXAPOD_runs/mkii_ramped_targets_v1/diagnostics/hexapod-fourbar-diagnose-20260905T181945Z-ec78eb1e/`.
Full analysis with ±12-sample peak windows remains under
`/home/orionh/HEXAPOD_runs/mkii_ramped_targets_v1/analysis/ramped_groups64_20260905T181945Z/`;
its digest is in the compact summary.

[summarize.py](summarize.py) reproduces the compact result from those original
files. Four focused [tests](test_summary.py) distinguish correct float32 ramps
from zero-order holds, delayed endpoints, changing endpoints, incorrect target
history, missing substeps and nonfinite inputs. All pass. Local file identities
are frozen in [SHA256SUMS](SHA256SUMS).

```sh
python3 summarize.py /path/to/report.json /path/to/analysis.json \
  /path/to/supervisor.json /path/to/source.SHA256SUMS /new/path/summary.json
python3 -m unittest discover -s . -p test_summary.py
```

Only CPU analysis and these artifact files were created for this review. No
GPU job, running source, shared coordination note, model or acceptance bound changed.
