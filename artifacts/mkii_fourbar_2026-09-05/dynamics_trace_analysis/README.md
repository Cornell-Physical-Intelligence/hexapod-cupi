# Offline analysis of physical four-bar diagnostic traces

[analyze.py](analyze.py) reads a genuine `hexapod.fourbar_diagnostic.v1` report
and its adjacent `trace_*.npz` files. It verifies SHA-256 hashes, numeric shape
and finite values, unique named columns, contiguous sample ranges, segment
lengths and report totals before producing numerical summaries. It refuses to
replace an existing output file. It does not manufacture a missing primary
report, run Isaac Sim, alter the asset or grant training admission.

```sh
python3 artifacts/mkii_fourbar_2026-09-05/dynamics_trace_analysis/analyze.py /path/to/report.json --out /path/to/new_analysis.json --window 8
python3 -m unittest discover -s artifacts/mkii_fourbar_2026-09-05/dynamics_trace_analysis -p 'test_*.py'
```

Python and NumPy are sufficient. The input report and all listed NPZ files must
be copied together before analysis. A genuine partial report can be analyzed
and remains explicitly labeled incomplete; a missing report is not analyzable
in this mode. Missing/truncated ranges cannot silently become completed data.

The output includes per-segment and global maxima/RMS for raw/applied torque,
P and D terms, pre/post joint velocities, finite-difference `dq/dt`, their
differences, cached versus direct-backend pre-state differences, and
`P+D+feedforward−raw demand`. It reports full closure-vector and axis-vector
components and norms, each leg's maxima, and relative C-pin point velocities.
Readback armature and gains are mapped by their recorded joint names, including
the motor model's own order; an absent order is flagged rather than guessed.

Peak demand, applied torque, full closure gap and axis error each have an exact
environment/sample/segment/control-step context and a snapshot of every named
field. The associated leg and foot net forces are explicit. A ±8-physics-sample
window is included by default and can cross segment boundaries; its value arrays
use the output's `trace_columns` vector. Bounds at the start/end of the run are
reported rather than padded. `--window 0` keeps only the peak samples.

Interpretation requires care:

- P, D, feedforward and delivered effort use the **pre-step** state; the post
  state is measured after physics. Do not reconstruct those efforts from the
  post-step velocity.
- Direct backend getters run after PD has read its cached input, before physics.
  Their difference is measured evidence, not proof of a cache fault by itself.
- PhysX's position and velocity corrections can make reported velocity differ
  from position finite differences. The analyzer reports both without claiming
  that disagreement necessarily indicates a defect.
- Solver drive gains may correctly be zero while the explicit motor model has
  nonzero PD gains. Both readbacks are retained separately.
- The trace contains six foot net forces and body velocities. It does not
  contain all-body contact forces or individual contact impulses.

Seven synthetic tests exercise cross-segment peak windows, multiple environments,
shuffled joint/column orders, readback mapping, pre/post timing, PD reconstruction,
hash tampering, missing/overlapping samples, invalid columns/nonfinite data and
explicitly incomplete reports. No cause is assigned from a maximum alone.
