# Complete D6 comparison: failed physical checks

Source0d1ceab;8 environments ×200 standing+700 driven control steps,
14,400 physics samples per environment. Host trace integrity and completion
passed. Physical admission is false: pin separation0.5055766mm,
passive-angle residual0.0289924rad, raw demand42.0734N·m and applied5.5N·m.
There were no non-foot contacts or resets. The body remained above0.1324m.

The primary report is preserved byte-for-byte. `summary.json` summarizes the
verified NPZ event analysis. Full NPZ files and the unabridged event windows
remain at `/home/orionh/HEXAPOD_runs/mkii_fourbar_diagnostics_v1/runs/hexapod-fourbar-diagnose-20260905T072121Z-16d523c4/`
and the sibling `analysis/d6_groups_20260905T072121Z/` directory.

`contact_geometry.json` independently reconstructs sphere-pad minima from the
recorded body poses (native XYZW), hashed URDF and terrain origin. Thirteen
environment-samples report zero loaded feet; initial occurrences have pad
minima within approximately−0.0002 to+0.003mm of the nominal plane while the
body moves downward. This is force loss very near contact, not proof of a
large ballistic flight. At the largest pin error, LR's pad is6.085mm above
its ground plane; LF,RF,RM remain loaded. That peak is not a whole-body fall.
Cached and direct joint data agree exactly throughout the trace.

Neither the completed trace nor these offline interpretations admit PPO.
The v5 native bilateral coupling is a separate candidate with unchanged live
closure, support and torque-envelope gates.
