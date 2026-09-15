# Independent terminal audit of evaluate_007

The deterministic native rollout stopped at **13.38 seconds because
`lm_tibia_pitch` crossed its lower limit**. It did not trigger either fall
predicate. The final angle was −0.0877675414 rad against a −0.0872664601 rad
lower limit, an excess of 0.0005010813 rad versus the unchanged 0.000002 rad
tolerance. This occurred at control index 668, substep 7, native sequence 5351.
The held target was +0.1962603033 rad, inside the joint limits.

At termination the root height was 0.0890876353 m and tilt 1.49685 degrees.
Across the complete recorded prefix, minimum height was 0.0837488174 m and
maximum tilt 2.57982 degrees. The fall conditions remain height below 0.045 m
or tilt above 0.85 rad. The recorded terminal mask exactly matches the
independently recomputed joint-limit predicate, with no timeout or reset.

All 5,352 native samples, 669 control endpoints and 5,352 contact packets are
finite and continuous. Recorded native motor input and the bound canonical
servo recurrence agree exactly. All 13 file hashes in the two native receipts
matches. There was one joint-limit violation, no speed-limit or recorded
nonfoot contact event, and no native error. Applied torque stayed below the
1.60001 N·m gate; peak requested torque was 6.38352 N·m. The trial remains
failed and incomplete; this audit grants no walking or Stage 2 acceptance.

The actual video contains 334 frames; 335 successful captures include the
initialization capture. Its last recorded frame is at 13.36 seconds, before
the final odd-numbered control endpoint. No causal attribution to exploration
variance, contact resolution or solver behavior follows from this audit.

`audit.py` reads the preserved run, binds exact source/model/input hashes and
recomputes the terminal mask, stream integrity and pure motor equation on CPU.
It never imports a simulator or modifies the run. Reproduction requires a new
output path, for example:

```sh
.venv/bin/python -B artifacts/restart_2026-09-14/paper_walk_execution_001/terminal_audit_evaluate_007/audit.py --output /tmp/evaluate007-audit-reproduction.json
```

`audit.json` is the original output; `SHA256.json` binds this audit's files.
The audit checks the saved native contact classifications rather than rerunning
the geometric patch classifier. Original numeric and visual failures are intact.
