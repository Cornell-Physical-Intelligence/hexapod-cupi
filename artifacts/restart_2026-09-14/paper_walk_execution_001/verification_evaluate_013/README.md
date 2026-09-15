# Independent native evaluation013 verification

`RESULT_002.json` completes verification of all three013 trials:24,400native physics
steps and497,495saved contact patches. The original failed verdicts are reproducible.
No case passes and no Stage2 qualification is granted.

The exact19-body/18-joint/153-shape robot, source017 acquisition, model/geometry
identity, no-error native lifecycle and owned-container cleanup are checked in
`BINDING_REVIEW.json`. All original input bytes remained unchanged. `verify_002.py`
recomputes original scores, every float32PD/derating/native input, pre/post state
continuity, all control endpoints, joint/speed bounds and every saved contact point's
body/mesh frame, toe/shaft classification and force aggregation. Mesh-clearance
channels are checked against the unchanged bounds; every mesh vertex was not
independently transformed again.

## Whole-trial physical outcome

All three trials retain all controls(1000,1000,1050) and physics steps(8000,8000,8400).
There are no joint-position or SDK joint-speed violations, falls/terminations,
truncations, internal resets, active nonfoot contacts or applied-torque cap breaches.
Peak SDK speeds are18.203,25.011 and18.203rad/s, below the native50.2655rad/s bound.
The1.6Nm cap remains intact. This clean acquisition does not imply successful motion
or quiet standing.

## Comparison with012 at the unchanged scoring windows

Forward uses2s settling then18s measurement. Quiet uses4s settling then16s measurement.
Stop commands forward for8s, then zero, with2s settling followed by11s measurement.

| Metric |012 |013 |
|---|---:|---:|
| Forward mean signed COM velocity(m/s), requested+0.05 |−0.0000472 |−0.0000157 |
| Quiet worst joint RMS(rad/s), limit0.03 |0.049253 |0.558420 |
| Quiet target-step p95(rad/20ms), limit0.002 |0.002582 |0.036786 |
| Quiet missing-six-toe samples(out of6400), limit0 |0 |6400 |
| Stop worst joint RMS(rad/s), limit0.03 |0.480717 |0.005963 |
| Stop target-step p95(rad/20ms), limit0.002 |0.007147 |0.0000922 |
| Stop missing-six-toe samples(out of4400), limit0 |38 |3 |
| Whole stop trial native speed violations |1 |0 |

013forward still fails tracking. The quiet trial additionally fails joint range and
requested saturation gates. The stop trial fails the six-toe-support gate: passing
its other quiet measurements is not a passed stop trial, and its preceding forward
motion never tracked the command.

Quiet's6400missing-support samples have a concrete pattern. LF is unsupported for
all16s; LR is missing6217samples andRM6251. RF andRR never lose support; LM loses
support4samples. The number of supporting toes is3for6120samples,4for220,5for56
and2for4; it is never6. This is predominantly a fixed three-foot support pattern,
not evidence of a successful alternating tripod gait.

Stop's three missing samples occur at10.865,10.8675 and17.3675s, all atRR. They form
two episodes lasting5ms and2.5ms. Each affected patch has zero normal force and a
valid nearly vertical normal; these are not the inactive zero-normal tuples seen
in earlier diagnostic failures. The classifier and zero-force aggregation exactly
reproduce the saved support flags. The underlying reason for zero force is not
isolated here, and the original zero-loss criterion remains unchanged.

The two zero-command windows settle into different postures: mean root heights are
about0.10167m for quiet and0.07083m after stop. Their histories also differ. This is
observed path dependence, not proof of its cause. Fixed-window raw action clipping
fractions are22.42% and61.11%, respectively; forward is55.56%.

## Preserved CPU compatibility finding

`verify.py` and `RESULT.json` preserve the initial strict dictionary-equality failure.
The only score differences were percentile arithmetic between localNumPy2.5 and the
recorded native runtime: quiet p95 differed1.12e−8rad, stop p95 differed2.18e−11rad.
Every gate status and bound matched. `verify_002.py` explicitly reproduces the
archived float32percentile rank/interpolation, obtaining the recorded values exactly.
It changes no maintained scorer, gate or input. `RESULT_002.json` is the successful
successor and records both values and the compatibility method.
